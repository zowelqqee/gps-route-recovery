"""Greedy one-active-route manager with event-aligned junction decisions.

Only the chosen child propagates. Siblings are immutable diagnostic metadata
unless bounded rollback explicitly activates one of them. Local probabilities
are recomputed from scratch at every junction and are never multiplied into a
whole-trip posterior.

Junction alignment is **event-centric**. A detected gyro turn is preserved as
an immutable physical observation (:class:`PhysicalTurn`); it is matched to a
reachable map junction despite odometer error, and its frozen signed angle
scores the outgoing edges. The gyro is only re-integrated around the crossing
time for junctions where no strong turn was detected - a genuine no-turn
measurement. See ``docs/EVENT_ALIGNED_JUNCTIONS.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from geotrace.coordinates import wrap_angle as _wrap
from geotrace.pacman_tracker.bend_anchor import match_curvature_event
from geotrace.pacman_tracker.config import BeamConfig, SinglePathConfig
from geotrace.pacman_tracker.manager import HypothesisManager
from geotrace.pacman_tracker.roadmap import RoadGeometry
from geotrace.pacman_tracker.state import HypothesisSet
from geotrace.pacman_tracker.turns import (
    HeadingIntegrator, TurnEvent, detect_turns, turn_log_likelihood)


def turn_kind(angle: float, threshold_deg: float = 25.0) -> str:
    threshold = math.radians(threshold_deg)
    if angle > threshold:
        return "left"
    if angle < -threshold:
        return "right"
    return "straight"


def local_probabilities(scores: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        return np.zeros(0)
    shifted = values - float(np.max(values))
    weights = np.exp(shifted)
    return weights / max(float(weights.sum()), 1e-300)


@dataclass
class PhysicalTurn:
    """An immutable physical turn observation from the gyro.

    ``signed_angle`` is frozen at detection and is never recomputed against a
    later crossing time. The event may wait, pending, to be associated with a
    map junction; the measurement has already happened.
    """

    id: int
    t_start: float
    t_peak: float
    t_end: float
    signed_angle: float
    peak_rate: float
    sigma_angle: float = math.radians(9.0)
    quality: float = 1.0
    soft: bool = False
    """A gentle turn below the strong-event threshold - real, but it could also
    be a lane change or a curved street, so it only steers a junction that has
    a matching moderate-angle successor and the decision is left provisional."""
    d_event: float = float("nan")
    ingested: bool = False
    consumed: bool = False
    expired: bool = False
    is_bend: bool = False
    """Classified as intra-edge road curvature, not a junction turn - excluded
    from junction matching, straight-commit revision, rollback and interval
    anchoring (see ``_classify_bends`` and ``bend_anchor.py``)."""
    matched_junction_offset: float = float("nan")
    matched_t: float = float("nan")
    matched_decision_index: int = -1

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "t_start": round(self.t_start, 2),
            "t_peak": round(self.t_peak, 2),
            "t_end": round(self.t_end, 2),
            "signed_angle_deg": round(math.degrees(self.signed_angle), 2),
            "turn_kind": turn_kind(self.signed_angle),
            "peak_rate_rads": round(self.peak_rate, 4),
            "soft": self.soft,
            "sigma_angle_deg": round(math.degrees(self.sigma_angle), 2),
            "d_event_m": (round(self.d_event, 2)
                          if math.isfinite(self.d_event) else None),
            "consumed": self.consumed,
            "expired": self.expired,
            "is_bend": self.is_bend,
            "matched_junction_offset_m": (round(self.matched_junction_offset, 2)
                                          if math.isfinite(self.matched_junction_offset)
                                          else None),
            "matched_t": (round(self.matched_t, 2)
                          if math.isfinite(self.matched_t) else None),
            "matched_decision_index": self.matched_decision_index,
        }


@dataclass
class TurnMatch:
    """A chosen (junction, successor) pairing for one pending decision."""

    turn: PhysicalTurn
    successors: list[int]
    turns_map: np.ndarray = field(repr=False)
    successor_index: int
    residual_m: float
    sigma_pos_m: float
    turn_ll: float
    distance_ll: float
    joint_score: float


def turn_kind_from_deg(angle_deg: float) -> str:
    return turn_kind(math.radians(angle_deg))


@dataclass
class DormantAlternative:
    edge: int
    map_turn_rad: float
    score: float
    probability: float
    activated: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "edge": self.edge,
            "map_turn_deg": round(math.degrees(self.map_turn_rad), 2),
            "turn_kind": turn_kind(self.map_turn_rad),
            "score": round(self.score, 4),
            "local_probability": round(self.probability, 6),
            "activated": self.activated,
        }


@dataclass
class SinglePathDecision:
    t_cross: float
    t_decision: float
    incoming_edge: int
    chosen_edge: int
    measured_turn_rad: float
    sigma_rad: float
    local_probability: float
    margin_log: float
    low_confidence: bool
    alternatives: list[DormantAlternative]
    parent_route: Any = field(repr=False)
    boundary_offset: float = field(repr=False)
    offset_bias: float = field(repr=False)
    distance_at_decision: float = 0.0
    distance_to_junction_at_turn_m: float = 0.0
    crossing_tolerance_m: float = 0.0
    rollback_from_edge: Optional[int] = None
    event_driven: bool = False
    event_id: Optional[int] = None
    event_residual_m: float = 0.0
    event_age_s: float = 0.0
    offset_bias_correction_m: float = 0.0
    soft_event: bool = False
    provisional: bool = False
    provisional_resolved: bool = False
    provisional_alternative_edge: Optional[int] = None
    provisional_outcome: str = ""
    _prov_events_seen: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "t_cross": round(self.t_cross, 2),
            "t_decision": round(self.t_decision, 2),
            "incoming_edge": self.incoming_edge,
            "chosen_edge": self.chosen_edge,
            "measured_turn_deg": round(math.degrees(self.measured_turn_rad), 2),
            "measured_turn_kind": turn_kind(self.measured_turn_rad),
            "sigma_deg": round(math.degrees(self.sigma_rad), 2),
            "local_probability": round(self.local_probability, 6),
            "margin_log": round(self.margin_log, 4),
            "low_confidence": self.low_confidence,
            "event_driven": self.event_driven,
            "soft_event": self.soft_event,
            "provisional": self.provisional,
            "provisional_resolved": self.provisional_resolved,
            "provisional_outcome": self.provisional_outcome,
            "provisional_alternative_edge": self.provisional_alternative_edge,
            "event_id": self.event_id,
            "event_residual_m": round(self.event_residual_m, 2),
            "event_age_s": round(self.event_age_s, 2),
            "offset_bias_correction_m": round(self.offset_bias_correction_m, 2),
            "distance_at_decision_m": round(self.distance_at_decision, 2),
            "distance_to_junction_at_turn_m": round(
                self.distance_to_junction_at_turn_m, 2),
            "crossing_tolerance_m": round(self.crossing_tolerance_m, 2),
            "rollback_from_edge": self.rollback_from_edge,
            "alternatives": [a.to_json() for a in self.alternatives],
        }


@dataclass
class _PendingDecision:
    t_cross: float
    window: float
    incoming_edge: int
    boundary_offset: float
    parent_route: Any
    offset_bias: float
    tolerance_m: float
    t_arrived: float = float("nan")
    match: Optional[TurnMatch] = None


class SinglePathManager(HypothesisManager):
    """Drop-in manager that keeps exactly one propagating route state."""

    def __init__(self, geometry: RoadGeometry, cfg: BeamConfig,
                 single_cfg: SinglePathConfig, integrator: HeadingIntegrator,
                 turn_events: Sequence[TurnEvent] = ()) -> None:
        super().__init__(geometry, cfg, integrator)
        self.single_cfg = single_cfg
        if not turn_events and integrator is not None and integrator.times.size:
            turn_events = detect_turns(integrator.times, integrator.rate, 0.0)
        self.turn_events = list(turn_events)
        self.turns: list[PhysicalTurn] = [
            PhysicalTurn(id=i, t_start=e.t_start, t_peak=e.t_peak, t_end=e.t_end,
                         signed_angle=e.delta_psi, peak_rate=e.peak_rate,
                         quality=min(1.0, abs(e.peak_rate) / 0.3))
            for i, e in enumerate(self.turn_events)
        ]
        # Soft tier: gentler turns the strong detector misses. On the review
        # data a real 38 deg map turn a driver takes wide integrates to ~24 deg
        # at a peak rate below the strong threshold, and the junction then gets
        # committed straight off a mistimed window. A soft event can steer such
        # a junction, but only one with a matching moderate-angle successor, and
        # the decision stays provisional.
        if single_cfg.soft_turn_enabled and integrator is not None and integrator.times.size:
            strong_spans = [(e.t_start - 1.0, e.t_end + 1.0) for e in self.turn_events]
            soft = detect_turns(
                integrator.times, integrator.rate, 0.0,
                min_rate_rads=single_cfg.soft_turn_min_rate_rads,
                min_delta_rad=math.radians(single_cfg.soft_turn_min_angle_deg))
            for e in soft:
                if any(a <= e.t_peak <= b for a, b in strong_spans):
                    continue
                if abs(e.delta_psi) >= math.radians(single_cfg.event_min_angle_deg):
                    continue  # would already be a strong event
                self.turns.append(PhysicalTurn(
                    id=len(self.turns), t_start=e.t_start, t_peak=e.t_peak,
                    t_end=e.t_end, signed_angle=e.delta_psi,
                    peak_rate=e.peak_rate, soft=True,
                    quality=min(1.0, abs(e.peak_rate) / 0.3)))
            self.turns.sort(key=lambda tn: tn.t_peak)
            for i, tn in enumerate(self.turns):
                tn.id = i
            # A lane change commonly appears as two adjacent, opposite soft
            # events. Neither half is a junction turn: together they restore
            # almost the same heading. Suppress the pair before either half can
            # be matched independently to a nearby fork.
            for first, second in zip(self.turns[:-1], self.turns[1:]):
                if not (first.soft and second.soft):
                    continue
                gap = second.t_start - first.t_end
                if gap < 0.0 or gap > single_cfg.soft_turn_pair_max_gap_s:
                    continue
                a, b = abs(first.signed_angle), abs(second.signed_angle)
                balanced = min(a, b) / max(a, b, 1e-9)
                if (np.sign(first.signed_angle) != np.sign(second.signed_angle)
                        and balanced >= single_cfg.soft_turn_pair_min_balance_ratio
                        and abs(first.signed_angle + second.signed_angle)
                        <= math.radians(
                            single_cfg.soft_turn_pair_max_net_angle_deg)):
                    first.is_bend = second.is_bend = True
                    first.consumed = second.consumed = True
        self.pending: Optional[_PendingDecision] = None
        self.decisions: list[SinglePathDecision] = []
        self._handled_contradictions: set[int] = set()
        self.rollback_count = 0
        self.rollback_fixed_wrong = 0
        self.rollback_events: list[dict[str, Any]] = []
        self._alt_tries: dict[int, int] = {}
        self.gyro_bias = 0.0
        self.gyro_bias_sigma = 0.004
        self.dead_end_active = False
        self._committed_junction_offsets: list[float] = []
        self._last_commit_distance: float = 0.0
        self._angle_blocked: set[int] = set()
        self._last_turn_anchor: Optional[dict[str, float]] = None
        self.turn_intervals: list[dict[str, Any]] = []
        # Accepted intra-edge bend anchors, drained by the tracker.
        self.bend_anchors: list[dict[str, Any]] = []
        self._provisional_log: list[dict[str, Any]] = []
        # A short OSM connector may be only the first half of one physical
        # manoeuvre. Once its net two-node angle selects a continuation, keep
        # that exact second edge for the next immediate commit.
        self._compound_continuation: Optional[tuple[int, int]] = None
        # True only while traversing the reverse twin used to leave a terminal
        # edge. At its mouth the car body yaw has the opposite sign from the
        # map trajectory turn, because the vehicle is still backing up.
        self._backing_from_terminal = False
        # Drained by the tracker: a time to anchor a distance interval at, and
        # the index of the newest not-yet-consumed interval record.
        self.interval_anchor_request: Optional[float] = None
        self._intervals_emitted: int = 0
        self.event_stats = {
            "detected": len(self.turns), "matched": 0, "straight_commits": 0,
            "expired_unconsumed": 0, "double_use_prevented": 0,
            "residuals_m": [],
        }

    # -------------------------------------------------------- distance history

    def _remember_distance(self, t: float, distance: float) -> None:
        # Longer retention than the beam manager: a turn event may stay pending
        # while the odometer walks up to its junction, and its d_event must
        # remain recoverable.
        self._distance_times.append(float(t))
        self._distance_values.append(float(distance))
        cutoff = float(t) - max(self.single_cfg.event_max_age_s + 15.0, 40.0)
        first = int(np.searchsorted(self._distance_times, cutoff))
        if first > 1:
            del self._distance_times[: first - 1]
            del self._distance_values[: first - 1]

    def _distance_at(self, when: float, fallback: float) -> float:
        if not self._distance_times:
            return fallback
        return float(np.interp(when, self._distance_times, self._distance_values,
                               left=self._distance_values[0],
                               right=self._distance_values[-1]))

    def _ingest_turns(self, t: float, distance: float) -> list["PhysicalTurn"]:
        newly: list[PhysicalTurn] = []
        for turn in self.turns:
            if turn.ingested or t < turn.t_end + self.single_cfg.event_settle_s:
                continue
            turn.d_event = self._distance_at(turn.t_peak, distance)
            sigma = float(self.integrator.sigma(
                turn.t_start, turn.t_end, self.gyro_bias_sigma,
                math.radians(self.cfg.turn_model_sigma_deg)))
            if turn.soft:
                sigma = math.hypot(
                    sigma, math.radians(self.single_cfg.soft_turn_sigma_inflation_deg))
            turn.sigma_angle = sigma
            turn.ingested = True
            newly.append(turn)
        return newly

    def _successors(self, edge: int, history: tuple[int, ...] = ()) -> list[int]:
        """``geometry.successors`` with any manually blocked OSM node-pair
        transitions removed - see
        ``SinglePathConfig.blocked_successor_node_pairs``."""
        raw = self.geometry.successors(edge, history)
        blocked = self.single_cfg.blocked_successor_node_pairs
        if not blocked:
            return raw
        edges = self.geometry.network.edges
        return [e for e in raw if (edges[e].u, edges[e].v) not in blocked]

    def _junction_gap_deg(self, edge: int, tail: tuple[int, ...],
                          signed_angle: float) -> float:
        """Smallest ``||map turn| - |measured||`` over sign-compatible outgoing
        turns at the node at the end of ``edge`` - how well a junction here
        could explain the event. Large means no junction explanation."""
        best = 999.0
        for nxt in self._successors(edge, tail or (edge,)):
            jt = self.geometry.junction_turn(edge, nxt)
            if np.sign(jt) == np.sign(signed_angle) or abs(signed_angle) < math.radians(15.0):
                best = min(best, abs(math.degrees(abs(jt)) - math.degrees(abs(signed_angle))))
        return best

    def _event_turn_scores(self, turns_map: np.ndarray, turn: PhysicalTurn
                           ) -> tuple[np.ndarray, np.ndarray]:
        """Score successors while preserving soft-turn eligibility.

        A soft event is evidence for a moderate corner in the event's
        direction.  The eligibility mask must survive through the argmax;
        otherwise an eligible corner merely admits the junction and an
        ineligible near-straight edge can still win its likelihood comparison.
        """
        turns_map = np.asarray(turns_map, dtype=float)
        eligible = np.ones(turns_map.shape, dtype=bool)
        if turn.soft:
            cfg = self.single_cfg
            eligible = (
                (np.abs(turns_map)
                 >= math.radians(cfg.soft_turn_map_turn_min_deg))
                & (np.abs(turns_map)
                   <= math.radians(cfg.soft_turn_map_turn_max_deg))
                & (np.sign(turns_map) == np.sign(turn.signed_angle))
            )
        scores = turn_log_likelihood(
            turns_map, turn.signed_angle, turn.sigma_angle)
        return np.where(eligible, scores, -1e18), eligible

    def _road_class_rank(self, edge: int) -> float:
        """Coarse OSM functional class, used only as a shallow-fork prior."""
        highway = self.geometry.network.edges[int(edge)].highway
        if isinstance(highway, (list, tuple)):
            highway = highway[0] if highway else ""
        value = str(highway or "").lower()
        is_link = value.endswith("_link")
        base = value[:-5] if is_link else value
        rank = {
            "motorway": 6.0, "trunk": 5.0, "primary": 4.0,
            "secondary": 3.0, "tertiary": 2.0,
            "residential": 1.0, "unclassified": 1.0,
            "living_street": 0.5, "service": 0.0,
        }.get(base, 1.0)
        return rank - (1.0 if is_link else 0.0)

    def _shallow_road_class_prior(self, incoming_edge: int,
                                  successors: Sequence[int]) -> np.ndarray:
        incoming_rank = self._road_class_rank(incoming_edge)
        penalty = self.single_cfg.shallow_road_class_downgrade_penalty_log
        return np.array([
            -penalty * max(0.0, incoming_rank - self._road_class_rank(edge))
            for edge in successors
        ], dtype=float)

    def _classify_bends(self, hs: HypothesisSet,
                        newly: Sequence["PhysicalTurn"]) -> None:
        """Mark a settled gyro event as intra-edge road curvature when the
        committed active edge's own polyline explains its shape uniquely and no
        junction here would. A bend turn is consumed so it never drives a
        junction, a straight-commit revision, a rollback or an interval."""
        cfg = self.single_cfg
        if not len(hs) or not len(hs.routes):
            return
        offset_bias = float(hs.offset_bias[0])
        # Committed edge chain with the route offset at the start of each edge.
        # The odometer may have lagged (a bend at high speed is exactly where it
        # does), so bind the event to the committed edge whose span contains its
        # d_event, not to whichever edge is active right now.
        node = hs.routes[0]
        chain: list[tuple[int, float]] = []
        while node is not None:
            chain.append((int(node.edge), float(node.offset)))
            node = node.parent
        chain.reverse()
        # The odometer lags most exactly at a high-speed bend, so a bend event's
        # d_event can point past the committed frontier - at an edge the route
        # has not reached yet. Add forward candidate edges (<=2 hops) so the
        # event can still be bound, and consumed, before the junction matcher
        # gets it. These are candidates for the shape match, not a route commit.
        f_edge, f_off = chain[-1]
        f_end = f_off + float(self.geometry.lengths[f_edge])
        hist = tuple(e for e, _ in chain[-self._restriction_depth:])
        frontier = [(f_edge, f_end, hist)]
        for _ in range(2):
            nxt_frontier: list[tuple[int, float, tuple[int, ...]]] = []
            for fe, fend, fh in frontier:
                for s in self._successors(fe, fh or (fe,)):
                    s = int(s)
                    chain.append((s, fend))
                    nxt_frontier.append(
                        (s, fend + float(self.geometry.lengths[s]),
                         (fh + (s,))[-self._restriction_depth:]))
            frontier = nxt_frontier
        gt, gr = self.integrator.times, self.integrator.rate
        for turn in newly:
            if turn.consumed or turn.soft or turn.is_bend:
                continue
            if abs(turn.peak_rate) > cfg.bend_event_max_peak_rate_rads:
                continue
            if (turn.t_end - turn.t_start) < cfg.bend_event_min_duration_s:
                continue
            rho = float(turn.d_event) - offset_bias      # route arc at the event
            edge = s_at_peak = edge_off = None
            for e, off in chain:
                el = float(self.geometry.lengths[e])
                if off <= rho <= off + el:
                    edge, s_at_peak, edge_off = e, rho - off, off
                    break
            if edge is None:
                continue                       # event not on the committed path
            w0, w1 = turn.t_start - 3.0, turn.t_end + 3.0
            m = (gt >= w0) & (gt <= w1)
            if int(m.sum()) < 5:
                continue
            pr = self.geometry.profile(edge)
            bm = match_curvature_event(
                pr.s, pr.heading, pr.length, gt[m], gr[m] - self.gyro_bias,
                turn.t_peak, min_margin=cfg.bend_min_uniqueness_margin,
                max_angle_resid_rad=math.radians(cfg.bend_max_shape_rms_deg),
                min_shape_corr=cfg.bend_min_rate_corr)
            if bm is None:
                continue
            if self._junction_gap_deg(edge, (edge,), turn.signed_angle) < cfg.bend_junction_gap_reject_deg:
                continue                       # a junction here could explain it
            turn.is_bend = True
            turn.consumed = True
            self.bend_anchors.append({
                "event_id": turn.id,
                "t_peak": float(turn.t_peak),
                "t_start": float(turn.t_start),
                "t_end": float(turn.t_end),
                "edge": int(edge),
                "route_offset_m": float(edge_off),
                "s_map_anchor_m": float(bm.s_peak),
                "s_est_at_event_m": float(s_at_peak),
                "residual_s_m": float(bm.s_peak - s_at_peak),
                "v_bar_ms": float(bm.v_bar),
                "shape_rms_deg": round(math.degrees(bm.angle_resid_rad), 2),
                "rate_corr": round(bm.shape_corr, 3),
                "uniqueness_margin": round(bm.margin, 3),
                "map_delta_psi_deg": round(math.degrees(bm.map_delta_psi), 1),
                "gyro_delta_psi_deg": round(math.degrees(bm.gyro_delta_psi), 1),
                "applied_position": False,
                "applied_speed": False,
                "t_applied": float("nan"),
                "reject_reason": "",
            })

    def _revise_recent_straight_commits(self, hs: HypothesisSet,
                                        newly: Sequence["PhysicalTurn"],
                                        t: float) -> HypothesisSet:
        """A turn event that has just settled may belong to a junction the
        odometer had already walked past and committed straight - the lag can
        be tens of seconds. Associate it using joint position and angle
        evidence, then rewind immediately if it selects a dormant sibling.
        """
        offset_bias = float(hs.offset_bias[0]) if len(hs) else 0.0
        recent = [d for d in self.decisions[-4:] if not d.event_driven]
        for turn in newly:
            if turn.consumed:
                continue
            best: Optional[tuple[float, float, int, SinglePathDecision,
                                 list[int], np.ndarray]] = None
            for di, d in enumerate(self.decisions):
                if d not in recent:
                    continue
                res = abs(d.boundary_offset - (turn.d_event - offset_bias))
                if res > self.single_cfg.event_match_max_tol_m:
                    continue
                if (turn.soft and res
                        > self.single_cfg.soft_turn_max_position_residual_m):
                    continue
                succ = self._successors(
                    d.incoming_edge,
                    d.parent_route.tail(self._restriction_depth))
                if len(succ) < 2:
                    continue
                tmap = np.array([
                    self.geometry.junction_turn(d.incoming_edge, e)
                    for e in succ])
                ll, eligible = self._event_turn_scores(tmap, turn)
                if not bool(np.any(eligible)):
                    continue
                k = int(np.argmax(ll))
                angle_res = abs(float(_wrap(
                    np.array([tmap[k] - turn.signed_angle]))[0]))
                if angle_res > math.radians(40.0):
                    continue
                probabilities = local_probabilities(ll)
                if (turn.soft and float(probabilities[k])
                        < self.single_cfg.soft_turn_min_probability):
                    continue
                pos_scale = max(float(d.crossing_tolerance_m),
                                self.single_cfg.event_match_min_tol_m, 1.0)
                angle_scale = max(float(turn.sigma_angle), math.radians(6.0))
                joint_cost = ((res / pos_scale) ** 2
                              + (angle_res / angle_scale) ** 2)
                candidate = (joint_cost, res, di, d, list(succ), ll)
                if best is None or candidate[0] < best[0]:
                    best = candidate
            if best is None:
                continue
            _, best_res, di, best_d, succ, ll = best
            k = int(np.argmax(ll))
            late_choice = int(succ[k])
            probabilities = local_probabilities(ll)
            best_d.provisional = True
            if late_choice != best_d.chosen_edge:
                best_d.provisional_alternative_edge = late_choice
                best_d.provisional_outcome = "switched_by_late_turn"
            else:
                best_d.provisional_resolved = True
                best_d.provisional_outcome = "confirmed_by_late_turn"
            best_d.measured_turn_rad = float(turn.signed_angle)
            best_d.sigma_rad = float(turn.sigma_angle)
            best_d.local_probability = float(probabilities[k])
            order = np.argsort(-ll, kind="stable")
            second = float(ll[order[1]]) if len(order) > 1 else float("-inf")
            best_d.margin_log = float(ll[k] - second)
            score_by_edge = {
                int(edge): (float(ll[i]), float(probabilities[i]))
                for i, edge in enumerate(succ)
            }
            for alternative in best_d.alternatives:
                revised = score_by_edge.get(alternative.edge)
                if revised is not None:
                    alternative.score, alternative.probability = revised
            best_d.event_driven = True
            best_d.soft_event = bool(turn.soft)
            best_d.event_id = turn.id
            best_d.event_residual_m = float(
                best_d.boundary_offset + offset_bias - turn.d_event)
            best_d.event_age_s = float(t - turn.t_peak)
            turn.consumed = True
            turn.matched_junction_offset = best_d.boundary_offset
            turn.matched_t = float(t)
            turn.matched_decision_index = di
            self.event_stats["matched"] += 1
            self.event_stats["residuals_m"].append(float(best_res))
            self._provisional_log.append({
                "decision_t": round(best_d.t_decision, 2),
                "incoming_edge": best_d.incoming_edge,
                "greedy_edge": best_d.chosen_edge,
                "late_turn_id": turn.id, "late_turn_deg": round(math.degrees(turn.signed_angle), 1),
                "late_turn_soft": turn.soft, "position_residual_m": round(best_res, 1),
                "late_turn_points_at": late_choice,
                "outcome": best_d.provisional_outcome,
            })
            if late_choice != best_d.chosen_edge:
                active = self._switch_provisional(
                    hs, best_d, di, t, trigger="late_turn_switch")
                if active is not None:
                    best_d.provisional_resolved = True
                    hs = active
                    recent = [d for d in self.decisions[-4:]
                              if not d.event_driven]
        return hs

    # ----------------------------------------------------------------- advance

    def advance(self, hs: HypothesisSet, distance: float, t: float,
                speed: float, sigma_s: float = 0.0) -> HypothesisSet:
        self._remember_distance(t, distance)
        if len(hs) == 0:
            return hs
        if len(hs) != 1:
            hs = hs.take(np.array([int(np.argmax(hs.logw))]))
            hs.logw[:] = 0.0
        newly = self._ingest_turns(t, distance)
        self._expire_turns(t)
        if newly and self.single_cfg.bend_classification_enabled:
            self._classify_bends(hs, newly)
        if newly:
            hs = self._revise_recent_straight_commits(hs, newly, t)
        if self.single_cfg.provisional_fork_enabled:
            hs = self._resolve_provisional_forks(hs, t)

        for _ in range(8):
            if self.dead_end_active:
                break
            edge = int(hs.edge[0])
            route_offset = float(hs.route_offset[0])
            offset_bias = float(hs.offset_bias[0])
            length = float(self.geometry.lengths[edge])
            r_junction = route_offset + length
            s = distance - route_offset - offset_bias
            tolerance = float(self.crossing_tolerance(hs, distance, sigma_s)[0])

            if self.pending is None:
                odometer_reached = s >= length - tolerance
                match = None
                if self._has_live_strong_turn(t) or odometer_reached:
                    match = self._match_turn_to_junction(
                        r_junction, offset_bias, distance, sigma_s, t, edge,
                        hs.routes[0])
                if match is None and not odometer_reached:
                    break
                t_cross = (match.turn.t_peak if match is not None else float(
                    np.clip(t + (length - s) / max(abs(speed), 1.0),
                            t - 10.0, t + 10.0)))
                self.pending = _PendingDecision(
                    t_cross=t_cross, window=self._turn_window(speed),
                    incoming_edge=edge,
                    boundary_offset=r_junction,
                    parent_route=hs.routes[0], offset_bias=offset_bias,
                    tolerance_m=tolerance, t_arrived=float(t), match=match)
            elif self.pending.match is None:
                # Still watching for a late turn event over this junction.
                match = self._match_turn_to_junction(
                    self.pending.boundary_offset, self.pending.offset_bias,
                    distance, sigma_s, t, self.pending.incoming_edge,
                    self.pending.parent_route)
                if match is not None:
                    self.pending.match = match
                    self.pending.t_cross = match.turn.t_peak

            ready = t >= self.pending.t_cross + self.pending.window
            if self.pending.match is None and ready:
                settle_by = self._turn_settle_deadline(
                    self.pending.t_cross, self.pending.window)
                if settle_by is not None:
                    # A strong turn is still being detected/settled over this
                    # crossing - wait for it before committing as a no-turn.
                    ready = t >= settle_by + self.single_cfg.straight_commit_extra_wait_s
            if not ready:
                break

            hs = self._commit(hs, distance, t)
            if len(hs) == 0:
                break

        if self.single_cfg.rollback_enabled:
            hs = self._maybe_rollback(hs, distance, t, sigma_s)
            if self.dead_end_active and len(hs):
                hs = self._activate_alternative(hs, t, event=None,
                                                trigger="dead_end")
        return hs

    def _turn_settle_deadline(self, t_cross: float, window: float
                              ) -> Optional[float]:
        """Latest ``t_end + settle`` of any not-yet-resolved turn (strong or
        soft) whose span could belong to this crossing, or None if there is
        none - so a junction is not committed straight while the turn that
        belongs to it is still being detected."""
        span = window + self.single_cfg.event_settle_s + 4.0
        floor = math.radians(self.single_cfg.soft_turn_min_angle_deg)
        deadlines = [
            tn.t_end + self.single_cfg.event_settle_s
            for tn in self.turns
            if not tn.consumed and not tn.expired
            and abs(tn.signed_angle) >= floor
            and tn.t_start <= t_cross + span
            and tn.t_end + self.single_cfg.event_settle_s >= t_cross - span
        ]
        return max(deadlines) if deadlines else None

    def _has_live_strong_turn(self, t: float) -> bool:
        floor = math.radians(self.single_cfg.event_min_angle_deg)
        soft_floor = math.radians(self.single_cfg.soft_turn_min_angle_deg)
        return any(
            tn.ingested and not tn.consumed and not tn.expired
            and abs(tn.signed_angle) >= (soft_floor if tn.soft else floor)
            and t - tn.t_peak <= self.single_cfg.event_max_age_s
            for tn in self.turns)

    def _expire_turns(self, t: float) -> None:
        for turn in self.turns:
            if (turn.ingested and not turn.consumed and not turn.expired
                    and abs(turn.signed_angle)
                    >= math.radians(self.single_cfg.event_min_angle_deg)
                    and t - turn.t_peak > self.single_cfg.event_max_age_s):
                turn.expired = True
                self.event_stats["expired_unconsumed"] += 1

    # ------------------------------------------------- event/junction matching

    def _match_turn_to_junction(self, r_junction: float, offset_bias: float,
                                distance: float, sigma_s: float, t: float,
                                incoming_edge: int, route: Any
                                ) -> Optional[TurnMatch]:
        successors = self._successors(
            incoming_edge, route.tail(self._restriction_depth))
        if not successors:
            return None
        turns_map = np.array([
            self.geometry.junction_turn(incoming_edge, e) for e in successors])
        # Position uncertainty: shared sigma_D plus route-length scale sigma.
        sigma_pos = math.hypot(float(sigma_s), self._route_map_sigma(distance))
        cfg = self.single_cfg
        tol_event = float(np.clip(
            cfg.event_match_k_sigma * sigma_pos + cfg.event_match_drift_allowance_m,
            cfg.event_match_min_tol_m, cfg.event_match_max_tol_m))

        best: Optional[TurnMatch] = None
        for turn in self.turns:
            if not turn.ingested or turn.consumed or turn.expired:
                continue
            if turn.soft:
                # A soft event steers only a junction that offers a moderate
                # corner turning the soft event's way.
                _, eligible = self._event_turn_scores(turns_map, turn)
                if not bool(np.any(eligible)):
                    continue
            elif abs(turn.signed_angle) < math.radians(cfg.event_min_angle_deg):
                continue
            if t - turn.t_peak > cfg.event_max_age_s:
                continue
            residual = r_junction - (turn.d_event - offset_bias)
            if abs(residual) > tol_event:
                continue
            if (turn.soft and abs(residual)
                    > cfg.soft_turn_max_position_residual_m):
                continue
            if abs(residual) > cfg.event_max_junction_distance_m:
                continue
            # A turn whose measured angle matches none of this junction's
            # outgoing edges does not belong here - it is a missed turn at a
            # junction we already committed straight, and must fall through to
            # a contradiction rather than be force-fitted forward.
            if float(np.min(np.abs(_wrap(turns_map - turn.signed_angle)))) \
                    > math.radians(cfg.event_min_angle_deg) + 3.0 * turn.sigma_angle:
                if turn.id not in self._angle_blocked:
                    self._angle_blocked.add(turn.id)
                    self.event_stats["double_use_prevented"] += 1
                continue
            turn_ll, _ = self._event_turn_scores(turns_map, turn)
            k = int(np.argmax(turn_ll))
            dz = residual / max(sigma_pos, 1.0)
            distance_ll = -0.5 * dz * dz * cfg.event_distance_prior_weight
            joint = float(turn_ll[k]) + distance_ll
            if best is None or joint > best.joint_score:
                best = TurnMatch(
                    turn=turn, successors=list(int(e) for e in successors),
                    turns_map=turns_map, successor_index=k,
                    residual_m=float(residual), sigma_pos_m=float(sigma_pos),
                    turn_ll=float(turn_ll[k]), distance_ll=float(distance_ll),
                    joint_score=joint)
        return best

    def _route_map_sigma(self, distance: float) -> float:
        """Route-length scale uncertainty since the last committed junction,
        bounded. Mirrors HypothesisSet.map_distance_sigma."""
        floor = self.cfg.map_distance_sigma_floor_m
        cap = self.cfg.map_distance_sigma_max_m
        rel = self.cfg.map_distance_relative_sigma
        travelled = max(0.0, abs(distance - self._last_commit_distance))
        return float(min(math.hypot(floor, rel * travelled), cap))

    # ------------------------------------------------------------------ commit

    def _commit(self, hs: HypothesisSet, distance: float, t: float) -> HypothesisSet:
        pending = self.pending
        assert pending is not None
        successors = self._successors(
            pending.incoming_edge,
            pending.parent_route.tail(self._restriction_depth))
        self.pending = None
        terminal_backtrack = False
        if not successors and self.single_cfg.terminal_backtrack_enabled:
            reverse = self.geometry.network.edges[
                pending.incoming_edge].reverse_index
            if reverse is not None:
                successors = [int(reverse)]
                terminal_backtrack = True
                self._backing_from_terminal = True
                self.stats.reversals += 1
        if not successors:
            self.stats.dead_ends += 1
            self.dead_end_active = True
            hs.spawned[0] = True
            return hs

        turns = (np.zeros(1, dtype=float) if terminal_backtrack else np.array([
            self.geometry.junction_turn(pending.incoming_edge, edge)
            for edge in successors
        ]))
        match = pending.match
        if match is not None and match.turn.consumed:
            match = None  # consumed since it was matched; fall back to geometry
        # A soft event only steers if it decisively picks one successor.
        soft_event = match is not None and match.turn.soft
        if soft_event:
            probe, eligible = self._event_turn_scores(turns, match.turn)
            if not bool(np.any(eligible)):
                match = None
                soft_event = False
            else:
                probability = float(local_probabilities(probe).max())
            if (match is not None
                    and probability < self.single_cfg.soft_turn_min_probability):
                match = None
                soft_event = False

        if match is not None:
            measured = float(match.turn.signed_angle)
            sigma = float(match.turn.sigma_angle)
            event_driven = True
        else:
            lo = pending.t_cross - pending.window
            hi = pending.t_cross + pending.window
            # A strong turn that belongs to a *different* junction can bleed
            # into this window. Excise its span so the no-turn measurement is
            # the heading change this junction is actually responsible for.
            measured = self.integrator.delta(lo, hi, self.gyro_bias)
            for tn in self.turns:
                floor = math.radians(self.single_cfg.event_min_angle_deg)
                if (abs(tn.signed_angle) >= floor and tn.t_start < hi
                        and tn.t_end > lo
                        and not (tn.consumed
                                 and tn.matched_junction_offset
                                 == pending.boundary_offset)):
                    measured = (self.integrator.delta(lo, min(tn.t_start, hi),
                                                      self.gyro_bias)
                                + self.integrator.delta(max(tn.t_end, lo), hi,
                                                        self.gyro_bias))
            sigma = self.integrator.sigma(
                lo, hi, self.gyro_bias_sigma,
                math.radians(self.cfg.turn_model_sigma_deg))
            event_driven = False
            self.event_stats["straight_commits"] += 1

        if match is not None:
            if self._backing_from_terminal and not terminal_backtrack:
                scores = turn_log_likelihood(
                    -turns, float(match.turn.signed_angle),
                    float(match.turn.sigma_angle))
            else:
                scores, _ = self._event_turn_scores(turns, match.turn)
            compound_next: list[Optional[int]] = [None] * len(successors)
            self._compound_continuation = None
        else:
            effective_turns = turns.copy()
            compound_next = [None] * len(successors)
            # A swept physical manoeuvre can cross two OSM nodes separated by
            # a tiny connector. Compare the net two-node angle with the gyro:
            # scoring either artificial node alone can reverse the decision.
            if (not self._backing_from_terminal and len(successors) > 1
                    and abs(measured) < math.radians(
                        self.single_cfg.soft_turn_min_angle_deg)):
                history = pending.parent_route.tail(self._restriction_depth)
                for i, edge_value in enumerate(successors):
                    edge = int(edge_value)
                    direct_residual = float(np.abs(_wrap(np.array(
                        [turns[i] - measured]))[0]))
                    if direct_residual <= math.radians(
                            self.single_cfg.compound_plan_max_residual_deg):
                        # This successor's own single-hop angle already
                        # explains the gyro measurement well. Trust it rather
                        # than a lookahead through a further, unrelated edge -
                        # otherwise a candidate that is a clean direct match
                        # can lose to one that only matches two hops out.
                        continue
                    if self.geometry.lengths[edge] > float(
                            self.single_cfg.compound_connector_max_m):
                        continue
                    edge_history = (history + (edge,))[-self._restriction_depth:]
                    following = self._successors(edge, edge_history)
                    if not following:
                        continue
                    # A terminal branch can appear angle-perfect only because
                    # the map split one manoeuvre at an artificial node. Do
                    # not pre-plan into it when a continuing second edge is
                    # available; a moving trajectory cannot remain there.
                    continuing = []
                    for nxt in following:
                        next_history = (edge_history + (int(nxt),))[
                            -self._restriction_depth:]
                        if self._successors(int(nxt), next_history):
                            continuing.append(int(nxt))
                    if continuing:
                        following = continuing
                    totals = np.array([
                        float(_wrap(np.array([
                            turns[i] + self.geometry.junction_turn(edge, nxt)
                        ]))[0])
                        for nxt in following
                    ])
                    residuals = np.abs(_wrap(totals - measured))
                    j = int(np.argmin(residuals))
                    if residuals[j] >= direct_residual:
                        continue
                    effective_turns[i] = totals[j]
                    compound_next[i] = int(following[j])
            score_turns = (-effective_turns
                           if self._backing_from_terminal and not terminal_backtrack
                           else effective_turns)
            scores = turn_log_likelihood(score_turns, measured, sigma)
            if abs(measured) < math.radians(
                    self.single_cfg.soft_turn_min_angle_deg):
                scores = scores + self._shallow_road_class_prior(
                    pending.incoming_edge, successors)
            planned = self._compound_continuation
            self._compound_continuation = None
            if planned is not None and planned[0] == pending.incoming_edge:
                for i, edge in enumerate(successors):
                    if int(edge) != planned[1]:
                        scores[i] = -1e18
        probabilities = local_probabilities(scores)
        order = np.argsort(-scores, kind="stable")
        best = int(order[0])
        second = float(scores[order[1]]) if len(order) > 1 else float("-inf")
        margin = float(scores[best] - second)
        alternatives = [
            DormantAlternative(int(successors[i]), float(turns[i]),
                               float(scores[i]), float(probabilities[i]))
            for i in order
        ]
        chosen_edge = int(successors[best])
        compound_residual = abs(float(_wrap(np.array([
            effective_turns[best] - measured
        ]))[0])) if match is None else float("inf")
        if (match is None and compound_next[best] is not None
                and compound_residual <= math.radians(
                    self.single_cfg.compound_plan_max_residual_deg)):
            self._compound_continuation = (
                chosen_edge, int(compound_next[best]))
        if self._backing_from_terminal and not terminal_backtrack:
            # The exit choice has consumed the one observable reverse-gear
            # sign inversion. Beyond the mouth we resume the normal forward
            # road model.
            self._backing_from_terminal = False

        if event_driven:
            d_turn = match.turn.d_event
            residual = float(match.residual_m)
            event_id = match.turn.id
            event_age = float(t - match.turn.t_peak)
        else:
            centre = self.integrator.centroid(
                pending.t_cross - pending.window,
                pending.t_cross + pending.window, self.gyro_bias)
            d_turn = (self._distance_at(centre, distance)
                      if centre is not None else float(distance))
            residual = pending.boundary_offset + pending.offset_bias - d_turn
            event_id = None
            event_age = 0.0

        distance_to_junction = pending.boundary_offset + pending.offset_bias - d_turn
        low = (margin < self.single_cfg.low_confidence_margin_log
               or probabilities[best] < self.single_cfg.min_confident_probability)

        child = hs.take(np.array([0]))
        self.dead_end_active = False
        child.edge[0] = chosen_edge
        child.route_offset[0] = pending.boundary_offset
        child.offset_bias[0] = pending.offset_bias
        child.routes[0] = pending.parent_route.child(
            chosen_edge, float(t), pending.boundary_offset)
        child.ids[0] = child.allocate_ids(1)[0]
        child.logw[0] = 0.0
        child.spawned[0] = False
        child.turn_angle[0] = float(turns[best])
        child.turn_t[0] = np.nan
        child.turn_window[0] = pending.window
        child.born_t[0] = float(t)

        offset_correction = 0.0
        if event_driven:
            # The event says the car was AT this junction at t_peak: the true D
            # then was pending.boundary_offset. Fold the residual into this
            # route's offset_bias so every following junction is realigned. The
            # gain is Kalman-like in the position uncertainty: when D is already
            # good (small sigma_pos) the correction is near-zero; when D lags
            # badly it is close to full.
            target_bias = match.turn.d_event - pending.boundary_offset
            innovation = float(np.clip(
                target_bias - child.offset_bias[0],
                -self.single_cfg.event_offset_max_correction_m,
                self.single_cfg.event_offset_max_correction_m))
            sp2 = float(match.sigma_pos_m) ** 2
            r_ev = self.single_cfg.event_offset_anchor_sigma_m ** 2
            gain = min(sp2 / (sp2 + r_ev), self.single_cfg.event_offset_gain)
            offset_correction = gain * innovation
            child.offset_bias[0] += offset_correction
            prior_sigma = float(self.map_distance_sigma(child, match.turn.d_event)[0])
            child.map_sigma_anchor[0] = math.sqrt(max(
                (1.0 - gain) * prior_sigma * prior_sigma,
                self.cfg.map_distance_sigma_floor_m ** 2))
            child.map_anchor_distance[0] = match.turn.d_event
            match.turn.consumed = True
            match.turn.matched_junction_offset = pending.boundary_offset
            match.turn.matched_t = float(t)
            match.turn.matched_decision_index = len(self.decisions)
            self.event_stats["matched"] += 1
            self.event_stats["residuals_m"].append(abs(float(match.residual_m)))
        elif abs(float(turns[best]) - measured) < 2.0 * sigma and abs(measured) > 1e-6:
            # No detected event, but the crossing window shows a real bend:
            # the original damped route-local timing nudge.
            measurement = d_turn - pending.boundary_offset
            innovation = float(np.clip(
                measurement - child.offset_bias[0],
                -self.cfg.offset_max_correction_m,
                self.cfg.offset_max_correction_m))
            prior_sigma = float(self.map_distance_sigma(child, d_turn)[0])
            r = self.cfg.offset_anchor_sigma_m ** 2
            gain = min(prior_sigma * prior_sigma / (prior_sigma * prior_sigma + r),
                       self.cfg.offset_correction_gain)
            offset_correction = gain * innovation
            child.offset_bias[0] += offset_correction
            child.map_sigma_anchor[0] = math.sqrt(max(
                (1.0 - gain) * prior_sigma * prior_sigma,
                self.cfg.map_distance_sigma_floor_m ** 2))
            child.map_anchor_distance[0] = d_turn
            self.stats.offset_corrections += 1

        decision = SinglePathDecision(
            t_cross=pending.t_cross, t_decision=float(t),
            incoming_edge=pending.incoming_edge, chosen_edge=chosen_edge,
            measured_turn_rad=float(measured), sigma_rad=float(sigma),
            local_probability=float(probabilities[best]), margin_log=margin,
            low_confidence=bool(low), alternatives=alternatives,
            parent_route=pending.parent_route,
            boundary_offset=pending.boundary_offset,
            offset_bias=pending.offset_bias, distance_at_decision=float(distance),
            distance_to_junction_at_turn_m=float(distance_to_junction),
            crossing_tolerance_m=pending.tolerance_m,
            event_driven=event_driven, event_id=event_id,
            event_residual_m=float(residual), event_age_s=event_age,
            offset_bias_correction_m=float(offset_correction),
            soft_event=bool(soft_event))
        # A fork the local turn score could not resolve is held provisional: the
        # greedy best still propagates (one active branch), the runner-up is
        # kept live, and the next strong turn events are asked to choose. Only a
        # genuine turn-vs-turn ambiguity qualifies - not straight vs a 3 deg
        # bend, where the runner-up leads nowhere different.
        if (self.single_cfg.provisional_fork_enabled and low and len(order) > 1
                and float(scores[order[1]]) > -1e17):
            runner_up = int(successors[int(order[1])])
            direction_split = abs(float(_wrap(
                np.array([turns[best] - turns[int(order[1])]]))[0]))
            if (runner_up != chosen_edge
                    and direction_split >= math.radians(
                        self.single_cfg.provisional_min_direction_split_deg)
                    and float(probabilities[int(order[1])])
                    >= self.single_cfg.provisional_min_alternative_probability):
                decision.provisional = True
                decision.provisional_alternative_edge = runner_up
        self.decisions.append(decision)
        self._committed_junction_offsets.append(pending.boundary_offset)
        self._last_commit_distance = float(distance)
        self.stats.branches += 1
        self.stats.children += len(successors)
        self.stats.turns_scored += len(successors)

        # Turn-to-turn map-distance interval. Both endpoints are junctions the
        # route reached because a physical gyro turn selected them (soft
        # distance prior only), so the segment between them is independent of
        # the odometer it is about to constrain - not circular. Whether it is
        # *clean* enough to use is a quality assessment, not a length cutoff.
        if event_driven:
            b_local_p = float(probabilities[best])
            b_angle_z = self._match_angle_z(match)
            b_kind_ok = (turn_kind(match.turn.signed_angle)
                         == turn_kind(float(turns[best]))
                         or abs(match.turn.signed_angle) < math.radians(20.0))
            anchor = self._last_turn_anchor
            if anchor is not None:
                self.turn_intervals.append(self._interval_quality(
                    anchor, match, pending, b_local_p, b_angle_z))
            self._last_turn_anchor = {
                "event_id": match.turn.id, "t_peak": float(match.turn.t_peak),
                "d_event": float(match.turn.d_event),
                "junction_offset": float(pending.boundary_offset),
                "residual": float(match.residual_m),
                "angle_z": b_angle_z,
                "kind_ok": bool(b_kind_ok),
                "local_probability": b_local_p,
                "decision_index": len(self.decisions) - 1,
                "sigma_pos_m": float(match.sigma_pos_m),
            }
            # Ask the tracker to (re)anchor a distance interval at this turn.
            self.interval_anchor_request = float(match.turn.t_peak)
        return child

    @staticmethod
    def _match_angle_z(match: "TurnMatch") -> float:
        """|measured - chosen map turn| / sigma for the winning successor."""
        chosen = float(match.turns_map[match.successor_index])
        d = float(_wrap(np.array([chosen - match.turn.signed_angle]))[0])
        return abs(d) / max(match.turn.sigma_angle, 1e-6)

    # -------------------------------------------- provisional fork resolution

    def _sequence_branch_score(self, start_edge: int, prev_edge: int,
                               observed: Sequence[float], sigma: float
                               ) -> tuple[bool, float, list[int]]:
        """Best bounded graph path from ``start_edge`` that explains the ordered
        list of strong turn angles ``observed`` - each matched to one junction,
        every other junction near-straight. Returns
        ``(has_consistent_path, total_turn_loglik, edge_chain)``. Uses only
        signed turn angle, direction, count and graph legality - never
        map-distance agreement with D."""
        near = math.radians(28.0)
        match_tol = math.radians(35.0)
        max_depth = 4 + 5 * len(observed)
        best: list = [False, -1e18, []]

        def rec(edge: int, prev: int, k: int, depth: int, acc: list[int],
                ll: float) -> None:
            if k == len(observed):
                if ll > best[1]:
                    best[0], best[1], best[2] = True, ll, acc[:]
                return
            if depth >= max_depth:
                return
            for nxt in self._successors(edge, (prev, edge)):
                jt = self.geometry.junction_turn(edge, nxt)
                dz = float(_wrap(np.array([jt - observed[k]]))[0])
                if abs(dz) <= match_tol and (np.sign(jt) == np.sign(observed[k])
                                             or abs(observed[k]) < near):
                    rec(nxt, edge, k + 1, depth + 1, acc + [nxt],
                        ll + float(turn_log_likelihood(
                            np.array([jt]), float(observed[k]), sigma)[0]))
                elif abs(jt) < near:
                    rec(nxt, edge, k, depth + 1, acc + [nxt], ll)

        rec(int(start_edge), int(prev_edge), 0, 0, [], 0.0)
        return bool(best[0]), float(best[1]), list(best[2])

    def _resolve_provisional_forks(self, hs: HypothesisSet, t: float
                                   ) -> HypothesisSet:
        cfg = self.single_cfg
        for di, d in enumerate(self.decisions):
            if not d.provisional or d.provisional_resolved:
                continue
            if d.provisional_alternative_edge is None:
                # provisional only for confidence (a late turn confirmed the
                # greedy edge); nothing to switch to.
                if d.provisional_outcome == "late_turn_flagged":
                    d.provisional_resolved = True
                    d.provisional_outcome = "confirmed_by_late_turn"
                continue
            later = [tn for tn in self.turns
                     if tn.ingested and not tn.soft and not tn.consumed
                     and abs(tn.signed_angle) >= math.radians(cfg.event_min_angle_deg)
                     and tn.t_peak > d.t_cross
                     and t >= tn.t_end + cfg.event_settle_s]
            if not later or len(later) <= d._prov_events_seen:
                continue
            d._prov_events_seen = len(later)
            n_use = min(len(later), cfg.provisional_max_events_to_resolve)
            observed = [tn.signed_angle for tn in later[:n_use]]
            sigma = math.radians(self.cfg.turn_model_sigma_deg + 3.0)
            a_ok, a_ll, a_chain = self._sequence_branch_score(
                d.chosen_edge, d.incoming_edge, observed, sigma)
            b_ok, b_ll, b_chain = self._sequence_branch_score(
                d.provisional_alternative_edge, d.incoming_edge, observed, sigma)
            # Per-turn mean log-likelihood: a path threaded through the 35 deg
            # matching tolerance scores much worse than a clean one.
            a_fit = a_ll / max(len(observed), 1) if a_ok else -1e18
            b_fit = b_ll / max(len(observed), 1) if b_ok else -1e18
            clean = math.log(0.5)  # ~ one turn matched within ~1 sigma
            outcome = "unresolved"
            switch = False
            # One observed turn is enough when it is topologically possible
            # from exactly one branch. Waiting for an arbitrary second event
            # only propagates the wrong active suffix further.
            decisive = (a_ok != b_ok
                        or n_use >= cfg.provisional_max_events_to_resolve)
            if decisive and a_ok and not b_ok and a_fit > clean:
                outcome = "confirmed_greedy"
            elif decisive and b_ok and not a_ok and b_fit > clean:
                outcome, switch = "switched", True
            elif decisive and a_ok and b_ok:
                if (a_ll - b_ll >= cfg.provisional_sequence_margin
                        and a_fit > clean):
                    outcome = "confirmed_greedy"
                elif (b_ll - a_ll >= cfg.provisional_sequence_margin
                      and b_fit > clean):
                    outcome, switch = "switched", True
            if outcome == "unresolved" and len(later) >= cfg.provisional_max_events_to_resolve:
                outcome = "forced_greedy"
            self._provisional_log.append({
                "decision_t": round(d.t_decision, 2),
                "incoming_edge": d.incoming_edge,
                "greedy_edge": d.chosen_edge,
                "alternative_edge": d.provisional_alternative_edge,
                "observed_turns_deg": [round(math.degrees(x), 1) for x in observed],
                "greedy_seq_consistent": a_ok, "greedy_seq_loglik": round(a_ll, 3),
                "alt_seq_consistent": b_ok, "alt_seq_loglik": round(b_ll, 3),
                "margin": round(a_ll - b_ll, 3),
                "outcome": outcome, "resolved_at_t": round(t, 2),
            })
            if outcome in ("unresolved",):
                continue
            d.provisional_resolved = True
            d.provisional_outcome = outcome
            if switch:
                active = self._switch_provisional(hs, d, di, t)
                if active is not None:
                    return active
        return hs

    def _switch_provisional(self, hs: HypothesisSet, decision: "SinglePathDecision",
                            di: int, t: float,
                            trigger: str = "provisional_sequence_switch"
                            ) -> Optional[HypothesisSet]:
        """Rewind the committed chain to a resolved provisional fork and take
        its alternative. One active branch throughout - a buffered replay of
        the junctions after it, not a parallel expansion."""
        alt_edge = decision.provisional_alternative_edge
        alt = next((a for a in decision.alternatives if a.edge == alt_edge), None)
        if alt is None or alt.activated:
            return None
        alt.activated = True
        decision.rollback_from_edge = decision.chosen_edge
        decision.chosen_edge = alt_edge
        active = hs.take(np.array([0]))
        active.edge[0] = alt_edge
        active.route_offset[0] = decision.boundary_offset
        active.offset_bias[0] = decision.offset_bias
        active.routes[0] = decision.parent_route.child(
            alt_edge, float(t), decision.boundary_offset)
        active.ids[0] = active.allocate_ids(1)[0]
        active.logw[0] = 0.0
        active.spawned[0] = False
        self.decisions = self.decisions[: di + 1]
        self._committed_junction_offsets = self._committed_junction_offsets[: di + 1]
        self.turn_intervals = [iv for iv in self.turn_intervals
                               if iv["decision_index"] <= di]
        for turn in self.turns:
            if turn.matched_decision_index > di:
                turn.consumed = False
                turn.matched_decision_index = -1
                turn.matched_junction_offset = float("nan")
                turn.matched_t = float("nan")
        self._last_turn_anchor = None
        self.pending = None
        self.dead_end_active = False
        self.rollback_count += 1
        self.rollback_events.append({
            "t": round(float(t), 2), "trigger": trigger,
            "rewound_to_decision_t": round(decision.t_decision, 2),
            "from_edge": int(decision.rollback_from_edge),
            "to_edge": int(alt_edge),
            "target_was_low_confidence": bool(decision.low_confidence),
        })
        return active

    def _interval_quality(self, anchor: dict, match: "TurnMatch",
                          pending: "_PendingDecision", b_local_p: float,
                          b_angle_z: float) -> dict[str, Any]:
        """Assess a candidate turn-to-turn interval on evidence quality, not
        length. Every field is a risk the old hard duration/distance cutoff was
        standing in for, badly."""
        a_dec = int(anchor["decision_index"])
        b_dec = len(self.decisions) - 1
        between = self.decisions[a_dec + 1:b_dec]        # interior junctions
        l_map = float(pending.boundary_offset - anchor["junction_offset"])
        d_span = float(match.turn.d_event - anchor["d_event"])

        def unresolved(d) -> bool:
            if getattr(d, "provisional", False):
                return d.provisional_outcome not in (
                    "confirmed_greedy", "confirmed_by_late_turn", "switched")
            return bool(d.low_confidence)

        interior_low_conf = sum(d.low_confidence for d in between)
        interior_provisional = sum(1 for d in between if unresolved(d))
        # Any unresolved fork *before* endpoint A means the whole committed
        # chain up to here - and therefore this interval's map length - could be
        # on the wrong branch. The route between the endpoints being locally
        # consistent does not rescue it.
        upstream_unresolved = sum(
            1 for d in self.decisions[: a_dec + 1] if unresolved(d))
        rolled_back_inside = any(
            ev.get("rewound_to_decision_t", -1e18) >= anchor["t_peak"]
            for ev in self.rollback_events)
        a_res = abs(float(anchor["residual"]))
        b_res = abs(float(match.residual_m))
        a_sig = float(anchor["sigma_pos_m"])
        b_sig = float(match.sigma_pos_m)
        b_chosen_turn = float(match.turns_map[match.successor_index])
        b_kind_ok = (turn_kind(match.turn.signed_angle)
                     == turn_kind(b_chosen_turn)
                     or abs(match.turn.signed_angle) < math.radians(20.0))
        # A clean endpoint: the turn selected ONE successor decisively (a
        # peaked local probability, not a coin toss), that successor turns the
        # way the gyro did, and the odometer places the junction within the
        # bounded match tolerance. The raw angle residual is only a diagnostic
        # - a driver who cuts a corner leaves 20-35 deg between their line and
        # the map's node angle, which does not make the junction wrong.
        a_ok = (a_res <= 2.5 * a_sig + 80.0
                and float(anchor.get("local_probability", 1.0)) >= 0.85
                and bool(anchor.get("kind_ok", True)))
        b_ok = (b_res <= 2.5 * b_sig + 80.0 and b_local_p >= 0.85 and b_kind_ok)
        return {
            "event_a_id": anchor["event_id"], "event_b_id": match.turn.id,
            "t_a": round(anchor["t_peak"], 2),
            "t_b": round(float(match.turn.t_peak), 2),
            "duration_s": round(float(match.turn.t_peak - anchor["t_peak"]), 2),
            "junction_a_offset_m": round(anchor["junction_offset"], 2),
            "junction_b_offset_m": round(pending.boundary_offset, 2),
            "map_length_m": round(l_map, 2),
            "odometer_span_m": round(d_span, 2),
            "discrepancy_m": round(d_span - l_map, 2),
            "endpoint_residuals_m": [round(float(anchor["residual"]), 2),
                                     round(float(match.residual_m), 2)],
            "endpoint_a_angle_z": round(float(anchor.get("angle_z", 0.0)), 2),
            "endpoint_b_angle_z": round(b_angle_z, 2),
            "endpoint_a_local_probability":
                round(float(anchor.get("local_probability", 1.0)), 4),
            "endpoint_b_local_probability": round(b_local_p, 4),
            "endpoint_a_ok": bool(a_ok), "endpoint_b_ok": bool(b_ok),
            "interior_junctions": len(between),
            "interior_low_confidence": int(interior_low_conf),
            "interior_provisional_unresolved": int(interior_provisional),
            "upstream_unresolved_forks": int(upstream_unresolved),
            "rolled_back_inside": bool(rolled_back_inside),
            "unambiguous_committed_path": True,
            "both_endpoints_turn_anchored": True,
            "independent_event_count": 2,
            "decision_index": b_dec,
            "applied": False, "reject_reason": "",
        }

    # ---------------------------------------------------------------- rollback

    def _maybe_rollback(self, hs: HypothesisSet, distance: float, t: float,
                        sigma_s: float) -> HypothesisSet:
        for turn in self.turns:
            if turn.id in self._handled_contradictions:
                continue
            if not turn.ingested or turn.consumed:
                continue
            if t < turn.t_end + self.single_cfg.event_settle_s:
                continue
            if abs(turn.signed_angle) < math.radians(25.0):
                continue
            # A strong turn is a contradiction once it has expired unmatched or
            # the active route has driven well past the junction it belonged to.
            active_pos = distance - float(hs.route_offset[0]) - float(hs.offset_bias[0])
            drove_past = (math.isfinite(turn.d_event)
                          and turn.d_event - float(hs.offset_bias[0])
                          < float(hs.route_offset[0])
                          - self.single_cfg.event_match_max_tol_m)
            if not (turn.expired or drove_past):
                continue
            self._handled_contradictions.add(turn.id)
            return self._activate_alternative(hs, t, event=turn,
                                              trigger="unexplained_turn")
        return hs

    def _rewind_target(self, t: float, event: Optional[PhysicalTurn]):
        want_kind = turn_kind(event.signed_angle) if event is not None else None
        recent = list(reversed(self.decisions))[: self.single_cfg.rollback_max_depth]
        pools: list[list] = [[], [], []]
        for decision in recent:
            if t - decision.t_decision > self.single_cfg.rollback_max_age_s:
                break
            if (t - decision.t_decision
                    > self.single_cfg.rollback_replay_window_s):
                break
            tried = self._alt_tries.get(id(decision), 0)
            if tried >= self.single_cfg.rollback_max_alternatives_per_junction:
                continue
            sibs = [a for a in decision.alternatives
                    if a.edge != decision.chosen_edge and not a.activated]
            if not sibs:
                continue
            dir_ok = (want_kind is not None
                      and any(turn_kind(a.map_turn_rad) == want_kind for a in sibs))
            if (self.single_cfg.rollback_require_low_confidence
                    and not decision.low_confidence):
                continue
            if dir_ok and decision.low_confidence:
                pools[0].append(decision)
            if dir_ok:
                pools[1].append(decision)
            pools[2].append(decision)
        if self.single_cfg.rollback_sibling_direction_match:
            for pool in pools:
                if pool:
                    return pool[0]
            return None
        return pools[2][0] if pools[2] else None

    def _activate_alternative(self, hs: HypothesisSet, t: float,
                              event: Optional[PhysicalTurn] = None,
                              trigger: str = "unexplained_turn") -> HypothesisSet:
        decision = self._rewind_target(t, event)
        if decision is None:
            return hs
        want_kind = turn_kind(event.signed_angle) if event is not None else None
        sibs = [a for a in decision.alternatives
                if a.edge != decision.chosen_edge and not a.activated]
        if want_kind is not None:
            sibs.sort(key=lambda a: turn_kind(a.map_turn_rad) != want_kind)
        alt = sibs[0]
        alt.activated = True
        self._alt_tries[id(decision)] = self._alt_tries.get(id(decision), 0) + 1
        old = decision.chosen_edge
        decision.rollback_from_edge = old
        decision.chosen_edge = alt.edge
        active = hs.take(np.array([0]))
        active.edge[0] = alt.edge
        active.route_offset[0] = decision.boundary_offset
        active.offset_bias[0] = decision.offset_bias
        active.routes[0] = decision.parent_route.child(
            alt.edge, float(t), decision.boundary_offset)
        active.ids[0] = active.allocate_ids(1)[0]
        active.logw[0] = 0.0
        active.spawned[0] = False
        idx = self.decisions.index(decision)
        self.decisions = self.decisions[: idx + 1]
        self._committed_junction_offsets = self._committed_junction_offsets[: idx + 1]
        self.turn_intervals = [iv for iv in self.turn_intervals
                               if iv["decision_index"] <= idx]
        # Release turn events consumed by the discarded decisions.
        for turn in self.turns:
            if turn.matched_decision_index > idx:
                turn.consumed = False
                turn.matched_decision_index = -1
                turn.matched_junction_offset = float("nan")
                turn.matched_t = float("nan")
        self._last_turn_anchor = None
        for di, d in enumerate(self.decisions):
            if d.event_driven and d.event_id is not None:
                tn = next((x for x in self.turns if x.id == d.event_id), None)
                if tn is not None and math.isfinite(tn.d_event):
                    self._last_turn_anchor = {
                        "event_id": tn.id, "t_peak": tn.t_peak,
                        "d_event": tn.d_event,
                        "junction_offset": d.boundary_offset,
                        "residual": d.event_residual_m,
                        "angle_z": 0.0,
                        "local_probability": d.local_probability,
                        "decision_index": di,
                        "sigma_pos_m": self.single_cfg.event_match_max_tol_m}
        self.pending = None
        self.dead_end_active = False
        self.rollback_count += 1
        self.rollback_events.append({
            "t": round(float(t), 2), "trigger": trigger,
            "rewound_to_decision_t": round(decision.t_decision, 2),
            "depth": len(self.decisions),
            "from_edge": int(old), "to_edge": int(alt.edge),
            "target_was_low_confidence": bool(decision.low_confidence),
            "event_delta_deg": (round(math.degrees(event.signed_angle), 1)
                                if event is not None else None),
        })
        return active

    # ----------------------------------------------- global distance coherence

    def apply_common_distance_shift(self, delta: float, after_t: float) -> None:
        """A map interval has revised the global odometer by ``delta`` metres.

        Every odometer-frame quantity this manager holds must move with it, or
        the next junction match / crossing tolerance is computed against a D
        that has jumped out from under it. The route's *map* position is held
        fixed by the caller shifting ``offset_bias`` by the same amount, so this
        is not a second correction of the same error - it keeps the two frames
        consistent. Turn events before ``after_t`` are inside or before the
        measured segment and are left alone (their d_event was part of what the
        interval measured); events after it get the full shift.
        """
        if not math.isfinite(delta) or abs(delta) < 1e-9:
            return
        self._last_commit_distance += delta
        for turn in self.turns:
            if turn.ingested and not turn.consumed and turn.t_peak >= after_t:
                turn.d_event += delta
        if (self._last_turn_anchor is not None
                and self._last_turn_anchor["t_peak"] >= after_t):
            self._last_turn_anchor["d_event"] += delta

    # ------------------------------------------------------------- pass-throughs

    def resolve_turns(self, hs: HypothesisSet, t: float, gyro_bias: float,
                      gyro_bias_sigma: float, speed: float = 10.0,
                      distance: float = 0.0) -> None:
        self.gyro_bias = float(gyro_bias)
        self.gyro_bias_sigma = float(gyro_bias_sigma)

    def merge(self, hs: HypothesisSet) -> HypothesisSet:
        return hs

    def prune(self, hs: HypothesisSet, distance: float, t: float) -> HypothesisSet:
        return hs

    def diagnostics(self) -> dict[str, Any]:
        residuals = self.event_stats["residuals_m"]
        return {
            "active_population": 1,
            "decisions": [d.to_json() for d in self.decisions],
            "decision_count": len(self.decisions),
            "event_driven_decisions": sum(d.event_driven for d in self.decisions),
            "low_confidence_count": sum(d.low_confidence for d in self.decisions),
            "rollback_count": self.rollback_count,
            "rollback_events": self.rollback_events,
            "dormant_alternatives": sum(max(0, len(d.alternatives) - 1)
                                        for d in self.decisions),
            "soft_event_decisions": sum(d.soft_event for d in self.decisions),
            "provisional_forks": sum(d.provisional for d in self.decisions),
            "provisional_resolved": sum(
                d.provisional and d.provisional_resolved for d in self.decisions),
            "provisional_switched": sum(
                d.provisional_outcome == "switched" for d in self.decisions),
            "provisional_log": self._provisional_log,
            "turn_intervals": self.turn_intervals,
            "bend_anchors": self.bend_anchors,
            "bend_events": sum(tn.is_bend for tn in self.turns),
            "turn_events": {
                **{k: v for k, v in self.event_stats.items() if k != "residuals_m"},
                "mean_abs_residual_m": (round(float(np.mean(residuals)), 2)
                                        if residuals else None),
                "p95_abs_residual_m": (round(float(np.percentile(residuals, 95)), 2)
                                       if residuals else None),
            },
            "physical_turns": [tn.to_json() for tn in self.turns],
        }
