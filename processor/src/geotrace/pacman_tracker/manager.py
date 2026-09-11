"""Population management: cross junctions, score the turn, merge, prune.

Two changes from the previous design carry most of the weight.

*Branching is scored by the turn actually taken.* When a hypothesis crosses a
node it records which way the map says it went; a few seconds later, once the
gyro evidence spanning the crossing has matured, that angle is compared with
the measured heading change. A hypothesis that claims a 90-degree left while
the gyro integrated -84 degrees is discriminated immediately and decisively -
where pointwise curvature matching had almost nothing to say, because
Petersburg streets are straight and all the information is at the junctions.

*There is no splitting along the road.* Along-road uncertainty is one global
number now (``sigma_D``), so it does not have to be represented by spawning
siblings at different distances - which was multiplying through every junction
and producing 571 000 branches and 310 000 beam-culled hypotheses on a single
20-minute trip. The population is now the set of distinct road states, and
merging keeps it that way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from geotrace.pacman_tracker.config import BeamConfig
from geotrace.pacman_tracker.roadmap import RoadGeometry
from geotrace.pacman_tracker.state import HypothesisSet, RouteNode, concat
from geotrace.pacman_tracker.turns import HeadingIntegrator, turn_log_likelihood


@dataclass
class PruneEvent:
    """Why one hypothesis was removed. The death report is built from these."""

    t: float
    hypothesis_id: int
    edge: int
    reason: str
    log_weight: float
    best_log_weight: float
    threshold: float
    rank: int
    population: int
    s: float
    normalized_rms: float

    def to_json(self) -> dict[str, Any]:
        return {
            "t": round(self.t, 2),
            "hypothesis_id": self.hypothesis_id,
            "edge": self.edge,
            "reason": self.reason,
            "log_weight": round(self.log_weight, 3),
            "best_log_weight": round(self.best_log_weight, 3),
            "pruning_threshold": round(self.threshold, 3),
            "rank": self.rank,
            "population": self.population,
            "distance_along_edge_m": round(self.s, 2),
            "normalized_rms": round(self.normalized_rms, 3),
        }


@dataclass
class ManagerStats:
    branches: int = 0
    children: int = 0
    merged: int = 0
    turns_scored: int = 0
    pruned_weight: int = 0
    pruned_beam: int = 0
    dead_ends: int = 0
    reversals: int = 0
    offset_corrections: int = 0
    events: list[PruneEvent] = field(default_factory=list)
    max_events: int = 6000

    def note(self, event: PruneEvent) -> None:
        self.events.append(event)
        if len(self.events) > self.max_events:
            del self.events[: len(self.events) // 2]


class HypothesisManager:
    def __init__(self, geometry: RoadGeometry, cfg: BeamConfig,
                 integrator: Optional[HeadingIntegrator] = None) -> None:
        self.geometry = geometry
        self.cfg = cfg
        self.integrator = integrator
        self.stats = ManagerStats()
        self._restriction_depth = max(2, geometry.network.restriction_history_limit)
        self._distance_times: list[float] = []
        self._distance_values: list[float] = []
        # Raw junction-alignment observations, drained by the tracker each
        # step. Entries are (turn centre time, D-map_coordinate, *global*
        # hypothesis masses, route-evidence ids).  The raw value is preserved
        # here; subtracting the route-local offset bias would erase the common
        # odometer component before the global filter can see it.
        self.pending_drift: list[tuple[float, "np.ndarray", "np.ndarray",
                                       "np.ndarray"]] = []
        self.last_removals: dict[int, str] = {}
        self._last_removal_t: Optional[float] = None

    def map_distance_sigma(self, hs: HypothesisSet, distance: float) -> np.ndarray:
        return hs.map_distance_sigma(
            distance, self.cfg.map_distance_relative_sigma,
            self.cfg.map_distance_sigma_floor_m, self.cfg.map_distance_sigma_max_m)

    def crossing_tolerance(self, hs: HypothesisSet, distance: float,
                           sigma_s: float) -> np.ndarray:
        combined = np.hypot(float(sigma_s), self.map_distance_sigma(hs, distance))
        return np.clip(self.cfg.crossing_tolerance_sigma * combined,
                       self.cfg.crossing_tolerance_min_m,
                       self.cfg.crossing_tolerance_max_m)

    def crossing_ambiguity(self, hs: HypothesisSet, distance: float,
                           sigma_s: float) -> np.ndarray:
        """Where edge membership is uncertain enough to defer point scoring."""
        if len(hs) == 0:
            return np.zeros(0, dtype=bool)
        s = hs.s(distance)
        tolerance = self.crossing_tolerance(hs, distance, sigma_s)
        lengths = self.geometry.lengths[hs.edge]
        return (s <= tolerance) | (s >= lengths - tolerance)

    def _remember_distance(self, t: float, distance: float) -> None:
        self._distance_times.append(float(t))
        self._distance_values.append(float(distance))
        cutoff = float(t) - 20.0
        first = int(np.searchsorted(self._distance_times, cutoff))
        if first > 1:
            del self._distance_times[: first - 1]
            del self._distance_values[: first - 1]

    # ------------------------------------------------------------- junctions

    def advance(self, hs: HypothesisSet, distance: float, t: float,
                speed: float, sigma_s: float = 0.0) -> HypothesisSet:
        """Move hypotheses across the nodes the global distance has reached.

        A crossing is not an instant. The odometer's distance and the map's
        route length disagree by tens of metres over a few kilometres, so there
        is a stretch of road either side of a node over which it is genuinely
        unknown whether the turn has happened yet. Successors are spawned on
        entering that stretch and the parent survives beside them until it is
        clear of it - both readings of the same evidence, left for the turn
        score to settle.

        Branching only at the exact point was how the true route died even when
        the distance came from the withheld GPS: the car turned, the hypothesis
        had not reached the node, and no hypothesis existed on the road the car
        was now on.
        """
        self._remember_distance(t, distance)
        if self._last_removal_t != float(t):
            self.last_removals = {}
            self._last_removal_t = float(t)
        for _ in range(6):
            s = hs.s(distance)
            lengths = self.geometry.lengths[hs.edge]
            tolerance = self.crossing_tolerance(hs, distance, sigma_s)
            early = np.nonzero((s >= lengths - tolerance) & ~hs.spawned)[0]
            gone = np.nonzero(s >= lengths + tolerance)[0]
            backward = np.nonzero(s < -tolerance)[0]
            if early.size == 0 and gone.size == 0 and backward.size == 0:
                break
            hs = self._transfer(hs, early, gone, backward, t, speed, distance)
            if len(hs) == 0:
                break
        return hs

    def _transfer(self, hs: HypothesisSet, early: np.ndarray, gone: np.ndarray,
                  backward: np.ndarray, t: float, speed: float,
                  distance: float = 0.0) -> HypothesisSet:
        geo = self.geometry
        cfg = self.cfg
        keep = np.ones(len(hs), dtype=bool)
        keep[gone] = False
        keep[backward] = False
        for i in gone:
            self.last_removals[int(hs.ids[int(i)])] = "edge_end_exceeded_crossing_tolerance"
        for i in backward:
            self.last_removals[int(hs.ids[int(i)])] = "edge_start_exceeded_crossing_tolerance"
        # Spawn from every hypothesis entering a crossing zone, and from any
        # that is already past one without having spawned (a long step, or a
        # very short edge).
        forward = np.unique(np.concatenate([early, gone])) if (early.size or gone.size) \
            else np.zeros(0, dtype=np.int64)

        parents: list[int] = []
        edges: list[int] = []
        offsets: list[float] = []
        logw: list[float] = []
        turns: list[float] = []
        cross_times: list[float] = []
        routes: list[RouteNode] = []
        window = self._turn_window(speed)
        s_now = hs.s(distance)

        for i in forward:
            i = int(i)
            if bool(hs.spawned[i]):
                continue
            hs.spawned[i] = True
            edge = int(hs.edge[i])
            length = float(geo.lengths[edge])
            history = hs.routes[i].tail(self._restriction_depth)
            successors = geo.successors(edge, history)
            if not successors:
                self.stats.dead_ends += 1
                continue
            self.stats.branches += 1
            offset = float(hs.route_offset[i]) + length
            # When this hypothesis expects to be *at* the node, not when it
            # entered the zone around it - the turn score compares a window
            # centred on the crossing against the map's angle, and a window
            # centred twelve metres early matches the wrong stretch of gyro.
            t_cross = t + (length - float(s_now[i])) / max(abs(speed), 1.0)
            t_cross = float(np.clip(t_cross, t - 10.0, t + 10.0))
            for successor in successors[: cfg.max_children]:
                parents.append(i)
                edges.append(int(successor))
                offsets.append(offset)
                logw.append(float(hs.logw[i]))
                turns.append(geo.junction_turn(edge, int(successor)))
                cross_times.append(t_cross)
                routes.append(hs.routes[i].child(int(successor), t, offset))

        for i in backward:
            i = int(i)
            parent = hs.routes[i].parent
            if parent is None:
                # Nothing behind it in its own route: it waits at the start.
                keep[i] = True
                continue
            self.stats.reversals += 1
            parents.append(i)
            edges.append(int(parent.edge))
            offsets.append(float(parent.offset))
            logw.append(float(hs.logw[i]))
            turns.append(0.0)
            cross_times.append(float("nan"))
            routes.append(parent)

        survivors = hs.take(np.nonzero(keep)[0])
        if not parents:
            return survivors

        children = hs.take(np.asarray(parents, dtype=np.int64))
        children.edge = np.asarray(edges, dtype=np.int64)
        children.route_offset = np.asarray(offsets, dtype=float)
        children.logw = np.asarray(logw, dtype=float)
        children.routes = routes
        children.ids = children.allocate_ids(len(parents))
        children.born_t = np.full(len(parents), float(t))
        children.turn_angle = np.asarray(turns, dtype=float)
        children.turn_t = np.asarray(cross_times, dtype=float)
        children.turn_window = np.full(len(parents), window)
        children.spawned = np.zeros(len(parents), dtype=bool)
        # A strong anchor belongs to the route history. Keep an older one while
        # it matures so crossing a short following edge cannot overwrite it.
        candidate = ((np.abs(children.turn_angle)
                      >= math.radians(cfg.offset_min_turn_deg))
                     & ~np.isfinite(children.anchor_turn_t))
        children.anchor_turn_angle[candidate] = children.turn_angle[candidate]
        children.anchor_turn_t[candidate] = children.turn_t[candidate]
        children.anchor_turn_window[candidate] = children.turn_window[candidate]
        children.anchor_turn_map_offset[candidate] = children.route_offset[candidate]
        self.stats.children += len(parents)

        merged = concat(survivors, children)
        merged._next_id = max(hs._next_id, children._next_id)
        return merged

    def _turn_window(self, speed: float) -> float:
        """How long the car spends inside a junction, from its own speed."""
        span = 2.5 * self.geometry.cfg.junction_smooth_m
        return float(np.clip(span / max(abs(speed), 2.0), 1.5, 6.0))

    # ----------------------------------------------------------- turn scoring

    def _resolve_alignment_anchors(self, hs: HypothesisSet, t: float,
                                   distance: float, gyro_bias: float,
                                   gyro_bias_sigma: float) -> None:
        """Fuse mature, route-carried turn anchors into distance alignment.

        A turn centroid observes the physical distance at an OSM route
        boundary. This makes ``D_turn - mapped_boundary`` a direct noisy
        measurement of ``offset_bias`` without consulting GPS.
        """
        due = np.nonzero(np.isfinite(hs.anchor_turn_t)
                         & (t >= hs.anchor_turn_t + hs.anchor_turn_window))[0]
        if due.size == 0:
            return
        keys: dict[tuple[float, float], list[int]] = {}
        for i in due:
            keys.setdefault((float(hs.anchor_turn_t[i]),
                             float(hs.anchor_turn_window[i])), []).append(int(i))
        for (t_cross, window), members in keys.items():
            lo, hi = t_cross - window, t_cross + window
            measured = self.integrator.delta(lo, hi, gyro_bias)
            sigma_angle = self.integrator.sigma(
                lo, hi, gyro_bias_sigma, math.radians(self.cfg.turn_model_sigma_deg))
            centre = self.integrator.centroid(lo, hi, gyro_bias)
            if centre is None:
                continue
            idx = np.asarray(members, dtype=np.int64)
            agree = np.abs(hs.anchor_turn_angle[idx] - measured) < 2.0 * sigma_angle
            chosen = idx[agree]
            if chosen.size == 0:
                continue
            distance_at_turn = float(np.interp(
                centre, self._distance_times, self._distance_values,
                left=self._distance_values[0], right=float(distance)))
            measurement = distance_at_turn - hs.anchor_turn_map_offset[chosen]
            # Before clipping: the raw, unbounded disagreement is what carries
            # the global-odometer information. The clip below exists to keep a
            # single bad anchor from wrenching one route's alignment; it must
            # not silently discard the evidence that *every* route is drifting.
            # Exact histories can carry microscopic mass after several
            # branches.  The independent present claim is the directed road;
            # histories that reconverged there share that evidence and are
            # deliberately aggregated rather than counted twice.
            evidence_ids = np.asarray(hs.edge[chosen], np.int64)
            self.pending_drift.append(
                (float(centre), np.asarray(measurement, float),
                 hs.weights()[chosen], evidence_ids))
            innovation = np.clip(measurement - hs.offset_bias[chosen],
                                 -self.cfg.offset_max_correction_m,
                                 self.cfg.offset_max_correction_m)
            prior_sigma = self.map_distance_sigma(hs, distance_at_turn)[chosen]
            r = self.cfg.offset_anchor_sigma_m ** 2
            gain = prior_sigma * prior_sigma / (prior_sigma * prior_sigma + r)
            gain = np.minimum(gain, self.cfg.offset_correction_gain)
            hs.offset_bias[chosen] += gain * innovation
            posterior = np.sqrt(np.maximum((1.0 - gain) * prior_sigma * prior_sigma,
                                           self.cfg.map_distance_sigma_floor_m ** 2))
            hs.map_sigma_anchor[chosen] = posterior
            hs.map_anchor_distance[chosen] = distance_at_turn
            self.stats.offset_corrections += int(chosen.size)
        hs.anchor_turn_t[due] = np.nan

    def resolve_turns(self, hs: HypothesisSet, t: float, gyro_bias: float,
                      gyro_bias_sigma: float, speed: float = 10.0,
                      distance: float = 0.0) -> None:
        """Score matured junction turns against the measured heading change."""
        if self.integrator is None or len(hs) == 0:
            return
        self._resolve_alignment_anchors(hs, t, distance, gyro_bias, gyro_bias_sigma)
        due = np.nonzero(np.isfinite(hs.turn_t) & (t >= hs.turn_t + hs.turn_window))[0]
        if due.size == 0:
            return
        # Group by (crossing time, window) so the integral is computed once for
        # every hypothesis that crossed the same junction at the same instant.
        keys: dict[tuple[float, float], list[int]] = {}
        for i in due:
            keys.setdefault((float(hs.turn_t[i]), float(hs.turn_window[i])), []).append(int(i))
        for (t_cross, window), members in keys.items():
            lo, hi = t_cross - window, t_cross + window
            measured = self.integrator.delta(lo, hi, gyro_bias)
            sigma = self.integrator.sigma(
                lo, hi, gyro_bias_sigma, math.radians(self.cfg.turn_model_sigma_deg))
            idx = np.asarray(members, dtype=np.int64)
            hs.logw[idx] += turn_log_likelihood(hs.turn_angle[idx], measured, sigma)
            self.stats.turns_scored += len(members)
        hs.turn_t[due] = np.nan
        hs.logw -= hs.logw.max()

    # ---------------------------------------------------------------- merging

    def merge(self, hs: HypothesisSet) -> HypothesisSet:
        """Fold hypotheses that have become the same road state.

        Two hypotheses on the same directed edge at the same distance along it
        have the same future whatever their pasts, so one representative carries
        the combined mass and the better route. This is what makes the
        population scale with the number of distinct road states rather than
        exponentially with the number of junctions crossed.
        """
        n = len(hs)
        if n < 2:
            return hs
        cell = np.floor((hs.route_offset + hs.offset_bias)
                        / max(self.cfg.merge_s_tol_m, 1e-6)).astype(np.int64)
        # A pending anchor is part of the future state: merging it with a route
        # that has no such observation would silently discard the correction.
        pending = np.where(np.isfinite(hs.anchor_turn_t),
                           np.rint(hs.anchor_turn_map_offset
                                   / max(self.cfg.merge_s_tol_m, 1e-6)),
                           -9_000_000_000).astype(np.int64)
        key = np.rec.fromarrays([hs.edge, cell, pending], names="edge,cell,pending")
        _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
        if len(counts) == n:
            return hs
        inverse = np.asarray(inverse).reshape(-1)

        order = np.lexsort((-hs.logw, inverse))
        first = np.ones(len(order), dtype=bool)
        first[1:] = inverse[order][1:] != inverse[order][:-1]
        representatives = np.sort(order[first])

        group_max = np.full(len(counts), -np.inf)
        np.maximum.at(group_max, inverse, hs.logw)
        acc = np.zeros(len(counts))
        np.add.at(acc, inverse, np.exp(hs.logw - group_max[inverse]))
        combined = group_max + np.log(np.maximum(acc, 1e-300))

        self.stats.merged += n - len(representatives)
        removed = np.ones(n, dtype=bool)
        removed[representatives] = False
        for i in np.nonzero(removed)[0]:
            self.last_removals[int(hs.ids[i])] = "merged_into_equivalent_route_state"
        out = hs.take(representatives)
        out.logw = combined[inverse[representatives]]
        return out

    # ---------------------------------------------------------------- pruning

    def prune(self, hs: HypothesisSet, distance: float, t: float) -> HypothesisSet:
        cfg = self.cfg
        n = len(hs)
        if n == 0:
            return hs
        best = float(hs.logw.max())
        threshold = best - cfg.prune_log_margin
        rms = hs.normalized_rms()
        s = hs.s(distance)
        ranks = np.empty(n, dtype=np.int64)
        ranks[hs.order()] = np.arange(n)

        below = hs.logw < threshold
        hs.strikes = np.where(below, hs.strikes + 1, 0).astype(np.int32)
        doomed = below & (hs.strikes >= cfg.prune_patience)
        if n - int(doomed.sum()) < cfg.min_hypotheses:
            allowed = max(0, n - cfg.min_hypotheses)
            victims = np.nonzero(doomed)[0]
            if victims.size > allowed:
                worst = victims[np.argsort(hs.logw[victims])][:allowed]
                doomed = np.zeros(n, dtype=bool)
                doomed[worst] = True

        for i in np.nonzero(doomed)[0]:
            self.last_removals[int(hs.ids[i])] = (
                "log_weight_below_margin_for_%d_steps" % cfg.prune_patience)
            self.stats.note(PruneEvent(
                t=t, hypothesis_id=int(hs.ids[i]), edge=int(hs.edge[i]),
                reason="log_weight_below_margin_for_%d_steps" % cfg.prune_patience,
                log_weight=float(hs.logw[i]), best_log_weight=best,
                threshold=threshold, rank=int(ranks[i]), population=n,
                s=float(s[i]), normalized_rms=float(rms[i])))
        self.stats.pruned_weight += int(doomed.sum())
        hs = hs.take(np.nonzero(~doomed)[0])

        if len(hs) > cfg.max_hypotheses:
            order = hs.order()
            dropped = order[cfg.max_hypotheses :]
            best = float(hs.logw.max())
            rms = hs.normalized_rms()
            s = hs.s(distance)
            for pos, i in enumerate(dropped[:200], start=cfg.max_hypotheses):
                self.last_removals[int(hs.ids[i])] = (
                    "beam_limit_%d_exceeded" % cfg.max_hypotheses)
                self.stats.note(PruneEvent(
                    t=t, hypothesis_id=int(hs.ids[i]), edge=int(hs.edge[i]),
                    reason="beam_limit_%d_exceeded" % cfg.max_hypotheses,
                    log_weight=float(hs.logw[i]), best_log_weight=best,
                    threshold=float(hs.logw[order[cfg.max_hypotheses - 1]]),
                    rank=pos, population=len(hs), s=float(s[i]),
                    normalized_rms=float(rms[i])))
            self.stats.pruned_beam += int(dropped.size)
            hs = hs.take(np.sort(order[: cfg.max_hypotheses]))

        hs.logw -= hs.logw.max()
        return hs
