"""Ground truth, survival metrics, and the answer to "why did the right Pacman die?".

Everything in this module runs against the **withheld** GPS and therefore
everything in it is off-limits to the tracker. It is called after
:meth:`PacmanTracker.run` has produced its result, or - for the death report -
through :class:`~geotrace.pacman_tracker.tracker.StepObserver`, which is handed
a read-only view of the population and returns nothing to it.

The first success criterion of this rewrite is not metres of error. It is
``ground_truth_edge_survival_rate``: the fraction of the outage during which the
road the car was actually on is still represented by *some* living hypothesis.
A tracker that reports a confident wrong street is worse than one that reports
four streets and includes the right one; the second can be narrowed, the first
cannot be recovered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from geotrace.coordinates import LocalFrame
from geotrace.models import LocationSample
from geotrace.pacman_tracker.curvature import MatchDiagnostics
from geotrace.pacman_tracker.manager import HypothesisManager, PruneEvent
from geotrace.pacman_tracker.motion import ImuSample
from geotrace.pacman_tracker.state import HypothesisSet
from geotrace.pacman_tracker.tracker import StepObserver, TrackerResult
from geotrace.road_graph import RoadNetwork


# ------------------------------------------------------------- ground truth


@dataclass
class GroundTruth:
    """Map-matched reference track. Built from withheld GPS, post hoc."""

    times: np.ndarray
    xy: np.ndarray
    edges: np.ndarray
    s: np.ndarray
    match_offset_m: np.ndarray
    matched_fraction: float
    notes: dict[str, Any] = field(default_factory=dict)

    def edge_at(self, t: float) -> Optional[int]:
        if len(self.times) == 0 or t < self.times[0] - 2.0 or t > self.times[-1] + 2.0:
            return None
        i = int(np.clip(np.searchsorted(self.times, t) - 1, 0, len(self.times) - 1))
        return int(self.edges[i]) if self.edges[i] >= 0 else None

    def position_at(self, t: float) -> Optional[np.ndarray]:
        if len(self.times) < 2 or t < self.times[0] or t > self.times[-1]:
            return None
        return np.array([np.interp(t, self.times, self.xy[:, 0]),
                         np.interp(t, self.times, self.xy[:, 1])])


def map_match_reference(
    reference: Sequence[LocationSample],
    network: RoadNetwork,
    frame: LocalFrame,
    step_s: float = 1.0,
    candidates: int = 6,
    snap_radius_m: float = 60.0,
    sigma_snap_m: float = 12.0,
    beta_m: float = 20.0,
) -> GroundTruth:
    """Offline HMM map matching of the withheld GPS.

    Standard Newson-Krumm: emission from the perpendicular snap distance,
    transition from the disagreement between the driven graph distance and the
    straight-line GPS distance. Offline and acausal on purpose - this is the
    reference the tracker is scored against, so it may use the whole track at
    once, and it must not be confused with anything the tracker does.
    """
    fixes = [f for f in reference if f.is_usable]
    if not fixes:
        return GroundTruth(np.zeros(0), np.zeros((0, 2)), np.zeros(0, np.int64),
                           np.zeros(0), np.zeros(0), 0.0, {"reason": "no usable reference fixes"})
    # Thin to `step_s` so the transition model sees real displacement.
    picked: list[LocationSample] = [fixes[0]]
    for fix in fixes[1:]:
        if fix.monotonic_time - picked[-1].monotonic_time >= step_s:
            picked.append(fix)
    times = np.array([f.monotonic_time for f in picked])
    xy = frame.to_local_array([f.latitude for f in picked], [f.longitude for f in picked])

    cand: list[list[tuple[int, float, float]]] = []
    for point in xy:
        options: list[tuple[int, float, float]] = []
        for index in network.nearest_edges(point, k=candidates, radius=snap_radius_m):
            s, offset = network.project(point, index)
            if offset <= snap_radius_m:
                options.append((int(index), float(s), float(offset)))
        cand.append(options)

    n = len(picked)
    log_prob: list[np.ndarray] = []
    back: list[np.ndarray] = []
    first = cand[0]
    log_prob.append(
        np.array([-0.5 * (o / sigma_snap_m) ** 2 for _, _, o in first]) if first else np.zeros(0)
    )
    back.append(np.full(len(first), -1, dtype=np.int64))

    for i in range(1, n):
        prev, cur = cand[i - 1], cand[i]
        if not cur:
            log_prob.append(np.zeros(0))
            back.append(np.zeros(0, dtype=np.int64))
            continue
        if not prev or log_prob[-1].size == 0:
            log_prob.append(np.array([-0.5 * (o / sigma_snap_m) ** 2 for _, _, o in cur]))
            back.append(np.full(len(cur), -1, dtype=np.int64))
            continue
        gps_step = float(np.linalg.norm(xy[i] - xy[i - 1]))
        budget = gps_step + 3.0 * beta_m + 60.0
        reach = [
            network.reachable_within_distance(edge, s, budget)
            for edge, s, _ in prev
        ]
        scores = np.full((len(prev), len(cur)), -np.inf)
        for a, (edge_a, s_a, _) in enumerate(prev):
            for b, (edge_b, s_b, _) in enumerate(cur):
                if edge_b == edge_a:
                    route = abs(s_b - s_a)
                elif edge_b in reach[a]:
                    route = reach[a][edge_b][0] + s_b
                else:
                    continue
                scores[a, b] = log_prob[i - 1][a] - abs(route - gps_step) / beta_m
        emit = np.array([-0.5 * (o / sigma_snap_m) ** 2 for _, _, o in cur])
        best_prev = np.argmax(scores, axis=0)
        best_val = scores[best_prev, np.arange(len(cur))]
        dead = ~np.isfinite(best_val)
        # A gap the graph cannot bridge restarts the chain instead of killing it.
        best_val = np.where(dead, log_prob[i - 1].max() - 10.0, best_val)
        best_prev = np.where(dead, -1, best_prev)
        log_prob.append(best_val + emit)
        back.append(best_prev.astype(np.int64))

    edges = np.full(n, -1, dtype=np.int64)
    s_out = np.zeros(n)
    offsets = np.full(n, np.nan)
    j = int(np.argmax(log_prob[-1])) if log_prob[-1].size else -1
    for i in range(n - 1, -1, -1):
        if j < 0 or not cand[i]:
            j = -1
            continue
        edge, s, offset = cand[i][j]
        edges[i], s_out[i], offsets[i] = edge, s, offset
        j = int(back[i][j])
        if j < 0 and i > 0:
            j = int(np.argmax(log_prob[i - 1])) if log_prob[i - 1].size else -1
    matched = float(np.mean(edges >= 0))
    return GroundTruth(
        times=times, xy=xy, edges=edges, s=s_out, match_offset_m=offsets,
        matched_fraction=matched,
        notes={
            "fixes": len(fixes), "nodes": n,
            "median_snap_m": float(np.nanmedian(offsets)) if np.any(np.isfinite(offsets)) else None,
            "p95_snap_m": float(np.nanpercentile(offsets, 95)) if np.any(np.isfinite(offsets)) else None,
        },
    )


# ------------------------------------------------------------ death report


@dataclass
class TraceRecord:
    """One step's worth of numbers about the ground-truth hypothesis."""

    t: float
    gt_edge: int
    alive: bool
    rank: Optional[int]
    hypothesis_id: Optional[int]
    log_weight: Optional[float]
    weight: Optional[float]
    best_log_weight: float
    pruning_threshold: float
    distance: Optional[float]
    route_offset: Optional[float]
    offset_bias: Optional[float]
    s: Optional[float]
    sigma_s: Optional[float]
    map_distance_sigma: Optional[float]
    crossing_tolerance: Optional[float]
    edge_length: Optional[float]
    mapped_length_since_root: Optional[float]
    latest_crossing_t: Optional[float]
    latest_anchor_distance: Optional[float]
    pending_anchor_t: Optional[float]
    disappearance_reason: Optional[str]
    v: Optional[float]
    gyro_residual: Optional[float]
    normalized_z: Optional[float]
    map_curvature: Optional[float]
    predicted_yaw_rate: Optional[float]
    actual_yaw_rate: float
    normalized_rms: Optional[float]
    strikes: Optional[int]
    population: int
    stationary: bool

    def to_json(self) -> dict[str, Any]:
        def r(v, n=4):
            return None if v is None else round(float(v), n)

        return {
            "t": round(self.t, 2),
            "ground_truth_edge": self.gt_edge,
            "alive": self.alive,
            "rank": self.rank,
            "hypothesis_id": self.hypothesis_id,
            "log_weight": r(self.log_weight, 3),
            "weight": r(self.weight, 6),
            "best_log_weight": r(self.best_log_weight, 3),
            "pruning_threshold": r(self.pruning_threshold, 3),
            "physical_distance_m": r(self.distance, 2),
            "route_offset_m": r(self.route_offset, 2),
            "offset_bias_m": r(self.offset_bias, 2),
            "distance_along_edge_m": r(self.s, 2),
            "sigma_s_m": r(self.sigma_s, 2),
            "map_distance_sigma_m": r(self.map_distance_sigma, 2),
            "crossing_tolerance_m": r(self.crossing_tolerance, 2),
            "edge_length_m": r(self.edge_length, 2),
            "mapped_length_since_root_m": r(self.mapped_length_since_root, 2),
            "latest_crossing_t": r(self.latest_crossing_t, 2),
            "latest_anchor_distance_m": r(self.latest_anchor_distance, 2),
            "pending_anchor_t": r(self.pending_anchor_t, 2),
            "disappearance_reason": self.disappearance_reason,
            "velocity_ms": r(self.v, 3),
            "gyro_residual_rads": r(self.gyro_residual, 5),
            "normalized_z": r(self.normalized_z, 3),
            "map_curvature_radm": r(self.map_curvature, 6),
            "predicted_yaw_rate_rads": r(self.predicted_yaw_rate, 5),
            "actual_gyro_yaw_rate_rads": r(self.actual_yaw_rate, 5),
            "normalized_rms": r(self.normalized_rms, 3),
            "strikes": self.strikes,
            "population": self.population,
            "stationary": self.stationary,
        }


class GroundTruthObserver(StepObserver):
    """Watches the ground-truth hypothesis and captures the moment it dies.

    It reads the population; it never writes to it. The tracker's behaviour is
    identical with and without an observer attached, which is what makes the
    numbers it produces worth anything.
    """

    def __init__(self, truth: GroundTruth, pre_death_s: float = 12.0,
                 keep_events: int = 400) -> None:
        self.truth = truth
        self.pre_death_s = pre_death_s
        self.trace: list[TraceRecord] = []
        self.first_loss_t: Optional[float] = None
        self.permanent_loss_t: Optional[float] = None
        self.death_window: list[TraceRecord] = []
        self.death_prune_events: list[PruneEvent] = []
        self.last_known_ids: set[int] = set()
        self._alive_history: list[tuple[float, bool]] = []
        self._events_seen = 0
        self._keep_events = keep_events
        self.samples = 0
        self.alive_steps = 0

    def observe(self, t: float, hs: HypothesisSet, sample: ImuSample,
                match: MatchDiagnostics, manager: HypothesisManager,
                distance: float = 0.0, sigma_s: float = 0.0,
                speed: float = 0.0) -> None:
        gt_edge = self.truth.edge_at(t)
        if gt_edge is None:
            return
        self.samples += 1
        order = hs.order()
        ranks = np.empty(len(hs), dtype=np.int64)
        ranks[order] = np.arange(len(hs))
        hits = np.nonzero(hs.edge == gt_edge)[0]
        best_logw = float(hs.logw.max()) if len(hs) else float("nan")
        threshold = best_logw - manager.cfg.prune_log_margin
        weights = hs.weights() if len(hs) else np.zeros(0)
        rms = hs.normalized_rms() if len(hs) else np.zeros(0)
        s_values = hs.s(distance) if len(hs) else np.zeros(0)
        map_sigma = manager.map_distance_sigma(hs, distance) if len(hs) else np.zeros(0)
        crossing_tolerance = (manager.crossing_tolerance(hs, distance, sigma_s)
                              if len(hs) else np.zeros(0))

        if hits.size:
            self.alive_steps += 1
            i = int(hits[np.argmax(hs.logw[hits])])
            # The match arrays predate merging/splitting/pruning, so this
            # hypothesis may have been created since, or moved. Look it up by
            # id; a miss means "no gyro residual for this one this step", not
            # "use whatever is at this index".
            j = match.index_of(int(hs.ids[i]))
            root = hs.routes[i]
            while root.parent is not None:
                root = root.parent
            record = TraceRecord(
                t=t, gt_edge=gt_edge, alive=True, rank=int(ranks[i]),
                hypothesis_id=int(hs.ids[i]), log_weight=float(hs.logw[i]),
                weight=float(weights[i]), best_log_weight=best_logw,
                pruning_threshold=threshold, distance=float(distance),
                route_offset=float(hs.route_offset[i]),
                offset_bias=float(hs.offset_bias[i]), s=float(s_values[i]),
                sigma_s=float(sigma_s), map_distance_sigma=float(map_sigma[i]),
                crossing_tolerance=float(crossing_tolerance[i]),
                edge_length=float(manager.geometry.lengths[hs.edge[i]]),
                mapped_length_since_root=float(hs.route_offset[i] - root.offset),
                latest_crossing_t=float(hs.routes[i].entered_t),
                latest_anchor_distance=float(hs.map_anchor_distance[i]),
                pending_anchor_t=(float(hs.anchor_turn_t[i])
                                  if np.isfinite(hs.anchor_turn_t[i]) else None),
                disappearance_reason=None, v=float(speed),
                gyro_residual=float(match.residual[j]) if j is not None else None,
                normalized_z=float(match.z[j]) if j is not None else None,
                map_curvature=float(match.kappa[j]) if j is not None else None,
                predicted_yaw_rate=(float(match.omega_map[j]) if j is not None else None),
                actual_yaw_rate=float(sample.yaw_rate),
                normalized_rms=float(rms[i]), strikes=int(hs.strikes[i]),
                population=len(hs), stationary=bool(sample.stationary),
            )
            self.last_known_ids = {int(x) for x in hs.ids[hits]}
        else:
            record = TraceRecord(
                t=t, gt_edge=gt_edge, alive=False, rank=None, hypothesis_id=None,
                log_weight=None, weight=None, best_log_weight=best_logw,
                pruning_threshold=threshold, distance=float(distance),
                route_offset=None, offset_bias=None, s=None, sigma_s=float(sigma_s),
                map_distance_sigma=None, crossing_tolerance=None, edge_length=None,
                mapped_length_since_root=None, latest_crossing_t=None,
                latest_anchor_distance=None, pending_anchor_t=None,
                disappearance_reason=("; ".join(sorted({
                    manager.last_removals[i] for i in self.last_known_ids
                    if i in manager.last_removals})) or None), v=None,
                gyro_residual=None, normalized_z=None,
                map_curvature=None, predicted_yaw_rate=None,
                actual_yaw_rate=float(sample.yaw_rate), normalized_rms=None, strikes=None,
                population=len(hs), stationary=bool(sample.stationary),
            )
            if self.first_loss_t is None:
                self.first_loss_t = t
                # Freeze the run-up: this is the answer to "why did it die?".
                self.death_window = [
                    r for r in self.trace if r.t >= t - self.pre_death_s
                ] + [record]
                self.death_prune_events = [
                    e for e in manager.stats.events
                    if e.hypothesis_id in self.last_known_ids or abs(e.t - t) <= 1.0
                ][-40:]
        self.trace.append(record)
        self._alive_history.append((t, record.alive))
        if len(self.trace) > 40000:
            self.trace = self.trace[-20000:]

    def finish(self) -> None:
        """Compute the point after which the truth never came back."""
        last_alive = None
        for t, alive in self._alive_history:
            if alive:
                last_alive = t
        if last_alive is not None and self._alive_history and not self._alive_history[-1][1]:
            for t, alive in self._alive_history:
                if t > last_alive:
                    self.permanent_loss_t = t
                    break
        elif last_alive is None and self._alive_history:
            self.permanent_loss_t = self._alive_history[0][0]

    def death_report(self) -> dict[str, Any]:
        self.finish()
        return {
            "first_ground_truth_loss_t": (None if self.first_loss_t is None
                                          else round(self.first_loss_t, 2)),
            "permanent_ground_truth_loss_t": (None if self.permanent_loss_t is None
                                              else round(self.permanent_loss_t, 2)),
            "observed_steps": self.samples,
            "steps_with_truth_alive": self.alive_steps,
            "pre_death_trace": [r.to_json() for r in self.death_window],
            "prune_events_at_death": [e.to_json() for e in self.death_prune_events],
        }


# ------------------------------------------------------------------ metrics


def survival_metrics(result: TrackerResult, truth: GroundTruth,
                     network: RoadNetwork, near_m: float = 25.0) -> dict[str, Any]:
    """Survival, position error, and corridor coverage, at output ticks."""
    alive: list[bool] = []
    in_top: dict[int, list[bool]] = {1: [], 3: [], 5: []}
    near: list[bool] = []
    errors: list[float] = []
    covered: list[bool] = []
    areas: list[float] = []
    confidences: dict[str, int] = {}
    ticks = 0

    for frame in result.frames:
        gt_edge = truth.edge_at(frame.t)
        gt_xy = truth.position_at(frame.t)
        if gt_edge is None or gt_xy is None:
            continue
        ticks += 1
        edges_ranked = [state.edge for state in frame.top]
        alive.append(bool(np.any(frame.alive_edges == gt_edge)))
        for k in in_top:
            in_top[k].append(gt_edge in edges_ranked[:k])
        pos = np.asarray(frame.position, dtype=float)
        errors.append(float(np.linalg.norm(pos - gt_xy)))
        near.append(any(
            float(np.linalg.norm(np.asarray(network.edges[st.edge].position(st.s)) - gt_xy)) <= near_m
            for st in frame.top))
        covered.append(frame.corridors.contains(gt_xy))
        areas.append(frame.corridors.area_m2())
        key = frame.corridors.confidence.value
        confidences[key] = confidences.get(key, 0) + 1

    def frac(values: list[bool]) -> Optional[float]:
        return round(float(np.mean(values)), 4) if values else None

    return {
        "ticks": ticks,
        "ground_truth_edge_survival_rate": frac(alive),
        "survival_top1": frac(in_top[1]),
        "survival_top3": frac(in_top[3]),
        "survival_top5": frac(in_top[5]),
        f"survival_top_hypothesis_within_{int(near_m)}m": frac(near),
        "position_error_m": {
            "mean": round(float(np.mean(errors)), 2) if errors else None,
            "median": round(float(np.median(errors)), 2) if errors else None,
            "p95": round(float(np.percentile(errors, 95)), 2) if errors else None,
            "max": round(float(np.max(errors)), 2) if errors else None,
        },
        "corridor_coverage": frac(covered),
        "corridor_mean_area_m2": round(float(np.mean(areas)), 1) if areas else None,
        "corridor_median_area_m2": round(float(np.median(areas)), 1) if areas else None,
        "confidence_histogram": confidences,
    }




# --------------------------------------------------- speed and distance truth


def _reference_speed(trip) -> tuple[np.ndarray, np.ndarray]:
    fixes = [f for f in trip.reference_locations if f.is_usable and f.has_valid_speed]
    if not fixes:
        return np.zeros(0), np.zeros(0)
    return (np.array([f.monotonic_time for f in fixes]),
            np.array([float(f.speed) for f in fixes]))


def speed_metrics(result: TrackerResult, trip) -> dict[str, Any]:
    """How good is the one speed estimate, and what does that cost in metres?

    The position error a speed bias buys is ``delta_s ~ delta_v * T``, so the
    30 s and 60 s distance errors below are the numbers that matter: the true
    hypothesis died at about 60 m of real lag, and anchors are 30-60 s apart.
    """
    rt, rv = _reference_speed(trip)
    if rt.size == 0 or not result.speed_trace:
        return {"reason": "no reference speed"}
    times = np.array([s.t for s in result.speed_trace])
    est = np.array([s.speed_ms for s in result.speed_trace])
    inside = (times >= rt[0]) & (times <= rt[-1])
    times, est = times[inside], est[inside]
    if times.size < 5:
        return {"reason": "no overlap with the reference"}
    truth = np.interp(times, rt, rv)
    moving = truth > 2.0

    def window_error(seconds: float) -> Optional[float]:
        """Integrated distance error over a sliding window of this length."""
        if times.size < 3:
            return None
        dt = float(np.median(np.diff(times)))
        span = max(2, int(round(seconds / max(dt, 1e-6))))
        if times.size <= span:
            return None
        cum_e = np.concatenate([[0.0], np.cumsum(est * dt)])
        cum_t = np.concatenate([[0.0], np.cumsum(truth * dt)])
        de = cum_e[span:] - cum_e[:-span]
        dt_ = cum_t[span:] - cum_t[:-span]
        return round(float(np.mean(np.abs(de - dt_))), 1)

    def window_worst(seconds: float) -> Optional[float]:
        """Worst local distance error, not the average of them.

        A route dies on its worst stretch, not its mean one, so the maximum
        over the sliding windows is the number that predicts a lost hypothesis.
        """
        if times.size < 3:
            return None
        dt = float(np.median(np.diff(times)))
        span = max(2, int(round(seconds / max(dt, 1e-6))))
        if times.size <= span:
            return None
        cum_e = np.concatenate([[0.0], np.cumsum(est * dt)])
        cum_t = np.concatenate([[0.0], np.cumsum(truth * dt)])
        return round(float(np.max(np.abs((cum_e[span:] - cum_e[:-span])
                                         - (cum_t[span:] - cum_t[:-span])))), 1)

    sigma_v = np.array([s.sigma_speed_ms for s in result.speed_trace])[inside]
    with np.errstate(divide="ignore", invalid="ignore"):
        zv = np.where(sigma_v > 1e-9, (est - truth) / sigma_v, np.inf)

    dt = float(np.median(np.diff(times)))
    return {
        "bias_ms": round(float(np.mean(est - truth)), 3),
        "bias_while_moving_ms": (round(float(np.mean(est[moving] - truth[moving])), 3)
                                 if moving.any() else None),
        "mae_ms": round(float(np.mean(np.abs(est - truth))), 3),
        "rmse_ms": round(float(np.sqrt(np.mean((est - truth) ** 2))), 3),
        "correlation": (round(float(np.corrcoef(est, truth)[0, 1]), 3)
                        if float(np.std(est)) > 1e-9 else None),
        "correlation_while_moving": (
            round(float(np.corrcoef(est[moving], truth[moving])[0, 1]), 3)
            if moving.sum() > 5 and float(np.std(est[moving])) > 1e-9 else None),
        "std_ratio": (round(float(np.std(est) / np.std(truth)), 3)
                      if float(np.std(truth)) > 1e-9 else None),
        "distance_error_10s_m": window_error(10.0),
        "distance_error_30s_m": window_error(30.0),
        "distance_error_60s_m": window_error(60.0),
        "max_distance_error_30s_m": window_worst(30.0),
        "max_distance_error_60s_m": window_worst(60.0),
        "speed_within_1_sigma": round(float(np.mean(np.abs(zv) <= 1.0)), 4),
        "speed_within_2_sigma": round(float(np.mean(np.abs(zv) <= 2.0)), 4),
        "estimated_distance_m": round(float(np.sum(est * dt)), 1),
        "true_distance_m": round(float(np.sum(truth * dt)), 1),
    }


def distance_calibration(result: TrackerResult, trip) -> dict[str, Any]:
    """Is ``sigma_D`` telling the truth?

    The single most important honesty check in the system. An error of 80 m
    with ``sigma_D = 100 m`` is a filter that knows what it does not know; an
    error of 2 km with ``sigma_D = 15 m`` is worse than useless, because
    everything downstream believes it. The target is the textbook one,
    ``P(|D - D_true| <= 2 sigma) ~ 95 %``.
    """
    rt, rv = _reference_speed(trip)
    if rt.size == 0 or not result.speed_trace:
        return {"reason": "no reference speed"}
    times = np.array([s.t for s in result.speed_trace])
    est = np.array([s.distance_m for s in result.speed_trace])
    sigma = np.array([s.sigma_distance_m for s in result.speed_trace])
    inside = (times >= rt[0]) & (times <= rt[-1])
    times, est, sigma = times[inside], est[inside], sigma[inside]
    if times.size < 5:
        return {"reason": "no overlap with the reference"}

    # True distance travelled since the outage began, by integrating the
    # withheld GPS speed on the same timeline.
    dense_dt = float(np.median(np.diff(rt)))
    truth_cum = np.cumsum(rv * dense_dt)
    truth = np.interp(times, rt, truth_cum)
    truth = truth - truth[0] + est[0]

    error = est - truth
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sigma > 1e-9, error / sigma, np.inf)
    return {
        "final_distance_m": round(float(est[-1]), 1),
        "final_true_distance_m": round(float(truth[-1]), 1),
        "distance_ratio": round(float(est[-1] / truth[-1]), 3) if abs(truth[-1]) > 1 else None,
        "mean_abs_error_m": round(float(np.mean(np.abs(error))), 1),
        "median_abs_error_m": round(float(np.median(np.abs(error))), 1),
        "max_abs_error_m": round(float(np.max(np.abs(error))), 1),
        "final_error_m": round(float(error[-1]), 1),
        "final_sigma_m": round(float(sigma[-1]), 1),
        "within_1_sigma": round(float(np.mean(np.abs(z) <= 1.0)), 4),
        "within_2_sigma": round(float(np.mean(np.abs(z) <= 2.0)), 4),
        "median_abs_z": round(float(np.median(np.abs(z[np.isfinite(z)]))), 3),
    }
