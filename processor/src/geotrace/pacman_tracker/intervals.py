"""Interval distance constraints: the only absolute distance the IMU can get.

Between two confidently identified road events - two turns, or two stops - the
selected route has a map length. That is direct information about

    integral of v dt  over the interval  =  L_map

and it is exactly what the estimator is missing. Every other GPS-free speed
source is either short-horizon (the accelerometer), sparse and low-speed (the
lateral anchors), or saturating (the spectral model). None of them constrains
how far the car actually went along a long fast straight.

It must not be applied as ``v = L / dt``: the vehicle accelerates and brakes
inside the interval, and pinning the *instantaneous* speed at the end of it to
the *average* over it would be wrong in exactly the places it matters. It is an
integral constraint, and the clean way to apply one in a forward filter is to
carry the anchor distance in the state:

    x = [ D, v, b_a, D_anchor ]

``D_anchor`` is frozen at the moment the interval opens (F is identity on it,
no process noise), so its covariance keeps growing correctly against ``D``.
When the interval closes,

    h(x) = D - D_anchor          H = [1, 0, 0, -1]         z = L_map

is an ordinary linear measurement. The correlation it induces between ``D`` and
``b_a`` is what makes the accelerometer bias observable over the interval - the
filter learns not just where it is but why it drifted.

**Anti-circularity.** Map distance may only inform speed when the route is
already supported by evidence that did not come from the speed estimate. The
gate here is agreement, not confidence: the constraint is emitted only when the
surviving high-weight hypotheses independently agree on the interval length
(:meth:`IntervalConstraint.from_hypotheses`). If the belief is genuinely
ambiguous the hypotheses disagree about ``L`` and nothing is applied, so a route
can never be confirmed by having its own assumed length fed back as truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np


@dataclass
class IntervalConfig:
    enabled: bool = False
    """Off by default so the baseline is the pipeline as it was; the
    ablation turns it on."""

    kind_turn: bool = True
    """Turn-to-turn intervals."""

    kind_stop: bool = True
    """Stop-to-stop intervals. Stronger, because both endpoints additionally
    pin v = 0, but rarer."""

    min_length_m: float = 60.0
    """Below this the map length is comparable to its own uncertainty."""

    max_length_m: float = 3000.0
    max_duration_s: float = 240.0
    """Beam (multi-route) turn-interval bounds. A very long beam interval
    accumulates too much route ambiguity across the population to trust. The
    committed single path uses the quality gate below instead, because there
    the segment is one committed chain and its trustworthiness is a property of
    its endpoints, not its length."""

    interval_offset_bias_fraction: float = 0.5
    """When an accepted single-path interval corrects D by delta, this fraction
    of it is allowed to advance the route's along-edge position (the car really
    was further along); the rest is folded into offset_bias so the per-junction
    event alignment is not double-counted. 0 keeps the route position fixed,
    1 lets it follow D."""
    interval_offset_bias_max_m: float = 45.0
    """Clip on the route-position advance from one interval."""

    interval_sanity_max_length_m: float = 12000.0
    interval_sanity_max_duration_s: float = 1200.0
    """Single-path turn-to-turn interval hard bounds - numerical / runtime
    sanity only, deliberately far above the beam values. A clean long interval
    is the most informative measurement there is for accelerometer scale and
    must not be refused for being long. Acceptance is decided by
    :meth:`PacmanTracker._interval_reject_reason` on endpoint and route-segment
    evidence quality."""

    min_mass: float = 0.60
    """Total weight of the hypotheses that must agree on the length.

    A majority, not a plurality. The failure this guards against is a single
    dominant hypothesis "agreeing with itself": its agreeing set has zero
    spread and looks maximally confident, which is precisely the circular case
    where a route would be confirmed by its own assumed length."""

    max_spread_frac: float = 0.06
    """How much the agreeing hypotheses may disagree about the length, as a
    fraction of it. This is the anti-circularity gate."""

    sigma_map_frac: float = 0.04
    """OSM polyline length error plus the driver's line through it. A car does
    not drive the centreline, and the polyline is a simplification of the road."""

    sigma_map_floor_m: float = 8.0

    sigma_timing_s: float = 1.0
    """Turn-event timing uncertainty, converted to metres at the current speed."""

    min_routes: int = 2
    """Distinct routes that must independently report the same drift. One
    hypothesis agreeing with itself is not corroboration."""

    max_drift_m: float = 120.0
    """Beyond this the likely explanation is a wrong route, not a wrong
    odometer, and applying it would drag the speed onto a fiction."""

    max_innovation_sigma: float = 4.0
    """Robust gate. A constraint that disagrees this badly is more likely a
    wrong route than a wrong distance."""

    min_route_mass: float = 0.08
    """Minimum *global* belief mass carried by every corroborating route.

    Without this, a 99%-mass leader plus a 1%-mass sibling can call itself two
    independent routes.  That is still the leader agreeing with itself for all
    practical purposes, and is the circular feedback this module must forbid.
    """

    min_effective_routes: float = 1.5
    """Inverse-Herfindahl effective route count of the agreeing set."""

    event_match_margin_s: float = 2.0
    """A map anchor must fall within this margin of an IMU-detected turn."""

    event_settle_s: float = 6.5
    """Wait this long after a detected turn ends before selecting its best
    route-supported anchor.  Hypothesis-specific crossing windows mature at
    slightly different times; applying each one separately double-counts a
    single physical turn."""


@dataclass
class IntervalConstraint:
    """One accepted interval, ready to be applied as a measurement."""

    t_start: float
    t_end: float
    length_m: float
    sigma_m: float
    kind: str
    mass: float
    spread_m: float
    n_hypotheses: int

    # filled in when applied, for diagnostics
    ins_length_m: float = float("nan")
    discrepancy_m: float = float("nan")
    innovation_sigma: float = float("nan")
    applied: bool = False
    reject_reason: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "t_start": round(self.t_start, 2),
            "t_end": round(self.t_end, 2),
            "duration_s": round(self.t_end - self.t_start, 2),
            "map_length_m": round(self.length_m, 1),
            "ins_length_m": (None if not math.isfinite(self.ins_length_m)
                             else round(self.ins_length_m, 1)),
            "discrepancy_m": (None if not math.isfinite(self.discrepancy_m)
                              else round(self.discrepancy_m, 1)),
            "sigma_m": round(self.sigma_m, 1),
            "innovation_sigma": (None if not math.isfinite(self.innovation_sigma)
                                 else round(self.innovation_sigma, 2)),
            "route_mass": round(self.mass, 3),
            "route_spread_m": round(self.spread_m, 1),
            "n_hypotheses": self.n_hypotheses,
            "applied": self.applied,
            "reject_reason": self.reject_reason,
        }


def route_length_between(hypothesis_offsets: Sequence[float],
                         anchor_offsets: Sequence[float]) -> np.ndarray:
    """Map distance each hypothesis has covered since the anchor.

    A hypothesis's position along its route is ``D - route_offset``, so the map
    distance it traversed between two times is just the change in its own route
    offset plus the change in D. Working in offsets keeps this independent of
    the very D the constraint is about to correct.
    """
    return np.asarray(anchor_offsets, dtype=float) - np.asarray(hypothesis_offsets, dtype=float)


def propose(cfg: IntervalConfig, t_start: float, t_end: float, kind: str,
            lengths: np.ndarray, weights: np.ndarray,
            speed: float) -> Optional[IntervalConstraint]:
    """Decide whether the route belief agrees well enough to use its length.

    ``lengths`` is each surviving hypothesis's own map distance over the
    interval and ``weights`` their normalised weights. Agreement between
    independently-scored hypotheses - not the confidence of any one of them -
    is what makes the length safe to believe.
    """
    if not cfg.enabled or lengths.size == 0:
        return None
    if kind == "turn" and not cfg.kind_turn:
        return None
    if kind == "stop" and not cfg.kind_stop:
        return None
    duration = t_end - t_start
    if duration <= 0 or duration > cfg.max_duration_s:
        return None

    order = np.argsort(-weights)
    lengths, weights = lengths[order], weights[order]
    if lengths.size < 2:
        return IntervalConstraint(t_start, t_end, float(lengths[0]), float("inf"),
                                  kind, float(weights[0]), 0.0, 1,
                                  reject_reason="only one hypothesis; cannot corroborate")
    centre = float(lengths[0])
    tol = max(cfg.max_spread_frac * max(abs(centre), 1.0), cfg.sigma_map_floor_m)
    agree = np.abs(lengths - centre) <= tol
    mass = float(weights[agree].sum())
    if mass < cfg.min_mass:
        return IntervalConstraint(t_start, t_end, centre, float("inf"), kind, mass,
                                  float(lengths.std()), int(agree.sum()),
                                  reject_reason=f"route mass {mass:.2f} below {cfg.min_mass}")

    w = weights[agree] / max(weights[agree].sum(), 1e-12)
    length = float(np.sum(w * lengths[agree]))
    spread = float(np.sqrt(max(np.sum(w * (lengths[agree] - length) ** 2), 0.0)))
    if not (cfg.min_length_m <= length <= cfg.max_length_m):
        return IntervalConstraint(t_start, t_end, length, float("inf"), kind, mass,
                                  spread, int(agree.sum()),
                                  reject_reason=f"length {length:.0f} m out of range")

    sigma = math.sqrt(
        (cfg.sigma_map_frac * length) ** 2
        + cfg.sigma_map_floor_m ** 2
        + spread ** 2
        + (cfg.sigma_timing_s * max(abs(speed), 1.0)) ** 2
    )
    return IntervalConstraint(t_start, t_end, length, sigma, kind, mass, spread,
                              int(agree.sum()))


@dataclass
class DriftObservation:
    """Common-mode disagreement between the odometer and the map, in metres.

    At a junction turn every surviving hypothesis measures ``D_turn - map
    offset of the node it claims to have crossed``. The part of that
    disagreement which is *common* to independently-scored routes cannot be a
    property of any one route - it is the global odometer being short or long,
    and it belongs in ``D``. The part that differs between routes is genuine
    route-alignment and stays in each hypothesis's own ``offset_bias``.

    Splitting the two is what keeps this from double-counting the alignment
    that the route manager already performs.
    """

    t: float
    drift_m: float
    sigma_m: float
    mass: float
    spread_m: float
    n_hypotheses: int
    n_routes: int
    event_kind: str = "turn"
    effective_routes: float = 0.0
    route_ids: list[int] = field(default_factory=list)
    anchor_t: float = float("nan")
    anchor_drift_m: float = float("nan")
    ins_length_m: float = float("nan")
    map_length_m: float = float("nan")
    discrepancy_m: float = float("nan")
    accel_bias_before: float = float("nan")
    accel_bias_after: float = float("nan")
    accel_scale_before: float = float("nan")
    accel_scale_after: float = float("nan")
    distance_before: float = float("nan")
    distance_after: float = float("nan")
    states_modified: int = 0
    lag_s: float = 0.0
    anchor_started: bool = False
    applied: bool = False
    innovation_sigma: float = float("nan")
    reject_reason: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "t": round(self.t, 2),
            "drift_m": round(self.drift_m, 2),
            "sigma_m": round(self.sigma_m, 2),
            "route_mass": round(self.mass, 3),
            "spread_m": round(self.spread_m, 2),
            "n_hypotheses": self.n_hypotheses,
            "n_routes": self.n_routes,
            "event_kind": self.event_kind,
            "effective_routes": round(self.effective_routes, 3),
            "route_ids": self.route_ids,
            "anchor_t": (None if not math.isfinite(self.anchor_t)
                         else round(self.anchor_t, 2)),
            "anchor_drift_m": (None if not math.isfinite(self.anchor_drift_m)
                                else round(self.anchor_drift_m, 2)),
            "ins_length_m": (None if not math.isfinite(self.ins_length_m)
                              else round(self.ins_length_m, 2)),
            "map_length_m": (None if not math.isfinite(self.map_length_m)
                              else round(self.map_length_m, 2)),
            "discrepancy_m": (None if not math.isfinite(self.discrepancy_m)
                               else round(self.discrepancy_m, 2)),
            "accel_bias_before": (None if not math.isfinite(self.accel_bias_before)
                                   else round(self.accel_bias_before, 5)),
            "accel_bias_after": (None if not math.isfinite(self.accel_bias_after)
                                  else round(self.accel_bias_after, 5)),
            "accel_scale_before": (None if not math.isfinite(self.accel_scale_before)
                                    else round(self.accel_scale_before, 5)),
            "accel_scale_after": (None if not math.isfinite(self.accel_scale_after)
                                   else round(self.accel_scale_after, 5)),
            "distance_before": (None if not math.isfinite(self.distance_before)
                                 else round(self.distance_before, 2)),
            "distance_after": (None if not math.isfinite(self.distance_after)
                                else round(self.distance_after, 2)),
            "states_modified": self.states_modified,
            "lag_s": round(self.lag_s, 2),
            "anchor_started": self.anchor_started,
            "applied": self.applied,
            "innovation_sigma": (None if not math.isfinite(self.innovation_sigma)
                                 else round(self.innovation_sigma, 2)),
            "reject_reason": self.reject_reason,
        }


def common_drift(cfg: IntervalConfig, t: float, innovations: np.ndarray,
                 weights: np.ndarray, route_ids: np.ndarray,
                 speed: float) -> DriftObservation:
    """Extract the odometer drift that several *different* routes agree on.

    The anti-circularity gate is corroboration by disagreeing parties: the
    observation is emitted only when hypotheses on distinct routes, carrying a
    majority of the weight, independently report the same disagreement. A
    single dominant hypothesis agrees with itself perfectly and is refused,
    because that is exactly the loop where a route would be confirmed by having
    its own assumed length fed back as truth.
    """
    n = int(innovations.size)
    if not cfg.enabled or n == 0:
        return DriftObservation(t, 0.0, float("inf"), 0.0, 0.0, n, 0,
                                reject_reason="no hypotheses")
    innovations = np.asarray(innovations, dtype=float)
    weights = np.asarray(weights, dtype=float)
    route_ids = np.asarray(route_ids)
    finite = np.isfinite(innovations) & np.isfinite(weights) & (weights > 0)
    innovations, weights, route_ids = (innovations[finite], weights[finite],
                                       route_ids[finite])
    if innovations.size == 0:
        return DriftObservation(t, 0.0, float("inf"), 0.0, 0.0, n, 0,
                                reject_reason="zero weight")

    # Aggregate duplicate descendants before measuring agreement.  The input
    # weights are deliberately *global* hypothesis masses and are not
    # renormalised here: a 2% subset of the belief must remain 2%, not become a
    # fictitious unanimous consensus merely because all other hypotheses were
    # filtered out by the turn-angle gate.
    unique = np.unique(route_ids)
    group_w = np.zeros(unique.size)
    group_y = np.zeros(unique.size)
    for j, rid in enumerate(unique):
        take = route_ids == rid
        group_w[j] = float(weights[take].sum())
        group_y[j] = float(np.sum(weights[take] * innovations[take])
                           / max(group_w[j], 1e-12))
    keep = group_w >= cfg.min_route_mass
    unique, group_w, group_y = unique[keep], group_w[keep], group_y[keep]
    n_routes = int(unique.size)
    if n_routes < cfg.min_routes:
        return DriftObservation(t, 0.0, float("inf"), 0.0, 0.0, n, n_routes,
                                reject_reason=f"{n_routes} distinct routes; "
                                              "cannot corroborate")

    # Find the heaviest robust cluster rather than centring on outliers.  Each
    # route gets one vote weighted by its actual belief mass.
    best = np.zeros(n_routes, dtype=bool)
    best_mass = -1.0
    for seed in group_y:
        tol = max(cfg.max_spread_frac * max(abs(float(seed)), 1.0),
                  cfg.sigma_map_floor_m)
        candidate = np.abs(group_y - seed) <= tol
        candidate_mass = float(group_w[candidate].sum())
        if candidate_mass > best_mass:
            best, best_mass = candidate, candidate_mass
    agree = best
    mass = float(group_w[agree].sum())
    if mass < cfg.min_mass:
        centre = float(np.sum(group_w * group_y) / max(group_w.sum(), 1e-12))
        return DriftObservation(t, centre, float("inf"), mass,
                                float(group_y.std()), n, n_routes,
                                reject_reason=f"route mass {mass:.2f} below "
                                              f"{cfg.min_mass}")
    agreeing_routes = int(agree.sum())
    if agreeing_routes < cfg.min_routes:
        return DriftObservation(t, float(group_y[agree][0]), float("inf"), mass,
                                float(group_y.std()), n, agreeing_routes,
                                reject_reason="agreeing set is a single route")

    wa = group_w[agree] / mass
    effective = float(1.0 / max(np.sum(wa * wa), 1e-12))
    if effective < cfg.min_effective_routes:
        return DriftObservation(t, float(np.sum(wa * group_y[agree])),
                                float("inf"), mass, float(group_y[agree].std()),
                                n, agreeing_routes, effective_routes=effective,
                                route_ids=[int(x) for x in unique[agree]],
                                reject_reason=f"effective routes {effective:.2f} below "
                                              f"{cfg.min_effective_routes}")

    drift = float(np.sum(wa * group_y[agree]))
    spread = float(np.sqrt(max(np.sum(wa * (group_y[agree] - drift) ** 2), 0.0)))
    sigma = math.sqrt(
        cfg.sigma_map_floor_m ** 2
        + spread ** 2
        + (cfg.sigma_timing_s * max(abs(speed), 1.0)) ** 2
    )
    return DriftObservation(t, drift, sigma, mass, spread, n, agreeing_routes,
                            effective_routes=effective,
                            route_ids=[int(x) for x in unique[agree]])
