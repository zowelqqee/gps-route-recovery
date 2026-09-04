"""Quality metrics.

Position error is only defined against a reference track, which exists only for
a synthetically corrupted trip. On a real outage there is nothing to compare
against and every error field is reported as null rather than as a comforting
number.

    error_t      = |p_hat_t - p_ref_t|
    Coverage_95  = #{t : p_ref_t in P_0.95,t} / T
    MeanArea     = (1/T) sum_t Area(P_0.95,t)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from geotrace.gps_quality import GateResult, GPSState
from geotrace.polygons import UncertaintySet


@dataclass
class ErrorSeries:
    times: list[float] = field(default_factory=list)
    errors: list[float] = field(default_factory=list)

    def add(self, t: float, error: float) -> None:
        self.times.append(float(t))
        self.errors.append(float(error))

    def __len__(self) -> int:
        return len(self.errors)

    def stats(self) -> dict[str, Optional[float]]:
        if not self.errors:
            return {
                "mean_m": None, "median_m": None, "p95_m": None,
                "max_m": None, "count": 0,
            }
        arr = np.asarray(self.errors, dtype=float)
        return {
            "mean_m": float(np.mean(arr)),
            "median_m": float(np.median(arr)),
            "p95_m": float(np.percentile(arr, 95)),
            "max_m": float(np.max(arr)),
            "count": int(arr.size),
        }


def interpolate_reference(
    ref_times: Sequence[float], ref_xy: np.ndarray, query_times: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of the reference track onto the estimate timeline.

    Returns (positions, valid_mask); queries outside the reference span are
    marked invalid instead of being extrapolated.
    """
    ref_times = np.asarray(ref_times, dtype=float)
    query = np.asarray(query_times, dtype=float)
    if ref_times.size == 0 or query.size == 0:
        return np.zeros((len(query), 2)), np.zeros(len(query), dtype=bool)
    valid = (query >= ref_times[0]) & (query <= ref_times[-1])
    east = np.interp(query, ref_times, ref_xy[:, 0])
    north = np.interp(query, ref_times, ref_xy[:, 1])
    return np.column_stack([east, north]), valid


def coverage_and_area(
    sets: Sequence[UncertaintySet],
    reference_times: Sequence[float],
    reference_xy: np.ndarray,
) -> dict[str, Any]:
    """Coverage_95 and MeanArea over the whole run."""
    if not sets:
        return {"coverage_95": None, "mean_area_m2": None, "median_area_m2": None, "samples": 0}
    times = [s.t for s in sets]
    reference, valid = interpolate_reference(reference_times, reference_xy, times)
    areas = np.array([s.total_area_m2 for s in sets], dtype=float)

    covered = 0
    counted = 0
    for i, item in enumerate(sets):
        if not valid[i]:
            continue
        counted += 1
        if item.contains(reference[i]):
            covered += 1
    return {
        "coverage_95": (covered / counted) if counted else None,
        "mean_area_m2": float(np.mean(areas)) if areas.size else None,
        "median_area_m2": float(np.median(areas)) if areas.size else None,
        "max_area_m2": float(np.max(areas)) if areas.size else None,
        "samples": counted,
    }


def branch_accuracy(
    sets: Sequence[UncertaintySet],
    reference_times: Sequence[float],
    reference_xy: np.ndarray,
) -> dict[str, Any]:
    """top-1 branch accuracy and top-3 branch recall.

    A branch is "correct" when the reference position falls inside its polygon.
    top-1 asks whether the highest-probability branch is the correct one; top-3
    asks whether any of the three most probable ones is.
    """
    if not sets:
        return {"top1_accuracy": None, "top3_recall": None, "evaluated": 0,
                "mean_branch_count": None}
    times = [s.t for s in sets]
    reference, valid = interpolate_reference(reference_times, reference_xy, times)
    import shapely

    top1 = 0
    top3 = 0
    counted = 0
    branch_counts = []
    for i, item in enumerate(sets):
        branch_counts.append(len(item.components))
        if not valid[i] or not item.components:
            continue
        counted += 1
        point = shapely.points(float(reference[i][0]), float(reference[i][1]))
        hits = [bool(shapely.contains(c.geometry, point)) for c in item.components]
        if hits and hits[0]:
            top1 += 1
        if any(hits[:3]):
            top3 += 1
    return {
        "top1_accuracy": (top1 / counted) if counted else None,
        "top3_recall": (top3 / counted) if counted else None,
        "evaluated": counted,
        "mean_branch_count": float(np.mean(branch_counts)) if branch_counts else None,
        "max_branch_count": int(np.max(branch_counts)) if branch_counts else None,
    }


def gate_metrics(
    history: Sequence[GateResult],
    corrupted_times: Optional[set[float]] = None,
) -> dict[str, Any]:
    """False-rejection and false-acceptance rates of the GPS gates.

    ``corrupted_times`` is the set of monotonic timestamps the fault injector
    touched. Without it the two rates cannot be computed and are reported null:
    on a real trip nobody knows which fixes were lies.
    """
    tested = [h for h in history if h.monotonic_time is not None and "dropout" not in h.reasons]
    out: dict[str, Any] = {
        "fixes_tested": len(tested),
        "fixes_accepted": sum(1 for h in tested if h.accepted),
        "fixes_rejected": sum(1 for h in tested if not h.accepted),
        "rejected_good_fraction": None,
        "accepted_false_fraction": None,
    }
    if corrupted_times is None:
        out["note"] = (
            "No fault manifest available, so 'good' and 'false' fixes cannot be "
            "distinguished. Both rates require a synthetically corrupted trip."
        )
        return out

    def is_corrupt(h: GateResult) -> bool:
        return h.monotonic_time is not None and round(h.monotonic_time, 3) in corrupted_times

    good = [h for h in tested if not is_corrupt(h)]
    bad = [h for h in tested if is_corrupt(h)]
    out["good_fixes"] = len(good)
    out["false_fixes"] = len(bad)
    if good:
        out["rejected_good_fraction"] = sum(1 for h in good if not h.accepted) / len(good)
    if bad:
        out["accepted_false_fraction"] = sum(1 for h in bad if h.accepted) / len(bad)
    return out


def trust_recovery_time(history: Sequence[GateResult]) -> dict[str, Any]:
    """How long after GPS came back before the monitor trusted it again.

    Measured from the first accepted fix following a LOST period to the moment
    the state reaches TRUSTED.
    """
    recoveries: list[float] = []
    pending: Optional[float] = None
    was_lost = False
    for item in history:
        if item.monotonic_time is None:
            continue
        if item.state is GPSState.LOST:
            was_lost = True
            pending = None
        elif was_lost and item.accepted and pending is None:
            pending = item.monotonic_time
        if pending is not None and item.state is GPSState.TRUSTED:
            recoveries.append(item.monotonic_time - pending)
            pending = None
            was_lost = False
    return {
        "recovery_events": len(recoveries),
        "mean_recovery_s": float(np.mean(recoveries)) if recoveries else None,
        "max_recovery_s": float(np.max(recoveries)) if recoveries else None,
        "recoveries_s": [round(r, 2) for r in recoveries],
    }


def outage_end_error(
    series: ErrorSeries, outage_end_t: Optional[float], tolerance_s: float = 2.0
) -> Optional[float]:
    """Position error at the moment the outage ended - the number that matters
    most, because that is the worst case of the dead-reckoning span."""
    if outage_end_t is None or not series.errors:
        return None
    times = np.asarray(series.times, dtype=float)
    index = int(np.argmin(np.abs(times - outage_end_t)))
    if abs(times[index] - outage_end_t) > tolerance_s:
        return None
    return float(series.errors[index])


def compute_error_series(
    estimate_times: Sequence[float],
    estimate_xy: np.ndarray,
    reference_times: Sequence[float],
    reference_xy: np.ndarray,
) -> ErrorSeries:
    series = ErrorSeries()
    if len(estimate_times) == 0 or len(reference_times) == 0:
        return series
    reference, valid = interpolate_reference(reference_times, reference_xy, estimate_times)
    for i, t in enumerate(estimate_times):
        if not valid[i]:
            continue
        series.add(t, float(np.linalg.norm(np.asarray(estimate_xy[i]) - reference[i])))
    return series


@dataclass
class MetricsBundle:
    """Everything that ends up in metrics.json."""

    algorithm: str
    has_reference: bool
    position_error: dict[str, Any] = field(default_factory=dict)
    baselines: dict[str, Any] = field(default_factory=dict)
    polygons: dict[str, Any] = field(default_factory=dict)
    branches: dict[str, Any] = field(default_factory=dict)
    gps_gates: dict[str, Any] = field(default_factory=dict)
    trust_recovery: dict[str, Any] = field(default_factory=dict)
    parking: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "has_reference_track": self.has_reference,
            "position_error": self.position_error,
            "baselines": self.baselines,
            "polygons": self.polygons,
            "branches": self.branches,
            "gps_gates": self.gps_gates,
            "trust_recovery": self.trust_recovery,
            "parking": self.parking,
            "runtime": self.runtime,
            "notes": self.notes,
        }
