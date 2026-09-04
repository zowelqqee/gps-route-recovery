"""Deliberate corruption of a recorded GPS track.

Every fault is applied to a *copy* of the trip. The clean track is kept in
``reference-samples.jsonl`` so error metrics can be computed, and the exact
description of what was done is written to ``faults.json``.

The reference track is only ground truth in this synthetic sense. During a real
outage on a real drive there is no ground truth at all, and the report labels it
accordingly.

Supported faults (specification part 9):

  dropout   z_broken = empty  over [start, start+duration]
  offset    E += dE, N += dN
  drift     E += k*t, N += m*t
  jumps     z_broken = z + eps,  eps ~ N(0, Sigma)
  false_recovery   a few plausible-looking but wrong fixes right after the
                   receiver comes back, which is what real hardware does
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from geotrace.coordinates import LocalFrame, heading_to_course
from geotrace.loader import utcnow
from geotrace.models import LocationSample, Trip

FAULT_KINDS = ("dropout", "offset", "drift", "jumps", "false_recovery")


class FaultError(ValueError):
    pass


@dataclass
class FaultSpec:
    """One fault to apply, timed relative to the start of the trip."""

    kind: str
    start_s: float = 0.0
    duration_s: Optional[float] = None
    east: float = 0.0
    north: float = 0.0
    drift_east_ms: float = 0.0
    drift_north_ms: float = 0.0
    sigma_m: float = 250.0
    count: int = 3
    accuracy_m: Optional[float] = None
    """Accuracy the corrupted fixes claim to have. Real receivers under-report
    during a failure, which is exactly why accuracy alone cannot be trusted."""

    def __post_init__(self) -> None:
        if self.kind not in FAULT_KINDS:
            raise FaultError(
                f"unknown fault '{self.kind}'. Supported: {', '.join(FAULT_KINDS)}"
            )
        if self.start_s < 0:
            raise FaultError("--start must be >= 0 (seconds from the start of the trip)")
        if self.duration_s is not None and self.duration_s <= 0:
            raise FaultError("--duration must be > 0")

    @property
    def end_s(self) -> float:
        return math.inf if self.duration_s is None else self.start_s + self.duration_s

    def covers(self, t_rel: float) -> bool:
        return self.start_s <= t_rel < self.end_s

    def to_json(self, affected: int = 0) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if self.kind == "offset":
            params = {"east_m": self.east, "north_m": self.north}
        elif self.kind == "drift":
            params = {"east_m_per_s": self.drift_east_ms, "north_m_per_s": self.drift_north_ms}
        elif self.kind == "jumps":
            params = {"sigma_m": self.sigma_m}
        elif self.kind == "false_recovery":
            params = {"count": self.count, "sigma_m": self.sigma_m}
        if self.accuracy_m is not None:
            params["reported_accuracy_m"] = self.accuracy_m
        return {
            "kind": self.kind,
            "start_s": self.start_s,
            "duration_s": self.duration_s,
            "params": params,
            "affected_samples": affected,
        }


@dataclass
class FaultReport:
    applied: list[dict[str, Any]] = field(default_factory=list)
    removed_samples: int = 0
    modified_samples: int = 0
    seed: Optional[int] = None
    source_trip: Optional[str] = None

    def to_json(self) -> dict[str, Any]:
        return {
            "source_trip": self.source_trip,
            "created_at": utcnow().isoformat().replace("+00:00", "Z"),
            "seed": self.seed,
            "removed_samples": self.removed_samples,
            "modified_samples": self.modified_samples,
            "faults": self.applied,
            "note": (
                "reference-samples.jsonl holds the uncorrupted track. It is "
                "ground truth only because this corruption was synthetic; a real "
                "GPS failure leaves no ground truth behind."
            ),
        }


def inject_faults(
    trip: Trip,
    faults: Sequence[FaultSpec],
    seed: int = 42,
    frame: Optional[LocalFrame] = None,
) -> tuple[Trip, FaultReport]:
    """Return a corrupted copy of ``trip`` plus a description of what was done."""
    usable = trip.usable_locations
    if not usable:
        raise FaultError("trip has no usable GPS fixes to corrupt")
    if not faults:
        raise FaultError("no faults specified")

    rng = np.random.default_rng(seed)
    frame = frame or LocalFrame(usable[0].latitude, usable[0].longitude)
    t0 = trip.t0

    broken = copy.deepcopy(trip)
    broken.reference_locations = [copy.deepcopy(s) for s in trip.locations]
    report = FaultReport(seed=seed, source_trip=trip.metadata.trip_id)

    counters = {id(f): 0 for f in faults}
    out: list[LocationSample] = []

    for sample in broken.locations:
        if not sample.is_usable:
            out.append(sample)
            continue
        t_rel = sample.monotonic_time - t0
        east, north = frame.to_local(sample.latitude, sample.longitude)
        dropped = False
        touched = False

        for spec in faults:
            if spec.kind == "false_recovery" or not spec.covers(t_rel):
                continue
            if spec.kind == "dropout":
                dropped = True
                counters[id(spec)] += 1
                break
            if spec.kind == "offset":
                east += spec.east
                north += spec.north
            elif spec.kind == "drift":
                elapsed = t_rel - spec.start_s
                east += spec.drift_east_ms * elapsed
                north += spec.drift_north_ms * elapsed
            elif spec.kind == "jumps":
                east += float(rng.normal(0.0, spec.sigma_m))
                north += float(rng.normal(0.0, spec.sigma_m))
            counters[id(spec)] += 1
            touched = True
            if spec.accuracy_m is not None:
                sample.horizontal_accuracy = spec.accuracy_m

        if dropped:
            report.removed_samples += 1
            continue
        if touched:
            sample.latitude, sample.longitude = frame.to_geo(east, north)
            sample.synthetic = True
            report.modified_samples += 1
        out.append(sample)

    broken.locations = out

    # false_recovery is applied last: it inserts new fixes just after a dropout.
    for spec in faults:
        if spec.kind != "false_recovery":
            continue
        added = _inject_false_recovery(broken, spec, frame, t0, rng)
        counters[id(spec)] += added
        report.modified_samples += added

    for spec in faults:
        report.applied.append(spec.to_json(counters[id(spec)]))

    broken.locations.sort(key=lambda s: s.monotonic_time)
    broken.faults = report.to_json()
    broken.metadata.location_sample_count = len(broken.locations)
    broken.metadata.notes = (
        (broken.metadata.notes or "")
        + " | GPS artificially corrupted by geotrace inject-fault"
    ).strip(" |")
    return broken, report


def _inject_false_recovery(
    trip: Trip,
    spec: FaultSpec,
    frame: LocalFrame,
    t0: float,
    rng: np.random.Generator,
) -> int:
    """Insert a few wrong-but-plausible fixes at the moment GPS comes back.

    A receiver that has just re-acquired satellites typically emits several
    fixes hundreds of metres off, with an optimistic accuracy, before it
    settles. Those are exactly the points the RECOVERING state exists to reject.
    """
    reference = {s.monotonic_time: s for s in trip.reference_locations if s.is_usable}
    if not reference:
        return 0
    times = sorted(reference)
    start = t0 + spec.start_s
    chosen = [t for t in times if t >= start][: spec.count]
    if not chosen:
        return 0

    existing = {round(s.monotonic_time, 3) for s in trip.locations}
    added = 0
    for i, t in enumerate(chosen):
        truth = reference[t]
        east, north = frame.to_local(truth.latitude, truth.longitude)
        # Error shrinks as the receiver converges.
        scale = spec.sigma_m * (1.0 - i / max(1, spec.count))
        east += float(rng.normal(0.0, max(scale, 20.0)))
        north += float(rng.normal(0.0, max(scale, 20.0)))
        lat, lon = frame.to_geo(east, north)
        fix = LocationSample(
            monotonic_time=t,
            latitude=lat,
            longitude=lon,
            wall_time=truth.wall_time,
            horizontal_accuracy=spec.accuracy_m if spec.accuracy_m is not None else 12.0,
            speed=truth.speed,
            speed_accuracy=truth.speed_accuracy,
            course=truth.course,
            course_accuracy=truth.course_accuracy,
            altitude=truth.altitude,
            synthetic=True,
        )
        if round(t, 3) in existing:
            trip.locations = [s for s in trip.locations if round(s.monotonic_time, 3) != round(t, 3)]
        trip.locations.append(fix)
        added += 1
    return added


def parse_fault_args(
    kind: str,
    start: float,
    duration: Optional[float],
    east: float = 0.0,
    north: float = 0.0,
    drift_east: float = 0.0,
    drift_north: float = 0.0,
    sigma: float = 250.0,
    count: int = 3,
    accuracy: Optional[float] = None,
) -> FaultSpec:
    """Turn CLI arguments into a FaultSpec, with useful error messages."""
    spec = FaultSpec(
        kind=kind,
        start_s=start,
        duration_s=duration,
        east=east,
        north=north,
        drift_east_ms=drift_east,
        drift_north_ms=drift_north,
        sigma_m=sigma,
        count=count,
        accuracy_m=accuracy,
    )
    if kind == "dropout" and duration is None:
        raise FaultError("--fault dropout needs --duration (seconds)")
    if kind == "offset" and east == 0.0 and north == 0.0:
        raise FaultError("--fault offset needs --east and/or --north (metres)")
    if kind == "drift" and drift_east == 0.0 and drift_north == 0.0:
        raise FaultError("--fault drift needs --drift-east and/or --drift-north (m/s)")
    return spec


def scenario_offset_dropout_recovery(
    start_s: float = 45.0,
    offset_duration_s: float = 25.0,
    dropout_duration_s: float = 45.0,
    east: float = 900.0,
    north: float = -600.0,
    false_points: int = 4,
    sigma_m: float = 220.0,
) -> list[FaultSpec]:
    """The composite failure the specification asks for:

    GPS first drifts off by a constant offset, then disappears entirely, then
    comes back with several false points before it settles.
    """
    dropout_start = start_s + offset_duration_s
    recovery_start = dropout_start + dropout_duration_s
    return [
        FaultSpec(kind="offset", start_s=start_s, duration_s=offset_duration_s,
                  east=east, north=north, accuracy_m=14.0),
        FaultSpec(kind="dropout", start_s=dropout_start, duration_s=dropout_duration_s),
        FaultSpec(kind="false_recovery", start_s=recovery_start, count=false_points,
                  sigma_m=sigma_m, accuracy_m=12.0),
    ]
