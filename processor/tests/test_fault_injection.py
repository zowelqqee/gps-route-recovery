"""Deliberate GPS corruption."""

from __future__ import annotations

import math

import numpy as np
import pytest

from geotrace.coordinates import LocalFrame, haversine_m
from geotrace.fault_injection import (
    FaultError,
    FaultSpec,
    inject_faults,
    parse_fault_args,
    scenario_offset_dropout_recovery,
)
from geotrace.models import LocationSample, Trip, TripMetadata


def make_trip(n: int = 120) -> Trip:
    """A car driving due east at 10 m/s from a point in Saint Petersburg."""
    frame = LocalFrame(59.9311, 30.3609)
    locations = []
    for i in range(n):
        lat, lon = frame.to_geo(10.0 * i, 0.0)
        locations.append(
            LocationSample(
                monotonic_time=1000.0 + i, latitude=lat, longitude=lon,
                horizontal_accuracy=8.0, speed=10.0, course=90.0, course_accuracy=5.0,
                speed_accuracy=1.0,
            )
        )
    return Trip(metadata=TripMetadata(trip_id="test"), locations=locations)


FRAME = LocalFrame(59.9311, 30.3609)


def test_dropout_removes_fixes_in_the_window() -> None:
    trip = make_trip()
    broken, report = inject_faults(
        trip, [FaultSpec(kind="dropout", start_s=30, duration_s=45)], frame=FRAME
    )
    assert report.removed_samples == 45
    assert len(broken.locations) == len(trip.locations) - 45
    times = [s.monotonic_time - trip.t0 for s in broken.locations]
    assert not any(30 <= t < 75 for t in times)


def test_dropout_leaves_everything_outside_the_window_alone() -> None:
    trip = make_trip()
    broken, _ = inject_faults(
        trip, [FaultSpec(kind="dropout", start_s=30, duration_s=45)], frame=FRAME
    )
    original = {s.monotonic_time: (s.latitude, s.longitude) for s in trip.locations}
    for sample in broken.locations:
        assert (sample.latitude, sample.longitude) == original[sample.monotonic_time]


def test_offset_shifts_by_exactly_the_requested_metres() -> None:
    """E_broken = E + dE, N_broken = N + dN."""
    trip = make_trip()
    broken, _ = inject_faults(
        trip,
        [FaultSpec(kind="offset", start_s=20, duration_s=30, east=4000.0, north=-2500.0)],
        frame=FRAME,
    )
    clean = {s.monotonic_time: s for s in trip.locations}
    moved = [s for s in broken.locations if 20 <= s.monotonic_time - trip.t0 < 50]
    assert len(moved) == 30
    for sample in moved:
        e0, n0 = FRAME.to_local(clean[sample.monotonic_time].latitude,
                                clean[sample.monotonic_time].longitude)
        e1, n1 = FRAME.to_local(sample.latitude, sample.longitude)
        assert e1 - e0 == pytest.approx(4000.0, abs=0.5)
        assert n1 - n0 == pytest.approx(-2500.0, abs=0.5)
        assert sample.synthetic


def test_drift_grows_linearly_with_time() -> None:
    """E_broken = E + k t."""
    trip = make_trip()
    k = 3.0
    broken, _ = inject_faults(
        trip,
        [FaultSpec(kind="drift", start_s=10, duration_s=60, drift_east_ms=k)],
        frame=FRAME,
    )
    clean = {s.monotonic_time: s for s in trip.locations}
    for sample in broken.locations:
        elapsed = sample.monotonic_time - trip.t0 - 10
        if not 0 <= elapsed < 60:
            continue
        e0, _ = FRAME.to_local(clean[sample.monotonic_time].latitude,
                               clean[sample.monotonic_time].longitude)
        e1, _ = FRAME.to_local(sample.latitude, sample.longitude)
        assert e1 - e0 == pytest.approx(k * elapsed, abs=0.5)


def test_jumps_are_gaussian_around_the_truth() -> None:
    """z_broken = z + eps, eps ~ N(0, Sigma)."""
    trip = make_trip(400)
    sigma = 200.0
    broken, _ = inject_faults(
        trip, [FaultSpec(kind="jumps", start_s=0, duration_s=400, sigma_m=sigma)],
        seed=3, frame=FRAME,
    )
    clean = {s.monotonic_time: s for s in trip.locations}
    errors = []
    for sample in broken.locations:
        e0, n0 = FRAME.to_local(clean[sample.monotonic_time].latitude,
                                clean[sample.monotonic_time].longitude)
        e1, n1 = FRAME.to_local(sample.latitude, sample.longitude)
        errors.append((e1 - e0, n1 - n0))
    arr = np.array(errors)
    assert abs(arr.mean()) < sigma * 0.25
    assert arr.std() == pytest.approx(sigma, rel=0.2)


def test_the_clean_track_is_preserved_as_a_reference() -> None:
    trip = make_trip()
    broken, _ = inject_faults(
        trip, [FaultSpec(kind="dropout", start_s=30, duration_s=45)], frame=FRAME
    )
    assert len(broken.reference_locations) == len(trip.locations)
    for a, b in zip(broken.reference_locations, trip.locations):
        assert (a.latitude, a.longitude) == (b.latitude, b.longitude)


def test_corrupted_fixes_are_flagged_synthetic() -> None:
    trip = make_trip()
    broken, _ = inject_faults(
        trip, [FaultSpec(kind="offset", start_s=10, duration_s=20, east=500.0)], frame=FRAME
    )
    touched = [s for s in broken.locations if s.synthetic]
    assert len(touched) == 20
    assert all(not s.synthetic for s in broken.reference_locations)


def test_a_corrupted_fix_can_claim_excellent_accuracy() -> None:
    """The point of the whole exercise: a lying receiver still reports 12 m."""
    trip = make_trip()
    broken, _ = inject_faults(
        trip,
        [FaultSpec(kind="offset", start_s=10, duration_s=20, east=4000.0, accuracy_m=12.0)],
        frame=FRAME,
    )
    lying = [s for s in broken.locations if s.synthetic]
    assert all(s.horizontal_accuracy == 12.0 for s in lying)


def test_the_manifest_records_what_was_done() -> None:
    trip = make_trip()
    broken, report = inject_faults(
        trip, [FaultSpec(kind="offset", start_s=10, duration_s=20, east=900.0, north=-600.0)],
        seed=7, frame=FRAME,
    )
    payload = broken.faults
    assert payload["seed"] == 7
    assert payload["faults"][0]["kind"] == "offset"
    assert payload["faults"][0]["params"]["east_m"] == 900.0
    assert payload["faults"][0]["affected_samples"] == 20


def test_the_composite_scenario_produces_all_three_phases() -> None:
    """Offset, then dropout, then a return with false points."""
    trip = make_trip(300)
    faults = scenario_offset_dropout_recovery(
        start_s=45.0, offset_duration_s=25.0, dropout_duration_s=45.0
    )
    broken, report = inject_faults(trip, faults, seed=2, frame=FRAME)
    kinds = [f["kind"] for f in broken.faults["faults"]]
    assert kinds == ["offset", "dropout", "false_recovery"]

    times = {round(s.monotonic_time - trip.t0) for s in broken.locations}
    assert not (times & set(range(70, 115))), "the dropout window must be empty"
    assert report.removed_samples == 45

    # The fixes just after the dropout are wrong but present.
    after = [s for s in broken.locations if 115 <= s.monotonic_time - trip.t0 < 119]
    assert after and all(s.synthetic for s in after)
    clean = {s.monotonic_time: s for s in trip.locations}
    displaced = [
        haversine_m(s.latitude, s.longitude,
                    clean[s.monotonic_time].latitude, clean[s.monotonic_time].longitude)
        for s in after
    ]
    assert max(displaced) > 50.0, "false recovery points must actually be wrong"


def test_injection_is_reproducible_for_a_seed() -> None:
    trip = make_trip()
    spec = [FaultSpec(kind="jumps", start_s=10, duration_s=50, sigma_m=150.0)]
    a, _ = inject_faults(make_trip(), spec, seed=11, frame=FRAME)
    b, _ = inject_faults(make_trip(), spec, seed=11, frame=FRAME)
    assert [s.latitude for s in a.locations] == [s.latitude for s in b.locations]

    c, _ = inject_faults(make_trip(), spec, seed=12, frame=FRAME)
    assert [s.latitude for s in a.locations] != [s.latitude for s in c.locations]


def test_multiple_faults_compose() -> None:
    trip = make_trip(200)
    broken, _ = inject_faults(
        trip,
        [
            FaultSpec(kind="offset", start_s=10, duration_s=20, east=100.0),
            FaultSpec(kind="drift", start_s=10, duration_s=20, drift_east_ms=2.0),
        ],
        frame=FRAME,
    )
    clean = {s.monotonic_time: s for s in trip.locations}
    sample = next(s for s in broken.locations if round(s.monotonic_time - trip.t0) == 20)
    e0, _ = FRAME.to_local(clean[sample.monotonic_time].latitude, clean[sample.monotonic_time].longitude)
    e1, _ = FRAME.to_local(sample.latitude, sample.longitude)
    assert e1 - e0 == pytest.approx(100.0 + 2.0 * 10.0, abs=1.0)


# --------------------------------------------------------------- validation


def test_unknown_fault_is_rejected() -> None:
    with pytest.raises(FaultError, match="unknown fault"):
        FaultSpec(kind="teleport")


def test_dropout_requires_a_duration() -> None:
    with pytest.raises(FaultError, match="duration"):
        parse_fault_args("dropout", start=10, duration=None)


def test_offset_requires_a_displacement() -> None:
    with pytest.raises(FaultError, match="east"):
        parse_fault_args("offset", start=10, duration=20)


def test_drift_requires_a_rate() -> None:
    with pytest.raises(FaultError, match="drift"):
        parse_fault_args("drift", start=10, duration=20)


def test_negative_start_is_rejected() -> None:
    with pytest.raises(FaultError):
        FaultSpec(kind="dropout", start_s=-5, duration_s=10)


def test_a_trip_with_no_fixes_cannot_be_corrupted() -> None:
    empty = Trip(metadata=TripMetadata(trip_id="empty"))
    with pytest.raises(FaultError, match="no usable GPS"):
        inject_faults(empty, [FaultSpec(kind="dropout", start_s=0, duration_s=1)])
