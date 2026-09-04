"""GPS gates and the TRUSTED / SUSPECT / LOST / RECOVERING state machine."""

from __future__ import annotations

import math

import numpy as np
import pytest

from geotrace.config import GPSQualityConfig
from geotrace.ekf import ExtendedKalmanFilter
from geotrace.gps_quality import (
    GPSQualityMonitor,
    GPSState,
    mahalanobis_gate,
    measurement_sigma,
    physical_gate,
    state_intervals,
)
from geotrace.models import LocationSample

CFG = GPSQualityConfig()
MAX_ACCEL = 6.0


def fix(t: float, lat: float = 59.9311, lon: float = 30.3609, accuracy: float = 10.0,
        speed: float = 10.0, course: float = 90.0) -> LocationSample:
    return LocationSample(
        monotonic_time=t, latitude=lat, longitude=lon,
        horizontal_accuracy=accuracy, speed=speed, speed_accuracy=1.0,
        course=course, course_accuracy=5.0,
    )


# ------------------------------------------------------------ physical gate


def test_physical_gate_accepts_a_reachable_point() -> None:
    """d_max = v dt + 0.5 a_max dt^2 + m."""
    passed, d_max = physical_gate(12.0, dt=1.0, previous_speed=10.0, cfg=CFG, max_accel=MAX_ACCEL)
    assert passed
    assert d_max == pytest.approx(10.0 + 0.5 * MAX_ACCEL + CFG.physical_margin_m)


def test_physical_gate_rejects_a_teleport() -> None:
    """The classic Saint Petersburg failure: a 4 km jump in one second."""
    passed, _ = physical_gate(4000.0, dt=1.0, previous_speed=10.0, cfg=CFG, max_accel=MAX_ACCEL)
    assert not passed


def test_physical_gate_is_exactly_at_the_boundary() -> None:
    _, d_max = physical_gate(0.0, dt=2.0, previous_speed=8.0, cfg=CFG, max_accel=MAX_ACCEL)
    assert physical_gate(d_max, 2.0, 8.0, CFG, MAX_ACCEL)[0]
    assert not physical_gate(d_max + 0.01, 2.0, 8.0, CFG, MAX_ACCEL)[0]


def test_physical_gate_allows_more_after_a_longer_gap() -> None:
    """After a 60 s outage the car really could be a kilometre away."""
    assert physical_gate(900.0, dt=60.0, previous_speed=14.0, cfg=CFG, max_accel=MAX_ACCEL)[0]


def test_physical_gate_margin_is_configurable() -> None:
    tight = GPSQualityConfig(physical_margin_m=1.0)
    assert not physical_gate(30.0, dt=1.0, previous_speed=1.0, cfg=tight, max_accel=MAX_ACCEL)[0]
    loose = GPSQualityConfig(physical_margin_m=100.0)
    assert physical_gate(30.0, dt=1.0, previous_speed=1.0, cfg=loose, max_accel=MAX_ACCEL)[0]


# --------------------------------------------------------- Mahalanobis gate


def test_mahalanobis_of_a_zero_residual_is_zero() -> None:
    passed, d2 = mahalanobis_gate([0.0, 0.0], np.eye(2) * 100.0, CFG.mahalanobis_threshold)
    assert passed and d2 == pytest.approx(0.0)


def test_mahalanobis_matches_the_definition() -> None:
    """D^2 = r^T S^-1 r."""
    S = np.array([[4.0, 0.0], [0.0, 25.0]])
    r = np.array([2.0, 5.0])
    _, d2 = mahalanobis_gate(r, S, 100.0)
    assert d2 == pytest.approx(1.0 + 1.0)


def test_mahalanobis_threshold_is_the_two_dof_chi_square() -> None:
    """The default 9.21 is chi^2(2) at p = 0.99."""
    from scipy.stats import chi2

    assert CFG.mahalanobis_threshold == pytest.approx(chi2.ppf(0.99, df=2), abs=1e-2)


def test_mahalanobis_rejects_beyond_the_threshold() -> None:
    S = np.eye(2)
    passed, d2 = mahalanobis_gate([3.0, 0.0], S, CFG.mahalanobis_threshold)
    assert d2 == pytest.approx(9.0)
    assert passed, "9.0 is just inside the 9.21 threshold"

    passed, d2 = mahalanobis_gate([3.2, 0.0], S, CFG.mahalanobis_threshold)
    assert d2 == pytest.approx(10.24)
    assert not passed


def test_mahalanobis_scales_with_the_covariance() -> None:
    """The same residual is fine when the filter is uncertain and not when it is
    confident. This is the whole point of using it rather than a fixed radius."""
    r = [50.0, 0.0]
    assert not mahalanobis_gate(r, np.eye(2) * 100.0, CFG.mahalanobis_threshold)[0]
    assert mahalanobis_gate(r, np.eye(2) * 10000.0, CFG.mahalanobis_threshold)[0]


def test_mahalanobis_on_a_singular_covariance_fails_closed() -> None:
    passed, d2 = mahalanobis_gate([1.0, 1.0], np.zeros((2, 2)), CFG.mahalanobis_threshold)
    assert not passed and d2 == math.inf


def test_measurement_sigma_has_a_floor() -> None:
    """Receivers under-report accuracy; the sigma must not collapse."""
    assert measurement_sigma(fix(0.0, accuracy=0.5), CFG) == CFG.min_accuracy_sigma_m
    assert measurement_sigma(fix(0.0, accuracy=30.0), CFG) == pytest.approx(30.0)
    assert measurement_sigma(fix(0.0, accuracy=-1.0), CFG) == CFG.max_horizontal_accuracy_m


# --------------------------------------------------------- the state machine


def _promote_to_trusted(monitor: GPSQualityMonitor, start: float = 0.0) -> float:
    """Feed enough consistent fixes to reach TRUSTED. Returns the last fix time."""
    t = start
    for i in range(CFG.recover_count):
        t = start + i
        monitor.update(fix(t), (10.0 * i, 0.0), predicted_speed=10.0)
    assert monitor.state is GPSState.TRUSTED
    return t


def test_a_trip_starts_untrusted() -> None:
    """Nothing is trusted until several consistent fixes have arrived."""
    assert GPSQualityMonitor(CFG).state is GPSState.LOST


def test_promotion_needs_several_consecutive_good_fixes() -> None:
    monitor = GPSQualityMonitor(CFG)
    # A car moving steadily east at 10 m/s, one fix a second.
    for i in range(CFG.recover_count - 1):
        monitor.update(fix(float(i)), (10.0 * i, 0.0), predicted_speed=10.0)
        assert monitor.state is not GPSState.TRUSTED
    last = CFG.recover_count - 1
    monitor.update(fix(float(last)), (10.0 * last, 0.0), predicted_speed=10.0)
    assert monitor.state is GPSState.TRUSTED


def test_trusted_to_suspect_on_one_bad_fix() -> None:
    monitor = GPSQualityMonitor(CFG)
    t = _promote_to_trusted(monitor)
    monitor.update(fix(t + 1), (4000.0, 0.0), predicted_speed=10.0)  # a 4 km jump
    assert monitor.state is GPSState.SUSPECT


def test_suspect_to_lost_after_repeated_bad_fixes() -> None:
    monitor = GPSQualityMonitor(CFG)
    t = _promote_to_trusted(monitor)
    for i in range(CFG.suspect_to_lost_count):
        monitor.update(fix(t + 1 + i), (4000.0 + i, 0.0), predicted_speed=10.0)
    assert monitor.state is GPSState.LOST


def test_dropout_moves_to_lost() -> None:
    """No fix at all for longer than lost_gap_s is itself a failure."""
    monitor = GPSQualityMonitor(CFG)
    t = _promote_to_trusted(monitor)
    assert monitor.note_gap(t + CFG.lost_gap_s + 1.0)
    assert monitor.state is GPSState.LOST


def test_a_short_gap_does_not_trigger_lost() -> None:
    monitor = GPSQualityMonitor(CFG)
    t = _promote_to_trusted(monitor)
    assert not monitor.note_gap(t + CFG.lost_gap_s - 0.5)
    assert monitor.state is GPSState.TRUSTED


def test_recovery_passes_through_recovering_and_needs_several_fixes() -> None:
    """The point of RECOVERING: a receiver coming back emits false fixes first."""
    monitor = GPSQualityMonitor(CFG)
    t = _promote_to_trusted(monitor)
    monitor.note_gap(t + 60.0)
    assert monitor.state is GPSState.LOST

    t += 60.0
    monitor.update(fix(t), (600.0, 0.0), predicted_speed=10.0)
    assert monitor.state is GPSState.RECOVERING
    for i in range(1, CFG.recover_count - 1):
        monitor.update(fix(t + i), (600.0 + 10 * i, 0.0), predicted_speed=10.0)
        assert monitor.state is GPSState.RECOVERING, "trust must not return early"
    monitor.update(
        fix(t + CFG.recover_count), (600.0 + 10 * CFG.recover_count, 0.0), predicted_speed=10.0
    )
    assert monitor.state is GPSState.TRUSTED


def test_a_false_fix_during_recovery_drops_back_to_lost() -> None:
    monitor = GPSQualityMonitor(CFG)
    t = _promote_to_trusted(monitor)
    monitor.note_gap(t + 60.0)
    t += 60.0
    monitor.update(fix(t), (600.0, 0.0), predicted_speed=10.0)
    assert monitor.state is GPSState.RECOVERING
    monitor.update(fix(t + 1.0), (9000.0, 9000.0), predicted_speed=10.0)  # implausible
    assert monitor.state is GPSState.LOST


def test_poor_accuracy_alone_is_rejected() -> None:
    monitor = GPSQualityMonitor(CFG)
    result = monitor.update(fix(0.0, accuracy=CFG.max_horizontal_accuracy_m + 10), (0.0, 0.0))
    assert not result.accepted and "accuracy_too_poor" in result.reasons


def test_good_accuracy_is_not_sufficient_on_its_own() -> None:
    """The failure mode the specification calls out: a confident wrong fix."""
    monitor = GPSQualityMonitor(CFG)
    t = _promote_to_trusted(monitor)
    result = monitor.update(fix(t + 1, accuracy=5.0), (4000.0, 0.0), predicted_speed=10.0)
    assert not result.accepted
    assert "physical_gate" in result.reasons
    assert result.sigma_m == pytest.approx(5.0), "the fix claimed to be accurate"


def test_a_coherent_fix_outside_graph_coverage_is_not_rejected() -> None:
    """A stale or clipped road graph cannot manufacture a GPS outage."""
    monitor = GPSQualityMonitor(CFG)
    t = _promote_to_trusted(monitor)
    result = monitor.update(
        fix(t + 1), (35.0, 5.0), predicted_speed=10.0,
        road_distance_m=CFG.max_distance_to_road_m + 50.0,
    )
    assert result.accepted
    assert result.road_distance_m == pytest.approx(CFG.max_distance_to_road_m + 50.0)


def test_software_simulated_fixes_are_rejected() -> None:
    monitor = GPSQualityMonitor(CFG)
    sample = fix(0.0)
    sample.source_information = {"is_simulated_by_software": True}
    result = monitor.update(sample, (0.0, 0.0))
    assert not result.accepted and "simulated_by_software" in result.reasons


def test_course_is_ignored_below_the_speed_threshold() -> None:
    """CoreLocation course is meaningless at walking pace."""
    monitor = GPSQualityMonitor(CFG)
    _promote_to_trusted(monitor)
    slow = fix(10.0, speed=1.0, course=270.0)  # opposite to the filter heading
    result = monitor.update(slow, (35.0, 0.0), predicted_heading=0.0, predicted_speed=1.0)
    assert "course_mismatch" not in result.reasons


def test_course_mismatch_is_caught_when_moving() -> None:
    monitor = GPSQualityMonitor(CFG)
    _promote_to_trusted(monitor)
    wrong = fix(10.0, speed=12.0, course=270.0)  # heading west
    result = monitor.update(
        wrong, (35.0, 0.0), predicted_heading=0.0, predicted_speed=12.0  # filter says east
    )
    assert not result.accepted and "course_mismatch" in result.reasons


def test_speed_mismatch_is_caught() -> None:
    monitor = GPSQualityMonitor(CFG)
    _promote_to_trusted(monitor)
    result = monitor.update(fix(10.0, speed=40.0), (35.0, 0.0), predicted_speed=2.0)
    assert not result.accepted and "speed_mismatch" in result.reasons


def test_invalid_fixes_are_refused() -> None:
    monitor = GPSQualityMonitor(CFG)
    bad = LocationSample(monotonic_time=1.0, latitude=59.9, longitude=30.3,
                         horizontal_accuracy=-1.0)
    assert not monitor.update(bad, (0.0, 0.0)).accepted


def test_the_mahalanobis_gate_is_dropped_once_the_filter_has_lost_track() -> None:
    """A dead-reckoning solution that has run free for a minute is not a valid
    reference. Gating returning fixes against it would make the filter defend
    its own drift and never recover."""
    monitor = GPSQualityMonitor(CFG)
    _promote_to_trusted(monitor)
    monitor.note_gap(100.0)
    assert monitor.state is GPSState.LOST
    P = np.eye(6) * 100.0
    # 80 m from the prediction: far outside the 9.21 gate a tracking filter
    # would apply, but the gate is not applied at all in LOST.
    result = monitor.update(
        fix(101.0), (80.0, 0.0), predicted_xy=(0.0, 0.0), covariance=P, predicted_speed=10.0
    )
    assert result.mahalanobis is None
    assert "mahalanobis_gate" not in result.reasons
    assert result.accepted


def test_while_lost_a_fix_is_checked_against_the_previous_fix_not_the_filter() -> None:
    """Recovery is rebuilt from consistency between fixes."""
    monitor = GPSQualityMonitor(CFG)
    _promote_to_trusted(monitor)
    monitor.note_gap(200.0)
    assert monitor.state is GPSState.LOST

    # First fix back: nothing to compare against yet, so it is taken on trust.
    monitor.update(fix(201.0, speed=10.0, course=90.0), (0.0, 0.0), predicted_speed=10.0)
    assert monitor.state is GPSState.RECOVERING

    # A fix claiming to drive east at 10 m/s that actually moved 10 m north.
    result = monitor.update(
        fix(202.0, speed=10.0, course=90.0), (0.0, 10.0), predicted_speed=10.0
    )
    assert not result.accepted
    assert "course_inconsistent_with_previous_fix" in result.reasons
    assert monitor.state is GPSState.LOST


def test_a_consistent_chain_of_fixes_restores_trust_after_a_long_outage() -> None:
    """The receiver comes back and behaves; trust must return."""
    monitor = GPSQualityMonitor(CFG)
    _promote_to_trusted(monitor)
    monitor.note_gap(200.0)

    for i in range(CFG.recover_count + 1):
        monitor.update(
            fix(201.0 + i, speed=10.0, course=90.0), (10.0 * i, 0.0), predicted_speed=10.0
        )
    assert monitor.state is GPSState.TRUSTED


def test_scattered_false_recovery_fixes_never_restore_trust() -> None:
    """The specification's exact failure mode: several wrong points right after
    the signal returns. They are individually plausible but mutually
    inconsistent, so the chain RECOVERING needs never forms."""
    monitor = GPSQualityMonitor(CFG)
    _promote_to_trusted(monitor)
    monitor.note_gap(200.0)

    scatter = [(0.0, 0.0), (0.0, 40.0), (30.0, -35.0), (-20.0, 25.0), (45.0, 30.0)]
    for i, point in enumerate(scatter):
        monitor.update(fix(201.0 + i, speed=10.0, course=90.0), point, predicted_speed=10.0)
    assert monitor.state is not GPSState.TRUSTED


def test_state_intervals_collapse_the_history() -> None:
    monitor = GPSQualityMonitor(CFG)
    _promote_to_trusted(monitor)
    intervals = state_intervals(monitor.history)
    assert intervals[-1]["state"] == GPSState.TRUSTED.value
    assert intervals[-1]["end_s"] >= intervals[-1]["start_s"]


def test_summary_counts_rejections_by_reason() -> None:
    monitor = GPSQualityMonitor(CFG)
    t = _promote_to_trusted(monitor)
    monitor.update(fix(t + 1), (4000.0, 0.0), predicted_speed=10.0)
    summary = monitor.summary()
    assert summary["fixes_rejected"] == 1
    assert summary["rejection_reasons"]["physical_gate"] == 1


def test_simulated_fixes_are_rejected_by_default() -> None:
    """A spoofed location is not evidence about where a car was."""
    monitor = GPSQualityMonitor(CFG)
    sample = fix(0.0)
    sample.source_information = {"is_simulated_by_software": True}
    assert not monitor.update(sample, (0.0, 0.0)).accepted


def test_simulated_fixes_can_be_allowed_explicitly() -> None:
    """Every fix recorded in the iOS Simulator carries the flag, so the pipeline
    needs a deliberate, clearly named opt-in to be testable end to end."""
    cfg = GPSQualityConfig(allow_simulated_fixes=True)
    monitor = GPSQualityMonitor(cfg)
    sample = fix(0.0)
    sample.source_information = {"is_simulated_by_software": True}
    result = monitor.update(sample, (0.0, 0.0))
    assert "simulated_by_software" not in result.reasons
    assert result.accepted
