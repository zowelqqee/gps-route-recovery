"""Integrated map-distance constraints and their anti-circularity gates."""

import numpy as np
import pytest

from geotrace.pacman_tracker.intervals import IntervalConfig, common_drift
from geotrace.pacman_tracker.speed import BA, KA, GlobalSpeedTracker, SpeedConfig


def _cfg() -> IntervalConfig:
    return IntervalConfig(enabled=True)


def test_three_independent_routes_may_report_common_drift():
    obs = common_drift(
        _cfg(), 20.0, np.array([38.0, 41.0, 39.0]),
        np.array([0.34, 0.33, 0.33]), np.array([10, 11, 12]), 8.0)
    assert obs.reject_reason == ""
    assert obs.drift_m == pytest.approx(39.32, abs=0.1)
    assert obs.mass == pytest.approx(1.0)
    assert obs.n_routes == 3


def test_one_route_cannot_feed_its_own_length_back():
    obs = common_drift(
        _cfg(), 20.0, np.array([40.0]), np.array([1.0]), np.array([7]), 8.0)
    assert "cannot corroborate" in obs.reject_reason


def test_a_token_sibling_does_not_make_a_dominant_route_independent():
    obs = common_drift(
        _cfg(), 20.0, np.array([40.0, 41.0]),
        np.array([0.92, 0.08]), np.array([7, 8]), 8.0)
    assert "effective routes" in obs.reject_reason


def test_high_spread_routes_do_not_create_global_feedback():
    obs = common_drift(
        _cfg(), 20.0, np.array([40.0, -20.0, 5.0]),
        np.array([1 / 3, 1 / 3, 1 / 3]), np.array([1, 2, 3]), 8.0)
    assert obs.reject_reason


def test_duplicate_descendants_count_as_one_evidence_group():
    obs = common_drift(
        _cfg(), 20.0, np.array([38.0, 39.0, 40.0]),
        np.array([0.3, 0.3, 0.4]), np.array([5, 5, 5]), 8.0)
    assert "cannot corroborate" in obs.reject_reason
    assert obs.n_routes == 1


def _triangle(tracker: GlobalSpeedTracker, measured_accel: float = 0.7) -> None:
    for i in range(100):
        tracker.predict(measured_accel, 0.1)
        tracker.remember((i + 1) * 0.1)
    for i in range(100):
        tracker.predict(-measured_accel, 0.1)
        tracker.remember(10.0 + (i + 1) * 0.1)


def test_interval_is_integrated_distance_not_average_terminal_speed():
    tracker = GlobalSpeedTracker(SpeedConfig(accel_bias_rw=0.0), v0=0.0)
    tracker.open_interval(0.0)
    _triangle(tracker, 1.0)
    terminal_before = tracker.speed
    applied, ins, _ = tracker.apply_interval(105.0, 2.0)
    assert applied
    assert ins == pytest.approx(100.0, abs=0.2)
    # The average speed is 5.25 m/s. An interval update must not set the
    # instantaneous terminal speed to that value.
    assert abs(tracker.speed - 5.25) > 2.0
    assert abs(tracker.speed - terminal_before) < 2.0


def test_map_interval_moves_accel_scale_in_the_observable_direction():
    cfg = SpeedConfig(
        accel_scale_enabled=True, accel_bias_rw=0.0,
        initial_accel_bias_sigma=0.01, initial_accel_scale_sigma=0.35)
    tracker = GlobalSpeedTracker(cfg, v0=0.0, accel_bias0=0.0)
    tracker.open_interval(0.0)
    _triangle(tracker, 0.7)  # true profile is +/-1.0 and covers 100 m
    before = tracker.accel_scale
    applied, ins, _ = tracker.apply_interval(100.0, 2.0)
    assert applied and ins == pytest.approx(70.0, abs=0.2)
    assert tracker.accel_scale > before
    assert tracker.x[BA] == pytest.approx(0.0, abs=0.05)


def test_scale_stays_unobservable_without_an_interval():
    cfg = SpeedConfig(accel_scale_enabled=True, accel_scale_rw=0.0)
    tracker = GlobalSpeedTracker(cfg, v0=0.0)
    prior = tracker.accel_scale
    _triangle(tracker, 0.7)
    assert tracker.accel_scale == pytest.approx(prior)
    assert tracker.x[KA] == pytest.approx(prior)


def test_fixed_lag_revises_only_the_bounded_recent_history():
    cfg = SpeedConfig(fixed_lag_s=5.0, accel_bias_rw=0.0)
    tracker = GlobalSpeedTracker(cfg, v0=0.0)
    tracker.open_interval(0.0)
    reports = {}
    for i in range(200):
        tracker.predict(1., 0.1)
        t = (i + 1) * 0.1
        tracker.remember(t)
        if i in (139, 179, 199):
            reports[round(t)] = tracker.report(t)
    old_14 = reports[14].distance_m
    old_18 = reports[18].distance_m
    applied, _, _ = tracker.apply_interval(230.0, 2.0, event_t=20.0)
    assert applied
    assert reports[14].distance_m == pytest.approx(old_14)
    assert reports[18].distance_m != pytest.approx(old_18)
    assert tracker.last_interval_states_modified <= 52


@pytest.mark.parametrize("lag_s", [0.0, 5.0, 10.0, 20.0, 30.0])
def test_fixed_lag_horizon_is_bounded(lag_s):
    tracker = GlobalSpeedTracker(
        SpeedConfig(fixed_lag_s=lag_s, accel_bias_rw=0.0), v0=0.0)
    tracker.open_interval(0.0)
    for i in range(300):
        tracker.predict(0.4 if i < 150 else -0.4, 0.1)
        tracker.remember((i + 1) * 0.1)
    assert tracker.apply_interval(110.0, 3.0, event_t=30.0)[0]
    expected = 1 if lag_s == 0.0 else min(300, int(round(lag_s / 0.1)) + 1)
    assert abs(tracker.last_interval_states_modified - expected) <= 1


def test_non_distance_speed_sources_cannot_calibrate_accel_scale():
    cfg = SpeedConfig(accel_scale_enabled=True, accel_scale_rw=0.0)
    tracker = GlobalSpeedTracker(cfg, v0=4.0)
    prior = tracker.accel_scale
    prior_variance = tracker.P[KA, KA]
    for _ in range(100):
        tracker.predict(0.8, 0.1)
        tracker.spectral_update(5.0, 2.0, 0.1)
        tracker.lateral_anchor(5.0 * 0.2, 0.2, 0.1)
    assert tracker.accel_scale == pytest.approx(prior)
    assert tracker.P[KA, KA] == pytest.approx(prior_variance)
