"""The one speed filter: it must integrate, stay honest, and be anchorable.

There is one car, so there is one of these. Everything that used to let each
road hypothesis hold its own opinion about how fast the car was going lives
here now, and these tests are what stop it drifting back.
"""

import math

import numpy as np
import pytest

from geotrace.pacman_tracker.speed import BA, D, GlobalSpeedTracker, SpeedConfig, V


def _tracker(v0=10.0, **kw):
    cfg = SpeedConfig(**kw)
    return GlobalSpeedTracker(cfg, v0=v0)


def test_measurement_authority_guards_are_production_defaults():
    cfg = SpeedConfig()
    assert cfg.zupt_motion_guard_enabled
    assert cfg.lateral_consensus_guard_enabled
    assert cfg.spectral_saturation_guard_enabled


def test_constant_acceleration_integrates_exactly():
    t = _tracker(v0=0.0)
    for _ in range(100):                       # 10 s at 1 m/s^2
        t.predict(1.0, 0.1)
    assert t.speed == pytest.approx(10.0, abs=1e-9)
    assert t.distance == pytest.approx(50.0, abs=0.06)


def test_bias_is_subtracted_before_integration():
    t = GlobalSpeedTracker(SpeedConfig(), v0=0.0, accel_bias0=0.4)
    for _ in range(100):
        t.predict(0.4, 0.1)
    assert abs(t.speed) < 1e-9
    assert abs(t.distance) < 1e-6


def test_uncertainty_grows_open_loop():
    """A dead-reckoner that does not become less sure over time is lying."""
    t = _tracker()
    sigmas = []
    for _ in range(600):                       # 60 s
        t.predict(0.0, 0.1)
        sigmas.append((t.sigma_distance, t.sigma_speed))
    d = [a for a, _ in sigmas]
    v = [b for _, b in sigmas]
    assert d == sorted(d) and v == sorted(v)
    assert d[-1] > 100.0, "60 s of open-loop integration is worth 100+ m of doubt"


def test_a_stop_pins_the_speed_without_collapsing_the_variance():
    cfg = SpeedConfig()
    t = GlobalSpeedTracker(cfg, v0=3.0)
    for step in range(300):
        t.predict(0.0, 0.1)
        t.zero_velocity(0.0, 0.1, 0.0, run_s=0.1 * (step + 1))
    assert abs(t.speed) < 0.3
    assert t.sigma_speed >= cfg.v_sigma_floor_ms - 1e-9


def test_a_stop_measures_both_biases():
    """Both converge, at very different rates - and that is correct.

    The gyro's true rate while parked really is zero, so its bias is sharply
    observable. The accelerometer's parked offset is 0.65 m/s^2 away from its
    driving offset on this recorder, so the observation is deliberately loose
    and convergence is partial. A filter that snapped ``b_a`` to the parked
    value would be confidently wrong the moment the car moved.
    """
    t = GlobalSpeedTracker(SpeedConfig(), v0=0.0)
    for _ in range(1200):                      # two minutes parked
        t.predict(0.25, 0.1)
        t.zero_velocity(0.25, 0.1, 0.004, run_s=120.0)
    assert 0.10 < t.x[BA] < 0.30
    assert t.gyro_bias == pytest.approx(0.004, abs=0.001)


def test_a_brief_false_stop_does_not_pin_the_speed():
    """A variance detector cannot tell smooth cruising from idling.

    12 m/s cannot become 0 during one second the detector itself called quiet:
    a quiet stretch is one in which the accelerometer reported no braking. The
    stop is discounted on those grounds, not on statistical ones.
    """
    t = _tracker(v0=12.0)
    for step in range(10):                     # 1 s of false stop
        t.predict(0.0, 0.1)
        t.zero_velocity(0.0, 0.1, 0.0, run_s=0.1 * (step + 1))
    assert t.speed > 10.0


def test_a_real_stop_is_still_decisive():
    t = _tracker(v0=12.0)
    for step in range(300):                    # 30 s of genuine stop
        t.predict(0.0, 0.1)
        t.zero_velocity(0.0, 0.1, 0.0, run_s=0.1 * (step + 1))
    assert abs(t.speed) < 0.5


def test_a_stop_reached_by_braking_is_accepted_at_once():
    """The credible case: the speed estimate is already low when it arrives."""
    t = _tracker(v0=1.5)
    for step in range(20):
        t.predict(0.0, 0.1)
        t.zero_velocity(0.0, 0.1, 0.0, run_s=0.1 * (step + 1))
    assert abs(t.speed) < 0.5


def test_spectral_road_speed_rejects_a_false_zupt_without_mutating_state():
    t = _tracker(v0=12.0, zupt_motion_guard_enabled=True)
    x0, p0 = t.x.copy(), t.P.copy()
    accepted = t.zero_velocity(0.0, 0.1, 0.0, run_s=6.0,
                               spectral_speed=11.0)
    assert accepted is False
    assert np.array_equal(t.x, x0)
    assert np.array_equal(t.P, p0)
    assert t.counts["zupt_motion_rejected"] == 1


def test_low_spectral_speed_does_not_block_a_real_zupt():
    t = _tracker(v0=1.0, zupt_motion_guard_enabled=True)
    accepted = t.zero_velocity(0.0, 0.1, 0.0, run_s=2.0,
                               spectral_speed=1.0)
    assert accepted is True
    assert t.counts["zupt"] == 1


# ------------------------------------------------------- v = a_lat / omega


def test_a_steady_bend_recovers_the_speed():
    t = _tracker(v0=5.0)
    true_v, radius = 14.0, 60.0
    omega = true_v / radius
    for _ in range(60):
        t.predict(0.0, 0.1)
        t.lateral_anchor(true_v * omega, omega, 0.1)
    assert t.speed == pytest.approx(true_v, abs=1.5)


def test_it_works_at_any_radius():
    """The ratio is scale-free: a motorway sweep and a street corner both work."""
    for radius in (20.0, 60.0, 200.0):
        t = _tracker(v0=4.0)
        omega = 12.0 / radius
        for _ in range(100):
            t.predict(0.0, 0.1)
            t.lateral_anchor(12.0 * omega, omega, 0.1)
        assert t.speed == pytest.approx(12.0, abs=2.0), radius


def test_a_right_turn_reads_the_same_as_a_left_turn():
    t = _tracker(v0=4.0)
    omega = -12.0 / 50.0
    for _ in range(100):
        t.predict(0.0, 0.1)
        t.lateral_anchor(12.0 * omega, omega, 0.1)
    assert t.speed == pytest.approx(12.0, abs=2.0)


def test_the_anchor_weakens_smoothly_as_the_turn_flattens():
    """No threshold decides whether this is usable - its variance does.

    sigma_v^2 = sigma_a^2/omega^2 + a_lat^2 sigma_omega^2/omega^4, so a barely
    perceptible drift in the steering yields a measurement too vague to matter
    and a real turn yields a sharp one.
    """
    reported = []
    for omega in (0.03, 0.08, 0.2, 0.5):
        t = _tracker(v0=10.0)
        out = t.lateral_anchor(10.0 * omega, omega, 0.1)
        assert out is not None
        reported.append(out[1])
    assert reported == sorted(reported, reverse=True)


def test_an_implausible_ratio_is_refused():
    t = _tracker(v0=8.0)
    before = t.speed
    t.lateral_anchor(12.0, 0.03, 0.1)          # implies 400 m/s
    assert t.speed == pytest.approx(before)
    assert t.counts["lateral_rejected"] >= 1


def test_lateral_and_longitudinal_disagreeing_in_sign_is_refused():
    """In circular motion a_lat points into the turn. A confident disagreement
    is not a turn - it is the mount moving, or a bump."""
    t = _tracker(v0=8.0)
    before = t.speed
    t.lateral_anchor(-2.0, 0.2, 0.1)
    assert t.speed == pytest.approx(before)


def test_lateral_consensus_guard_softens_only_a_two_source_disagreement():
    plain = _tracker(v0=15.0, lateral_correlation_s=0.0,
                     lateral_consensus_guard_enabled=False)
    guarded = _tracker(v0=15.0, lateral_correlation_s=0.0,
                       lateral_consensus_guard_enabled=True)
    for tracker in (plain, guarded):
        tracker.lateral_anchor(8.0 * 0.2, 0.2, 0.1,
                               spectral_speed=14.0)
    assert guarded.speed > plain.speed
    assert guarded.counts["lateral_consensus_downweighted"] == 1

    lone_disagreement = _tracker(
        v0=15.0, lateral_correlation_s=0.0,
        lateral_consensus_guard_enabled=True)
    lone_disagreement.lateral_anchor(8.0 * 0.2, 0.2, 0.1,
                                     spectral_speed=7.0)
    assert lone_disagreement.counts["lateral_consensus_downweighted"] == 0


def test_delayed_lateral_anchor_uses_velocity_at_measurement_time():
    """A lagged observation's residual belongs to the retained state."""
    cfg = SpeedConfig(
        lateral_delayed_correction_enabled=True,
        lateral_anchor_delay_s=0.5,
        d_sigma_floor_m=0.0,
        v_sigma_floor_ms=0.0,
        accel_bias_sigma_floor=0.0,
    )
    delayed = GlobalSpeedTracker(cfg, v0=5.0)
    delayed.remember(1.0)
    delayed.predict(30.0, 0.5)  # reception-time v is 20 m/s

    # The observation says v(1.0) = 10 m/s. Its innovation is +5 m/s, not
    # -10 m/s relative to the reception-time state.
    before = delayed.speed
    delayed.lateral_anchor(2.0, 0.2, 0.1, t=1.5)
    assert delayed.speed > before


def test_zero_anchor_delay_is_a_true_no_op():
    """``lateral_anchor_delay_s == 0`` must reduce to today's undelayed update
    exactly - the history lookup lands on the current tick itself, so no
    replay happens. This is the plumbing's own correctness check."""
    kw = dict(d_sigma_floor_m=0.0, v_sigma_floor_ms=0.0,
              accel_bias_sigma_floor=0.0)
    baseline = GlobalSpeedTracker(SpeedConfig(**kw), v0=5.0)
    flagged = GlobalSpeedTracker(
        SpeedConfig(lateral_delayed_correction_enabled=True,
                    lateral_anchor_delay_s=0.0, **kw),
        v0=5.0)
    for tracker in (baseline, flagged):
        tracker.remember(1.0)
        tracker.predict(30.0, 0.5)
    baseline.lateral_anchor(2.0, 0.2, 0.1)
    flagged.lateral_anchor(2.0, 0.2, 0.1, t=1.5)
    assert flagged.speed == pytest.approx(baseline.speed)
    assert flagged.distance == pytest.approx(baseline.distance)
    assert flagged.counts["lateral_delayed"] == 0


def test_delayed_correction_defaults_off():
    cfg = SpeedConfig()
    assert cfg.lateral_delayed_correction_enabled is False

    without_flag = GlobalSpeedTracker(SpeedConfig(), v0=5.0)
    with_flag_off = GlobalSpeedTracker(SpeedConfig(lateral_anchor_delay_s=0.5), v0=5.0)
    for tracker in (without_flag, with_flag_off):
        tracker.remember(1.0)
        tracker.predict(30.0, 0.5)
        tracker.lateral_anchor(2.0, 0.2, 0.1, t=1.5)
    assert with_flag_off.speed == pytest.approx(without_flag.speed)
    assert with_flag_off.counts["lateral_delayed"] == 0


def test_a_shock_is_not_a_turn():
    t = _tracker(v0=8.0)
    before = t.speed
    t.lateral_anchor(20.0 * 0.3, 0.3, 0.1, shock=True)
    assert t.speed == pytest.approx(before)


def test_predictive_spectral_guard_weakens_a_plateau_downward_pull():
    plain = _tracker(v0=20.0, spectral_correlation_s=0.0,
                     spectral_saturation_guard_enabled=False)
    guarded = _tracker(v0=20.0, spectral_correlation_s=0.0,
                       spectral_saturation_guard_enabled=True)
    for tracker in (plain, guarded):
        tracker.predict(0.0, 0.1)
        tracker.spectral_update(13.0, 2.0, 0.1)
    assert guarded.speed > plain.speed
    assert guarded.counts["spectral_guarded"] == 1


def test_predictive_spectral_guard_keeps_upward_and_braking_updates_normal():
    upward = _tracker(v0=9.0, spectral_correlation_s=0.0,
                      spectral_saturation_guard_enabled=True)
    upward.predict(0.0, 0.1)
    upward.spectral_update(13.0, 2.0, 0.1)
    assert upward.counts["spectral_guarded"] == 0

    braking = _tracker(v0=20.0, spectral_correlation_s=0.0,
                       spectral_saturation_guard_enabled=True)
    braking.predict(-1.0, 0.1)
    braking.spectral_update(13.0, 2.0, 0.1)
    assert braking.counts["spectral_guarded"] == 0


# ------------------------------------------------------------------ rails


def test_the_envelope_only_fires_outside_the_band():
    t = _tracker(v0=12.0)
    before = t.speed
    t.envelope(20.0, 0.1)
    assert t.speed == pytest.approx(before)


def test_sustained_reversing_is_railed_out():
    """Sign matters: v < 0 makes the map predict the road turning the wrong way."""
    cfg = SpeedConfig()
    t = GlobalSpeedTracker(cfg, v0=-8.0)
    for _ in range(300):
        t.predict(0.0, 0.1)
        t.envelope(16.7, 0.1)
    assert t.speed > -cfg.reverse_speed_limit_ms - 0.5


def test_the_envelope_does_not_drag_a_legitimate_speed_through_zero():
    """A car exceeding the posted limit is common; the rail must nudge, not
    dominate. Applied at face sigma ten times a second it used to pull the
    estimate down through the limit, past zero, into the reversing rail."""
    t = _tracker(v0=15.0)
    for _ in range(300):
        t.predict(0.0, 0.1)
        t.envelope(11.0, 0.1)
    assert 8.0 < t.speed < 15.5


def test_covariance_stays_positive_semidefinite():
    rng = np.random.default_rng(3)
    t = _tracker(v0=9.0)
    for i in range(2000):
        t.predict(float(rng.normal(0, 0.4)), 0.1)
        if i % 37 == 0:
            t.zero_velocity(0.1, 0.1, 0.0, run_s=5.0)
        if i % 11 == 0:
            t.lateral_anchor(9.0 * 0.15, 0.15, 0.1)
        t.envelope(16.7, 0.1)
        assert np.all(np.isfinite(t.P))
        assert np.all(np.linalg.eigvalsh(t.P) > -1e-9)
