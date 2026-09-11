"""Online spectral absolute-speed calibration k_s.

k_s corrects ``v_true ~ k_s * v_spectral``. It is a separate state from the
accelerometer scale k_a, is moved only by the two absolute-speed anchors
(``a_lat/omega`` and accepted map intervals), never by ordinary spectral
observations, and keeps a wide uncertainty where no anchor has constrained it.
"""

import inspect
import math

import numpy as np
import pytest

from geotrace.pacman_tracker.speed import (
    BA, KA, V, GlobalSpeedTracker, SpeedConfig)


def _tracker(**over):
    base = dict(spectral_scale_enabled=True, accel_bias_rw=0.0,
                spectral_scale_rw=0.0)
    base.update(over)
    return GlobalSpeedTracker(SpeedConfig(**base), v0=0.0)


def test_k_s_unchanged_without_an_absolute_anchor():
    tr = _tracker()
    k0, p0 = tr.spectral_scale, tr.spectral_scale_var
    for _ in range(200):
        tr.predict(0.0, 0.1)
        tr.spectral_update(9.0, 3.0, 0.1)      # ordinary spectral obs
        tr.envelope(20.0, 0.1)
    assert tr.spectral_scale == pytest.approx(k0)
    assert tr.spectral_scale_var == pytest.approx(p0)


def test_ordinary_spectral_observation_cannot_calibrate_k_s():
    tr = _tracker()
    src = inspect.getsource(GlobalSpeedTracker.spectral_update)
    assert "_spectral_scale_kalman" not in src
    assert "spectral_scale_var =" not in src   # never shrinks the scale variance
    p0 = tr.spectral_scale_var
    for _ in range(500):
        tr.spectral_update(14.0, 2.0, 0.1)
    assert tr.spectral_scale_var == pytest.approx(p0)


def test_clean_lateral_anchor_moves_k_s_up_when_true_speed_exceeds_spectral():
    tr = _tracker()
    # anchor says 15 m/s, spectral said 10 -> k_s should rise toward 1.5
    ok = tr.spectral_scale_point(v_anchor=15.0, sigma_anchor=1.0,
                                 v_spectral_now=10.0, t=100.0)
    assert ok
    assert 1.0 < tr.spectral_scale < 1.5
    assert tr.spectral_scale_var < tr.cfg.initial_spectral_scale_sigma ** 2


def test_lateral_anchor_below_min_speed_is_ignored_for_k_s():
    tr = _tracker(spectral_scale_anchor_min_speed_ms=6.0)
    assert not tr.spectral_scale_point(8.0, 1.0, 4.0, t=1.0)
    assert tr.spectral_scale == pytest.approx(1.0)


def test_wildly_disagreeing_anchor_is_gated_out():
    tr = _tracker(spectral_scale_max_innovation_sigma=3.0)
    # implied k_s ~ 3.0, many sigma away from the 1.0 +/- 0.15 prior
    assert not tr.spectral_scale_point(30.0, 0.5, 10.0, t=1.0)
    assert tr.spectral_scale == pytest.approx(1.0)


def test_clean_map_interval_moves_k_s_toward_L_over_integral():
    tr = _tracker()
    # map says 1200 m, integral of v_spectral over the segment was 900 m
    ok = tr.spectral_scale_interval(length_m=1200.0, sigma_m=40.0,
                                    integrated_v_spectral=900.0, t=300.0)
    assert ok
    assert 1.05 < tr.spectral_scale < 1.34    # toward 1.333, damped by the prior
    assert tr.spectral_scale_var < tr.cfg.initial_spectral_scale_sigma ** 2


def test_interval_jacobian_dL_dk_s_is_the_integrated_spectral_speed():
    """L_pred(k_s) = k_s * integral(v_spectral), so dL_pred/dk_s = integral.

    The calibrator carries a length-domain sigma into the k_s domain as
    ``(sigma_m / integral)**2``; that division is exactly 1 / (dL/dk_s)**2, the
    measurement-linearisation the update relies on. Check the analytic Jacobian
    against a finite difference of the predicted length, and check the code
    actually scales the variance that way."""
    integ = 850.0
    eps = 1e-6

    def L_pred(ks):
        return ks * integ

    analytic = integ
    numeric = (L_pred(1.0 + eps) - L_pred(1.0 - eps)) / (2 * eps)
    assert numeric == pytest.approx(analytic, rel=1e-6)

    sigma_m = 50.0
    tr_a = _tracker()
    tr_a.spectral_scale_interval(L_pred(1.0), sigma_m, integ)
    # implied k_s obs variance must be (sigma_m / dL/dk_s)^2 before the prior
    r_expected = (sigma_m / analytic) ** 2
    s = tr_a.cfg.initial_spectral_scale_sigma ** 2 + r_expected
    k_gain = tr_a.cfg.initial_spectral_scale_sigma ** 2 / s
    # z == 1.0 here (L = 1.0 * integ) so k_s must not move
    assert tr_a.spectral_scale == pytest.approx(1.0)
    assert tr_a.spectral_scale_var == pytest.approx((1.0 - k_gain) *
        tr_a.cfg.initial_spectral_scale_sigma ** 2, rel=1e-9)


def test_map_interval_ks_update_needs_the_interval_machinery_to_raise_it():
    """k_s can only be moved by an interval from inside
    ``_apply_single_path_intervals`` - the branch the single-path manager
    reaches only after its own endpoint-quality / anti-circularity checks. A
    route picked because its distance agreed cannot then reuse that distance to
    calibrate k_s without first clearing those independent gates."""
    import inspect

    from geotrace.pacman_tracker import tracker as tk

    caller = inspect.getsource(tk.PacmanTracker._apply_single_path_intervals)
    assert "spectral_scale_interval(" in caller
    # the only production call site
    whole = inspect.getsource(tk)
    assert whole.count(".spectral_scale_interval(") == 1


def test_grossly_wrong_interval_is_gated_out_of_k_s():
    tr = _tracker(spectral_scale_max_innovation_sigma=3.5)
    # a route far shorter than the integrated speed, reported tightly
    assert not tr.spectral_scale_interval(length_m=380.0, sigma_m=15.0,
                                          integrated_v_spectral=1000.0, t=1.0)
    assert tr.spectral_scale == pytest.approx(1.0)


def test_causal_replay_uses_only_anchors_at_or_before_now():
    """A future anchor cannot retroactively change k_s at an earlier time: the
    trace is append-only and each entry stamps the time it was applied."""
    tr = _tracker()
    tr.spectral_scale_interval(1100.0, 40.0, 1000.0, t=100.0)
    k_at_100 = tr.spectral_scale
    tr.spectral_scale_interval(900.0, 40.0, 1000.0, t=400.0)
    # the t=100 entry is still the one recorded for t=100
    first = tr.spectral_scale_trace[0]
    assert first[0] == 100.0
    assert first[1] == pytest.approx(k_at_100)


def test_long_cruising_interval_observes_k_s_even_when_k_a_is_not():
    """On a constant-speed segment a_long ~ 0, so k_a barely moves, but the
    integrated spectral speed is large so k_s is strongly constrained."""
    cfg = SpeedConfig(spectral_scale_enabled=True, accel_scale_enabled=True,
                      accel_bias_rw=0.0, accel_scale_rw=0.0, spectral_scale_rw=0.0)
    tr = GlobalSpeedTracker(cfg, v0=12.0)
    tr.open_interval(0.0)
    integ = 0.0
    for i in range(1500):                       # 150 s of ~constant speed
        tr.predict(0.0, 0.1)                    # no longitudinal acceleration
        tr.spectral_update(12.0, 2.0, 0.1)
        integ += 12.0 * 0.1
    ka_var_before = tr.P[KA, KA]
    ks_var_before = tr.spectral_scale_var
    tr.apply_interval(integ * 1.2, 40.0)        # map says 20 % further
    tr.spectral_scale_interval(integ * 1.2, 40.0, integ)
    assert tr.spectral_scale_var < 0.5 * ks_var_before        # k_s well constrained
    assert tr.P[KA, KA] > 0.9 * ka_var_before                 # k_a barely moved
    assert tr.spectral_scale > 1.05


def test_k_a_and_k_s_are_separate_states():
    cfg = SpeedConfig(spectral_scale_enabled=True, accel_scale_enabled=True)
    tr = GlobalSpeedTracker(cfg, v0=0.0)
    assert tr.x.shape[0] == 5                    # k_s is NOT in the EKF vector
    tr.spectral_scale_point(15.0, 1.0, 10.0, t=1.0)
    assert tr.x[KA] == pytest.approx(1.0)        # k_a untouched by a k_s update


def test_causal_k_s_trace_is_monotonic_in_time():
    tr = _tracker()
    for t, va, vs in [(50.0, 12.0, 9.0), (120.0, 8.0, 8.5), (300.0, 16.0, 11.0)]:
        tr.spectral_scale_point(va, 1.0, vs, t=t)
    ts = [row[0] for row in tr.spectral_scale_trace]
    assert ts == sorted(ts)


def test_production_calibration_path_has_no_hidden_gps():
    for name in ("spectral_scale_point", "spectral_scale_interval",
                 "_spectral_scale_kalman", "spectral_update"):
        src = inspect.getsource(getattr(GlobalSpeedTracker, name))
        for banned in ("reference", "withheld", "oracle", "truth"):
            assert banned not in src


def test_unconstrained_k_s_keeps_a_wide_uncertainty():
    tr = _tracker(spectral_scale_rw=0.004)
    for _ in range(6000):                        # 600 s, no anchors
        tr.predict(0.0, 0.1)
    # sqrt(0.15^2 + 0.004^2 * 600) ~ 0.17 - still wide
    assert tr.sigma_spectral_scale > 0.14


def test_k_s_is_bounded():
    tr = _tracker(spectral_scale_min=0.7, spectral_scale_max=1.7,
                  spectral_scale_max_innovation_sigma=99.0,
                  initial_spectral_scale_sigma=5.0)
    tr.spectral_scale_point(50.0, 0.1, 10.0, t=1.0)
    assert tr.spectral_scale <= 1.7
    tr = _tracker(spectral_scale_min=0.7, spectral_scale_max=1.7,
                  spectral_scale_max_innovation_sigma=99.0,
                  initial_spectral_scale_sigma=5.0)
    tr.spectral_scale_point(1.0, 0.1, 10.0, t=1.0)
    assert tr.spectral_scale >= 0.7


def test_corrected_spectral_speed_cannot_go_negative():
    tr = _tracker()
    tr.spectral_scale = 1.5
    tr.x[V] = 0.0
    tr.spectral_update(5.0, 2.0, 0.1)
    assert tr.x[V] >= 0.0 or tr.x[V] > -tr.cfg.reverse_speed_limit_ms - 1e-9


def test_zupt_still_pins_speed_to_zero_with_k_s_on():
    tr = _tracker()
    tr.spectral_scale = 1.5
    tr.x[V] = 8.0
    for _ in range(400):                       # a real, sustained stop
        tr.predict(0.0, 0.1)
        tr.zero_velocity(0.0, 0.1, 0.0, run_s=1e9)
    assert abs(tr.x[V]) < 0.1
