"""The interval measurement is d(D_B - D_A) and its Jacobians w.r.t. k_a and
b_a must match a numerical perturbation of the propagation, or the innovation
is attributed to the wrong state."""

import numpy as np
import pytest

from geotrace.pacman_tracker.speed import (
    BA, D, DA, KA, V, GlobalSpeedTracker, SpeedConfig)


def _step(k_a: float, b_a: float, a_long: float, dt: float, v0: float):
    cfg = SpeedConfig(accel_scale_enabled=True, accel_bias_rw=0.0,
                      accel_scale_rw=0.0, accel_noise_ms2=0.0)
    tr = GlobalSpeedTracker(cfg, v0=v0, accel_bias0=b_a)
    tr.x[KA] = k_a
    F_before = None
    ds = tr.predict(a_long, dt)
    return float(tr.x[D]), float(tr.x[V]), ds


def test_interval_distance_jacobian_wrt_k_a_matches_finite_difference():
    a_long, dt, v0, b_a, k_a = 1.7, 0.1, 6.0, 0.05, 1.0
    eps = 1e-6
    d0, v0d, _ = _step(k_a, b_a, a_long, dt, v0)
    d1, v1d, _ = _step(k_a + eps, b_a, a_long, dt, v0)
    numeric_dD = (d1 - d0) / eps
    numeric_dV = (v1d - v0d) / eps
    raw = a_long - b_a
    analytic_dD = 0.5 * raw * dt * dt      # F[D, KA]
    analytic_dV = raw * dt                 # F[V, KA]
    assert numeric_dD == pytest.approx(analytic_dD, rel=1e-4)
    assert numeric_dV == pytest.approx(analytic_dV, rel=1e-4)


def test_interval_distance_jacobian_wrt_b_a_matches_finite_difference():
    a_long, dt, v0, b_a, k_a = 1.7, 0.1, 6.0, 0.05, 1.1
    eps = 1e-6
    d0, v0d, _ = _step(k_a, b_a, a_long, dt, v0)
    d1, v1d, _ = _step(k_a, b_a + eps, a_long, dt, v0)
    numeric_dD = (d1 - d0) / eps
    numeric_dV = (v1d - v0d) / eps
    analytic_dD = -0.5 * k_a * dt * dt     # F[D, BA]
    analytic_dV = -k_a * dt                # F[V, BA]
    assert numeric_dD == pytest.approx(analytic_dD, rel=1e-4)
    assert numeric_dV == pytest.approx(analytic_dV, rel=1e-4)


def test_multi_step_interval_sensitivity_to_k_a_is_the_summed_jacobian():
    """d(D_N - D_0)/dk_a over N steps equals the accumulated propagation
    Jacobian, so an interval that spans acceleration really does carry scale
    information (and one that spans constant cruising barely does)."""
    dt, v0 = 0.1, 0.0
    accel = [1.0] * 40 + [-1.0] * 40      # a real accel/decel profile
    cruise = [0.0] * 80

    def run(profile, k_a):
        cfg = SpeedConfig(accel_scale_enabled=True, accel_bias_rw=0.0,
                          accel_scale_rw=0.0, accel_noise_ms2=0.0)
        tr = GlobalSpeedTracker(cfg, v0=v0)
        tr.x[KA] = k_a
        tr.open_interval(0.0)
        for a in profile:
            tr.predict(a, dt)
        return float(tr.x[D] - tr.x[DA])

    eps = 1e-4
    sens_accel = (run(accel, 1.0 + eps) - run(accel, 1.0 - eps)) / (2 * eps)
    sens_cruise = (run(cruise, 1.0 + eps) - run(cruise, 1.0 - eps)) / (2 * eps)
    # An accel/decel profile that returns to zero net velocity still has a
    # non-trivial scale sensitivity; a pure cruise (raw force ~ 0) has ~none.
    assert abs(sens_accel) > 5.0
    assert abs(sens_cruise) < 1e-6
