"""EKF predict/update behaviour."""

from __future__ import annotations

import math

import numpy as np
import pytest

from geotrace.config import MotionConfig
from geotrace.ekf import ExtendedKalmanFilter
from geotrace.motion_model import IDX_BA, IDX_E, IDX_N, IDX_PSI, IDX_V

CFG = MotionConfig()


def make(state=None) -> ExtendedKalmanFilter:
    return ExtendedKalmanFilter(CFG, initial_state=state or [0.0, 0.0, 10.0, 0.0, 0.0, 0.0])


def test_prediction_moves_the_state_along_the_heading() -> None:
    f = make()
    for _ in range(10):
        f.predict((0.0, 0.0, 0.0), 0.0, 0.1)
    assert f.position == pytest.approx([10.0, 0.0], abs=1e-9)


def test_covariance_grows_without_measurements() -> None:
    """Dead reckoning must become less certain, and say so."""
    f = make()
    before = f.P[IDX_E, IDX_E]
    for _ in range(100):
        f.predict((0.0, 0.0, 0.0), 0.0, 0.1)
    assert f.P[IDX_E, IDX_E] > before


def test_covariance_stays_symmetric_and_positive_definite() -> None:
    f = make()
    for i in range(200):
        f.predict((0.3 * math.sin(i / 10), 0.1, 0.0), 0.05, 0.1)
        if i % 10 == 0:
            f.update_position([f.x[IDX_E] + 3.0, f.x[IDX_N] - 2.0], 10.0)
    assert np.allclose(f.P, f.P.T, atol=1e-9)
    assert np.all(np.linalg.eigvalsh(f.P) > -1e-9)


def test_position_update_pulls_the_state_towards_the_measurement() -> None:
    f = make()
    f.P = np.diag([100.0, 100.0, 4.0, 0.3, 0.2, 0.01])
    f.update_position([50.0, 0.0], sigma=5.0)
    assert 25.0 < f.x[IDX_E] < 50.0, "a confident measurement should dominate a loose prior"


def test_a_precise_measurement_shrinks_the_covariance() -> None:
    f = make()
    before = f.P[IDX_E, IDX_E]
    f.update_position([1.0, 0.0], sigma=1.0)
    assert f.P[IDX_E, IDX_E] < before


def test_an_imprecise_measurement_barely_moves_the_state() -> None:
    f = make()
    f.P = np.diag([1.0, 1.0, 4.0, 0.3, 0.2, 0.01])
    f.update_position([500.0, 0.0], sigma=500.0)
    assert abs(f.x[IDX_E]) < 10.0


def test_speed_never_becomes_negative_after_an_update() -> None:
    f = make([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    f.update_speed(-50.0, sigma=0.1)
    assert f.speed >= 0.0


def test_heading_update_wraps_correctly() -> None:
    """A residual across the +/-pi boundary must be the short way round."""
    f = make([0.0, 0.0, 10.0, 3.0, 0.0, 0.0])
    f.update_heading(-3.0, sigma=0.05)
    assert f.heading > 3.0 or f.heading < -3.0, "must not rotate the long way"


def test_zero_velocity_update_drives_the_speed_to_zero() -> None:
    f = make([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    for _ in range(20):
        f.zero_velocity_update()
    assert f.speed == pytest.approx(0.0, abs=0.05)


def test_zupt_lets_the_filter_learn_the_accelerometer_bias() -> None:
    """The car is standing still but the accelerometer claims it is accelerating;
    repeated ZUPTs must attribute that to bias."""
    f = make([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    true_bias = 0.25
    for _ in range(400):
        f.predict((true_bias, 0.0, 0.0), 0.0, 0.1)
        f.zero_velocity_update()
    assert f.biases[0] == pytest.approx(true_bias, abs=0.1)


def test_gps_speed_updates_make_the_bias_observable() -> None:
    """The bias is what limits dead reckoning, and GPS speed is what pins it."""
    f = make([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    true_bias = 0.2
    for _ in range(600):
        f.predict((true_bias, 0.0, 0.0), 0.0, 0.1)
        f.update_speed(10.0, sigma=0.5)
    assert f.biases[0] == pytest.approx(true_bias, abs=0.08)


def test_integration_across_a_gap_is_skipped_and_the_covariance_inflated() -> None:
    f = make()
    before_position = f.position.copy()
    before_variance = f.P[IDX_E, IDX_E]
    f.predict((5.0, 0.0, 0.0), 1.0, CFG.max_gap_s + 5.0)
    assert f.skipped_gaps == 1
    assert f.position == pytest.approx(before_position), "must not invent a position"
    assert f.P[IDX_E, IDX_E] > before_variance, "must admit it lost track"


def test_dead_reckoning_drifts_and_the_filter_knows_it() -> None:
    """The honest baseline: with no GPS, error and reported sigma both grow."""
    f = make()
    sigmas = []
    for _ in range(600):
        f.predict((0.0, 0.0, 0.0), 0.0, 0.1)
        sigmas.append(math.sqrt(f.P[IDX_E, IDX_E] + f.P[IDX_N, IDX_N]))
    assert sigmas[-1] > sigmas[0]
    assert sigmas == sorted(sigmas)


def test_state_json_is_serialisable() -> None:
    import json

    json.dumps(make().state_json())
