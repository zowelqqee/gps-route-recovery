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


def test_predict_deadband_is_off_by_default_so_bias_learning_still_works() -> None:
    """The failure the opt-in deadband must not reintroduce: with GPS aiding
    available, a residual bias must still be learnable through repeated
    speed corrections, exactly as without the deadband feature at all."""
    f = make([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    true_bias = 0.2
    for _ in range(600):
        f.predict((true_bias, 0.0, 0.0), 0.0, 0.1)  # deadband defaults to False
        f.update_speed(10.0, sigma=0.5)
    assert f.biases[0] == pytest.approx(true_bias, abs=0.08)


def test_predict_deadband_holds_speed_through_a_long_unaided_cruise() -> None:
    """The bug this exists to fix: a residual bias too small to tell apart
    from noise must not be allowed to erase a real cruising speed over many
    minutes with no GPS to correct it."""
    residual = -0.6 * CFG.accel_deadband_ms2  # an unmodelled deceleration-like bias
    f = make([0.0, 0.0, 20.0, 0.0, 0.0, 0.0])
    for _ in range(15000):  # 1500 s at dt=0.1
        f.predict((residual, 0.0, 0.0), 0.0, 0.1, deadband=True)
    assert f.speed == pytest.approx(20.0)


def test_without_the_deadband_the_same_residual_erases_the_speed() -> None:
    """Contrast case: confirms the fix above is the deadband, not something
    else - the exact same residual, without it, drains speed to zero."""
    residual = -0.6 * CFG.accel_deadband_ms2
    f = make([0.0, 0.0, 20.0, 0.0, 0.0, 0.0])
    for _ in range(15000):
        f.predict((residual, 0.0, 0.0), 0.0, 0.1, deadband=False)
    assert f.speed == pytest.approx(0.0)


def test_yaw_trust_holds_the_heading_close_to_the_real_turn_after_a_shock() -> None:
    """The failure this exists to fix (trip-32c24d86, Kantemirovsky bridge): a
    shock can leave the phone at a new angle in its mount rather than
    bouncing back, and the gyro then keeps reporting that real rotation of
    the phone as if it were the car turning - 30 degrees measured against a
    real net turn of about 9. A discounted yaw rate must land much closer to
    the real turn than the raw, undiscounted signal would."""
    real_turn_rad = math.radians(9.0)
    measured_yaw_rate = math.radians(1.0)  # rad/s, sustained for 30 s -> 30 degrees raw
    gain = 0.3
    total_s = 30.0
    f = make([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    for _ in range(300):  # 30 s at dt=0.1
        f.predict((0.0, 0.0, 0.0), measured_yaw_rate, 0.1, yaw_trust=gain)
    raw_heading = measured_yaw_rate * total_s  # what full trust would have given
    assert f.heading == pytest.approx(gain * raw_heading, abs=1e-6)
    assert abs(f.heading - real_turn_rad) < abs(raw_heading - real_turn_rad)


def test_nudge_heading_partially_rotates_towards_the_target() -> None:
    f = make([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    f.nudge_heading(math.radians(40.0), 0.5)
    assert f.heading == pytest.approx(math.radians(20.0))


def test_nudge_heading_touches_only_psi_not_position_or_covariance() -> None:
    """Unlike update_heading, this must not drag position along via P's
    cross-terms - see RoadParticleFilter.heading_consensus, which carries no
    position information of its own."""
    f = make([5.0, 3.0, 10.0, 0.0, 0.0, 0.0])
    before_p = f.P.copy()
    f.nudge_heading(math.radians(90.0), 0.5)
    assert f.x[IDX_E] == pytest.approx(5.0)
    assert f.x[IDX_N] == pytest.approx(3.0)
    assert np.array_equal(f.P, before_p)


def test_without_yaw_trust_the_same_signal_overshoots_the_real_turn() -> None:
    """Contrast case: confirms the fix above is the discount, not something
    else - the exact same signal, fully trusted, overshoots the real turn."""
    measured_yaw_rate = math.radians(1.0)
    f = make([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    for _ in range(300):
        f.predict((0.0, 0.0, 0.0), measured_yaw_rate, 0.1)  # yaw_trust defaults to 1.0
    assert f.heading == pytest.approx(math.radians(30.0), abs=1e-6)


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
