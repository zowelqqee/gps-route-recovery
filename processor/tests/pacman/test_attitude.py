"""The gravity/attitude filter.

Gravity projection is the largest single error source in this pipeline: a
steady 0.3 degrees of unmodelled tilt leaks 0.05 m/s^2 into the forward axis,
which is 3 m/s of velocity error per minute. These tests pin the properties
that make the filter worth having.
"""

import math

import numpy as np
import pytest

from geotrace.pacman_tracker.attitude import (
    G,
    AttitudeConfig,
    AttitudeFilter,
    quat_from_gravity,
    quat_to_matrix,
)


def _tilted(roll: float, pitch: float) -> np.ndarray:
    """Specific force a stationary device reads at this roll and pitch."""
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    return G * np.array([-sp, sr * cp, cr * cp])


@pytest.mark.parametrize("roll_deg,pitch_deg", [(10.0, 0.0), (0.0, 7.0), (-5.0, 3.0)])
def test_a_stationary_tilt_is_recovered(roll_deg, pitch_deg):
    roll, pitch = math.radians(roll_deg), math.radians(pitch_deg)
    f = _tilted(roll, pitch)
    filt = AttitudeFilter(AttitudeConfig(), q0=quat_from_gravity(f))
    for _ in range(2000):
        filt.update(f, np.zeros(3), 0.02)
    got_roll, got_pitch = filt.roll_pitch()
    assert math.degrees(abs(got_roll - roll)) < 0.1
    assert math.degrees(abs(got_pitch - pitch)) < 0.1


def test_gravity_is_removed_so_a_parked_car_reads_zero():
    f = _tilted(math.radians(8.0), math.radians(-4.0))
    filt = AttitudeFilter(AttitudeConfig(), q0=quat_from_gravity(f))
    for _ in range(1500):
        rot = filt.update(f, np.zeros(3), 0.02)
    residual = rot @ f - np.array([0.0, 0.0, G])
    assert np.linalg.norm(residual) < 0.02


def test_the_experimental_filter_is_rejected_for_sustained_acceleration():
    """Pin the real failure that keeps the Mahony path out of production.

    A norm gate cannot distinguish gravity from gravity plus a modest
    horizontal acceleration. The filter therefore interprets the manoeuvre as
    pitch and deletes the signal. This is a characterization test, not a claim
    that the behaviour is good.
    """
    filt = AttitudeFilter(AttitudeConfig(), q0=quat_from_gravity(_tilted(0.0, 0.0)))
    push = np.array([1.5, 0.0, G])          # 1.5 m/s^2 forward, level device
    for _ in range(1000):                   # 20 s of it
        rot = filt.update(push, np.zeros(3), 0.02)
    world = rot @ push - np.array([0.0, 0.0, G])
    assert abs(world[0]) < 0.1
    assert abs(math.degrees(filt.roll_pitch()[1])) > 5.0
    assert AttitudeConfig().enabled is False


def test_zaru_recovers_the_gyro_bias_at_a_stop():
    filt = AttitudeFilter(AttitudeConfig())
    bias = np.array([0.004, -0.003, 0.006])
    for _ in range(500):
        filt.zero_rotation(bias, 0.02)
    assert np.allclose(filt.bias, bias, atol=1e-4)
    assert filt.stats()["zaru_steps"] == 500


def test_the_estimated_bias_stays_physically_bounded():
    cfg = AttitudeConfig()
    filt = AttitudeFilter(cfg)
    for _ in range(5000):
        filt.zero_rotation(np.full(3, 10.0), 0.02)
    assert np.all(np.abs(filt.bias) <= cfg.bias_limit_rads + 1e-12)


def test_the_rotation_matrix_stays_orthonormal():
    rng = np.random.default_rng(3)
    filt = AttitudeFilter(AttitudeConfig())
    for _ in range(2000):
        rot = filt.update(_tilted(0.05, -0.02) + rng.normal(0, 0.3, 3),
                          rng.normal(0, 0.2, 3), 0.02)
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9)
    assert abs(float(np.linalg.det(rot)) - 1.0) < 1e-9


def test_quat_to_matrix_matches_quat_from_gravity():
    f = _tilted(math.radians(12.0), math.radians(-6.0))
    rot = quat_to_matrix(quat_from_gravity(f))
    assert np.allclose(rot @ f, np.array([0.0, 0.0, G]), atol=1e-6)
