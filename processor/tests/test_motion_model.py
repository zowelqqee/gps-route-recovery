"""Motion model: state transition, gap guard, bias handling, Jacobian."""

from __future__ import annotations

import math

import numpy as np
import pytest

from geotrace.config import MotionConfig
from geotrace.coordinates import wrap_angle
from geotrace.models import MotionSample
from geotrace.motion_model import (
    IDX_BA,
    IDX_BW,
    IDX_E,
    IDX_N,
    IDX_PSI,
    IDX_V,
    STATE_DIM,
    build_imu_stream,
    estimate_initial_biases,
    longitudinal_acceleration,
    propagate_state,
    quaternion_to_matrix,
    rotate_device_to_world,
    transition_jacobian,
    yaw_rate_world,
)

CFG = MotionConfig()


def state(e=0.0, n=0.0, v=0.0, psi=0.0, ba=0.0, bw=0.0) -> np.ndarray:
    return np.array([e, n, v, psi, ba, bw], dtype=float)


# ------------------------------------------------------------- quaternions


def test_identity_quaternion_is_identity_rotation() -> None:
    assert np.allclose(quaternion_to_matrix([1, 0, 0, 0]), np.eye(3))


def test_quaternion_rotation_is_orthonormal() -> None:
    q = np.array([0.4, -0.2, 0.7, 0.1])
    R = quaternion_to_matrix(q)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_quaternion_is_normalised_before_use() -> None:
    """A quaternion that arrives un-normalised must still give a pure rotation."""
    R = quaternion_to_matrix([2.0, 0.0, 0.0, 0.0])
    assert np.allclose(R, np.eye(3))


def test_ninety_degree_yaw_maps_east_to_north() -> None:
    q = [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
    assert rotate_device_to_world([1, 0, 0], q) == pytest.approx([0, 1, 0], abs=1e-12)


def test_yaw_rate_world_takes_the_vertical_component() -> None:
    """However the phone is oriented, the car's yaw is the world Z rate."""
    q = [math.cos(0.6), 0.0, 0.0, math.sin(0.6)]
    assert yaw_rate_world([0.0, 0.0, 0.25], q) == pytest.approx(0.25)


def test_longitudinal_projection() -> None:
    """a_parallel = a_E cos(psi) + a_N sin(psi)."""
    assert longitudinal_acceleration([2.0, 0.0, 0.0], 0.0) == pytest.approx(2.0)
    assert longitudinal_acceleration([2.0, 0.0, 0.0], math.pi / 2) == pytest.approx(0.0, abs=1e-12)
    assert longitudinal_acceleration([0.0, 3.0, 0.0], math.pi / 2) == pytest.approx(3.0)
    # A purely lateral acceleration contributes nothing longitudinally.
    assert longitudinal_acceleration([0.0, 4.0, 0.0], 0.0) == pytest.approx(0.0, abs=1e-12)


def test_acceleration_unit_conversion() -> None:
    """CoreMotion reports g; the model needs m/s^2."""
    sample = MotionSample(monotonic_time=0.0, user_acceleration_g=(0.12, -0.03, 0.01))
    assert sample.user_acceleration_ms2 == pytest.approx((1.1768, -0.2942, 0.0981), abs=1e-4)


# --------------------------------------------------------- state transition


def test_straight_line_motion() -> None:
    """Constant speed, no turn: the car advances v*dt along its heading."""
    x = state(v=10.0, psi=0.0)
    out = propagate_state(x, 0.0, 0.0, 0.1, CFG)
    assert out[IDX_E] == pytest.approx(1.0)
    assert out[IDX_N] == pytest.approx(0.0, abs=1e-12)
    assert out[IDX_V] == pytest.approx(10.0)
    assert out[IDX_PSI] == pytest.approx(0.0)


def test_straight_line_accumulates_over_a_second() -> None:
    x = state(v=10.0)
    for _ in range(10):
        x = propagate_state(x, 0.0, 0.0, 0.1, CFG)
    assert x[IDX_E] == pytest.approx(10.0)
    assert x[IDX_N] == pytest.approx(0.0, abs=1e-9)


def test_straight_line_at_an_angle() -> None:
    x = state(v=10.0, psi=math.pi / 4)
    out = propagate_state(x, 0.0, 0.0, 1.0, CFG)
    assert out[IDX_E] == pytest.approx(10.0 / math.sqrt(2))
    assert out[IDX_N] == pytest.approx(10.0 / math.sqrt(2))


def test_acceleration_term() -> None:
    """E advances by v*dt + 0.5*a*dt^2 and v by a*dt."""
    x = state(v=5.0)
    out = propagate_state(x, 2.0, 0.0, 1.0, CFG)
    assert out[IDX_E] == pytest.approx(5.0 * 1.0 + 0.5 * 2.0 * 1.0)
    assert out[IDX_V] == pytest.approx(7.0)


def test_turn_uses_the_midpoint_heading() -> None:
    """The displacement is taken along psi_bar = psi + 0.5*w*dt, not psi."""
    w, dt, v = 0.4, 1.0, 10.0
    out = propagate_state(state(v=v), 0.0, w, dt, CFG)
    psi_bar = 0.5 * w * dt
    assert out[IDX_PSI] == pytest.approx(w * dt)
    assert out[IDX_E] == pytest.approx(v * dt * math.cos(psi_bar))
    assert out[IDX_N] == pytest.approx(v * dt * math.sin(psi_bar))


def test_quarter_turn_ends_heading_north() -> None:
    x = state(v=8.0, psi=0.0)
    w = math.pi / 2  # rad/s
    for _ in range(10):
        x = propagate_state(x, 0.0, w, 0.1, CFG)
    assert x[IDX_PSI] == pytest.approx(math.pi / 2, abs=1e-9)
    assert x[IDX_N] > 0 and x[IDX_E] > 0


def test_heading_is_wrapped() -> None:
    x = state(psi=3.0)
    out = propagate_state(x, 0.0, 1.0, 1.0, CFG)
    assert -math.pi < out[IDX_PSI] <= math.pi
    assert out[IDX_PSI] == pytest.approx(wrap_angle(4.0))


def test_stationary_car_does_not_move() -> None:
    x = state(v=0.0)
    for _ in range(100):
        x = propagate_state(x, 0.0, 0.0, 0.1, CFG)
    assert x[IDX_E] == pytest.approx(0.0)
    assert x[IDX_N] == pytest.approx(0.0)
    assert x[IDX_V] == pytest.approx(0.0)


def test_speed_never_goes_negative() -> None:
    """v_{t+1} = max(0, v + a dt): a car does not reverse under braking."""
    out = propagate_state(state(v=1.0), -5.0, 0.0, 1.0, CFG)
    assert out[IDX_V] == 0.0


def test_speed_is_capped_at_a_plausible_maximum() -> None:
    out = propagate_state(state(v=CFG.max_speed_ms - 0.1), 6.0, 0.0, 1.0, CFG)
    assert out[IDX_V] <= CFG.max_speed_ms


def test_bias_is_subtracted_from_the_measurement() -> None:
    """a_hat = a - b_a, w_hat = w - b_omega."""
    biased = propagate_state(state(v=5.0, ba=1.0), 3.0, 0.0, 1.0, CFG)
    clean = propagate_state(state(v=5.0), 2.0, 0.0, 1.0, CFG)
    assert biased[IDX_V] == pytest.approx(clean[IDX_V])
    assert biased[IDX_E] == pytest.approx(clean[IDX_E])

    biased = propagate_state(state(v=5.0, bw=0.2), 0.0, 0.5, 1.0, CFG)
    clean = propagate_state(state(v=5.0), 0.0, 0.3, 1.0, CFG)
    assert biased[IDX_PSI] == pytest.approx(clean[IDX_PSI])


def test_a_perfectly_estimated_bias_cancels_the_drift() -> None:
    """With b_a equal to the real bias, a stationary car stays put for 60 s."""
    x = state(v=0.0, ba=0.3)
    for _ in range(600):
        x = propagate_state(x, 0.3, 0.0, 0.1, CFG)
    assert math.hypot(x[IDX_E], x[IDX_N]) < 1e-6


def test_an_unestimated_bias_integrates_into_a_large_error() -> None:
    """The reason bias matters: 0.3 m/s^2 unmodelled is ~540 m after 60 s."""
    x = state(v=0.0)
    for _ in range(600):
        x = propagate_state(x, 0.3, 0.0, 0.1, CFG)
    assert x[IDX_E] == pytest.approx(0.5 * 0.3 * 60**2, rel=0.02)


def test_biases_are_unchanged_by_a_single_step() -> None:
    out = propagate_state(state(v=3.0, ba=0.11, bw=0.02), 1.0, 0.1, 0.1, CFG)
    assert out[IDX_BA] == pytest.approx(0.11)
    assert out[IDX_BW] == pytest.approx(0.02)


def test_acceleration_is_clipped_to_a_physical_limit() -> None:
    out = propagate_state(state(v=0.0), 500.0, 0.0, 1.0, CFG)
    assert out[IDX_V] == pytest.approx(CFG.max_accel_ms2)


# ------------------------------------------------------------- the gap guard


def test_integration_across_a_large_gap_is_refused() -> None:
    """Integrating over a hole in the IMU stream silently invents position."""
    with pytest.raises(ValueError, match="gap"):
        propagate_state(state(v=10.0), 0.0, 0.0, CFG.max_gap_s + 0.01, CFG)


def test_a_gap_exactly_at_the_limit_is_allowed() -> None:
    out = propagate_state(state(v=10.0), 0.0, 0.0, CFG.max_gap_s, CFG)
    assert out[IDX_E] == pytest.approx(10.0 * CFG.max_gap_s)


def test_the_gap_limit_is_configurable() -> None:
    cfg = MotionConfig(max_gap_s=5.0)
    out = propagate_state(state(v=10.0), 0.0, 0.0, 3.0, cfg)
    assert out[IDX_E] == pytest.approx(30.0)
    with pytest.raises(ValueError):
        propagate_state(state(v=10.0), 0.0, 0.0, 6.0, cfg)


def test_zero_and_negative_dt_are_no_ops() -> None:
    x = state(e=5.0, v=10.0)
    assert propagate_state(x, 1.0, 1.0, 0.0, CFG) == pytest.approx(x)
    assert propagate_state(x, 1.0, 1.0, -0.5, CFG) == pytest.approx(x)


def test_imu_stream_flags_a_gap() -> None:
    """A hole in the raw samples must be marked, not silently interpolated."""
    samples = [MotionSample(monotonic_time=t / 50.0) for t in range(100)]
    samples += [MotionSample(monotonic_time=10.0 + t / 50.0) for t in range(100)]
    stream = build_imu_stream(samples, CFG)
    assert any(c.gap_exceeded for c in stream.controls)


# -------------------------------------------------------------- Jacobian


def numeric_jacobian(x: np.ndarray, a: float, w: float, dt: float) -> np.ndarray:
    J = np.zeros((STATE_DIM, STATE_DIM))
    h = 1e-6
    for i in range(STATE_DIM):
        up, down = x.copy(), x.copy()
        up[i] += h
        down[i] -= h
        J[:, i] = (propagate_state(up, a, w, dt, CFG) - propagate_state(down, a, w, dt, CFG)) / (2 * h)
    return J


@pytest.mark.parametrize(
    "x,a,w,dt",
    [
        (state(v=10.0), 0.0, 0.0, 0.1),
        (state(e=12.0, n=-3.0, v=8.0, psi=0.7, ba=0.05, bw=0.01), 0.8, 0.15, 0.1),
        (state(v=2.0, psi=-2.5, ba=-0.2, bw=-0.03), -1.2, -0.4, 0.2),
        (state(v=25.0, psi=3.0), 1.5, 0.02, 0.05),
    ],
)
def test_analytic_jacobian_matches_finite_differences(x, a, w, dt) -> None:
    analytic = transition_jacobian(x, a, w, dt, CFG)
    numeric = numeric_jacobian(x, a, w, dt)
    assert np.abs(analytic - numeric).max() < 1e-6


def test_jacobian_is_flat_where_the_speed_clamp_is_active() -> None:
    """Below zero speed the max(0, .) is flat, so dv'/dv must be 0."""
    J = transition_jacobian(state(v=0.5), -10.0, 0.0, 1.0, CFG)
    assert J[IDX_V, IDX_V] == 0.0


# ------------------------------------------------------------ bias estimation


def test_initial_bias_from_a_stationary_period_is_signed() -> None:
    """The longitudinal bias is the projection of the world bias on the heading,
    not its magnitude - taking |b| would always inject a positive bias."""
    from geotrace.motion_model import ImuControl, ImuStream

    stream = ImuStream(
        controls=[
            ImuControl(t=i * 0.1, dt=0.1, a_long=0.0, yaw_rate=0.01,
                       a_world=(-0.25, 0.0, 0.0), is_quiet=True)
            for i in range(60)
        ]
    )
    b_a, b_w = estimate_initial_biases(stream, heading_rad=0.0, cfg=CFG)
    assert b_a == pytest.approx(-0.25)
    assert b_w == pytest.approx(0.01)

    # Driving the other way, the same world bias projects with the other sign.
    b_a, _ = estimate_initial_biases(stream, heading_rad=math.pi, cfg=CFG)
    assert b_a == pytest.approx(0.25)


def test_absurd_bias_estimates_are_rejected() -> None:
    from geotrace.motion_model import ImuControl, ImuStream

    stream = ImuStream(
        controls=[
            ImuControl(t=i * 0.1, dt=0.1, a_long=0.0, yaw_rate=0.0,
                       a_world=(9.0, 0.0, 0.0), is_quiet=True)
            for i in range(60)
        ]
    )
    assert estimate_initial_biases(stream, 0.0, CFG)[0] == 0.0


def test_bias_estimation_without_a_stationary_period_returns_zero() -> None:
    from geotrace.motion_model import ImuStream

    assert estimate_initial_biases(ImuStream(), 0.0, CFG) == (0.0, 0.0)
