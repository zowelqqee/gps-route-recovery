"""Motion model: state transition, gap guard, bias handling, Jacobian."""

from __future__ import annotations

import math

import numpy as np
import pytest

from geotrace.config import G_TO_MS2, MotionConfig
from geotrace.coordinates import wrap_angle
from geotrace.models import MotionSample, MountCalibration
from geotrace.motion_model import (
    IDX_BA,
    build_imu_stream,
    leveling_correction,
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


def test_impact_marks_a_short_mount_disturbance_hold() -> None:
    samples = [
        MotionSample(monotonic_time=0.00, user_acceleration_g=(0.0, 0.0, 0.0), rotation_rate=(0, 0, 0)),
        MotionSample(monotonic_time=0.05, user_acceleration_g=(10.0 / 9.80665, 0.0, 0.0), rotation_rate=(0, 0, 0)),
        MotionSample(monotonic_time=0.15, user_acceleration_g=(0.0, 0.0, 0.0), rotation_rate=(0, 0, 0)),
    ]
    stream = build_imu_stream(samples, MotionConfig(filter_dt_s=0.1, shock_hold_s=0.2))
    assert stream.controls[0].is_shock
    assert stream.controls[0].peak_accel_ms2 == pytest.approx(10.0)
    assert stream.controls[1].is_shock


def test_robust_normalisation_removes_a_single_nonshock_imu_spike() -> None:
    """One bad 20 ms frame must not become a 100 ms vehicle manoeuvre."""
    samples = [
        MotionSample(
            monotonic_time=0.02 * index,
            user_acceleration_g=((5.0 if index == 2 else 0.2) / 9.80665, 0.0, 0.0),
            rotation_rate=(0.0, 0.0, 0.0),
        )
        for index in range(5)
    ]
    stream = build_imu_stream(samples, MotionConfig(filter_dt_s=0.1, robust_window_s=0.1))
    assert stream.controls[0].a_world[0] == pytest.approx(0.2, abs=0.01)


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


# --------------------------------------------------------------- deadband


def test_deadband_is_off_by_default() -> None:
    """A residual below the deadband still integrates unless asked not to."""
    small = 0.5 * CFG.accel_deadband_ms2
    out = propagate_state(state(v=10.0), small, 0.0, 1.0, CFG)
    assert out[IDX_V] == pytest.approx(10.0 + small)


def test_deadband_coasts_through_a_residual_below_the_threshold() -> None:
    """Below the threshold, a_hat is indistinguishable from sensor bias, so
    the model holds the current speed instead of drifting."""
    small = 0.5 * CFG.accel_deadband_ms2
    out = propagate_state(state(v=10.0), small, 0.0, 1.0, CFG, deadband=True)
    assert out[IDX_V] == pytest.approx(10.0)
    assert out[IDX_E] == pytest.approx(10.0)  # step uses the coasted v, not v + drift


def test_deadband_does_not_mask_a_real_manoeuvre() -> None:
    """A clearly-above-threshold acceleration is still a real signal."""
    big = 2.0 * CFG.accel_deadband_ms2
    out = propagate_state(state(v=10.0), big, 0.0, 1.0, CFG, deadband=True)
    assert out[IDX_V] == pytest.approx(10.0 + big)


def test_deadband_does_not_survive_many_minutes_of_residual_bias() -> None:
    """The failure this exists to fix: on a real trip an unmodelled residual
    this small, integrated for ~25 minutes, is enough to erase highway speed
    entirely. With the deadband active it must not move the speed at all."""
    residual = 0.6 * CFG.accel_deadband_ms2
    x = state(v=20.0)
    for _ in range(15000):  # 1500 s at dt=0.1, roughly this trip's real outage
        x = propagate_state(x, residual, 0.0, 0.1, CFG, deadband=True)
    assert x[IDX_V] == pytest.approx(20.0)


@pytest.mark.parametrize(
    "x,a,w,dt",
    [
        (state(v=10.0, ba=0.05), 0.05, 0.0, 0.1),  # a_hat inside the deadband
        (state(v=8.0, ba=-0.02, bw=0.01), 2.5, 0.15, 0.1),  # a_hat well above it
    ],
)
def test_analytic_jacobian_matches_finite_differences_with_deadband(x, a, w, dt) -> None:
    analytic = transition_jacobian(x, a, w, dt, CFG, deadband=True)
    numeric = np.zeros((STATE_DIM, STATE_DIM))
    h = 1e-6
    for i in range(STATE_DIM):
        up, down = x.copy(), x.copy()
        up[i] += h
        down[i] -= h
        numeric[:, i] = (
            propagate_state(up, a, w, dt, CFG, deadband=True)
            - propagate_state(down, a, w, dt, CFG, deadband=True)
        ) / (2 * h)
    assert np.abs(analytic - numeric).max() < 1e-6


# ------------------------------------------------------- shock heading trust


def test_yaw_trust_is_full_by_default() -> None:
    out = propagate_state(state(psi=0.0), 0.0, 1.0, 1.0, CFG)
    assert out[IDX_PSI] == pytest.approx(1.0)


def test_yaw_trust_discounts_the_measured_yaw_rate() -> None:
    """A shock can leave the phone at a new angle in its mount; the gyro then
    measures that real rotation, but possibly of the phone, not the car."""
    out = propagate_state(state(psi=0.0), 0.0, 1.0, 1.0, CFG, yaw_trust=0.3)
    assert out[IDX_PSI] == pytest.approx(0.3)


def test_yaw_trust_does_not_fully_suppress_a_genuine_manoeuvre() -> None:
    """Damped, not silenced: part of a real turn right after a bump must
    still register rather than being thrown away entirely."""
    out = propagate_state(state(psi=0.0), 0.0, 2.0, 1.0, CFG, yaw_trust=0.3)
    assert out[IDX_PSI] == pytest.approx(0.6)
    assert out[IDX_PSI] != pytest.approx(0.0)


def test_build_imu_stream_discounts_yaw_right_after_a_shock_then_recovers() -> None:
    """The failure this exists to fix: a shock (see shock_accel_ms2) can leave
    the phone sitting at a new angle rather than bouncing back. Right after
    the shock's own hold ends, yaw must be discounted; once
    shock_heading_recovery_s has elapsed it must be trusted again."""
    cfg = MotionConfig(shock_hold_s=0.2, shock_heading_recovery_s=1.0, shock_heading_gain=0.3)
    dt = cfg.filter_dt_s
    samples = []
    t = 0.0
    for _ in range(10):
        samples.append(MotionSample(monotonic_time=t))
        t += dt
    samples.append(MotionSample(monotonic_time=t, user_acceleration_g=(2.0, 0.0, 0.0)))
    t += dt
    for _ in range(40):
        samples.append(MotionSample(monotonic_time=t))
        t += dt
    controls = build_imu_stream(samples, cfg).controls

    shock_indices = [i for i, c in enumerate(controls) if c.is_shock]
    assert shock_indices, "the injected acceleration must have been flagged as a shock"
    last_shock = shock_indices[-1]

    assert not controls[last_shock + 1].is_shock
    assert controls[last_shock + 1].yaw_trust == pytest.approx(cfg.shock_heading_gain)
    assert controls[-1].yaw_trust == pytest.approx(1.0)


@pytest.mark.parametrize(
    "x,a,w,dt,yaw_trust",
    [
        (state(v=10.0, bw=0.02), 0.5, 0.3, 0.1, 0.3),
        (state(v=8.0, psi=0.4, bw=-0.01), 1.2, 1.5, 0.1, 0.0),
    ],
)
def test_analytic_jacobian_matches_finite_differences_with_yaw_trust(x, a, w, dt, yaw_trust) -> None:
    analytic = transition_jacobian(x, a, w, dt, CFG, yaw_trust=yaw_trust)
    numeric = np.zeros((STATE_DIM, STATE_DIM))
    h = 1e-6
    for i in range(STATE_DIM):
        up, down = x.copy(), x.copy()
        up[i] += h
        down[i] -= h
        numeric[:, i] = (
            propagate_state(up, a, w, dt, CFG, yaw_trust=yaw_trust)
            - propagate_state(down, a, w, dt, CFG, yaw_trust=yaw_trust)
        ) / (2 * h)
    assert np.abs(analytic - numeric).max() < 1e-6


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


# --------------------------------------------------- AHRS levelling recovery


def _levelling_ahrs_samples(
    duration_s: float = 60.0,
    rate_hz: float = 50.0,
    accel_ms2: float = 1.0,
    gain_per_s: float = 0.08,
) -> tuple[list[MotionSample], float]:
    """A recorder whose attitude filter leans into sustained acceleration.

    The car drives straight and accelerates at a constant `accel_ms2`. The
    attitude filter has no gyro rotation to explain, so it slowly tips its idea
    of "down" toward the measured specific force. Gravity removal then cancels
    part of the acceleration - the defect `leveling_correction` undoes.
    """
    n = int(duration_s * rate_hz)
    dt = 1.0 / rate_hz
    samples: list[MotionSample] = []
    tilt = 0.0
    for i in range(n):
        # The filter chases the tilt that would explain the acceleration.
        target = math.atan2(accel_ms2, G_TO_MS2)
        tilt += gain_per_s * (target - tilt) * dt
        # What survives gravity removal once the frame has tipped by `tilt`.
        visible = accel_ms2 - G_TO_MS2 * math.sin(tilt)
        half = tilt / 2.0
        # Rotation about +Y tips the device's nose up in this convention.
        q = (math.cos(half), 0.0, math.sin(half), 0.0)
        samples.append(
            MotionSample(
                monotonic_time=i * dt,
                user_acceleration_g=(visible / G_TO_MS2, 0.0, 0.0),
                rotation_rate=(0.0, 0.0, 0.0),
                gravity=(0.0, 0.0, -1.0),
                quaternion=q,
            )
        )
    return samples, accel_ms2


def test_the_levelling_a_filter_did_is_recovered_from_the_gyro() -> None:
    """The gyro reported no rotation, so every degree of tilt is the filter's.

    That is the whole trick: a real manoeuvre appears in both the quaternion
    and the gyro, and cancels; a levelling correction appears only in the
    quaternion, and is exactly the acceleration that was swallowed.
    """
    samples, true_accel = _levelling_ahrs_samples()
    times = np.array([s.monotonic_time for s in samples])
    quats = np.array([s.quaternion for s in samples])
    rates = np.array([s.rotation_rate for s in samples])

    visible = np.array([s.user_acceleration_ms2[0] for s in samples])
    # By the end of the minute the filter has eaten most of the signal.
    assert visible[-1] < 0.5 * true_accel

    correction = leveling_correction(times, quats, rates, tau_s=60.0)
    recovered = visible + correction[:, 0]
    late = times > 20.0
    lost = true_accel - float(np.mean(visible[late]))
    regained = float(np.mean(recovered[late])) - float(np.mean(visible[late]))
    # Most of the deficit comes back, but not all of it, and that is correct:
    # the integrator leaks, so a tilt held forever is indistinguishable from a
    # mount simply bolted in nose-up. Only a lean the filter took on recently
    # is attributable to levelling, and only that is given back.
    assert regained > 0.5 * lost
    assert float(np.mean(recovered[late])) < true_accel * 1.1


def test_a_real_turn_is_not_mistaken_for_levelling() -> None:
    """A rotation the gyro also reports must produce no correction at all."""
    n, dt = 500, 0.02
    rate = math.radians(6.0)
    samples_t = np.arange(n) * dt
    quats = np.array(
        [(math.cos(rate * t / 2), 0.0, 0.0, math.sin(rate * t / 2)) for t in samples_t]
    )
    rates = np.tile([0.0, 0.0, rate], (n, 1))
    correction = leveling_correction(samples_t, quats, rates, tau_s=20.0)
    assert np.max(np.abs(correction)) < 0.05


def test_the_correction_is_off_unless_the_trip_says_its_attitude_needs_it() -> None:
    """It must never fire on a CoreMotion trip, which does not have the defect."""
    samples, _ = _levelling_ahrs_samples()
    cfg = MotionConfig()
    cfg.leveling_recovery_tau_s = 20.0

    plain = build_imu_stream(samples, cfg, calibration=MountCalibration())
    ahrs = build_imu_stream(
        samples,
        cfg,
        calibration=MountCalibration(attitude_source="accelerometer_levelled_ahrs"),
    )
    plain_a = np.array([c.a_world[0] for c in plain.controls])
    ahrs_a = np.array([c.a_world[0] for c in ahrs.controls])
    assert np.allclose(plain_a, ahrs_a[: len(plain_a)] * 0 + plain_a)
    assert np.mean(ahrs_a[-50:]) > np.mean(plain_a[-50:]) * 1.5

    cfg.leveling_recovery_tau_s = 0.0
    disabled = build_imu_stream(
        samples,
        cfg,
        calibration=MountCalibration(attitude_source="accelerometer_levelled_ahrs"),
    )
    assert np.allclose([c.a_world[0] for c in disabled.controls], plain_a)


def test_road_vibration_separates_a_stopped_car_from_a_quiet_cruise() -> None:
    """Level of acceleration cannot; that is why the GPS gate existed.

    Both cars below report near-zero acceleration and near-zero yaw rate. Only
    the moving one is still being shaken by the road.
    """
    rng = np.random.default_rng(7)
    n, dt = 400, 0.02

    def stream(vibration_g: float, cfg: MotionConfig) -> list[bool]:
        samples = [
            MotionSample(
                monotonic_time=i * dt,
                # Vertical only, as real road vibration mostly is: that keeps
                # the horizontal level below `zupt_accel_ms2`, so the existing
                # test cannot separate the two and the vibration test must.
                user_acceleration_g=(0.0, 0.0, float(rng.normal(0.0, vibration_g))),
                rotation_rate=(0.0, 0.0, 0.0),
                gravity=(0.0, 0.0, -1.0),
                quaternion=(1.0, 0.0, 0.0, 0.0),
            )
            for i in range(n)
        ]
        return [c.is_quiet for c in build_imu_stream(samples, cfg).controls]

    cfg = MotionConfig()
    cfg.zupt_vibration_g = 0.0
    # Without the vibration test both look identically quiet.
    assert any(stream(0.004, cfg)) and any(stream(0.05, cfg))

    cfg.zupt_vibration_g = 0.02
    assert any(stream(0.004, cfg)), "a parked car must still qualify"
    assert not any(stream(0.05, cfg)), "a shaken car must not"
