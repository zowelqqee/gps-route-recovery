"""Vehicle motion model and IMU pre-processing.

State vector (specification section "Состояние"):

    X = [E, N, v, psi, b_a, b_omega]^T

  E, N      position in the local metric frame, metres
  v         forward speed along the vehicle axis, m/s (never negative)
  psi       vehicle heading, radians CCW from +E
  b_a       longitudinal accelerometer bias, m/s^2
  b_omega   yaw-rate gyro bias, rad/s

The phone is rigidly mounted, so the device->world rotation from CMDeviceMotion
can be used directly; the longitudinal component of the world-frame acceleration
is the only accelerometer channel the model consumes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

from geotrace.config import MotionConfig
from geotrace.coordinates import wrap_angle
from geotrace.models import MotionSample, MountCalibration

STATE_DIM = 6
IDX_E, IDX_N, IDX_V, IDX_PSI, IDX_BA, IDX_BW = range(STATE_DIM)


def quaternion_to_matrix(q: Sequence[float]) -> np.ndarray:
    """Rotation matrix R_WD from a (w, x, y, z) quaternion.

    ``a_W = R_WD(q) a_D`` - takes a vector in the device frame to the reference
    frame CMDeviceMotion was started with.
    """
    w, x, y, z = (float(c) for c in q)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        return np.eye(3)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotate_device_to_world(vector: Sequence[float], q: Sequence[float]) -> np.ndarray:
    return quaternion_to_matrix(q) @ np.asarray(vector, dtype=float)


def longitudinal_acceleration(a_world: Sequence[float], psi: float) -> float:
    """a_parallel = a_E cos(psi) + a_N sin(psi).

    Note that CMDeviceMotion's reference frame has +x pointing to magnetic or
    true north depending on the requested reference; :class:`ImuStream` maps it
    into the E/N convention before this is called.
    """
    a = np.asarray(a_world, dtype=float)
    return float(a[0] * math.cos(psi) + a[1] * math.sin(psi))


def yaw_rate_world(rotation_rate: Sequence[float], q: Sequence[float]) -> float:
    """Vertical component of the rotation rate = vehicle yaw rate.

    Rotating the body rate into the world frame and taking the Z (up) component
    is exactly the projection of the gyro onto the gravity axis, which is what a
    car's yaw is, regardless of how the phone is oriented in its holder.
    """
    return float(rotate_device_to_world(rotation_rate, q)[2])


@dataclass
class ImuControl:
    """One resampled IMU step handed to a filter."""

    t: float
    dt: float
    a_long: float
    """Raw longitudinal acceleration, m/s^2, before bias removal."""

    yaw_rate: float
    """Raw yaw rate, rad/s, before bias removal."""

    a_world: tuple[float, float, float] = (0.0, 0.0, 0.0)
    is_quiet: bool = False
    """The IMU is quiet: near-zero acceleration and near-zero yaw rate. This is
    a necessary but NOT sufficient condition for a stop - steady cruising looks
    the same. The consumer must also check its own speed estimate before
    applying a zero-velocity update."""

    gap_exceeded: bool = False
    """True when the source samples were further apart than max_gap_s. The
    filters must not integrate across such a step."""


@dataclass
class ImuStream:
    """Resamples raw CMDeviceMotion frames onto a fixed filter timeline.

    The phone records at 50 Hz. Running a 5000-particle filter at 50 Hz is
    wasteful, and the sample spacing is not perfectly regular anyway, so the
    stream is binned into fixed ``filter_dt_s`` steps with the mean acceleration
    and yaw rate inside each bin.
    """

    controls: list[ImuControl] = field(default_factory=list)
    heading_reference_offset: float = 0.0
    """Rotation applied to the CMDeviceMotion reference frame so that its X axis
    lines up with local East."""

    @property
    def times(self) -> np.ndarray:
        return np.array([c.t for c in self.controls], dtype=float)

    def __len__(self) -> int:
        return len(self.controls)


def build_imu_stream(
    motions: Sequence[MotionSample],
    cfg: MotionConfig,
    calibration: Optional[MountCalibration] = None,
    reference_heading_rad: Optional[float] = None,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
) -> ImuStream:
    """Turn raw motion samples into fixed-rate controls.

    ``reference_heading_rad`` is the true initial vehicle heading (from a
    trusted GPS course). CMDeviceMotion's reference frame is arbitrary in yaw,
    so the whole world frame is rotated once so that integrating the gyro from
    the first sample reproduces the real heading. This is the "gyro drives the
    turn, trusted GPS course corrects it" arrangement from the specification -
    the magnetometer is never used as the primary heading source.
    """
    stream = ImuStream()
    if not motions:
        return stream

    ordered = sorted(motions, key=lambda s: s.monotonic_time)
    t0 = t_start if t_start is not None else ordered[0].monotonic_time
    t1 = t_end if t_end is not None else ordered[-1].monotonic_time
    if t1 <= t0:
        return stream

    # World-frame acceleration and yaw rate for every raw sample.
    n = len(ordered)
    times = np.empty(n)
    a_world = np.empty((n, 3))
    yaw = np.empty(n)
    gyro_norm = np.empty(n)
    for i, sample in enumerate(ordered):
        times[i] = sample.monotonic_time
        rot = quaternion_to_matrix(sample.quaternion)
        a_world[i] = rot @ (np.asarray(sample.user_acceleration_ms2, dtype=float))
        yaw[i] = float((rot @ np.asarray(sample.rotation_rate, dtype=float))[2])
        gyro_norm[i] = float(np.linalg.norm(sample.rotation_rate))

    # Align the arbitrary CMDeviceMotion yaw reference with the true heading.
    offset = 0.0
    if reference_heading_rad is not None:
        offset = _estimate_reference_offset(
            a_world, yaw, times, reference_heading_rad, calibration
        )
    if offset:
        c, s = math.cos(offset), math.sin(offset)
        rot2 = np.array([[c, -s], [s, c]])
        a_world[:, :2] = a_world[:, :2] @ rot2.T
    stream.heading_reference_offset = offset

    dt = cfg.filter_dt_s
    n_steps = max(1, int(math.ceil((t1 - t0) / dt)))
    edges = t0 + dt * np.arange(n_steps + 1)
    # np.digitize puts each raw sample into its step bin.
    bins = np.clip(np.digitize(times, edges) - 1, 0, n_steps - 1)

    quiet_flags = _quiet_imu(times, a_world, gyro_norm, cfg)

    prev_time = t0
    for step in range(n_steps):
        mask = bins == step
        t_end_step = float(edges[step + 1])
        if not np.any(mask):
            stream.controls.append(
                ImuControl(
                    t=t_end_step,
                    dt=dt,
                    a_long=0.0,
                    yaw_rate=0.0,
                    is_quiet=False,
                    gap_exceeded=(t_end_step - prev_time) > cfg.max_gap_s,
                )
            )
            continue
        chunk_times = times[mask]
        gap = float(np.max(np.diff(chunk_times))) if chunk_times.size > 1 else 0.0
        gap = max(gap, float(chunk_times[0] - prev_time))
        mean_a = a_world[mask].mean(axis=0)
        stream.controls.append(
            ImuControl(
                t=t_end_step,
                dt=dt,
                a_long=0.0,  # filled per-filter: depends on the current psi
                yaw_rate=float(np.mean(yaw[mask])),
                a_world=(float(mean_a[0]), float(mean_a[1]), float(mean_a[2])),
                is_quiet=bool(np.all(quiet_flags[mask])),
                gap_exceeded=gap > cfg.max_gap_s,
            )
        )
        prev_time = float(chunk_times[-1])
    return stream


def _estimate_reference_offset(
    a_world: np.ndarray,
    yaw: np.ndarray,
    times: np.ndarray,
    reference_heading_rad: float,
    calibration: Optional[MountCalibration],
) -> float:
    """Yaw rotation that maps the CMDeviceMotion frame onto local E/N.

    If the calibration recorded a vehicle forward axis, the phone's forward
    direction at t0 is rotated onto the known initial heading. Otherwise the
    first strong acceleration burst (the calibration drive "start moving in a
    straight line") is assumed to point forward.
    """
    if calibration is not None and calibration.forward_axis_device is not None:
        fwd = np.asarray(calibration.forward_axis_device, dtype=float)
        rot = quaternion_to_matrix(calibration.reference_quaternion)
        fwd_world = rot @ fwd
        if np.linalg.norm(fwd_world[:2]) > 1e-6:
            measured = math.atan2(fwd_world[1], fwd_world[0])
            return float(wrap_angle(reference_heading_rad - measured))

    horizontal = a_world[:, :2]
    magnitude = np.linalg.norm(horizontal, axis=1)
    if magnitude.size == 0 or float(np.max(magnitude)) < 0.2:
        return 0.0
    # Average the direction of the strongest 10% of horizontal accelerations
    # during the first 30 s: on a straight-line start those all point forward.
    horizon = times <= times[0] + 30.0
    if not np.any(horizon):
        horizon = np.ones_like(magnitude, dtype=bool)
    candidate = magnitude.copy()
    candidate[~horizon] = 0.0
    threshold = np.quantile(candidate[candidate > 0], 0.9) if np.any(candidate > 0) else 0.0
    mask = candidate >= max(threshold, 0.2)
    if not np.any(mask):
        return 0.0
    mean_vec = horizontal[mask].mean(axis=0)
    if np.linalg.norm(mean_vec) < 1e-6:
        return 0.0
    measured = math.atan2(mean_vec[1], mean_vec[0])
    return float(wrap_angle(reference_heading_rad - measured))


def _quiet_imu(
    times: np.ndarray, a_world: np.ndarray, gyro_norm: np.ndarray, cfg: MotionConfig
) -> np.ndarray:
    """Mark samples where the IMU is quiet.

    Deliberately does not claim the vehicle is stopped - see ImuControl.is_quiet.
    """
    a_mag = np.linalg.norm(a_world[:, :2], axis=1)
    quiet = (a_mag < cfg.zupt_accel_ms2) & (gyro_norm < cfg.zupt_gyro_rads)
    if not np.any(quiet):
        return quiet
    # Require the quiet condition to hold for a whole window before trusting it.
    out = np.zeros_like(quiet)
    start = None
    for i, flag in enumerate(quiet):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            if times[i - 1] - times[start] >= cfg.zupt_window_s:
                out[start:i] = True
            start = None
    if start is not None and times[-1] - times[start] >= cfg.zupt_window_s:
        out[start:] = True
    return out


def propagate_state(
    state: np.ndarray, a_long: float, yaw_rate: float, dt: float, cfg: MotionConfig
) -> np.ndarray:
    """Exact transition from the specification.

        psi_{t+1} = wrap(psi_t + w_hat dt)
        psi_bar   = psi_t + 0.5 w_hat dt
        E_{t+1}   = E_t + v dt cos(psi_bar) + 0.5 a_hat dt^2 cos(psi_bar)
        N_{t+1}   = N_t + v dt sin(psi_bar) + 0.5 a_hat dt^2 sin(psi_bar)
        v_{t+1}   = max(0, v_t + a_hat dt)

    Biases are constant across a single step (random walk is applied in the
    covariance, not in the mean).
    """
    if dt <= 0:
        return state.copy()
    if dt > cfg.max_gap_s:
        raise ValueError(
            f"refusing to integrate across a {dt:.3f} s gap "
            f"(motion.max_gap_s = {cfg.max_gap_s} s)"
        )

    e, n, v, psi, b_a, b_w = state
    a_hat = float(np.clip(a_long - b_a, -cfg.max_accel_ms2, cfg.max_accel_ms2))
    w_hat = yaw_rate - b_w

    psi_bar = psi + 0.5 * w_hat * dt
    step = v * dt + 0.5 * a_hat * dt * dt
    out = np.empty_like(state)
    out[IDX_E] = e + step * math.cos(psi_bar)
    out[IDX_N] = n + step * math.sin(psi_bar)
    out[IDX_V] = min(max(0.0, v + a_hat * dt), cfg.max_speed_ms)
    out[IDX_PSI] = wrap_angle(psi + w_hat * dt)
    out[IDX_BA] = b_a
    out[IDX_BW] = b_w
    return out


def transition_jacobian(
    state: np.ndarray, a_long: float, yaw_rate: float, dt: float, cfg: MotionConfig
) -> np.ndarray:
    """Analytic dF/dX of :func:`propagate_state`.

    Derived by hand; ``tests/test_ekf.py`` checks it against a central finite
    difference of ``propagate_state`` so the two can never drift apart.
    """
    _e, _n, v, psi, b_a, b_w = state
    a_raw = a_long - b_a
    saturated = abs(a_raw) >= cfg.max_accel_ms2
    a_hat = float(np.clip(a_raw, -cfg.max_accel_ms2, cfg.max_accel_ms2))
    w_hat = yaw_rate - b_w

    psi_bar = psi + 0.5 * w_hat * dt
    cos_b, sin_b = math.cos(psi_bar), math.sin(psi_bar)
    step = v * dt + 0.5 * a_hat * dt * dt
    # d(step)/d(b_a) is zero once the acceleration clips.
    dstep_dba = 0.0 if saturated else -0.5 * dt * dt
    dpsibar_dbw = -0.5 * dt

    F = np.eye(STATE_DIM)
    F[IDX_E, IDX_V] = dt * cos_b
    F[IDX_E, IDX_PSI] = -step * sin_b
    F[IDX_E, IDX_BA] = dstep_dba * cos_b
    F[IDX_E, IDX_BW] = -step * sin_b * dpsibar_dbw

    F[IDX_N, IDX_V] = dt * sin_b
    F[IDX_N, IDX_PSI] = step * cos_b
    F[IDX_N, IDX_BA] = dstep_dba * sin_b
    F[IDX_N, IDX_BW] = step * cos_b * dpsibar_dbw

    v_next = v + a_hat * dt
    if v_next <= 0.0 or v_next >= cfg.max_speed_ms:
        # max(0, .) / clip is flat here, so the row is zero apart from itself.
        F[IDX_V, IDX_V] = 0.0
        F[IDX_V, IDX_BA] = 0.0
    else:
        F[IDX_V, IDX_V] = 1.0
        F[IDX_V, IDX_BA] = 0.0 if saturated else -dt

    F[IDX_PSI, IDX_BW] = -dt
    return F


def noise_jacobian(state: np.ndarray, dt: float) -> np.ndarray:
    """dF/du for the process noise: [accel noise, gyro noise, b_a rw, b_w rw]."""
    psi = state[IDX_PSI]
    cos_p, sin_p = math.cos(psi), math.sin(psi)
    G = np.zeros((STATE_DIM, 4))
    G[IDX_E, 0] = 0.5 * dt * dt * cos_p
    G[IDX_N, 0] = 0.5 * dt * dt * sin_p
    G[IDX_V, 0] = dt
    G[IDX_E, 1] = -0.5 * state[IDX_V] * dt * dt * sin_p
    G[IDX_N, 1] = 0.5 * state[IDX_V] * dt * dt * cos_p
    G[IDX_PSI, 1] = dt
    G[IDX_BA, 2] = 1.0
    G[IDX_BW, 3] = 1.0
    return G


def process_noise(dt: float, cfg: MotionConfig) -> np.ndarray:
    return np.diag(
        [
            cfg.accel_noise**2,
            cfg.gyro_noise**2,
            (cfg.accel_bias_rw**2) * dt,
            (cfg.gyro_bias_rw**2) * dt,
        ]
    )


def estimate_initial_biases(
    stream: "ImuStream", heading_rad: float, cfg: MotionConfig, max_seconds: float = 12.0
) -> tuple[float, float]:
    """Initial (b_a, b_omega) from the stationary period before the drive.

    The calibration instructions ask the driver to stand still for a few seconds
    precisely so this is available. While the car is provably not moving, every
    reading is bias.

    The accelerometer bias must be handled as a *vector*, not a magnitude. The
    horizontal bias is observable in the world frame while stationary; the
    longitudinal bias the motion model needs is its projection onto the vehicle
    heading:

        b_a = b_world . [cos(psi), sin(psi)]

    Taking |b_world| instead would inject a positive bias whatever the true
    sign, and 0.2 m/s^2 of phantom acceleration integrates to roughly 200 m over
    a 45 s outage - larger than everything else in the error budget.

    Note this operates on the *stream*, whose world frame has already been
    aligned with local East/North, not on the raw samples.
    """
    if not stream.controls:
        return 0.0, 0.0
    t0 = stream.controls[0].t
    quiet = [
        c for c in stream.controls
        if c.t - t0 <= max_seconds and c.is_quiet and not c.gap_exceeded
    ]
    if len(quiet) < 5:
        return 0.0, 0.0
    bias_world = np.mean([[c.a_world[0], c.a_world[1]] for c in quiet], axis=0)
    b_a = float(bias_world[0] * math.cos(heading_rad) + bias_world[1] * math.sin(heading_rad))
    b_w = float(np.mean([c.yaw_rate for c in quiet]))
    # Refuse absurd values: a real MEMS bias is small, anything larger means the
    # "stationary" period was not stationary.
    if abs(b_a) > 1.0:
        b_a = 0.0
    if abs(b_w) > 0.2:
        b_w = 0.0
    return b_a, b_w
