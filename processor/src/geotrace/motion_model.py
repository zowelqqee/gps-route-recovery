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

from geotrace.config import G_TO_MS2, MotionConfig
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
    a_long: Optional[float]
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

    is_shock: bool = False
    """A likely holder disturbance. The filters must not interpret its raw
    acceleration or rotation as a manoeuvre of the car."""

    yaw_trust: float = 1.0
    """Fraction of ``yaw_rate`` to trust once a shock's own hold has ended
    (see ``MotionConfig.shock_heading_recovery_s``). A shock can leave the
    phone at a new angle in its mount; the gyro then measures that real
    rotation, but of the phone, not necessarily the car."""

    peak_accel_ms2: float = 0.0
    peak_gyro_rads: float = 0.0


@dataclass
class ImuStream:
    """Resamples raw CMDeviceMotion frames onto a fixed filter timeline.

    The phone records at 50 Hz. Running a 5000-particle filter at 50 Hz is
    wasteful, and the sample spacing is not perfectly regular anyway, so the
    stream is robustly normalised, then binned into fixed ``filter_dt_s`` steps
    with median acceleration and yaw rate inside each bin.
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


def angular_velocity_from_quaternions(
    times: np.ndarray, quaternions: np.ndarray
) -> np.ndarray:
    """Body-frame angular velocity implied by a sequence of attitudes, rad/s.

    ``omega = 2 * conj(q) (x) dq/dt``. This is what the attitude *actually did*,
    which is not the same as what the gyro reported - and the difference is the
    point of `leveling_correction`.
    """
    q = np.asarray(quaternions, dtype=float)
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.where(norms < 1e-12, 1.0, norms)
    if len(q) < 2:
        return np.zeros((len(q), 3))
    # q and -q encode the same attitude. Differentiate a continuous lift.
    flips = np.r_[1.0, np.where(np.sum(q[1:] * q[:-1], axis=1) < 0.0, -1.0, 1.0)]
    q = q * np.cumprod(flips)[:, None]
    dq = np.gradient(q, times, axis=0)
    w, x, y, z = q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]
    dw, dx, dy, dz = dq[:, 0], dq[:, 1], dq[:, 2], dq[:, 3]
    return 2.0 * np.column_stack(
        [
            w * dx + x * dw + y * dz - z * dy,
            w * dy - x * dz + y * dw + z * dx,
            w * dz + x * dy - y * dx + z * dw,
        ]
    )


def leveling_correction(
    times: np.ndarray,
    quaternions: np.ndarray,
    rotation_rates: np.ndarray,
    tau_s: float,
    gravity_ms2: float = 9.80665,
) -> np.ndarray:
    """The horizontal acceleration an accelerometer-levelled AHRS absorbed.

    Such a filter has no way to tell a nose-up tilt from an acceleration
    forwards - both push the specific-force vector the same way - so under
    sustained acceleration it slowly leans, and the gravity subtraction then
    removes the acceleration along with the gravity. Measured on the vehicle
    logger, this costs 15% of the signal at a 15 s timescale and half of it at
    30 s.

    What makes it recoverable is that the lean is a rotation *nothing rotated*.
    The gyro reports the vehicle's real angular velocity; the quaternion
    sequence reports the real rotation plus the filter's own correction. The
    difference is the correction alone, and integrating it gives the tilt error
    the filter is currently carrying. Rotating gravity by that tilt gives back
    the acceleration it swallowed:

        residual_horizontal = -g * (delta x z_up)   with  delta = the tilt error

    The integral leaks with `tau_s` because the AHRS's own correction decays:
    a tilt taken on during one manoeuvre is given back over the seconds after
    it, and an integrator with no leak would keep charging on gyro bias alone.

    Returns an (N, 2) array to be *added* to the world-frame horizontal
    acceleration. Sign and magnitude were checked against GPS on five recorded
    days: fitting a free scale factor on top of this returned 0.75-1.27, i.e.
    what comes back is the amount that went missing, not a tuned fudge.
    """
    times = np.asarray(times, dtype=float)
    n = len(times)
    if n < 3 or tau_s <= 0.0:
        return np.zeros((n, 2))

    residual_body = angular_velocity_from_quaternions(times, quaternions) - np.asarray(
        rotation_rates, dtype=float
    )
    residual_world = np.empty((n, 3))
    for i in range(n):
        residual_world[i] = quaternion_to_matrix(quaternions[i]) @ residual_body[i]

    # Leaky integral of the correction rate: the tilt the filter is carrying.
    dt = np.diff(times, prepend=times[0])
    dt = np.clip(dt, 0.0, tau_s)
    decay = np.exp(-dt / tau_s)
    tilt = np.zeros((n, 2))
    carry = np.zeros(2)
    for i in range(n):
        carry = decay[i] * carry + residual_world[i, :2] * dt[i]
        tilt[i] = carry

    # delta x z_up = (delta_y, -delta_x); the residual left in the measurement
    # is -g times that, so adding it back is +g times it.
    return gravity_ms2 * np.column_stack([tilt[:, 1], -tilt[:, 0]])


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

    ordered = sorted({s.monotonic_time: s for s in motions}.values(), key=lambda s: s.monotonic_time)
    t0 = t_start if t_start is not None else ordered[0].monotonic_time
    t1 = t_end if t_end is not None else ordered[-1].monotonic_time
    if t1 <= t0:
        return stream
    ordered = [s for s in ordered if t0 <= s.monotonic_time <= t1]
    if not ordered:
        return stream

    # World-frame acceleration and yaw rate for every raw sample.
    n = len(ordered)
    times = np.empty(n)
    a_world = np.empty((n, 3))
    yaw = np.empty(n)
    gyro_norm = np.empty(n)
    quaternions = np.empty((n, 4))
    rotation_rates = np.empty((n, 3))
    forward_world = None
    if (calibration is not None and calibration.forward_axis_device is not None
            and calibration.attitude_source != "synthetic_fixed_world_legacy"):
        forward = np.asarray(calibration.forward_axis_device, dtype=float)
        if np.linalg.norm(forward) > 1e-9:
            forward = forward / np.linalg.norm(forward)
            forward_world = np.empty((n, 3))
    for i, sample in enumerate(ordered):
        times[i] = sample.monotonic_time
        quaternions[i] = sample.quaternion
        rotation_rates[i] = sample.rotation_rate
        rot = quaternion_to_matrix(sample.quaternion)
        if forward_world is not None:
            forward_world[i] = rot @ forward
        a_world[i] = rot @ (np.asarray(sample.user_acceleration_ms2, dtype=float))
        yaw[i] = float((rot @ np.asarray(sample.rotation_rate, dtype=float))[2])
        gyro_norm[i] = float(np.linalg.norm(sample.rotation_rate))

    accel_norm = np.linalg.norm(a_world, axis=1)
    # Road vibration, measured before anything is corrected: it is the one
    # stillness cue a moving car cannot suppress. See MotionConfig.zupt_vibration_g.
    vibration_g = _rolling_std(accel_norm, times, cfg.zupt_window_s) / G_TO_MS2

    # Undo the attitude filter's levelling, when the recorder is known to do it.
    if (
        cfg.leveling_recovery_tau_s > 0.0
        and calibration is not None
        and calibration.attitude_source == "accelerometer_levelled_ahrs"
    ):
        a_world[:, :2] += leveling_correction(
            times, quaternions, rotation_rates, cfg.leveling_recovery_tau_s
        )

    # A rigid mount directly measures forward acceleration in the vehicle
    # frame. Re-projecting it onto each road hypothesis would turn lateral
    # acceleration into acceleration/braking on the wrong branches.
    a_vehicle = None
    if forward_world is not None:
        a_vehicle = np.sum(a_world * forward_world, axis=1)
        a_vehicle = _robust_normalize(a_vehicle, times, window_s=cfg.robust_window_s,
            z=cfg.robust_hampel_z, floor=cfg.robust_accel_floor_ms2)
        if cfg.accel_smooth_window_s > 0:
            a_vehicle = _rolling_mean(a_vehicle, times, cfg.accel_smooth_window_s)

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

    # A phone occasionally produces a single implausible frame even when it
    # was not dropped hard enough to meet the separate shock threshold.  Mean
    # aggregation lets that frame leak directly into vehicle acceleration and
    # heading.  Hampel filtering is local (not global): it removes an isolated
    # spike but retains sustained braking and a real turn.
    a_world = _robust_normalize(
        a_world,
        times,
        window_s=cfg.robust_window_s,
        z=cfg.robust_hampel_z,
        floor=cfg.robust_accel_floor_ms2,
    )
    yaw = _robust_normalize(
        yaw,
        times,
        window_s=cfg.robust_window_s,
        z=cfg.robust_hampel_z,
        floor=cfg.robust_yaw_floor_rads,
    )

    # Plain boxcar on the spike-cleaned signal. See MotionConfig.accel_smooth_window_s.
    if cfg.accel_smooth_window_s > 0.0:
        a_world = _rolling_mean(a_world, times, cfg.accel_smooth_window_s)

    dt = cfg.filter_dt_s
    n_steps = max(1, int(math.ceil((t1 - t0) / dt)))
    edges = t0 + dt * np.arange(n_steps + 1)
    edges[-1] = t1
    # np.digitize puts each raw sample into its step bin.
    bounds = np.searchsorted(times, edges, side="left")
    bounds[-1] = len(times)

    quiet_flags = _quiet_imu(times, a_world, gyro_norm, vibration_g, cfg)
    shock_flags = (accel_norm >= cfg.shock_accel_ms2) | (gyro_norm >= cfg.shock_gyro_rads)

    prev_time = t0
    shock_until = float("-inf")
    heading_recovery_until = float("-inf")
    for step in range(n_steps):
        mask = slice(int(bounds[step]), int(bounds[step + 1]))
        t_end_step = float(edges[step + 1])
        step_dt = float(edges[step + 1] - edges[step])
        if bounds[step] == bounds[step + 1]:
            is_shock_now = t_end_step <= shock_until
            stream.controls.append(
                ImuControl(
                    t=t_end_step,
                    dt=step_dt,
                    a_long=0.0,
                    yaw_rate=0.0,
                    is_quiet=False,
                    gap_exceeded=(t_end_step - prev_time) > cfg.max_gap_s,
                    is_shock=is_shock_now,
                    yaw_trust=(
                        cfg.shock_heading_gain
                        if not is_shock_now and t_end_step <= heading_recovery_until
                        else 1.0
                    ),
                )
            )
            continue
        chunk_times = times[mask]
        gap = float(np.max(np.diff(chunk_times))) if chunk_times.size > 1 else 0.0
        gap = max(gap, float(chunk_times[0] - prev_time))
        mean_a = np.median(a_world[mask], axis=0)
        peak_accel = float(np.max(accel_norm[mask]))
        peak_gyro = float(np.max(gyro_norm[mask]))
        if np.any(shock_flags[mask]):
            shock_until = max(shock_until, t_end_step + cfg.shock_hold_s)
            heading_recovery_until = max(
                heading_recovery_until, shock_until + cfg.shock_heading_recovery_s
            )
        is_shock_now = t_end_step <= shock_until
        stream.controls.append(
            ImuControl(
                t=t_end_step,
                dt=step_dt,
                a_long=float(np.median(a_vehicle[mask])) if a_vehicle is not None else None,
                yaw_rate=float(np.median(yaw[mask])),
                a_world=(float(mean_a[0]), float(mean_a[1]), float(mean_a[2])),
                is_quiet=bool(np.all(quiet_flags[mask])),
                gap_exceeded=gap > cfg.max_gap_s,
                is_shock=is_shock_now,
                yaw_trust=(
                    cfg.shock_heading_gain
                    if not is_shock_now and t_end_step <= heading_recovery_until
                    else 1.0
                ),
                peak_accel_ms2=peak_accel,
                peak_gyro_rads=peak_gyro,
            )
        )
        prev_time = float(chunk_times[-1])
    return stream


def _robust_normalize(
    values: np.ndarray,
    times: np.ndarray,
    *,
    window_s: float,
    z: float,
    floor: float,
) -> np.ndarray:
    """Replace isolated local outliers using a median/MAD (Hampel) filter.

    The transform stays in physical units.  It is therefore a data-cleaning
    step, not a rescaling that could conceal genuine uncertainty downstream.
    """
    data = np.asarray(values, dtype=float)
    if len(data) < 3 or window_s <= 0.0 or z <= 0.0:
        return data.copy()
    spacing = np.diff(np.asarray(times, dtype=float))
    spacing = spacing[np.isfinite(spacing) & (spacing > 0.0)]
    if not len(spacing):
        return data.copy()
    radius = max(1, int(round(0.5 * window_s / float(np.median(spacing)))))

    def clean(series: np.ndarray) -> np.ndarray:
        padded = np.pad(series, radius, mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * radius + 1)
        median = np.median(windows, axis=-1)
        mad = np.median(np.abs(windows - median[:, None]), axis=-1)
        threshold = z * np.maximum(1.4826 * mad, floor)
        return np.where(np.abs(series - median) > threshold, median, series)

    if data.ndim == 1:
        return clean(data)
    return np.column_stack([clean(data[:, index]) for index in range(data.shape[1])])


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


def _rolling_std(values: np.ndarray, times: np.ndarray, window_s: float) -> np.ndarray:
    """Standard deviation of `values` over a centred window of `window_s`."""
    n = len(values)
    if n < 3 or window_s <= 0.0:
        return np.zeros(n)
    spacing = float(np.median(np.diff(times))) if n > 1 else 0.0
    width = max(3, int(round(window_s / spacing))) if spacing > 0 else 3
    c = np.cumsum(np.insert(values, 0, 0.0))
    c2 = np.cumsum(np.insert(values * values, 0, 0.0))
    i = np.arange(n)
    lo = np.maximum(i - width // 2, 0)
    hi = np.minimum(i + width // 2 + 1, n)
    count = hi - lo
    mean = (c[hi] - c[lo]) / count
    mean_sq = (c2[hi] - c2[lo]) / count
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def _rolling_mean(values: np.ndarray, times: np.ndarray, window_s: float) -> np.ndarray:
    """Boxcar average of `values` over a centred window of `window_s`.

    See `MotionConfig.accel_smooth_window_s` for what this is fitted to and
    what it does and does not fix."""
    n = len(values)
    if n < 3 or window_s <= 0.0:
        return values
    spacing = float(np.median(np.diff(times))) if n > 1 else 0.0
    width = max(3, int(round(window_s / spacing))) if spacing > 0 else 3
    # `axis=0` is load-bearing: without it np.insert flattens a (n, k) array
    # first, and the division below then broadcasts an (n,) numerator against
    # an (n, 1) count into an (n, n) matrix - a silent multi-gigabyte blow-up
    # instead of an error.
    c = np.cumsum(np.insert(values, 0, 0.0, axis=0), axis=0)
    i = np.arange(n)
    lo = np.maximum(i - width // 2, 0)
    hi = np.minimum(i + width // 2 + 1, n)
    count = (hi - lo).astype(float)
    if values.ndim > 1:
        count = count[:, None]
    return (c[hi] - c[lo]) / count


def _quiet_imu(
    times: np.ndarray,
    a_world: np.ndarray,
    gyro_norm: np.ndarray,
    vibration_g: np.ndarray,
    cfg: MotionConfig,
) -> np.ndarray:
    """Mark samples where the IMU is quiet.

    Deliberately does not claim the vehicle is stopped - see ImuControl.is_quiet.
    """
    a_mag = np.linalg.norm(a_world[:, :2], axis=1)
    quiet = (a_mag < cfg.zupt_accel_ms2) & (gyro_norm < cfg.zupt_gyro_rads)
    if cfg.zupt_vibration_g > 0.0:
        quiet &= vibration_g < cfg.zupt_vibration_g
    if not np.any(quiet):
        return quiet
    # Only mark quiet after the elapsed window; never backdate a stop.
    out = np.zeros_like(quiet)
    start = None
    for i, flag in enumerate(quiet):
        if i and times[i] - times[i - 1] > cfg.max_gap_s:
            start = None
        if flag and start is None:
            start = i
        if not flag:
            start = None
        if start is not None and times[i] - times[start] >= cfg.zupt_window_s:
            out[i] = True
    return out


def bounded_displacement(v: float, a: float, dt: float, vmax: float) -> tuple[float, float, float]:
    """Integral of clip(v + a*t, 0, vmax), and its derivatives in v and a.

    Braking stops displacement at the time speed reaches zero. Acceleration
    beyond the speed limit contributes constant-speed travel for the remainder.
    """
    active = dt
    tail_speed = 0.0
    if a < 0.0 and v + a * dt < 0.0:
        active = max(0.0, -v / a)
    elif a > 0.0 and v + a * dt > vmax:
        active = max(0.0, (vmax - v) / a)
        tail_speed = vmax
    distance = v * active + 0.5 * a * active**2 + tail_speed * (dt - active)
    return distance, active, 0.5 * active**2


def _bias_compensated_acceleration(
    a_long: float, b_a: float, cfg: MotionConfig, deadband: bool = False
) -> tuple[float, bool]:
    """a_hat with the physical clip and, optionally, the noise deadband applied.

    ``deadband`` defaults to off: the deadband trades away bias-learning
    sensitivity (see below) for immunity to long-run drift, which is only a
    good trade while GPS cannot corroborate either one. A caller with GPS
    aiding available - the common case - should leave it off so bias
    calibration keeps working exactly as before.

    Returns ``(a_hat, neutralized)``. ``neutralized`` is True whenever a_hat's
    value cannot be traced back to the raw input - either it saturated at the
    physical clip boundary, or (only when ``deadband`` is set) it fell inside
    the deadband and was forced to exactly zero (see
    ``MotionConfig.accel_deadband_ms2``). Both cases need the same Jacobian
    treatment: d(a_hat)/d(a_long) and d(a_hat)/d(b_a) are both zero, so
    `transition_jacobian` must agree with whichever of the two reasons
    applied here.
    """
    a_raw = a_long - b_a
    saturated = abs(a_raw) >= cfg.max_accel_ms2
    a_hat = float(np.clip(a_raw, -cfg.max_accel_ms2, cfg.max_accel_ms2))
    if deadband and not saturated and abs(a_hat) < cfg.accel_deadband_ms2:
        return 0.0, True
    return a_hat, saturated


def propagate_state(
    state: np.ndarray,
    a_long: float,
    yaw_rate: float,
    dt: float,
    cfg: MotionConfig,
    deadband: bool = False,
    yaw_trust: float = 1.0,
) -> np.ndarray:
    """Exact transition from the specification.

        psi_{t+1} = wrap(psi_t + w_hat dt)
        psi_bar   = psi_t + 0.5 w_hat dt
        E_{t+1}   = E_t + v dt cos(psi_bar) + 0.5 a_hat dt^2 cos(psi_bar)
        N_{t+1}   = N_t + v dt sin(psi_bar) + 0.5 a_hat dt^2 sin(psi_bar)
        v_{t+1}   = max(0, v_t + a_hat dt)

    Biases are constant across a single step (random walk is applied in the
    covariance, not in the mean). ``deadband`` - see
    `_bias_compensated_acceleration` - defaults to off. ``yaw_trust`` - see
    `MotionConfig.shock_heading_recovery_s` - discounts the measured yaw rate
    after a shock, defaulting to 1.0 (fully trusted).
    """
    if dt <= 0:
        return state.copy()
    if dt > cfg.max_gap_s:
        raise ValueError(
            f"refusing to integrate across a {dt:.3f} s gap "
            f"(motion.max_gap_s = {cfg.max_gap_s} s)"
        )

    e, n, v, psi, b_a, b_w = state
    a_hat, _ = _bias_compensated_acceleration(a_long, b_a, cfg, deadband)
    w_hat = (yaw_rate - b_w) * yaw_trust

    psi_bar = psi + 0.5 * w_hat * dt
    step, _, _ = bounded_displacement(v, a_hat, dt, cfg.max_speed_ms)
    out = np.empty_like(state)
    out[IDX_E] = e + step * math.cos(psi_bar)
    out[IDX_N] = n + step * math.sin(psi_bar)
    out[IDX_V] = min(max(0.0, v + a_hat * dt), cfg.max_speed_ms)
    out[IDX_PSI] = wrap_angle(psi + w_hat * dt)
    out[IDX_BA] = b_a
    out[IDX_BW] = b_w
    return out


def transition_jacobian(
    state: np.ndarray,
    a_long: float,
    yaw_rate: float,
    dt: float,
    cfg: MotionConfig,
    deadband: bool = False,
    yaw_trust: float = 1.0,
    a_long_heading_derivative: float = 0.0,
) -> np.ndarray:
    """Analytic dF/dX of :func:`propagate_state`. ``deadband`` and
    ``yaw_trust`` must match the values passed to `propagate_state` for this
    same step, or the two disagree on where a_hat/w_hat came from.

    Derived by hand; ``tests/test_ekf.py`` checks it against a central finite
    difference of ``propagate_state`` so the two can never drift apart.
    """
    _e, _n, v, psi, b_a, b_w = state
    a_hat, neutralized = _bias_compensated_acceleration(a_long, b_a, cfg, deadband)
    w_hat = (yaw_rate - b_w) * yaw_trust

    psi_bar = psi + 0.5 * w_hat * dt
    cos_b, sin_b = math.cos(psi_bar), math.sin(psi_bar)
    step, dstep_dv, dstep_da = bounded_displacement(v, a_hat, dt, cfg.max_speed_ms)
    # d(step)/d(b_a) is zero once the acceleration clips or falls in the deadband.
    dstep_dba = 0.0 if neutralized else -dstep_da
    da_dpsi = 0.0 if neutralized else a_long_heading_derivative
    # d(w_hat)/d(b_w) = -yaw_trust, so every derivative through w_hat picks up
    # the same factor.
    dpsibar_dbw = -0.5 * dt * yaw_trust

    F = np.eye(STATE_DIM)
    F[IDX_E, IDX_V] = dstep_dv * cos_b
    F[IDX_E, IDX_PSI] = -step * sin_b + dstep_da * da_dpsi * cos_b
    F[IDX_E, IDX_BA] = dstep_dba * cos_b
    F[IDX_E, IDX_BW] = -step * sin_b * dpsibar_dbw

    F[IDX_N, IDX_V] = dstep_dv * sin_b
    F[IDX_N, IDX_PSI] = step * cos_b + dstep_da * da_dpsi * sin_b
    F[IDX_N, IDX_BA] = dstep_dba * sin_b
    F[IDX_N, IDX_BW] = step * cos_b * dpsibar_dbw

    v_next = v + a_hat * dt
    if v_next <= 0.0 or v_next >= cfg.max_speed_ms:
        # max(0, .) / clip is flat here, so the row is zero apart from itself.
        F[IDX_V, IDX_V] = 0.0
        F[IDX_V, IDX_BA] = 0.0
    else:
        F[IDX_V, IDX_V] = 1.0
        F[IDX_V, IDX_BA] = 0.0 if neutralized else -dt
        F[IDX_V, IDX_PSI] = dt * da_dpsi

    F[IDX_PSI, IDX_BW] = -dt * yaw_trust
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
            cfg.accel_noise**2 / max(dt, 1e-12),
            cfg.gyro_noise**2 / max(dt, 1e-12),
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
