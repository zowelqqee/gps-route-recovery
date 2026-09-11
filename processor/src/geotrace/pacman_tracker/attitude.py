"""Attitude estimation and gravity compensation.

Gravity is 9.81 m/s^2 and the longitudinal channel is a projection of it, so
attitude error is the most expensive error in the whole estimator::

    sin(0.3 deg) * g  =  0.05 m/s^2  =  3 m/s after one minute

The recorder's own attitude carries exactly this defect by construction. Its
`attitude_source` is ``accelerometer_levelled_ahrs``: it steers its idea of
"down" toward the measured specific force. That works because gravity dominates
on average, and it fails while the vehicle accelerates, because sustained
forward acceleration is indistinguishable from a nose-up tilt. The filter leans
into it and then subtracts the very acceleration being measured.

This module estimates attitude from the raw signals instead, so the gyro carries
the fast dynamics and the accelerometer is consulted for roll/pitch only when
the vehicle is not accelerating hard enough to corrupt the reference. It is a
Mahony-style complementary filter - no ML, no magnetometer, and no yaw
reference, because yaw does not enter the gravity projection at all.

    q      device -> world, propagated by the gyro
    b_g    gyro bias, driven by the same correction
    a_w    R(q) f  -  g z_hat          gravity-compensated world acceleration

The accelerometer correction is gated on two conditions that together mean "the
specific force we are looking at is mostly gravity":

    | ||f|| - g |  <  accel_gate_ms2        magnitude looks like gravity alone
    ||omega||      <  gyro_gate_rads        not mid-manoeuvre

Both are necessary. Magnitude alone passes a steady turn, where the lateral and
centripetal terms rotate the apparent vertical without changing its length.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

G = 9.80665


@dataclass
class AttitudeConfig:
    enabled: bool = False
    """Use this filter instead of the recorder's own attitude.

    Off by default so the baseline is exactly the pipeline that existed before
    it; the ablation turns it on."""

    kp: float = 3.0
    """Proportional gain pulling roll/pitch toward the gravity reference, 1/s.

    Swept over 0.01-6.0 on both recordings: the parked-versus-moving
    longitudinal offset and the 60 s velocity error both fall monotonically as
    the gain rises, and flatten by 3.0. The intuition that a high gain would
    let manoeuvres tilt the estimate is wrong here because a car's horizontal
    acceleration alternates in sign over seconds while a mounting tilt does
    not, so the fast loop averages the former out and tracks the latter."""

    ki: float = 0.01
    """Integral gain, which is what actually estimates the gyro bias."""

    accel_gate_ms2: float = 0.6
    """How far ||f|| may sit from g and still be treated as gravity."""

    gyro_gate_rads: float = 0.15
    """Above this the vehicle is manoeuvring and the specific force is not a
    gravity reference."""

    horizontal_gate_ms2: float = 0.0
    """The gate that actually matters. A horizontal specific force adds to
    gravity in quadrature, so 0.5 m/s^2 of longitudinal acceleration moves
    ``||f||`` by only 0.013 m/s^2 - a norm gate is effectively blind to the one
    disturbance that tilts the gravity reference, which is precisely how the
    recorder's own AHRS ends up absorbing acceleration as pitch.

    Measured, and off because it makes things worse: the residual is computed
    from the current attitude, so a wrong attitude produces a large residual
    and the gate refuses the very correction that would fix it. At a 0.5 m/s^2
    gate that feedback drove the gravity-correction rate down to 0.39 and the
    60 s velocity MAE up from 4.0 to 20.2 m/s. Non-positive disables it."""

    bias_limit_rads: float = 0.05
    """Physical bound on the estimated gyro bias."""

    zaru_enabled: bool = True
    zaru_window_s: float = 1.0
    zaru_accel_std_ms2: float = 0.08
    zaru_gyro_std_rads: float = 0.01

    zaru_gain: float = 0.5
    """Rate, 1/s, at which a confirmed stop pulls the gyro bias onto the
    measured rate. A couple of seconds parked is enough to converge."""

    warmup_kp: float = 6.0
    warmup_s: float = 20.0
    """The filter starts level-agnostic, so it is allowed to converge quickly
    during the first seconds, which on these recordings are a standing start."""


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix R such that ``v_world = R @ v_device``."""
    w, x, y, z = q / max(float(np.linalg.norm(q)), 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_from_gravity(f_device: np.ndarray) -> np.ndarray:
    """Attitude with the measured specific force pointing at world +z.

    Yaw is arbitrary and irrelevant: gravity compensation and the forward/left
    projections used downstream are all invariant to it.
    """
    v = np.asarray(f_device, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    v = v / n
    target = np.array([0.0, 0.0, 1.0])
    axis = np.cross(v, target)
    s = float(np.linalg.norm(axis))
    c = float(np.dot(v, target))
    if s < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0]) if c > 0 else np.array([0.0, 1.0, 0.0, 0.0])
    axis = axis / s
    angle = math.atan2(s, c)
    return np.concatenate([[math.cos(angle / 2)], axis * math.sin(angle / 2)])


class AttitudeFilter:
    """Mahony complementary filter on the raw specific force and gyro."""

    def __init__(self, cfg: Optional[AttitudeConfig] = None,
                 q0: Optional[np.ndarray] = None) -> None:
        self.cfg = cfg or AttitudeConfig()
        self.q = np.array([1.0, 0.0, 0.0, 0.0]) if q0 is None else np.asarray(q0, float)
        self.bias = np.zeros(3)
        self.elapsed = 0.0
        self.corrections = 0
        self.steps = 0
        self.last_error = 0.0
        self.last_horizontal = 0.0
        self.zaru_steps = 0

    def update(self, f_device: np.ndarray, omega_device: np.ndarray,
               dt: float) -> np.ndarray:
        """Advance one step and return ``R`` (device -> world)."""
        cfg = self.cfg
        f = np.asarray(f_device, dtype=float)
        w = np.asarray(omega_device, dtype=float) - self.bias
        self.elapsed += dt
        self.steps += 1

        norm = float(np.linalg.norm(f))
        R = quat_to_matrix(self.q)
        usable = (abs(norm - G) < cfg.accel_gate_ms2
                  and float(np.linalg.norm(omega_device)) < cfg.gyro_gate_rads
                  and norm > 1e-6)
        if usable and cfg.horizontal_gate_ms2 > 0.0:
            residual = R @ f - np.array([0.0, 0.0, G])
            self.last_horizontal = float(math.hypot(residual[0], residual[1]))
            usable = self.last_horizontal < cfg.horizontal_gate_ms2
        if usable:
            # Error between where the accelerometer says "up" is and where the
            # current attitude thinks it is, expressed in the device frame.
            up_device = R.T @ np.array([0.0, 0.0, 1.0])
            error = np.cross(f / norm, up_device)
            self.last_error = float(np.linalg.norm(error))
            kp = cfg.warmup_kp if self.elapsed < cfg.warmup_s else cfg.kp
            w = w + kp * error
            self.bias = np.clip(self.bias - cfg.ki * error * dt,
                                -cfg.bias_limit_rads, cfg.bias_limit_rads)
            self.corrections += 1

        angle = float(np.linalg.norm(w)) * dt
        if angle > 1e-12:
            axis = w / float(np.linalg.norm(w))
            dq = np.concatenate([[math.cos(angle / 2)], axis * math.sin(angle / 2)])
            self.q = _quat_mul(self.q, dq)
            self.q = self.q / max(float(np.linalg.norm(self.q)), 1e-12)
        return quat_to_matrix(self.q)

    def zero_rotation(self, omega_device: np.ndarray, dt: float) -> None:
        """ZARU: the vehicle is provably parked, so the gyro reads pure bias.

        The gravity reference in :meth:`update` only ever sees the two axes
        orthogonal to gravity - rotation *about* gravity leaves the reference
        unchanged, so yaw bias is unobservable from it. A confirmed stop is the
        one moment all three axes are observable at once, which is why this is
        worth a separate entry point rather than a larger ``ki``.
        """
        cfg = self.cfg
        gain = min(1.0, cfg.zaru_gain * max(dt, 0.0))
        self.bias = np.clip(
            self.bias + gain * (np.asarray(omega_device, dtype=float) - self.bias),
            -cfg.bias_limit_rads, cfg.bias_limit_rads)
        self.zaru_steps += 1

    def roll_pitch(self) -> tuple[float, float]:
        """Roll and pitch of the *device*, radians. Diagnostics only."""
        R = quat_to_matrix(self.q)
        up = R.T @ np.array([0.0, 0.0, 1.0])      # world up, in device axes
        return (math.atan2(up[1], up[2]),
                math.atan2(-up[0], math.hypot(up[1], up[2])))

    def stats(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "gravity_corrections": self.corrections,
            "correction_fraction": round(self.corrections / max(self.steps, 1), 4),
            "gyro_bias_rads": [round(float(b), 6) for b in self.bias],
            "zaru_steps": self.zaru_steps,
        }
