"""Extended Kalman filter over [E, N, v, psi, b_a, b_omega].

This is the `ekf_dead_reckoning` baseline: IMU propagation with GPS updates,
no road graph at all. It is useful in its own right (it is what the particle
filter's gates are compared against) and it is the honest reference for how far
pure inertial dead reckoning drifts during an outage.

    predict:  X^- = f(X, u),        P^- = F P F^T + G Q G^T
    update:   K   = P^- H^T (H P^- H^T + R)^-1
              X   = X^- + K (z - H X^-)
              P   = (I - K H) P^-
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from geotrace.config import MotionConfig
from geotrace.coordinates import wrap_angle
from geotrace.motion_model import (
    IDX_BA,
    IDX_BW,
    IDX_E,
    IDX_N,
    IDX_PSI,
    IDX_V,
    STATE_DIM,
    longitudinal_acceleration,
    noise_jacobian,
    process_noise,
    propagate_state,
    transition_jacobian,
)


@dataclass
class EKFResult:
    times: list[float] = field(default_factory=list)
    states: list[np.ndarray] = field(default_factory=list)
    covariances: list[np.ndarray] = field(default_factory=list)

    def positions(self) -> np.ndarray:
        if not self.states:
            return np.zeros((0, 2))
        return np.array([[s[IDX_E], s[IDX_N]] for s in self.states])

    def position_sigmas(self) -> np.ndarray:
        if not self.covariances:
            return np.zeros(0)
        return np.array([math.sqrt(max(0.0, P[0, 0] + P[1, 1])) for P in self.covariances])


class ExtendedKalmanFilter:
    """EKF with an analytic Jacobian (see :mod:`geotrace.motion_model`)."""

    H_POS = np.zeros((2, STATE_DIM))
    H_POS[0, IDX_E] = 1.0
    H_POS[1, IDX_N] = 1.0

    def __init__(
        self,
        cfg: MotionConfig,
        initial_state: Optional[Sequence[float]] = None,
        initial_covariance: Optional[np.ndarray] = None,
    ) -> None:
        self.cfg = cfg
        self.x = (
            np.asarray(initial_state, dtype=float).copy()
            if initial_state is not None
            else np.zeros(STATE_DIM)
        )
        if initial_covariance is not None:
            self.P = np.asarray(initial_covariance, dtype=float).copy()
        else:
            self.P = np.diag([25.0, 25.0, 4.0, 0.35, 0.25, 0.01])
        self.skipped_gaps = 0
        """Steps refused because the timestamp gap was larger than max_gap_s."""

    # ------------------------------------------------------------- predict

    def predict(self, a_world: Sequence[float], yaw_rate: float, dt: float) -> None:
        if dt <= 0:
            return
        if dt > self.cfg.max_gap_s:
            # Do not integrate across a hole; inflate the covariance instead so
            # the filter honestly reports that it lost track.
            self.skipped_gaps += 1
            growth = (self.cfg.max_speed_ms * dt) ** 2
            self.P[IDX_E, IDX_E] += growth
            self.P[IDX_N, IDX_N] += growth
            self.P[IDX_PSI, IDX_PSI] += (math.pi / 2) ** 2
            return

        a_long = longitudinal_acceleration(a_world, float(self.x[IDX_PSI]))
        F = transition_jacobian(self.x, a_long, yaw_rate, dt, self.cfg)
        G = noise_jacobian(self.x, dt)
        Q = process_noise(dt, self.cfg)
        self.x = propagate_state(self.x, a_long, yaw_rate, dt, self.cfg)
        self.P = F @ self.P @ F.T + G @ Q @ G.T
        self.P = 0.5 * (self.P + self.P.T)

    # -------------------------------------------------------------- update

    def update_position(self, measurement: Sequence[float], sigma: float) -> np.ndarray:
        """Standard linear position update. Returns the residual."""
        z = np.asarray(measurement, dtype=float).reshape(2)
        H = self.H_POS
        R = np.eye(2) * (sigma**2)
        residual = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ residual
        self.x[IDX_PSI] = wrap_angle(self.x[IDX_PSI])
        self.x[IDX_V] = max(0.0, self.x[IDX_V])
        I_KH = np.eye(STATE_DIM) - K @ H
        # Joseph form: stays positive-definite even with a rough K.
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return residual

    def update_speed(self, speed: float, sigma: float) -> None:
        H = np.zeros((1, STATE_DIM))
        H[0, IDX_V] = 1.0
        self._scalar_update(H, np.array([speed]) - H @ self.x, sigma)

    def update_heading(self, heading_rad: float, sigma: float) -> None:
        H = np.zeros((1, STATE_DIM))
        H[0, IDX_PSI] = 1.0
        residual = np.array([wrap_angle(heading_rad - float(self.x[IDX_PSI]))])
        self._scalar_update(H, residual, sigma)

    def zero_velocity_update(self, sigma: float = 0.05) -> None:
        """ZUPT: while the car is provably still, v is 0 and the residual
        speed error is accumulated accelerometer bias."""
        H = np.zeros((1, STATE_DIM))
        H[0, IDX_V] = 1.0
        self._scalar_update(H, np.array([0.0 - float(self.x[IDX_V])]), sigma)

    def _scalar_update(self, H: np.ndarray, residual: np.ndarray, sigma: float) -> None:
        R = np.array([[sigma**2]])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ residual).reshape(-1)
        self.x[IDX_PSI] = wrap_angle(self.x[IDX_PSI])
        self.x[IDX_V] = max(0.0, self.x[IDX_V])
        I_KH = np.eye(STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def reanchor(
        self,
        position: Sequence[float],
        sigma: float,
        heading_rad: Optional[float] = None,
        speed: Optional[float] = None,
    ) -> None:
        """Hard reset onto a fix that trust has just been restored to.

        After a long outage the dead-reckoned state is not a prior worth
        blending with: it is stale by hundreds of metres and its covariance
        understates that. Letting the Kalman gain merge the two would drag the
        recovered position back towards the drift for many seconds. The bias
        estimates are the one thing worth keeping - they were learned before the
        outage and are still valid.
        """
        self.x[IDX_E] = float(position[0])
        self.x[IDX_N] = float(position[1])
        if heading_rad is not None:
            self.x[IDX_PSI] = wrap_angle(heading_rad)
        if speed is not None:
            self.x[IDX_V] = max(0.0, float(speed))
        variance = max(sigma, 1.0) ** 2
        self.P[IDX_E, :] = 0.0
        self.P[:, IDX_E] = 0.0
        self.P[IDX_N, :] = 0.0
        self.P[:, IDX_N] = 0.0
        self.P[IDX_E, IDX_E] = variance
        self.P[IDX_N, IDX_N] = variance
        if heading_rad is not None:
            self.P[IDX_PSI, IDX_PSI] = math.radians(20.0) ** 2
        self.P = 0.5 * (self.P + self.P.T)

    def inflate_for_mount_disturbance(
        self, position_noise_mpsqrt: float, heading_noise_radsqrt: float, dt: float
    ) -> None:
        """Represent a phone shock without integrating it as vehicle motion."""
        scale = max(0.0, float(dt))
        self.P[IDX_E, IDX_E] += (position_noise_mpsqrt**2) * scale
        self.P[IDX_N, IDX_N] += (position_noise_mpsqrt**2) * scale
        self.P[IDX_PSI, IDX_PSI] += (heading_noise_radsqrt**2) * scale
        self.P = 0.5 * (self.P + self.P.T)

    # ----------------------------------------------------------- accessors

    @property
    def position(self) -> np.ndarray:
        return self.x[[IDX_E, IDX_N]].copy()

    @property
    def speed(self) -> float:
        return float(self.x[IDX_V])

    @property
    def heading(self) -> float:
        return float(self.x[IDX_PSI])

    @property
    def biases(self) -> tuple[float, float]:
        return float(self.x[IDX_BA]), float(self.x[IDX_BW])

    def state_json(self) -> dict[str, Any]:
        return {
            "east": float(self.x[IDX_E]),
            "north": float(self.x[IDX_N]),
            "speed": float(self.x[IDX_V]),
            "heading_rad": float(self.x[IDX_PSI]),
            "accel_bias": float(self.x[IDX_BA]),
            "gyro_bias": float(self.x[IDX_BW]),
            "position_sigma_m": math.sqrt(max(0.0, self.P[0, 0] + self.P[1, 1])),
        }
