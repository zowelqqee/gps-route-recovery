"""The IMU step, and the GPS-free stop detector.

Everything that used to live here about *estimating* motion has moved to
:mod:`~geotrace.pacman_tracker.speed`, where there is one estimate for the one
car. What remains is the per-step record handed to the rest of the tracker, and
the detector that decides whether the car is moving at all - which is the single
thing the vibration signal is genuinely good at.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geotrace.pacman_tracker.config import MotionConfigP


@dataclass
class ImuSample:
    """One resampled step of the IMU timeline, shared by all hypotheses."""

    t: float
    dt: float
    a_long: float
    """Longitudinal (vehicle-forward) acceleration, m/s^2, bias not removed."""

    yaw_rate: float
    """Yaw rate about the world vertical, rad/s, bias not removed."""

    a_lat: float = 0.0
    """Lateral (horizontal, left-positive) acceleration, m/s^2, smoothed over
    the same window as ``yaw_rate_smooth``. The two form ``v = a_lat / omega``
    and must be smoothed identically or their ratio is not a speed."""

    yaw_rate_smooth: float = 0.0

    stationary: bool = False
    stationary_run_s: float = 0.0
    """How long the current quiet stretch has lasted. A stop is only credible
    if the car could have decelerated into it in that time."""

    """Vehicle is almost certainly not moving. Derived from IMU variance only -
    never from GPS, which does not exist during an outage."""

    shock: bool = False
    gap: bool = False
    accel_std: float = 0.0
    gyro_std: float = 0.0
    peak_accel: float = 0.0
    spectral_speed: float = float("nan")
    spectral_sigma: float = float("nan")


def _rolling_std(values: np.ndarray, half_window: int) -> np.ndarray:
    n = len(values)
    if n == 0:
        return values
    c1 = np.concatenate([[0.0], np.cumsum(values)])
    c2 = np.concatenate([[0.0], np.cumsum(values * values)])
    lo = np.maximum(np.arange(n) - half_window, 0)
    hi = np.minimum(np.arange(n) + half_window + 1, n)
    count = hi - lo
    mean = (c1[hi] - c1[lo]) / count
    var = (c2[hi] - c2[lo]) / count - mean * mean
    return np.sqrt(np.maximum(var, 0.0))


def detect_stationary(
    times: np.ndarray,
    a_long: np.ndarray,
    yaw_rate: np.ndarray,
    cfg: MotionConfigP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GPS-free stop detector.

    A stopped car is not silent - the engine idles - but its longitudinal
    acceleration and yaw rate both stop *varying*. Measured on the two review
    recordings this separates cleanly: standing still gives an accel std of
    0.008-0.011 m/s^2 against 0.08-0.30 while driving, and a yaw-rate std of
    0.0008 rad/s against 0.002-0.008. The thresholds are set for a low false
    positive rate (about 1 % of moving samples), because a false stop injected
    into a dead-reckoner is far more damaging than a missed one: the missed stop
    merely leaves the speed uncertainty wide.

    This is the one job the vibration signal does well. It is emphatically not a
    speedometer - see :mod:`~geotrace.pacman_tracker.spectral`.

    Returns ``(stationary, accel_std, gyro_std, run_seconds)``.
    """
    if len(times) == 0:
        return np.zeros(0, dtype=bool), np.zeros(0), np.zeros(0), np.zeros(0)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else cfg.dt_s
    half = max(1, int(round(0.5 * cfg.zupt_window_s / max(dt, 1e-6))))
    a_std = _rolling_std(np.nan_to_num(a_long), half)
    w_std = _rolling_std(np.nan_to_num(yaw_rate), half)
    quiet = (a_std < cfg.zupt_accel_std_ms2) & (w_std < cfg.zupt_gyro_std_rads)

    need = max(1, int(round(cfg.zupt_min_duration_s / max(dt, 1e-6))))
    if need > 1:
        run = np.zeros(len(quiet), dtype=np.int32)
        count = 0
        for i, q in enumerate(quiet):
            count = count + 1 if q else 0
            run[i] = count
        stationary = np.zeros(len(quiet), dtype=bool)
        i = len(quiet) - 1
        while i >= 0:
            if run[i] >= need:
                start = i - run[i] + 1
                stationary[start : i + 1] = True
                i = start - 1
            else:
                i -= 1
    else:
        stationary = quiet.copy()

    # How long each stationary sample has been stationary for.
    run_s = np.zeros(len(stationary))
    elapsed = 0.0
    for i, flag in enumerate(stationary):
        elapsed = elapsed + dt if flag else 0.0
        run_s[i] = elapsed
    return stationary, a_std, w_std, run_s
