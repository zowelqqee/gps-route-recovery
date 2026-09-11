"""Heading change over a window - the discrimination that actually works here.

Pointwise curvature matching asks "does the yaw rate right now match this
road's curvature right now". On the review recordings that question is nearly
content-free: on the true road |kappa| has a median of 0.00000 and a p90 of
0.00057 rad/m, and only 1.5 % of moving steps have anything worth matching.
Petersburg streets are straight, and what distinguishes routes is not the
streets - it is the **junctions between them**.

So the useful question is the integrated one::

              left +87 deg
                 /
    ------------o------------  straight +3 deg
                 \\
              right -91 deg

    gyro over the crossing:  -84 deg   ->  right is strongly favoured,
                                           straight and left are not

which compares a number the gyro measures well (integrated yaw over a few
seconds, drift-free at that timescale) against a number the map knows well (the
angle between two edges at a node). Neither side needs the map's polyline shape
to be right, and neither side needs an instantaneous speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from geotrace.coordinates import wrap_angle


@dataclass
class TurnEvent:
    """A distinct heading change, for diagnostics and reporting."""

    t_start: float
    t_end: float
    t_peak: float
    delta_psi: float
    peak_rate: float

    def to_json(self) -> dict[str, Any]:
        return {
            "t_start": round(self.t_start, 2),
            "t_end": round(self.t_end, 2),
            "t_peak": round(self.t_peak, 2),
            "delta_deg": round(math.degrees(self.delta_psi), 1),
            "peak_rate_rads": round(self.peak_rate, 4),
        }


class HeadingIntegrator:
    """Integrated yaw over any window of the trip, with an honest sigma.

    Integrated yaw is the one thing the gyro is genuinely good at over seconds:
    the noise averages down and the bias has not had time to matter. Over the
    whole outage it is worthless (thousands of degrees of accumulated drift on
    these recordings) - which is exactly why it is used over a few-second
    junction window and never as an absolute heading.
    """

    def __init__(self, times: Sequence[float], yaw_rate: Sequence[float],
                 gyro_noise_rads: float = 0.02) -> None:
        self.times = np.asarray(times, dtype=float)
        self.rate = np.asarray(yaw_rate, dtype=float)
        self.gyro_noise = float(gyro_noise_rads)
        if self.times.size:
            dt = np.diff(self.times, prepend=self.times[0])
            self.dt = np.where(dt > 0, dt, np.median(dt) if self.times.size > 1 else 0.1)
            self._cum = np.concatenate([[0.0], np.cumsum(self.rate * self.dt)])
            self._cum_dt = np.concatenate([[0.0], np.cumsum(self.dt)])
        else:
            self.dt = np.zeros(0)
            self._cum = np.zeros(1)
            self._cum_dt = np.zeros(1)

    def _index(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.times, t), 0, self.times.size))

    def delta(self, t0: float, t1: float, gyro_bias: float = 0.0) -> float:
        """Integrated yaw between two times, gyro bias removed."""
        if self.times.size == 0 or t1 <= t0:
            return 0.0
        i0, i1 = self._index(t0), self._index(t1)
        span = float(self._cum_dt[i1] - self._cum_dt[i0])
        return float(self._cum[i1] - self._cum[i0]) - float(gyro_bias) * span

    def sigma(self, t0: float, t1: float, gyro_bias_sigma: float = 0.004,
              model_sigma_rad: float = 0.15) -> float:
        """Uncertainty of that integral.

        Three terms: gyro white noise averaged over the window, the bias times
        the window length, and a model term for the fact that a driver's line
        through a junction is not the map's angle between two edges - they cut
        corners, they swing wide, and the node is not where the turn's centre
        actually is. The model term dominates, at about 9 degrees.
        """
        if self.times.size == 0 or t1 <= t0:
            return float(model_sigma_rad)
        i0, i1 = self._index(t0), self._index(t1)
        span = float(self._cum_dt[i1] - self._cum_dt[i0])
        step = float(np.median(self.dt)) if self.dt.size else 0.1
        white = self.gyro_noise * math.sqrt(max(span, 0.0) * step)
        drift = float(gyro_bias_sigma) * span
        return float(math.sqrt(white**2 + drift**2 + model_sigma_rad**2))


    def centroid(self, t0: float, t1: float, gyro_bias: float = 0.0) -> Optional[float]:
        """When, inside this window, the turning actually happened.

        The |omega|-weighted centre of the window. This is the observation that
        says *where along its route* a hypothesis really is: the map says the
        junction is at a given route length, the gyro says the turn happened at
        a given time, and the difference between them is how far the odometer
        has drifted from the map's idea of the same road.
        """
        if self.times.size == 0 or t1 <= t0:
            return None
        i0, i1 = self._index(t0), self._index(t1)
        if i1 <= i0:
            return None
        weight = np.abs(self.rate[i0:i1] - float(gyro_bias)) * self.dt[i0:i1]
        total = float(weight.sum())
        if total < 1e-9:
            return None
        return float((self.times[i0:i1] * weight).sum() / total)


def detect_turns(times: Sequence[float], yaw_rate: Sequence[float],
                 gyro_bias: float = 0.0, min_rate_rads: float = 0.06,
                 min_delta_rad: float = math.radians(25.0),
                 max_gap_s: float = 1.0) -> list[TurnEvent]:
    """Group sustained yaw into discrete turn events. Diagnostics only."""
    times = np.asarray(times, dtype=float)
    rate = np.asarray(yaw_rate, dtype=float) - float(gyro_bias)
    if times.size < 3:
        return []
    dt = float(np.median(np.diff(times)))
    active = np.abs(rate) > min_rate_rads
    events: list[TurnEvent] = []
    i = 0
    n = len(times)
    gap_steps = max(1, int(round(max_gap_s / max(dt, 1e-6))))
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i
        idle = 0
        while j + 1 < n and idle <= gap_steps:
            j += 1
            idle = 0 if active[j] else idle + 1
        seg = slice(i, j + 1)
        delta = float(np.sum(rate[seg]) * dt)
        if abs(delta) >= min_delta_rad:
            peak = int(np.argmax(np.abs(rate[seg]))) + i
            events.append(TurnEvent(t_start=float(times[i]), t_end=float(times[j]),
                                    t_peak=float(times[peak]), delta_psi=delta,
                                    peak_rate=float(rate[peak])))
        i = j + 1
    return events


def turn_log_likelihood(map_turn: np.ndarray, measured: float, sigma: float,
                        dof: float = 4.0) -> np.ndarray:
    """Robust score for "did the car take *this* turn?".

    Student-t, for the same reason every other score in this tracker is: a
    driver who swings wide round one corner, or a single knock on the phone
    during a crossing, must cost a hypothesis some weight and never all of it.
    """
    z = wrap_angle(np.asarray(map_turn, dtype=float) - float(measured)) / max(sigma, 1e-6)
    return -0.5 * (dof + 1.0) * np.log1p((z * z) / dof)
