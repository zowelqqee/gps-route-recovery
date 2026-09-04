"""Independent free-space tracker for the last manoeuvre before parking.

Unlike :mod:`geotrace.road_tracker`, this state is ENU and its signed velocity
is never clipped.  It consumes only recorded IMU/GPS plus a road prior; a
reference track is not accepted anywhere in this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from geotrace.config import Config
from geotrace.coordinates import LocalFrame, course_to_heading, wrap_angle
from geotrace.models import LocationSample
from geotrace.motion_model import ImuControl, longitudinal_acceleration


@dataclass
class ParkingResult:
    status: str
    position: tuple[float, float]
    covariance: np.ndarray
    confidence: float
    polygon_radius_m: float
    times: list[float] = field(default_factory=list)
    trajectory: list[tuple[float, float]] = field(default_factory=list)
    terminal_cluster_count: int = 0
    rejected_fixes: int = 0
    has_reverse_motion: bool = False
    reason: str = ""
    max_distance_from_road_m: Optional[float] = None

    def to_json(self, frame: LocalFrame) -> dict:
        lat, lon = frame.to_geo(*self.position)
        return {
            "status": self.status,
            "position": {"latitude": lat, "longitude": lon},
            "covariance_m2": self.covariance.round(3).tolist(),
            "confidence": round(self.confidence, 3),
            "polygon_radius_m": round(self.polygon_radius_m, 2),
            "terminal_cluster_fixes": self.terminal_cluster_count,
            "rejected_fixes": self.rejected_fixes,
            "has_reverse_motion": self.has_reverse_motion,
            "reason": self.reason,
            "max_distance_from_road_m": self.max_distance_from_road_m,
        }


class ParkingTracker:
    """Six-state, free-space terminal tracker: ``x,y,u,theta,b_a,b_w``."""

    def __init__(self, cfg: Config, frame: LocalFrame) -> None:
        self.cfg = cfg
        self.frame = frame
        self.state = np.zeros(6, dtype=float)
        self.P = np.diag([25.0, 25.0, 4.0, 0.8, 0.3, 0.03])

    def run(
        self,
        controls: Sequence[ImuControl],
        fixes: Sequence[LocationSample],
        ended_at: float,
        road_prior_xy: Sequence[float],
        road_heading: float,
        road_speed: float,
        calibration_present: bool,
        road_distance: Optional[callable] = None,
    ) -> ParkingResult:
        cfg = self.cfg.parking_tracker
        start = ended_at - cfg.window_s
        controls = [c for c in controls if start <= c.t <= ended_at]
        fixes = [f for f in fixes if start <= f.monotonic_time <= ended_at and f.is_usable]
        self.state[:] = [road_prior_xy[0], road_prior_xy[1], road_speed, road_heading, 0.0, 0.0]
        trajectory: list[tuple[float, float]] = []
        times: list[float] = []
        accepted: list[tuple[LocationSample, np.ndarray]] = []
        rejected = 0
        cursor = 0
        reverse = False
        for control in controls:
            self._predict(control)
            reverse = reverse or self.state[2] < -0.25
            while cursor < len(fixes) and fixes[cursor].monotonic_time <= control.t:
                fix = fixes[cursor]
                cursor += 1
                xy = np.asarray(self.frame.to_local(fix.latitude, fix.longitude), dtype=float)
                if not self._reachable(fix, xy, accepted, road_prior_xy, start):
                    rejected += 1
                    continue
                self._gps_update(fix, xy)
                accepted.append((fix, xy))
            trajectory.append((float(self.state[0]), float(self.state[1])))
            times.append(control.t)

        # No prediction after `ended_at`: signed velocity is a terminal stop
        # constraint, never an extra extrapolation after recording stopped.
        self.state[2] = 0.0
        terminal = self._terminal_cluster(accepted, ended_at)
        if terminal:
            points, weights = terminal
            centre = np.average(points, axis=0, weights=weights)
            scatter = float(np.sqrt(np.average(np.sum((points - centre) ** 2, axis=1), weights=weights)))
            radius = max(4.0, scatter * 2.0, math.sqrt(1.0 / weights.sum()) * 2.0)
            confidence = min(0.95, 0.55 + 0.08 * len(points) + (0.08 if calibration_present else 0.0))
            status = "CONFIDENT" if confidence >= 0.78 else "PROBABLE"
            reason = "robust low-speed terminal GPS cluster"
            self.state[:2] = centre
        elif accepted:
            radius, confidence, status = cfg.uncertain_radius_m, 0.32, "UNCERTAIN"
            reason = "no mutually consistent terminal GPS cluster"
        else:
            radius, confidence, status = cfg.uncertain_radius_m, 0.12, "INSUFFICIENT_DATA"
            reason = "no usable GPS in terminal window"
        if not calibration_present:
            confidence *= 0.65
            if status == "CONFIDENT":
                status = "PROBABLE"
            reason += "; mount calibration unavailable"
        max_road = None
        if road_distance and trajectory:
            max_road = float(max(road_distance(p) for p in trajectory))
        return ParkingResult(
            status=status, position=(float(self.state[0]), float(self.state[1])),
            covariance=np.diag([radius**2, radius**2]), confidence=confidence,
            polygon_radius_m=radius, times=times, trajectory=trajectory,
            terminal_cluster_count=(len(terminal[0]) if terminal else 0), rejected_fixes=rejected,
            has_reverse_motion=bool(reverse), reason=reason, max_distance_from_road_m=max_road,
        )

    def _predict(self, control: ImuControl) -> None:
        if control.dt <= 0 or control.gap_exceeded:
            return
        x, y, u, theta, ba, bw = self.state
        theta = float(wrap_angle(theta + (control.yaw_rate - bw) * control.dt))
        a = longitudinal_acceleration(control.a_world, theta) - ba
        ds = u * control.dt + 0.5 * a * control.dt * control.dt
        self.state[0] = x + ds * math.cos(theta)
        self.state[1] = y + ds * math.sin(theta)
        self.state[2] = u + a * control.dt  # signed: reverse is valid
        self.state[3] = theta
        self.P[:2, :2] += np.eye(2) * (self.cfg.parking_tracker.process_position_sigma_m * control.dt) ** 2

    def _reachable(self, fix, xy, accepted, road_prior_xy, start) -> bool:
        cfg = self.cfg.parking_tracker
        if fix.horizontal_accuracy is None or fix.horizontal_accuracy <= 0 or fix.horizontal_accuracy > cfg.max_accuracy_m:
            return False
        if accepted:
            prev, prev_xy = accepted[-1]
            dt = max(0.0, fix.monotonic_time - prev.monotonic_time)
            maximum = self.cfg.motion.max_speed_ms * dt + cfg.physical_margin_m
            return math.dist(xy, prev_xy) <= maximum
        dt = max(0.0, fix.monotonic_time - start)
        maximum = self.cfg.motion.max_speed_ms * dt + cfg.physical_margin_m
        return math.dist(xy, road_prior_xy) <= maximum

    def _gps_update(self, fix: LocationSample, xy: np.ndarray) -> None:
        sigma = max(float(fix.horizontal_accuracy or 8.0), self.cfg.gps.min_accuracy_sigma_m)
        gain = float(np.clip(30.0 / (30.0 + sigma * sigma), 0.08, 0.65))
        previous = self.state[:2].copy()
        self.state[:2] += gain * (xy - self.state[:2])
        if fix.has_valid_course and fix.has_valid_speed and float(fix.speed or 0.0) >= self.cfg.parking_tracker.low_speed_ms:
            self.state[3] = course_to_heading(float(fix.course or 0.0))
        if fix.has_valid_speed:
            observed_speed = float(fix.speed or 0.0)
            displacement = xy - previous
            direction = np.array([math.cos(self.state[3]), math.sin(self.state[3])])
            sign = -1.0 if float(np.dot(displacement, direction)) < 0 else 1.0
            self.state[2] = sign * observed_speed

    def _terminal_cluster(self, accepted, ended_at):
        cfg = self.cfg.parking_tracker
        tail = [(f, xy) for f, xy in accepted if f.monotonic_time >= ended_at - cfg.terminal_cluster_window_s
                and f.has_valid_speed and float(f.speed or 0.0) <= cfg.low_speed_ms]
        if len(tail) < cfg.min_cluster_fixes:
            return None
        points = np.asarray([xy for _, xy in tail])
        median = np.median(points, axis=0)
        kept = [(f, xy) for f, xy in tail if math.dist(xy, median) <= 35.0]
        if len(kept) < cfg.min_cluster_fixes:
            return None
        weights = np.asarray([1.0 / max(float(f.horizontal_accuracy or 5.0), 5.0) ** 2 for f, _ in kept])
        return np.asarray([xy for _, xy in kept]), weights
