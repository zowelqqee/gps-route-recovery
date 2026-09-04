"""Detection of broken GPS.

A single `horizontalAccuracy` is not evidence. A receiver that has just teleported
the car 4 km across the Neva will happily report 12 m. So every fix is tested
against several independent pieces of evidence:

  * a physical gate - could the car have got there at all;
  * a Mahalanobis gate against the filter's own predicted position;
  * consistency of the reported speed with the filter speed;
  * consistency of the reported course with the filter heading (only above
    `min_speed_for_course_ms` - below that CoreLocation course is noise);
  * distance to the nearest drivable road;
  * the recent history of the same fix stream.

The result drives a four-state machine:

    TRUSTED -> SUSPECT -> LOST -> RECOVERING -> TRUSTED

Returning to TRUSTED deliberately requires several consecutive consistent fixes.
That is what rejects the handful of false points a receiver emits in the first
second after it re-acquires a signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence

import numpy as np

from geotrace.config import GPSQualityConfig
from geotrace.coordinates import course_to_heading, wrap_angle
from geotrace.models import LocationSample


class GPSState(str, Enum):
    TRUSTED = "TRUSTED"
    SUSPECT = "SUSPECT"
    LOST = "LOST"
    RECOVERING = "RECOVERING"


@dataclass
class GateResult:
    """Outcome of testing one fix."""

    accepted: bool
    state: GPSState
    reasons: list[str] = field(default_factory=list)
    distance_m: Optional[float] = None
    max_distance_m: Optional[float] = None
    mahalanobis: Optional[float] = None
    road_distance_m: Optional[float] = None
    sigma_m: Optional[float] = None
    monotonic_time: Optional[float] = None

    def to_json(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "state": self.state.value,
            "reasons": self.reasons,
            "monotonic_time": self.monotonic_time,
            "distance_m": None if self.distance_m is None else round(self.distance_m, 2),
            "max_distance_m": None if self.max_distance_m is None else round(self.max_distance_m, 2),
            "mahalanobis": None if self.mahalanobis is None else round(self.mahalanobis, 3),
            "road_distance_m": None if self.road_distance_m is None else round(self.road_distance_m, 2),
            "sigma_m": None if self.sigma_m is None else round(self.sigma_m, 2),
        }


def physical_gate(
    distance_m: float, dt: float, previous_speed: float, cfg: GPSQualityConfig, max_accel: float
) -> tuple[bool, float]:
    """d_max = v*dt + 0.5*a_max*dt^2 + m; the fix is an outlier when d > d_max."""
    dt = max(dt, 0.0)
    d_max = previous_speed * dt + 0.5 * max_accel * dt * dt + cfg.physical_margin_m
    return distance_m <= d_max, d_max


def mahalanobis_gate(
    residual: Sequence[float], S: np.ndarray, threshold: float
) -> tuple[bool, float]:
    """D^2 = r^T S^-1 r, with S = H P H^T + R.

    Returns (passed, D^2). A singular S is treated as a failed test rather than
    raising: a degenerate covariance means the filter has no opinion and should
    not be allowed to bless an arbitrary fix.
    """
    r = np.asarray(residual, dtype=float).reshape(-1)
    S = np.asarray(S, dtype=float)
    try:
        solved = np.linalg.solve(S, r)
    except np.linalg.LinAlgError:
        return False, float("inf")
    d2 = float(r @ solved)
    if not math.isfinite(d2) or d2 < 0:
        return False, float("inf")
    return d2 <= threshold, d2


def measurement_sigma(sample: LocationSample, cfg: GPSQualityConfig) -> float:
    """Turn the reported accuracy into a usable sigma with a sane floor."""
    accuracy = sample.horizontal_accuracy
    if accuracy is None or accuracy < 0:
        return cfg.max_horizontal_accuracy_m
    return max(cfg.min_accuracy_sigma_m, cfg.accuracy_sigma_scale * float(accuracy))


class GPSQualityMonitor:
    """The TRUSTED / SUSPECT / LOST / RECOVERING state machine."""

    def __init__(self, cfg: GPSQualityConfig, max_accel_ms2: float = 6.0) -> None:
        self.cfg = cfg
        self.max_accel = max_accel_ms2
        self.state = GPSState.LOST
        """A trip starts with no established trust; the first consistent fixes
        promote it to TRUSTED."""

        self.consecutive_bad = 0
        self.consecutive_good = 0
        self.last_accepted: Optional[LocationSample] = None
        self.last_accepted_xy: Optional[tuple[float, float]] = None
        self.last_seen_time: Optional[float] = None
        self._untrusted_since: Optional[float] = None
        self.history: list[GateResult] = []
        self._bootstrap_done = False

    # ------------------------------------------------------------- helpers

    @property
    def is_trusted(self) -> bool:
        return self.state is GPSState.TRUSTED

    def note_gap(self, now: float) -> bool:
        """Call on every filter step. Returns True if a dropout pushed us to LOST."""
        if self.last_seen_time is None:
            return False
        if now - self.last_seen_time > self.cfg.lost_gap_s and self.state is not GPSState.LOST:
            self.state = GPSState.LOST
            self.consecutive_good = 0
            self._untrusted_since = now
            self.history.append(
                GateResult(
                    accepted=False,
                    state=self.state,
                    reasons=["dropout"],
                    monotonic_time=now,
                )
            )
            return True
        return False

    # ---------------------------------------------------------------- main

    def update(
        self,
        sample: LocationSample,
        measured_xy: Sequence[float],
        predicted_xy: Optional[Sequence[float]] = None,
        predicted_speed: float = 0.0,
        predicted_heading: Optional[float] = None,
        covariance: Optional[np.ndarray] = None,
        road_distance_m: Optional[float] = None,
    ) -> GateResult:
        """Test one fix and advance the state machine."""
        cfg = self.cfg
        reasons: list[str] = []
        sigma = measurement_sigma(sample, cfg)
        result = GateResult(
            accepted=True,
            state=self.state,
            sigma_m=sigma,
            road_distance_m=road_distance_m,
            monotonic_time=sample.monotonic_time,
        )

        # Every gate is evaluated, even after one has already failed: the full
        # reason list is what makes a rejection diagnosable afterwards.
        if not sample.is_usable:
            reasons.append("invalid_fix")

        if (
            not cfg.allow_simulated_fixes
            and sample.source_information
            and sample.source_information.get("is_simulated_by_software")
        ):
            reasons.append("simulated_by_software")

        # --- physical gate against the last accepted fix
        if self.last_accepted_xy is not None and self.last_accepted is not None:
            dt = sample.monotonic_time - self.last_accepted.monotonic_time
            distance = math.dist(measured_xy, self.last_accepted_xy)
            speed = predicted_speed
            if self.last_accepted.has_valid_speed:
                speed = max(speed, float(self.last_accepted.speed or 0.0))
            passed, d_max = physical_gate(distance, dt, speed, cfg, self.max_accel)
            result.distance_m = distance
            result.max_distance_m = d_max
            if not passed:
                reasons.append("physical_gate")

        # --- Innovation gate: GPS is a position measurement; the IMU is the
        # prediction. This test is intentionally independent of the road graph.
        # When IMU-only tracking has lasted a while, its uncertainty is included
        # in S rather than disabling the test or declaring a GPS point invalid.
        tracking = self.state in (GPSState.TRUSTED, GPSState.SUSPECT)

        if predicted_xy is not None and covariance is not None:
            residual = np.asarray(measured_xy, dtype=float) - np.asarray(predicted_xy, dtype=float)
            P_xy = np.asarray(covariance, dtype=float)[:2, :2].copy()
            if not tracking:
                since = 0.0 if self._untrusted_since is None else max(
                    0.0, sample.monotonic_time - self._untrusted_since
                )
                drift_sigma = (
                    cfg.recovery_position_sigma_m
                    + cfg.recovery_position_sigma_growth_mps * since
                )
                P_xy += np.eye(2) * (drift_sigma**2)
            S = P_xy + np.eye(2) * (sigma**2)
            passed, d2 = mahalanobis_gate(residual, S, cfg.mahalanobis_threshold)
            result.mahalanobis = d2
            if not passed:
                reasons.append("mahalanobis_gate")

        if tracking:

            if sample.has_valid_speed and self._bootstrap_done:
                if abs(float(sample.speed or 0.0) - predicted_speed) > cfg.max_speed_mismatch_ms:
                    reasons.append("speed_mismatch")

            if (
                predicted_heading is not None
                and sample.has_valid_course
                and sample.has_valid_speed
                and float(sample.speed or 0.0) >= cfg.min_speed_for_course_ms
                and predicted_speed >= cfg.min_speed_for_course_ms
                and self._bootstrap_done
            ):
                measured_heading = course_to_heading(float(sample.course or 0.0))
                error = abs(math.degrees(wrap_angle(measured_heading - predicted_heading)))
                if error > cfg.max_course_error_deg:
                    reasons.append("course_mismatch")
        else:
            reasons.extend(self._fix_to_fix_reasons(sample, measured_xy))

        result.accepted = not reasons
        result.reasons = reasons
        self._advance(result, sample, measured_xy)
        result.state = self.state
        self.history.append(result)
        self.last_seen_time = sample.monotonic_time
        return result

    def _advance(
        self,
        result: GateResult,
        sample: LocationSample,
        measured_xy: Sequence[float],
    ) -> None:
        cfg = self.cfg
        if result.accepted:
            self.consecutive_bad = 0
            self.consecutive_good += 1
            self.last_accepted = sample
            self.last_accepted_xy = (float(measured_xy[0]), float(measured_xy[1]))
            if self.state is GPSState.TRUSTED:
                pass
            elif self.state is GPSState.LOST:
                self.state = GPSState.RECOVERING
            elif self.state in (GPSState.RECOVERING, GPSState.SUSPECT):
                if self.consecutive_good >= cfg.recover_count:
                    self.state = GPSState.TRUSTED
                    self._bootstrap_done = True
                else:
                    self.state = GPSState.RECOVERING
            if self.consecutive_good >= cfg.recover_count:
                self.state = GPSState.TRUSTED
                self._bootstrap_done = True
                self._untrusted_since = None
        else:
            self.consecutive_good = 0
            self.consecutive_bad += 1
            if self.state is GPSState.TRUSTED:
                self.state = GPSState.SUSPECT
                self._untrusted_since = sample.monotonic_time
            elif self.state is GPSState.SUSPECT and self.consecutive_bad >= cfg.suspect_to_lost_count:
                self.state = GPSState.LOST
            elif self.state is GPSState.RECOVERING:
                self.state = GPSState.LOST

    def _fix_to_fix_reasons(
        self, sample: LocationSample, measured_xy: Sequence[float]
    ) -> list[str]:
        """Consistency of this fix with the previous accepted one.

        Used while the filter has lost track, where the only trustworthy
        reference is the fix stream itself. A receiver that has just
        re-acquired emits points that scatter: their implied speed and bearing
        disagree wildly with what they report, and the chain of consecutive
        agreements that RECOVERING requires never forms.
        """
        cfg = self.cfg
        reasons: list[str] = []
        previous = self.last_accepted
        if previous is None or self.last_accepted_xy is None:
            return reasons

        dt = sample.monotonic_time - previous.monotonic_time
        if dt <= 0:
            return ["out_of_order"]
        # After a long hole the implied speed/bearing say nothing about a fix.
        if dt > cfg.lost_gap_s:
            return reasons

        delta = (
            measured_xy[0] - self.last_accepted_xy[0],
            measured_xy[1] - self.last_accepted_xy[1],
        )
        step = math.hypot(*delta)
        implied_speed = step / dt

        if sample.has_valid_speed:
            if abs(float(sample.speed or 0.0) - implied_speed) > cfg.recovery_speed_mismatch_ms:
                reasons.append("speed_inconsistent_with_previous_fix")

        if (
            sample.has_valid_course
            and step >= cfg.recovery_min_step_m
            and implied_speed >= cfg.min_speed_for_course_ms
        ):
            implied_heading = math.atan2(delta[1], delta[0])
            measured_heading = course_to_heading(float(sample.course or 0.0))
            error = abs(math.degrees(wrap_angle(measured_heading - implied_heading)))
            if error > cfg.recovery_course_error_deg:
                reasons.append("course_inconsistent_with_previous_fix")

        return reasons

    # ---------------------------------------------------------- diagnostics

    def summary(self) -> dict[str, Any]:
        accepted = sum(1 for h in self.history if h.accepted)
        reasons: dict[str, int] = {}
        for h in self.history:
            for r in h.reasons:
                reasons[r] = reasons.get(r, 0) + 1
        return {
            "final_state": self.state.value,
            "fixes_tested": len(self.history),
            "fixes_accepted": accepted,
            "fixes_rejected": len(self.history) - accepted,
            "rejection_reasons": reasons,
        }


def state_intervals(history: Sequence[GateResult]) -> list[dict[str, Any]]:
    """Collapse the per-fix history into [start, end] intervals per state.

    Used by the report to shade the outage on the timeline.
    """
    out: list[dict[str, Any]] = []
    for item in history:
        if item.monotonic_time is None:
            continue
        if out and out[-1]["state"] == item.state.value:
            out[-1]["end_s"] = item.monotonic_time
        else:
            out.append(
                {
                    "state": item.state.value,
                    "start_s": item.monotonic_time,
                    "end_s": item.monotonic_time,
                }
            )
    return out
