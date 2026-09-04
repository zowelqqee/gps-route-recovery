"""Free-space ParkingTracker regressions; RoadTracker is never used as output."""

from __future__ import annotations

import numpy as np
import pytest

from geotrace.config import Config
from geotrace.models import LocationSample
from geotrace.motion_model import ImuControl
from geotrace.parking_tracker import ParkingTracker


def _fix(frame, t, xy, speed=0.2, accuracy=5.0):
    lat, lon = frame.to_geo(*xy)
    return LocationSample(t, lat, lon, horizontal_accuracy=accuracy, speed=speed)


def _controls(end=60.0, acceleration=0.0):
    return [ImuControl(t=float(t), dt=1.0, a_long=acceleration, yaw_rate=0.0,
                       a_world=(acceleration, 0.0, 0.0)) for t in range(1, int(end) + 1)]


def test_parking_endpoint_can_be_15m_off_road(fork_network) -> None:
    tracker = ParkingTracker(Config(), fork_network.frame)
    fixes = [_fix(fork_network.frame, 50 + i * 2, (300 + i * .2, 15)) for i in range(6)]
    result = tracker.run(_controls(), fixes, 60, (280, 0), 0.0, 0.0, True, fork_network.distance_to_road)
    assert result.status == "CONFIDENT"
    assert result.position[1] == pytest.approx(15, abs=2)
    assert fork_network.distance_to_road(result.position) > 10


def test_unreachable_one_kilometre_fix_is_rejected(fork_network) -> None:
    tracker = ParkingTracker(Config(), fork_network.frame)
    fixes = [_fix(fork_network.frame, 50 + i * 2, (300, 15)) for i in range(5)]
    fixes.append(_fix(fork_network.frame, 60, (1300, 15)))
    result = tracker.run(_controls(), fixes, 60, (280, 0), 0.0, 0.0, True)
    assert result.rejected_fixes >= 1
    assert result.position[0] < 310


def test_signed_velocity_reports_reverse_motion(fork_network) -> None:
    tracker = ParkingTracker(Config(), fork_network.frame)
    result = tracker.run(_controls(acceleration=-1.0), [], 60, (300, 0), 0.0, 5.0, True)
    assert result.has_reverse_motion


def test_missing_calibration_reduces_confidence(fork_network) -> None:
    fixes = [_fix(fork_network.frame, 50 + i * 2, (300, 15)) for i in range(6)]
    calibrated = ParkingTracker(Config(), fork_network.frame).run(_controls(), fixes, 60, (280, 0), 0, 0, True)
    legacy = ParkingTracker(Config(), fork_network.frame).run(_controls(), fixes, 60, (280, 0), 0, 0, False)
    assert legacy.confidence < calibrated.confidence
    assert legacy.status != "CONFIDENT"
