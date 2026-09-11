"""Integrated yaw over a junction - the discrimination that works in a city.

Pointwise curvature matching has almost nothing to say on this data: on the
true road |kappa| has a median of 0.00000 rad/m and only 1.5 % of moving steps
carry anything worth matching. Petersburg streets are straight, and what tells
routes apart is the angle turned at the junctions between them.
"""

import math

import numpy as np
import pytest

from geotrace.pacman_tracker.turns import (
    HeadingIntegrator,
    detect_turns,
    turn_log_likelihood,
)


def _profile(duration=30.0, dt=0.1, turns=()):
    t = np.arange(0.0, duration, dt)
    rate = np.zeros_like(t)
    for start, length, angle in turns:
        mask = (t >= start) & (t < start + length)
        rate[mask] = angle / length
    return t, rate


def test_integrated_yaw_recovers_the_turn():
    t, rate = _profile(turns=[(10.0, 3.0, math.radians(90.0))])
    h = HeadingIntegrator(t, rate)
    assert math.degrees(h.delta(8.0, 15.0)) == pytest.approx(90.0, abs=1.0)


def test_a_window_that_misses_the_turn_sees_nothing():
    t, rate = _profile(turns=[(10.0, 3.0, math.radians(90.0))])
    h = HeadingIntegrator(t, rate)
    assert abs(math.degrees(h.delta(20.0, 27.0))) < 1.0


def test_the_gyro_bias_is_removed():
    t, rate = _profile(turns=[(10.0, 3.0, math.radians(90.0))])
    bias = 0.01
    h = HeadingIntegrator(t, rate + bias)
    assert math.degrees(h.delta(8.0, 15.0, gyro_bias=bias)) == pytest.approx(90.0, abs=1.0)
    assert math.degrees(h.delta(8.0, 15.0, gyro_bias=0.0)) > 93.0


def test_sigma_grows_with_the_window():
    t, rate = _profile()
    h = HeadingIntegrator(t, rate)
    assert h.sigma(0.0, 2.0) < h.sigma(0.0, 20.0)


def test_sigma_is_dominated_by_the_drivers_line_not_the_gyro():
    """Over a few seconds the gyro is excellent; what is uncertain is that the
    driver's path through a junction is not the map's angle between two edges."""
    t, rate = _profile()
    h = HeadingIntegrator(t, rate)
    total = h.sigma(0.0, 4.0, model_sigma_rad=math.radians(9.0))
    without = h.sigma(0.0, 4.0, model_sigma_rad=0.0)
    assert without < 0.3 * total


def test_detect_turns_finds_each_turn_once():
    t, rate = _profile(duration=60.0, turns=[(10.0, 3.0, math.radians(90.0)),
                                             (30.0, 4.0, math.radians(-88.0))])
    events = detect_turns(t, rate)
    assert len(events) == 2
    assert math.degrees(events[0].delta_psi) == pytest.approx(90.0, abs=3.0)
    assert math.degrees(events[1].delta_psi) == pytest.approx(-88.0, abs=3.0)


def test_gentle_drift_is_not_a_turn():
    t = np.arange(0.0, 60.0, 0.1)
    rate = np.full_like(t, math.radians(0.5))       # 0.5 deg/s for a minute
    assert detect_turns(t, rate) == []


def test_the_score_picks_the_branch_that_was_driven():
    """A -84 degree measurement against left / straight / right."""
    options = np.array([math.radians(87.0), math.radians(3.0), math.radians(-91.0)])
    ll = turn_log_likelihood(options, math.radians(-84.0), math.radians(9.0))
    assert int(np.argmax(ll)) == 2
    assert ll[2] - ll[1] > 3.0, "the wrong branch should be clearly penalised"


def test_the_score_is_robust_to_one_bad_crossing():
    """A driver who swings wide costs a hypothesis weight, never all of it."""
    options = np.array([math.radians(90.0)])
    good = turn_log_likelihood(options, math.radians(88.0), math.radians(9.0))[0]
    awful = turn_log_likelihood(options, math.radians(0.0), math.radians(9.0))[0]
    assert good > awful
    assert awful > -30.0, "a Gaussian would have scored this around -50"


def test_going_straight_is_evidence_too():
    """Measuring no turn is not the absence of evidence: it rules out turning."""
    options = np.array([math.radians(90.0), math.radians(0.0), math.radians(-90.0)])
    ll = turn_log_likelihood(options, math.radians(1.0), math.radians(9.0))
    assert int(np.argmax(ll)) == 1
