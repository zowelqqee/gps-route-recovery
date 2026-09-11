"""Regression: a straight road carries no along-road information.

On a road with kappa == 0 every position along it predicts exactly the same yaw
rate, so a gyro reading is evidence about the gyro and about nothing else. A
filter that lets it sharpen position is manufacturing certainty out of a
symmetry.

Under the global-speed architecture the guarantee is structural rather than
delicate: hypotheses have no position state to sharpen. Along-road uncertainty
is one number owned by the speed filter, which never sees the map at all. What
remains to test is that the *scoring* stays neutral - that a straight road does
not silently rank one position ahead of another.
"""

import numpy as np
import pytest

from geotrace.pacman_tracker.config import PacmanConfig
from geotrace.pacman_tracker.curvature import CurvatureMatcher
from geotrace.pacman_tracker.motion import ImuSample
from geotrace.pacman_tracker.roadmap import RoadGeometry
from geotrace.pacman_tracker.speed import GlobalSpeedTracker, SpeedConfig
from geotrace.pacman_tracker.state import RouteNode, make_set
from geotrace.pacman_tracker.synthetic import arc, build_network, straight


def _population(offsets=(-50.0, -150.0, -250.0)):
    routes = [RouteNode.root(0, 0.0, o) for o in offsets]
    return make_set([0] * len(offsets), offsets, [0.0] * len(offsets), routes, 0.0)


@pytest.fixture
def matcher():
    network, _ = build_network([("Straight", straight(600.0), {})])
    cfg = PacmanConfig()
    return CurvatureMatcher(RoadGeometry(network, cfg.geometry), cfg.match)


def test_a_straight_road_does_not_separate_positions(matcher):
    """Three hypotheses 100 m apart on the same straight road are
    indistinguishable. If any gains weight, the ranking is an artefact."""
    hs = _population()
    rng = np.random.default_rng(0)
    for step in range(400):
        sample = ImuSample(t=0.1 * step, dt=0.1, a_long=0.0,
                           yaw_rate=float(rng.normal(0.0, 0.02)))
        matcher.score(hs, sample, distance=0.0, speed=10.0, gyro_bias=0.0, dt=0.1)
    assert float(np.ptp(hs.logw)) < 1e-9


def test_a_straight_road_cannot_shrink_the_along_road_uncertainty(matcher):
    """The structural half: sigma_s belongs to the speed filter, and the map
    has no path to it. Scoring a thousand steps must leave it untouched."""
    speed = GlobalSpeedTracker(SpeedConfig(), v0=10.0)
    for _ in range(50):
        speed.predict(0.0, 0.1)
    before = speed.sigma_distance
    hs = _population()
    for step in range(1000):
        matcher.score(hs, ImuSample(t=0.1 * step, dt=0.1, a_long=0.0, yaw_rate=0.001),
                      distance=speed.distance, speed=speed.speed, gyro_bias=0.0, dt=0.1)
    assert speed.sigma_distance == pytest.approx(before, abs=1e-12)


def test_the_curvature_matcher_has_no_way_to_change_the_speed(matcher):
    """v = omega / kappa is not available to it, by construction.

    On this data usable curvature exists on 1.5 % of steps and is wrong there by
    a median of -5.8 m/s; letting it set the speed closed the loop between which
    road the filter believed and how fast it thought it was going.
    """
    speed = GlobalSpeedTracker(SpeedConfig(), v0=10.0)
    hs = _population()
    matcher.score(hs, ImuSample(t=0.0, dt=0.1, a_long=0.0, yaw_rate=0.5),
                  distance=0.0, speed=speed.speed, gyro_bias=0.0, dt=0.1)
    assert speed.speed == pytest.approx(10.0)
    assert not hasattr(hs, "x"), "hypotheses must not carry a state vector at all"


def test_a_bend_does_separate_positions():
    """The counterpart: where the shape changes, position is observable."""
    network, _ = build_network([("Bend", arc(60.0, np.pi / 2, step_m=2.0), {})])
    cfg = PacmanConfig()
    match = CurvatureMatcher(RoadGeometry(network, cfg.geometry), cfg.match)
    hs = _population(offsets=(-5.0, -45.0))
    for step in range(200):
        match.score(hs, ImuSample(t=0.1 * step, dt=0.1, a_long=0.0, yaw_rate=8.0 / 60.0),
                    distance=0.0, speed=8.0, gyro_bias=0.0, dt=0.1)
    assert float(np.ptp(hs.logw)) > 0.0
