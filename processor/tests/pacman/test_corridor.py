"""Corridors: narrow, on the road, and honest about giving up.

A corridor is ``[s - k*sigma_D, s + k*sigma_D]`` walked along the hypothesis's
own route and buffered a few metres sideways - never a free-space disc, which
would claim the car might be in the courtyard next door while smearing away the
one thing the model does know, which is *which street*.
"""

import numpy as np
import pytest

from geotrace.pacman_tracker.config import PacmanConfig
from geotrace.pacman_tracker.corridor import Confidence, CorridorBuilder
from geotrace.pacman_tracker.roadmap import RoadGeometry
from geotrace.pacman_tracker.state import RouteNode, make_set

DISTANCE = 1000.0
"""Every hypothesis reads its position off the one global distance; what
differs between them is the length of the route that got them there."""


def _set(edges, s_values, logw, routes=None):
    offsets = [DISTANCE - float(s) for s in s_values]
    routes = routes or [RouteNode.root(e, 0.0, o) for e, o in zip(edges, offsets)]
    return make_set(list(edges), offsets, list(logw), routes, 0.0)


@pytest.fixture
def builder(junction_network):
    network, _ = junction_network
    cfg = PacmanConfig()
    geometry = RoadGeometry(network, cfg.geometry)
    names = {e.name: e.index for e in network.edges}
    return CorridorBuilder(geometry, cfg.corridor), network, names


def test_corridor_length_follows_the_global_sigma(builder):
    build, _, names = builder
    for sigma in (5.0, 20.0, 50.0):
        result = build.build(_set([names["A"]], [150.0], [0.0]), 0.0, DISTANCE, sigma)
        assert result.corridors[0].length_m == pytest.approx(
            2 * build.cfg.k_sigma * sigma, rel=0.05)


def test_a_corridor_is_a_ribbon_not_a_disc(builder):
    build, _, names = builder
    result = build.build(_set([names["A"]], [150.0], [0.0]), 0.0, DISTANCE, 100.0)
    disc = np.pi * (build.cfg.k_sigma * 100.0) ** 2
    assert result.area_m2() < disc / 10.0
    assert result.corridors[0].half_width_m == build.cfg.lateral_buffer_m


def test_it_contains_the_truth_when_the_belief_is_right(builder):
    build, network, names = builder
    result = build.build(_set([names["A"]], [150.0], [0.0]), 0.0, DISTANCE, 20.0)
    on_road = network.edges[names["A"]].position(160.0)
    assert result.contains(on_road)
    assert not result.contains((on_road[0], on_road[1] + 40.0))


def test_a_split_belief_is_ambiguous_and_still_narrow(builder):
    build, _, names = builder
    hs = _set([names["B"], names["C"], names["D"]], [50.0] * 3, [0.0, -0.1, -0.2])
    result = build.build(hs, 0.0, DISTANCE, 15.0)
    assert result.confidence is Confidence.AMBIGUOUS
    assert len(result.corridors) == 3
    assert all(c.length_m < 100.0 for c in result.corridors)


def test_a_hopeless_belief_says_so_instead_of_widening(builder):
    build, _, names = builder
    result = build.build(_set([names["A"]], [150.0], [0.0]), 0.0, DISTANCE, 400.0)
    assert result.confidence is Confidence.LOW_CONFIDENCE
    assert "sigma_s" in result.reason


def test_a_tight_single_branch_is_confident(builder):
    build, _, names = builder
    result = build.build(_set([names["A"]], [150.0], [0.0]), 0.0, DISTANCE, 10.0)
    assert result.confidence is Confidence.CONFIDENT


def test_a_belief_over_many_streets_is_never_confident(builder):
    """The top corridor can look excellent while the belief is everywhere. On
    the real recordings that combination reported CONFIDENT on a third of all
    ticks while the truth was a kilometre outside every corridor drawn."""
    build, _, names = builder
    hs = _set([names["A"], names["B"], names["C"], names["D"]], [50.0] * 4,
              [0.0, -0.4, -0.5, -0.6])
    result = build.build(hs, 0.0, DISTANCE, 10.0)
    assert result.confidence is Confidence.AMBIGUOUS
    assert "effective streets" in result.reason
    assert result.diagnostics["effective_streets"] > 3.0


def test_one_street_with_a_tight_belief_is_still_confident(builder):
    build, _, names = builder
    hs = _set([names["A"]] * 3, [50.0, 52.0, 54.0], [0.0, -0.05, -0.1])
    result = build.build(hs, 0.0, DISTANCE, 8.0)
    assert result.confidence is Confidence.CONFIDENT
    assert result.diagnostics["effective_streets"] == pytest.approx(1.0, abs=1e-6)


def test_the_corridor_spills_back_onto_the_edge_it_came_from(builder):
    build, _, names = builder
    root = RouteNode.root(names["A"], 0.0, DISTANCE - 310.0)
    route = root.child(names["C"], 1.0, DISTANCE - 10.0)
    hs = _set([names["C"]], [10.0], [0.0], routes=[route])
    result = build.build(hs, 0.0, DISTANCE, 40.0)
    assert names["A"] in result.corridors[0].edges


def test_the_corridor_stops_at_a_fork_rather_than_guessing(builder):
    build, _, names = builder
    result = build.build(_set([names["A"]], [290.0], [0.0]), 0.0, DISTANCE, 60.0)
    assert result.corridors[0].edges == [names["A"]]


def test_an_empty_population_is_low_confidence(builder):
    build, _, _ = builder
    result = build.build(_set([], [], []), 0.0, DISTANCE, 10.0)
    assert result.confidence is Confidence.LOW_CONFIDENCE
    assert result.corridors == []


def test_every_corridor_shares_the_one_sigma(builder):
    """Hypotheses on different streets are equally uncertain along them,
    because there is one car and one distance travelled."""
    build, _, names = builder
    hs = _set([names["B"], names["C"]], [50.0, 50.0], [0.0, -0.1])
    result = build.build(hs, 0.0, DISTANCE, 25.0)
    lengths = {round(c.length_m, 3) for c in result.corridors}
    assert len(lengths) == 1
