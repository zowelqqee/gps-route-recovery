"""Synthetic junction: branch on every legal successor, let the turn choose.

              B
             /
A ----------+--------- C
             \\
              D

Coming from A, hypotheses for B, C and D must all be created, none of them
pre-selected, and the one matching the *integrated* yaw over the crossing must
end up on top.
"""

import math

import numpy as np
import pytest

from geotrace.pacman_tracker.config import PacmanConfig
from geotrace.pacman_tracker.manager import HypothesisManager
from geotrace.pacman_tracker.roadmap import RoadGeometry
from geotrace.pacman_tracker.state import RouteNode, make_set
from geotrace.pacman_tracker.turns import HeadingIntegrator


def _drive(network, geometry, cfg, yaw_of_time, speed=8.0, seconds=40.0,
           start_s=250.0, dt=0.1):
    by_name = {e.name: e.index for e in network.edges}
    times = np.arange(0.0, seconds, dt)
    rates = np.array([yaw_of_time(t) for t in times])
    manager = HypothesisManager(geometry, cfg.beam, HeadingIntegrator(times, rates))

    offset = -start_s
    hs = make_set([by_name["A"]], [offset], [0.0],
                  [RouteNode.root(by_name["A"], 0.0, offset)], 0.0)
    for t in times:
        distance = speed * float(t)
        hs = manager.advance(hs, distance, float(t), speed)
        if len(hs) == 0:
            break
        manager.resolve_turns(hs, float(t), 0.0, 0.004)
        hs = manager.merge(hs)
        hs = manager.prune(hs, distance, float(t))
    return hs, by_name, manager


def _turn_profile(angle_rad, t_start, duration):
    def yaw(t):
        return angle_rad / duration if t_start <= t < t_start + duration else 0.0
    return yaw


def test_a_junction_spawns_every_legal_successor(junction_network):
    network, _ = junction_network
    cfg = PacmanConfig()
    geometry = RoadGeometry(network, cfg.geometry)
    hs, by_name, manager = _drive(network, geometry, cfg, lambda t: 0.0, seconds=8.0)
    assert {by_name["B"], by_name["C"], by_name["D"]} <= set(int(e) for e in hs.edge)
    assert manager.stats.branches == 1


def test_the_measured_turn_picks_the_branch(junction_network):
    """A left turn of pi/2 taken at 8 m/s, 50 m after the start."""
    network, _ = junction_network
    cfg = PacmanConfig()
    geometry = RoadGeometry(network, cfg.geometry)
    hs, by_name, manager = _drive(
        network, geometry, cfg,
        _turn_profile(math.pi / 2, 50.0 / 8.0, 3.0), seconds=30.0)
    assert manager.stats.turns_scored >= 3
    best = int(hs.edge[hs.order()[0]])
    assert best == by_name["B"], "the left turn in the gyro did not select the left branch"
    assert float(hs.weights()[hs.edge == by_name["B"]].sum()) > 0.8


def test_a_right_turn_picks_the_right_branch(junction_network):
    network, _ = junction_network
    cfg = PacmanConfig()
    geometry = RoadGeometry(network, cfg.geometry)
    hs, by_name, _ = _drive(network, geometry, cfg,
                            _turn_profile(-math.pi / 2, 50.0 / 8.0, 3.0), seconds=30.0)
    assert int(hs.edge[hs.order()[0]]) == by_name["D"]


def test_going_straight_selects_the_straight_branch(junction_network):
    """Measuring no turn is evidence, not the absence of it."""
    network, _ = junction_network
    cfg = PacmanConfig()
    geometry = RoadGeometry(network, cfg.geometry)
    hs, by_name, _ = _drive(network, geometry, cfg, lambda t: 0.0, seconds=30.0)
    assert int(hs.edge[hs.order()[0]]) == by_name["C"]


def test_one_bad_sample_does_not_kill_the_correct_branch(junction_network):
    """A pothole in the middle of a straight run must not be fatal."""
    network, _ = junction_network
    cfg = PacmanConfig()
    geometry = RoadGeometry(network, cfg.geometry)

    def yaw(t):
        return 2.0 if 12.0 <= t < 12.1 else 0.0

    hs, by_name, _ = _drive(network, geometry, cfg, yaw, seconds=30.0)
    assert by_name["C"] in set(int(e) for e in hs.edge)
    assert int(hs.edge[hs.order()[0]]) == by_name["C"]


def test_turn_restrictions_are_honoured(junction_network):
    from geotrace.road_graph import RoadNetwork, TurnRestriction

    network, _ = junction_network
    by_id = {e.name: e.edge_id for e in network.edges}
    restricted = RoadNetwork(
        network.graph, network.frame,
        turn_restrictions=[TurnRestriction(kind="no", from_edge=by_id["A"],
                                           to_edge=by_id["B"])])
    cfg = PacmanConfig()
    geometry = RoadGeometry(restricted, cfg.geometry)
    names = {e.name: e.index for e in restricted.edges}
    hs, _, _ = _drive(restricted, geometry, cfg, lambda t: 0.0, seconds=8.0)
    edges = set(int(e) for e in hs.edge)
    assert names["B"] not in edges
    assert {names["C"], names["D"]} <= edges


def test_the_branch_children_share_one_distance(junction_network):
    """Children differ in route, never in how far the car has travelled."""
    network, _ = junction_network
    cfg = PacmanConfig()
    geometry = RoadGeometry(network, cfg.geometry)
    hs, _, _ = _drive(network, geometry, cfg, lambda t: 0.0, seconds=8.0)
    distance = 8.0 * 7.9
    s = hs.s(distance)
    assert np.all(np.isfinite(s))
    # All three successors start at the same node, so all three offsets match.
    assert float(np.ptp(hs.route_offset)) == pytest.approx(0.0, abs=1e-9)
