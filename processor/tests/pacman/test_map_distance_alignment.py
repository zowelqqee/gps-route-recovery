"""Route distance is a noisy alignment between physical D and map length."""

import math

import numpy as np
import pytest

from geotrace.pacman_tracker.config import PacmanConfig
from geotrace.pacman_tracker.manager import HypothesisManager
from geotrace.pacman_tracker.roadmap import RoadGeometry
from geotrace.pacman_tracker.state import RouteNode, make_set
from geotrace.pacman_tracker.synthetic import build_network, straight
from geotrace.pacman_tracker.turns import HeadingIntegrator


def _chain(count=21, length=100.0):
    ways = [
        (f"E{i}", straight(length, start=(i * length, 0.0)), {})
        for i in range(count)
    ]
    network, _ = build_network(ways)
    cfg = PacmanConfig()
    geometry = RoadGeometry(network, cfg.geometry)
    names = {edge.name: edge.index for edge in network.edges}
    hs = make_set([names["E0"]], [0.0], [0.0],
                  [RouteNode.root(names["E0"], 0.0, 0.0)], 0.0)
    return HypothesisManager(geometry, cfg.beam), hs, names


def _advance_to(manager, hs, distance):
    for d in np.arange(0.0, distance, 25.0):
        hs = manager.advance(hs, float(d), float(d / 10.0), 10.0, 5.0)
        hs = manager.merge(hs)
    return manager.advance(hs, float(distance), float(distance / 10.0), 10.0, 5.0)


def test_physical_distance_longer_than_map_keeps_the_current_edge():
    manager, hs, names = _chain()
    hs = _advance_to(manager, hs, 2019.0)  # 0.95% longer over two kilometres
    assert names["E19"] in set(int(edge) for edge in hs.edge)


def test_physical_distance_shorter_than_map_spawns_the_next_edge():
    manager, hs, names = _chain()
    hs = _advance_to(manager, hs, 1981.0)  # true crossing may be 0.95% early
    assert names["E20"] in set(int(edge) for edge in hs.edge)


def test_map_distance_uncertainty_accumulates_on_a_long_straight():
    manager, hs, _ = _chain()
    near = float(manager.map_distance_sigma(hs, 100.0)[0])
    far = float(manager.map_distance_sigma(hs, 2000.0)[0])
    assert far > near
    assert far == pytest.approx(math.hypot(4.0, 20.0))


def test_reliable_turn_reanchors_distance_alignment():
    manager, hs, _ = _chain(count=2)
    times = np.arange(0.0, 30.0, 0.1)
    rate = np.where((times >= 12.0) & (times < 15.0), math.pi / 2 / 3.0, 0.0)
    manager.integrator = HeadingIntegrator(times, rate)
    for t in times[times <= 20.0]:
        manager._remember_distance(float(t), 8.0 * float(t))
    hs.anchor_turn_angle[:] = math.pi / 2
    hs.anchor_turn_t[:] = 11.5
    hs.anchor_turn_window[:] = 4.0
    hs.anchor_turn_map_offset[:] = 92.0
    before_sigma = float(manager.map_distance_sigma(hs, 160.0)[0])
    manager.resolve_turns(hs, 20.0, 0.0, 0.004, 8.0, 160.0)
    after_sigma = float(manager.map_distance_sigma(hs, 160.0)[0])
    assert hs.offset_bias[0] > 0.0
    assert hs.map_anchor_distance[0] > 100.0
    assert after_sigma < before_sigma


def test_crossing_zone_keeps_both_adjacent_edges_plausible(junction_network):
    network, _ = junction_network
    cfg = PacmanConfig()
    geometry = RoadGeometry(network, cfg.geometry)
    names = {edge.name: edge.index for edge in network.edges}
    manager = HypothesisManager(geometry, cfg.beam)
    hs = make_set([names["A"]], [0.0], [0.0],
                  [RouteNode.root(names["A"], 0.0, 0.0)], 0.0)
    hs = manager.advance(hs, 290.0, 29.0, 10.0, 5.0)
    edges = set(int(edge) for edge in hs.edge)
    assert names["A"] in edges
    assert {names["B"], names["C"], names["D"]} <= edges


def test_strong_turn_anchor_survives_a_short_following_edge():
    ways = [
        ("A", straight(100.0), {}),
        ("B", straight(5.0, start=(100.0, 0.0), heading_rad=math.pi / 2), {}),
        ("C", straight(100.0, start=(100.0, 5.0), heading_rad=math.pi / 2), {}),
    ]
    network, _ = build_network(ways)
    cfg = PacmanConfig()
    geometry = RoadGeometry(network, cfg.geometry)
    names = {edge.name: edge.index for edge in network.edges}
    manager = HypothesisManager(geometry, cfg.beam)
    hs = make_set([names["A"]], [0.0], [0.0],
                  [RouteNode.root(names["A"], 0.0, 0.0)], 0.0)
    hs = manager.advance(hs, 94.0, 9.4, 10.0, 5.0)
    child = np.nonzero(hs.edge == names["C"])[0]
    assert child.size == 1, "the short edge should be crossed in the same advance"
    assert np.isfinite(hs.anchor_turn_t[child[0]])
    assert hs.anchor_turn_angle[child[0]] == pytest.approx(math.pi / 2)


def test_map_distance_tolerance_is_bounded_after_repeated_distance():
    manager, hs, _ = _chain()
    sigma = manager.map_distance_sigma(hs, 1_000_000.0)
    tolerance = manager.crossing_tolerance(hs, 1_000_000.0, 500.0)
    assert float(sigma[0]) == manager.cfg.map_distance_sigma_max_m
    assert float(tolerance[0]) == manager.cfg.crossing_tolerance_max_m
