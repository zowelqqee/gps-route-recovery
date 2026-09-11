"""Population management: keep the right hypothesis alive, keep the beam small.

The objective is survival of the correct route, not an early tidy top-1, and
the two mechanisms that matter most are merging (which is what stops the
population growing as 3^junctions) and patient pruning.
"""

import math

import numpy as np
import pytest

from geotrace.pacman_tracker.config import BeamConfig, PacmanConfig
from geotrace.pacman_tracker.manager import HypothesisManager
from geotrace.pacman_tracker.roadmap import RoadGeometry
from geotrace.pacman_tracker.state import RouteNode, make_set
from geotrace.pacman_tracker.synthetic import build_network, straight


def _population(edges, offsets, logw, parents=None):
    routes = []
    for i, edge in enumerate(edges):
        if parents is None:
            routes.append(RouteNode.root(edge, 0.0, offsets[i]))
        else:
            root = RouteNode.root(parents[i], 0.0, 0.0)
            routes.append(root.child(edge, 0.0, offsets[i]))
    return make_set(edges, offsets, logw, routes, 0.0)


@pytest.fixture
def manager(junction_network):
    network, _ = junction_network
    cfg = PacmanConfig()
    geometry = RoadGeometry(network, cfg.geometry)
    names = {e.name: e.index for e in network.edges}
    return HypothesisManager(geometry, cfg.beam), geometry, names


def test_merge_combines_mass_and_keeps_the_better_route(manager):
    mgr, _, names = manager
    hs = _population([names["A"]] * 3, [0.0, 1.0, 80.0], [0.0, -0.1, -3.0])
    merged = mgr.merge(hs)
    assert len(merged) == 2
    assert float(np.exp(merged.logw).sum()) == pytest.approx(
        float(np.exp(hs.logw).sum()), rel=1e-9)


def test_merge_does_not_join_different_places(manager):
    mgr, _, names = manager
    hs = _population([names["A"]] * 2, [0.0, 200.0], [0.0, 0.0])
    assert len(mgr.merge(hs)) == 2


def test_merge_uses_corrected_route_offset(manager):
    mgr, _, names = manager
    hs = _population([names["A"]] * 2, [0.0, 0.0], [0.0, -0.1])
    hs.offset_bias[:] = [0.0, 20.0]
    assert len(mgr.merge(hs)) == 2


def test_merge_does_not_discard_a_pending_alignment_anchor(manager):
    mgr, _, names = manager
    hs = _population([names["A"]] * 2, [0.0, 0.0], [0.0, -0.1])
    hs.anchor_turn_t[0] = 10.0
    hs.anchor_turn_map_offset[0] = 100.0
    assert len(mgr.merge(hs)) == 2


def test_merge_collapses_routes_that_have_reconverged(manager):
    """Two histories arriving at the same edge at the same distance have the
    same future. This is what keeps the beam from growing exponentially."""
    mgr, _, names = manager
    hs = _population([names["C"]] * 2, [300.0, 300.0], [0.0, -0.5],
                     parents=[names["A"], names["A"]])
    merged = mgr.merge(hs)
    assert len(merged) == 1
    assert merged.logw[0] == pytest.approx(math.log(1.0 + math.exp(-0.5)), abs=1e-9)


def test_prune_needs_sustained_badness(manager):
    """One bad step must not remove a hypothesis - that is how truth dies."""
    mgr, _, names = manager
    mgr.cfg = BeamConfig(min_hypotheses=1)
    hs = _population([names["A"], names["C"]], [0.0, 0.0],
                     [0.0, -mgr.cfg.prune_log_margin - 5.0])
    for step in range(mgr.cfg.prune_patience - 1):
        hs = mgr.prune(hs, 10.0, float(step))
        assert len(hs) == 2, f"pruned after only {step + 1} bad steps"
    assert len(mgr.prune(hs, 10.0, 99.0)) == 1


def test_a_hypothesis_that_recovers_is_not_pruned(manager):
    mgr, _, names = manager
    mgr.cfg = BeamConfig(min_hypotheses=1)
    hs = _population([names["A"], names["C"]], [0.0, 0.0],
                     [0.0, -mgr.cfg.prune_log_margin - 5.0])
    for step in range(mgr.cfg.prune_patience - 2):
        hs = mgr.prune(hs, 10.0, float(step))
    hs.logw[:] = [0.0, -1.0]
    for step in range(mgr.cfg.prune_patience * 2):
        hs = mgr.prune(hs, 10.0, 100.0 + step)
    assert len(hs) == 2


def test_the_population_floor_is_respected(manager):
    mgr, _, names = manager
    mgr.cfg = BeamConfig(min_hypotheses=3, prune_log_margin=0.1, prune_patience=1)
    n = 10
    hs = _population([names["A"]] * n, list(np.linspace(0, 290, n)),
                     list(np.linspace(-50.0, 0.0, n)))
    for step in range(10):
        hs = mgr.prune(hs, 10.0, float(step))
    assert len(hs) >= 3


def test_the_beam_limit_keeps_the_best(manager):
    mgr, _, names = manager
    mgr.cfg = BeamConfig(max_hypotheses=5, min_hypotheses=1, prune_log_margin=1e9)
    n = 40
    hs = _population([names["A"]] * n, list(np.linspace(0, 290, n)),
                     list(np.linspace(-20.0, 0.0, n)))
    kept = mgr.prune(hs, 10.0, 0.0)
    assert len(kept) == 5
    assert float(kept.logw.max()) == pytest.approx(0.0)


def test_dead_end_hypotheses_are_dropped():
    network, _ = build_network([("Cul de sac", straight(100.0), {})])
    cfg = PacmanConfig()
    mgr = HypothesisManager(RoadGeometry(network, cfg.geometry), cfg.beam)
    hs = _population([0], [0.0], [0.0])
    assert len(mgr.advance(hs, 120.0, 1.0, 10.0)) == 0
    assert mgr.stats.dead_ends == 1


def test_reversing_returns_to_the_previous_edge(junction_network):
    network, _ = junction_network
    cfg = PacmanConfig()
    geometry = RoadGeometry(network, cfg.geometry)
    names = {e.name: e.index for e in network.edges}
    mgr = HypothesisManager(geometry, cfg.beam)
    hs = _population([names["C"]], [300.0], [0.0], parents=[names["A"]])
    out = mgr.advance(hs, 270.0, 1.0, -3.0)      # D fell well back behind the node
    assert int(out.edge[0]) == names["A"]
    assert mgr.stats.reversals == 1


def test_prune_events_record_why(manager):
    mgr, _, names = manager
    mgr.cfg = BeamConfig(min_hypotheses=1)
    hs = _population([names["A"], names["C"]], [0.0, 0.0],
                     [0.0, -mgr.cfg.prune_log_margin - 5.0])
    for step in range(mgr.cfg.prune_patience + 1):
        hs = mgr.prune(hs, 10.0, float(step))
    event = mgr.stats.events[-1].to_json()
    assert "log_weight_below_margin" in event["reason"]
    assert event["pruning_threshold"] < event["best_log_weight"]
    assert event["rank"] == 1


def test_hypotheses_carry_no_speed_of_their_own(manager):
    """The architectural invariant. If this ever fails, the map is deciding how
    fast the car went again."""
    mgr, _, names = manager
    hs = _population([names["A"]], [0.0], [0.0])
    for attribute in ("x", "P", "sigma_v", "velocity"):
        assert not hasattr(hs, attribute), attribute


def test_a_measured_turn_re_times_the_crossing(junction_network):
    """Odometer distance and map route length are different quantities.

    A hypothesis on exactly the right road still drifts along it - the car does
    not drive down the polyline centreline - and with one global D there is
    nothing to absorb that. A junction turn is an observation of *when* the
    crossing happened, so it is what puts the route offset back.
    """
    import numpy as np

    from geotrace.pacman_tracker.roadmap import RoadGeometry
    from geotrace.pacman_tracker.turns import HeadingIntegrator

    network, _ = junction_network
    cfg = PacmanConfig()
    geometry = RoadGeometry(network, cfg.geometry)
    names = {e.name: e.index for e in network.edges}

    # The gyro says the left turn happened two seconds *after* the crossing the
    # hypothesis assumed. At 8 m/s the physical crossing distance is therefore
    # about 16 m longer than its mapped route coordinate.
    dt, speed = 0.1, 8.0
    times = np.arange(0.0, 40.0, dt)
    rate = np.where((times >= 12.0) & (times < 15.0), math.pi / 2 / 3.0, 0.0)
    mgr = HypothesisManager(geometry, cfg.beam, HeadingIntegrator(times, rate))

    hs = _population([names["B"]], [92.0], [0.0], parents=[names["A"]])
    hs.turn_angle[:] = math.pi / 2
    hs.turn_t[:] = 11.5
    hs.turn_window[:] = 4.0
    hs.anchor_turn_angle[:] = math.pi / 2
    hs.anchor_turn_t[:] = 11.5
    hs.anchor_turn_window[:] = 4.0
    hs.anchor_turn_map_offset[:] = 92.0
    for t in times[times <= 20.0]:
        mgr._remember_distance(float(t), speed * float(t))
    before = float(hs.offset_bias[0])
    mgr.resolve_turns(hs, 20.0, 0.0, 0.004, speed, 160.0)

    assert mgr.stats.offset_corrections == 1
    assert float(hs.offset_bias[0]) > before, "a later physical turn needs a positive offset"
    assert abs(float(hs.offset_bias[0])) <= cfg.beam.offset_max_correction_m


def test_a_gentle_bend_does_not_re_time_anything(junction_network):
    """Only a real turn localises a crossing."""
    import numpy as np

    from geotrace.pacman_tracker.roadmap import RoadGeometry
    from geotrace.pacman_tracker.turns import HeadingIntegrator

    network, _ = junction_network
    cfg = PacmanConfig()
    geometry = RoadGeometry(network, cfg.geometry)
    names = {e.name: e.index for e in network.edges}
    times = np.arange(0.0, 40.0, 0.1)
    rate = np.full_like(times, math.radians(0.5))
    mgr = HypothesisManager(geometry, cfg.beam, HeadingIntegrator(times, rate))

    hs = _population([names["C"]], [300.0], [0.0], parents=[names["A"]])
    hs.turn_angle[:] = 0.0
    hs.turn_t[:] = 11.5
    hs.turn_window[:] = 4.0
    mgr.resolve_turns(hs, 20.0, 0.0, 0.004, 8.0)
    assert mgr.stats.offset_corrections == 0
    assert float(hs.offset_bias[0]) == 0.0
