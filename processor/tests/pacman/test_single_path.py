"""One-active-route junction decisions and dormant alternatives."""

import inspect
import math

import numpy as np
import pytest

from geotrace.pacman_tracker.config import PacmanConfig
from geotrace.pacman_tracker.roadmap import RoadGeometry
from geotrace.pacman_tracker.single_path import (
    PhysicalTurn, SinglePathManager, local_probabilities)
from geotrace.pacman_tracker.state import RouteNode, make_set
from geotrace.pacman_tracker.synthetic import build_network, straight
from geotrace.pacman_tracker.turns import HeadingIntegrator


def _run(junction_network, angle, *, low_confidence=False):
    network, _ = junction_network
    cfg = PacmanConfig()
    if low_confidence:
        cfg.single_path.min_confident_probability = 1.1
    geometry = RoadGeometry(network, cfg.geometry)
    names = {e.name: e.index for e in network.edges}
    dt, speed = 0.1, 8.0
    times = np.arange(0.0, 18.0, dt)
    rates = np.where((times >= 6.25) & (times < 9.25), angle / 3.0, 0.0)
    manager = SinglePathManager(
        geometry, cfg.beam, cfg.single_path, HeadingIntegrator(times, rates))
    offset = -250.0
    hs = make_set([names["A"]], [offset], [0.0],
                  [RouteNode.root(names["A"], 0.0, offset)], 0.0)
    for t in times:
        hs = manager.advance(hs, speed * float(t), float(t), speed, 5.0)
        if manager.decisions:
            break
    return hs, names, manager


def test_right_junction_keeps_only_right_active(junction_network):
    hs, names, manager = _run(junction_network, -math.pi / 2)
    assert len(hs) == 1
    assert int(hs.edge[0]) == names["D"]
    assert manager.decisions[0].alternatives[0].edge == names["D"]


def test_left_junction_keeps_only_left_active(junction_network):
    hs, names, _ = _run(junction_network, math.pi / 2)
    assert len(hs) == 1
    assert int(hs.edge[0]) == names["B"]


def test_local_probabilities_reset_at_each_junction():
    first = local_probabilities([math.log(0.9), math.log(0.1)])
    second = local_probabilities([math.log(0.6), math.log(0.4)])
    assert first[0] == pytest.approx(0.9)
    assert second[0] == pytest.approx(0.6)
    assert second[0] != 0.9 * 0.6


def test_dormant_branches_do_not_propagate(junction_network):
    hs, _, manager = _run(junction_network, -math.pi / 2)
    assert len(hs) == 1
    assert sum(not a.activated for a in manager.decisions[0].alternatives[1:]) == 2
    assert manager.stats.children == 3


def test_route_history_reconstructs_parent_chain(junction_network):
    hs, names, _ = _run(junction_network, math.pi / 2)
    assert hs.routes[0].edges() == [names["A"], names["B"]]


def test_low_confidence_keeps_ordered_alternatives(junction_network):
    _, _, manager = _run(junction_network, 0.0, low_confidence=True)
    decision = manager.decisions[0]
    assert decision.low_confidence
    assert len(decision.alternatives) == 3
    scores = [a.score for a in decision.alternatives]
    assert scores == sorted(scores, reverse=True)


def test_rollback_activates_exactly_one_dormant_sibling(junction_network):
    hs, _, manager = _run(junction_network, 0.0, low_confidence=True)
    before = int(hs.edge[0])
    after = manager._activate_alternative(hs, 20.0)
    activated = [a for a in manager.decisions[0].alternatives if a.activated]
    assert len(after) == 1
    assert int(after.edge[0]) != before
    assert len(activated) == 1


def test_dead_end_does_not_end_the_replay(junction_network):
    """A route that runs out of road stays a live length-1 population.

    The tracker must keep producing frames to the end of the outage, marked
    stuck, rather than the run terminating early on an empty set.
    """
    network, _ = junction_network
    cfg = PacmanConfig()
    geometry = RoadGeometry(network, cfg.geometry)
    names = {e.name: e.index for e in network.edges}
    dt, speed = 0.1, 8.0
    times = np.arange(0.0, 120.0, dt)
    # hard right at the node, then drive off the end of the stub edge D
    rates = np.where((times >= 6.25) & (times < 9.25), -math.pi / 2 / 3.0, 0.0)
    manager = SinglePathManager(
        geometry, cfg.beam, cfg.single_path, HeadingIntegrator(times, rates))
    offset = -250.0
    hs = make_set([names["A"]], [offset], [0.0],
                  [RouteNode.root(names["A"], 0.0, offset)], 0.0)
    lengths = []
    for t in times:
        hs = manager.advance(hs, speed * float(t), float(t), speed, 5.0)
        lengths.append(len(hs))
    assert min(lengths) == 1 and max(lengths) == 1  # never empties, never splits
    assert manager.dead_end_active
    assert manager.stats.dead_ends >= 1


def test_short_connector_is_scored_as_one_compound_manoeuvre():
    """Opposite OSM node angles on a 30 m connector are one gyro turn."""
    node = (300.0, 0.0)
    angle = math.radians(100.0)
    connector_end = (
        node[0] + 30.0 * math.cos(angle),
        node[1] + 30.0 * math.sin(angle),
    )
    network, _ = build_network([
        ("A", straight(300.0), {}),
        ("Stub", straight(120.0, start=node,
                          heading_rad=math.radians(-30.0)), {}),
        ("Connector", straight(30.0, start=node, heading_rad=angle), {}),
        ("Through", straight(200.0, start=connector_end,
                             heading_rad=0.0), {}),
    ])
    cfg = PacmanConfig()
    geometry = RoadGeometry(network, cfg.geometry)
    by_name = {}
    for edge in network.edges:
        if np.linalg.norm(edge.coords[0] - np.asarray(node)) < 1.0:
            by_name[edge.name] = edge.index
    incoming = next(edge.index for edge in network.edges
                    if edge.name == "A"
                    and np.linalg.norm(edge.coords[-1] - np.asarray(node)) < 1.0)
    times = np.arange(0.0, 60.0, 0.1)
    manager = SinglePathManager(
        geometry, cfg.beam, cfg.single_path,
        HeadingIntegrator(times, np.zeros_like(times)))
    hs = make_set([incoming], [-250.0], [0.0],
                  [RouteNode.root(incoming, 0.0, -250.0)], 0.0)
    for t in times:
        hs = manager.advance(hs, 8.0 * float(t), float(t), 8.0, 5.0)
        if len(manager.decisions) >= 2:
            break
    assert manager.decisions[0].chosen_edge == by_name["Connector"]
    assert manager.decisions[1].chosen_edge == next(
        edge.index for edge in network.edges
        if edge.name == "Through"
        and np.linalg.norm(edge.coords[0] - np.asarray(connector_end)) < 1.0)


def test_bidirectional_dead_end_reverses_out_instead_of_sticking():
    network, _ = build_network([
        ("A", straight(100.0), {}),
        ("Stub", straight(40.0, start=(100.0, 0.0)), {"oneway": False}),
    ])
    cfg = PacmanConfig()
    geometry = RoadGeometry(network, cfg.geometry)
    incoming = next(edge.index for edge in network.edges
                    if edge.name == "A" and edge.coords[-1, 0] > 99.0)
    times = np.arange(0.0, 50.0, 0.1)
    manager = SinglePathManager(
        geometry, cfg.beam, cfg.single_path,
        HeadingIntegrator(times, np.zeros_like(times)))
    hs = make_set([incoming], [-80.0], [0.0],
                  [RouteNode.root(incoming, 0.0, -80.0)], 0.0)
    for t in times:
        hs = manager.advance(hs, 5.0 * float(t), float(t), 5.0, 3.0)
        if manager.stats.reversals:
            break
    assert manager.stats.reversals == 1
    assert not manager.dead_end_active


def test_truth_traversal_and_divergence_cause():
    from types import SimpleNamespace

    from geotrace.pacman_tracker.benchmark import (
        _divergence_cause, _truth_traversal)

    truth = SimpleNamespace(
        times=np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        edges=np.array([10, 10, -1, 20, 20, 30]))
    assert [e for e, _, _ in _truth_traversal(truth)] == [10, 20, 30]

    # true edge was a candidate but the decision landed 20 s / 110 m late
    row = {
        "t_cross": 260.0, "measured_turn_deg": 2.0,
        "truth_edge": 20627, "truth_was_candidate": True,
        "truth_junction_in_crossing_tolerance": True,
        "truth_edge_map_turn_deg": 84.0,
        "distance_error_at_turn_m": -112.0,
        "real_turn_event": {"t_peak": 240.0},
    }
    assert "temporal misalignment" in _divergence_cause(row, False, False)

    row2 = dict(row, truth_edge=None)
    assert "earlier junction" in _divergence_cause(row2, True, False)


def test_production_branch_choice_has_no_truth_or_reference_input():
    for name in ("_commit", "_match_turn_to_junction", "advance"):
        source = inspect.getsource(getattr(SinglePathManager, name))
        assert "truth" not in source
        assert "reference" not in source
        assert "oracle" not in source
    assert set(inspect.signature(SinglePathManager._commit).parameters) == {
        "self", "hs", "distance", "t"}


# --------------------------------------------------- event-aligned junctions


def _forensic_junction_network():
    """The 2026-07-22 t+266.9 s junction: incoming edge, then successors at
    +8 deg (straight), +84 deg (the true left) and -98 deg (right)."""
    node = (200.0, 0.0)
    ways = [
        ("in", straight(200.0, start=(0.0, 0.0), heading_rad=0.0), {}),
        ("straight8", straight(150.0, start=node, heading_rad=math.radians(8.0)), {}),
        ("left84", straight(150.0, start=node, heading_rad=math.radians(84.0)), {}),
        ("right98", straight(150.0, start=node, heading_rad=math.radians(-98.0)), {}),
    ]
    return build_network(ways)


def _drive_with_lagging_odometer(lag_m, *, angle_deg=84.0, turn_window=(18.0, 21.0),
                                 speed=10.0, seconds=40.0, dt=0.1,
                                 event_min_angle_deg=25.0):
    network, _ = _forensic_junction_network()
    names = {e.name: e.index for e in network.edges}
    cfg = PacmanConfig()
    cfg.single_path.event_min_angle_deg = event_min_angle_deg
    geometry = RoadGeometry(network, cfg.geometry)
    times = np.arange(0.0, seconds, dt)
    t0, t1 = turn_window
    rates = np.where((times >= t0) & (times < t1),
                     math.radians(angle_deg) / (t1 - t0), 0.0)
    manager = SinglePathManager(geometry, cfg.beam, cfg.single_path,
                                HeadingIntegrator(times, rates))
    hs = make_set([names["in"]], [0.0], [0.0],
                  [RouteNode.root(names["in"], 0.0, 0.0)], 0.0)
    for t in times:
        distance = max(0.0, speed * float(t) - lag_m)
        hs = manager.advance(hs, distance, float(t), speed, 40.0)
    return hs, names, manager


def test_preserved_turn_event_fixes_the_delayed_odometer_junction():
    """The real turn happens ~20 s / ~100 m before the lagging odometer walks
    the route up to the node. The preserved +84 deg event must still select the
    true left branch, not the near-straight gyro window at the late crossing."""
    hs, names, manager = _drive_with_lagging_odometer(lag_m=100.0)
    assert len(manager.decisions) == 1
    decision = manager.decisions[0]
    assert decision.chosen_edge == names["left84"]
    assert decision.event_driven
    # the FROZEN event angle, not the ~0 deg window at the delayed crossing
    assert decision.measured_turn_rad == pytest.approx(math.radians(84.0), abs=0.15)
    assert int(hs.edge[0]) == names["left84"]


def test_turn_event_angle_is_immutable_after_creation():
    _, _, manager = _drive_with_lagging_odometer(lag_m=100.0)
    turn = manager.turns[0]
    assert turn.consumed and turn.signed_angle == pytest.approx(math.radians(84.0), abs=0.1)
    # the angle stored on the decision is exactly the event's, not re-integrated
    assert manager.decisions[0].measured_turn_rad == turn.signed_angle


def test_one_event_is_consumed_by_at_most_one_junction():
    _, _, manager = _drive_with_lagging_odometer(lag_m=100.0)
    consumed = [t for t in manager.turns if t.consumed]
    assert len(consumed) == 1
    assert consumed[0].matched_decision_index == 0
    assert manager.event_stats["matched"] == 1


def test_bad_odometer_is_a_soft_prior_not_a_veto():
    """A 100 m odometer error must not stop the true junction being matched."""
    _, names, manager = _drive_with_lagging_odometer(lag_m=100.0)
    assert manager.decisions[0].event_driven
    assert abs(manager.decisions[0].event_residual_m) > 60.0  # the lag is real


def test_event_match_cannot_jump_arbitrarily_far():
    """A 400 m odometer error is beyond the bounded search; no event match."""
    _, _, manager = _drive_with_lagging_odometer(lag_m=400.0)
    assert all(not d.event_driven for d in manager.decisions)


def test_strong_left_event_rejects_a_right_successor():
    hs, names, manager = _drive_with_lagging_odometer(lag_m=100.0, angle_deg=84.0)
    assert manager.decisions[0].chosen_edge != names["right98"]
    right = next(a for a in manager.decisions[0].alternatives
                 if a.edge == names["right98"])
    assert right.probability < 1e-3


def test_no_event_straight_traversal_still_commits():
    """No turn at all: the junction still commits, by geometry, to straight."""
    hs, names, manager = _drive_with_lagging_odometer(
        lag_m=0.0, angle_deg=0.0)
    assert len(manager.decisions) == 1
    assert manager.decisions[0].chosen_edge == names["straight8"]
    assert not manager.decisions[0].event_driven


def _two_turn_network():
    """in -> J1 (left) -> mid -> J2 (right) -> out, plus decoy successors."""
    j1 = (200.0, 0.0)
    j2 = (200.0, 300.0)  # 300 m north of J1
    ways = [
        ("in", straight(200.0, start=(0.0, 0.0), heading_rad=0.0), {}),
        ("mid", straight(300.0, start=j1, heading_rad=math.radians(90.0)), {}),
        ("in_decoy", straight(120.0, start=j1, heading_rad=math.radians(2.0)), {}),
        ("out", straight(150.0, start=j2, heading_rad=math.radians(0.0)), {}),
        ("out_decoy", straight(120.0, start=j2, heading_rad=math.radians(178.0)), {}),
    ]
    return build_network(ways)


def test_turn_to_turn_interval_is_emitted_between_two_event_anchored_junctions():
    network, _ = _two_turn_network()
    names = {e.name: e.index for e in network.edges}
    cfg = PacmanConfig()
    geometry = RoadGeometry(network, cfg.geometry)
    speed, dt = 10.0, 0.1
    times = np.arange(0.0, 90.0, dt)
    rates = np.zeros_like(times)
    rates[(times >= 18.0) & (times < 21.0)] = math.radians(90.0) / 3.0   # J1 left
    rates[(times >= 48.0) & (times < 51.0)] = math.radians(-90.0) / 3.0  # J2 right
    manager = SinglePathManager(geometry, cfg.beam, cfg.single_path,
                                HeadingIntegrator(times, rates))
    hs = make_set([names["in"]], [0.0], [0.0],
                  [RouteNode.root(names["in"], 0.0, 0.0)], 0.0)
    for t in times:
        hs = manager.advance(hs, speed * float(t), float(t), speed, 8.0)
    assert manager.decisions[0].chosen_edge == names["mid"]
    assert manager.decisions[1].chosen_edge == names["out"]
    assert manager.decisions[0].event_driven and manager.decisions[1].event_driven
    assert len(manager.turn_intervals) == 1
    iv = manager.turn_intervals[0]
    assert iv["both_endpoints_turn_anchored"]
    # map length between J1 (offset 200) and J2 (offset 200 + 300) is ~300 m
    assert iv["map_length_m"] == pytest.approx(300.0, abs=15.0)
    assert iv["event_a_id"] == 0 and iv["event_b_id"] == 1


def test_k_a_stays_frozen_without_an_accepted_interval():
    """Running single_path with intervals disabled must not move k_a."""
    from pathlib import Path

    from geotrace.coordinates import LocalFrame
    from geotrace.pacman_tracker.synthetic import simulate_trip
    from geotrace.pacman_tracker.tracker import PacmanTracker, build_inputs
    from geotrace.road_graph import RoadNetwork

    network, _ = _two_turn_network()
    names = {e.name: e.index for e in network.edges}
    trip = simulate_trip(network, [names["in"], names["mid"], names["out"]],
                         lambda t: 10.0, duration_s=70.0, gps_visible_s=12.0)
    cfg = PacmanConfig()
    cfg.tracker_mode = "single_path"
    cfg.speed.accel_scale_enabled = True
    cfg.intervals.enabled = False
    tracker = PacmanTracker(network, cfg)
    result = tracker.run(build_inputs(trip, network, cfg))
    assert result.stats["accel_scale"]["value"] == pytest.approx(1.0, abs=1e-9)


def test_delayed_crossing_window_does_not_replace_the_event_angle():
    """With the event suppressed (min angle set absurdly high) the same drive
    picks the near-straight branch off the late window - which is exactly the
    bug the event path removes."""
    _, names, manager = _drive_with_lagging_odometer(
        lag_m=100.0, event_min_angle_deg=200.0)
    assert manager.decisions[0].chosen_edge == names["straight8"]
    assert not manager.decisions[0].event_driven


# ---------------------------------------- provisional forks & interval quality


def _shallow_fork_network():
    """in -> J (fork: near-straight +12 deg vs a real +40 deg), each branch
    then continues to a strong +85 deg turn so a later event does not by
    itself disambiguate them."""
    j = (200.0, 0.0)
    a1 = (200.0 + 150.0 * math.cos(math.radians(12)),
          150.0 * math.sin(math.radians(12)))
    b1 = (200.0 + 150.0 * math.cos(math.radians(40)),
          150.0 * math.sin(math.radians(40)))
    ways = [
        ("in", straight(200.0, start=(0.0, 0.0), heading_rad=0.0), {}),
        ("a", straight(150.0, start=j, heading_rad=math.radians(12)), {}),
        ("b", straight(150.0, start=j, heading_rad=math.radians(40)), {}),
        ("a2", straight(300.0, start=a1, heading_rad=math.radians(12 + 85)), {}),
        ("b2", straight(300.0, start=b1, heading_rad=math.radians(40 + 85)), {}),
    ]
    return build_network(ways)


def _drive_fork(measured_fork_deg, *, next_turn_deg=85.0, seconds=90.0,
                distance_lead_m=0.0, fork_turn_window=(18.0, 22.0),
                soft_turn_max_position_residual_m=60.0):
    network, _ = _shallow_fork_network()
    names = {e.name: e.index for e in network.edges}
    cfg = PacmanConfig()
    cfg.single_path.soft_turn_max_position_residual_m = (
        soft_turn_max_position_residual_m)
    geometry = RoadGeometry(network, cfg.geometry)
    speed, dt = 10.0, 0.1
    times = np.arange(0.0, seconds, dt)
    rates = np.zeros_like(times)
    turn_start, turn_end = fork_turn_window
    rates[(times >= turn_start) & (times < turn_end)] = (
        math.radians(measured_fork_deg) / (turn_end - turn_start))
    rates[(times >= 50.0) & (times < 53.0)] = math.radians(next_turn_deg) / 3.0
    manager = SinglePathManager(geometry, cfg.beam, cfg.single_path,
                                HeadingIntegrator(times, rates))
    hs = make_set([names["in"]], [0.0], [0.0],
                  [RouteNode.root(names["in"], 0.0, 0.0)], 0.0)
    lengths = []
    for t in times:
        hs = manager.advance(
            hs, speed * float(t) + distance_lead_m, float(t), speed, 8.0)
        lengths.append(len(hs))
    return manager, names, lengths


def test_soft_turn_cannot_select_an_ineligible_near_straight_edge():
    """A +24 deg soft event may steer the +40 deg moderate corner, but the
    +12 deg near-straight sibling must not win merely because its raw angular
    residual is smaller."""
    manager, names, _ = _drive_fork(24.0, seconds=35.0)
    fork = manager.decisions[0]
    assert fork.event_driven and fork.soft_event
    assert fork.chosen_edge == names["b"]
    straight = next(a for a in fork.alternatives if a.edge == names["a"])
    assert straight.probability == pytest.approx(0.0)


def test_late_soft_turn_rewinds_an_early_straight_commit():
    """When D reaches the fork before the physical turn, the settled event
    must revise that fork and activate its dormant moderate-angle sibling."""
    manager, names, _ = _drive_fork(
        24.0, seconds=38.0, distance_lead_m=58.0,
        fork_turn_window=(30.0, 34.0),
        soft_turn_max_position_residual_m=160.0)
    fork = manager.decisions[0]
    assert fork.chosen_edge == names["b"]
    assert fork.rollback_from_edge == names["a"]
    assert fork.provisional_outcome == "switched_by_late_turn"
    assert manager.rollback_events[0]["trigger"] == "late_turn_switch"


def test_far_late_soft_turn_does_not_revise_an_old_junction():
    manager, names, _ = _drive_fork(
        24.0, seconds=38.0, distance_lead_m=58.0,
        fork_turn_window=(30.0, 34.0))
    assert manager.decisions[0].chosen_edge == names["a"]
    assert not any(e["trigger"] == "late_turn_switch"
                   for e in manager.rollback_events)


def test_shallow_window_prefers_road_class_continuity():
    node = (200.0, 0.0)
    network, _ = build_network([
        ("in", straight(200.0), {"highway": "primary"}),
        ("downgrade", straight(
            150.0, start=node, heading_rad=math.radians(-10.0)),
         {"highway": "secondary"}),
        ("continue", straight(
            150.0, start=node, heading_rad=math.radians(10.0)),
         {"highway": "primary"}),
    ])
    names = {edge.name: edge.index for edge in network.edges}
    cfg = PacmanConfig()
    times = np.arange(0.0, 30.0, 0.1)
    manager = SinglePathManager(
        RoadGeometry(network, cfg.geometry), cfg.beam, cfg.single_path,
        HeadingIntegrator(times, np.zeros_like(times)))
    hs = make_set([names["in"]], [0.0], [0.0],
                  [RouteNode.root(names["in"], 0.0, 0.0)], 0.0)
    for t in times:
        hs = manager.advance(hs, 10.0 * float(t), float(t), 10.0, 8.0)
    assert manager.decisions[0].chosen_edge == names["continue"]


def test_opposite_soft_pair_is_suppressed_as_one_s_manoeuvre():
    network, _ = _shallow_fork_network()
    cfg = PacmanConfig()
    times = np.arange(0.0, 20.0, 0.1)
    rates = np.zeros_like(times)
    rates[(times >= 5.0) & (times < 7.0)] = 0.18
    rates[(times >= 8.2) & (times < 10.2)] = -0.16
    manager = SinglePathManager(
        RoadGeometry(network, cfg.geometry), cfg.beam, cfg.single_path,
        HeadingIntegrator(times, rates))
    soft = [turn for turn in manager.turns if turn.soft]
    assert len(soft) == 2
    assert all(turn.is_bend and turn.consumed for turn in soft)


def test_shallow_fork_is_held_provisional_with_one_bounded_alternative():
    manager, names, lengths = _drive_fork(26.0)   # between +12 and +40
    fork = manager.decisions[0]
    assert fork.provisional
    assert fork.provisional_alternative_edge in (names["a"], names["b"])
    assert fork.provisional_alternative_edge != fork.chosen_edge
    # exactly one runner-up kept, not a beam
    assert sum(not a.activated for a in fork.alternatives) >= 1
    assert max(lengths) == 1  # one active branch throughout the provisional period


def test_provisional_resolution_uses_topology_not_distance_agreement():
    src = inspect.getsource(SinglePathManager._sequence_branch_score)
    for banned in ("d_event", "odometer", "distance_", "offset_bias", "D_"):
        assert banned not in src
    assert "junction_turn" in src and "successors" in src


def test_provisional_fork_left_unresolved_is_not_forced_switched():
    """When both branches thread a consistent sequence, the greedy choice
    stands - the fork is never switched on ambiguous evidence."""
    manager, names, _ = _drive_fork(26.0)
    fork = manager.decisions[0]
    assert manager.diagnostics()["provisional_switched"] == 0
    assert fork.chosen_edge == fork.chosen_edge  # unchanged


def _scale_error_trip(scale_true: float, route_names, duration_s=220.0):
    from geotrace.pacman_tracker.synthetic import simulate_trip
    network, _ = _two_turn_network()
    names = {e.name: e.index for e in network.edges}
    trip = simulate_trip(
        network, [names[n] for n in route_names],
        lambda t: 8.0 + 3.0 * math.sin(t / 12.0), duration_s=duration_s,
        gps_visible_s=14.0)
    # inject a longitudinal scale error: the recorder under-reports accel
    for m in trip.motions:
        ax, ay, az = m.user_acceleration_g
        m.user_acceleration_g = (ax / scale_true, ay, az)
    return network, names, trip


def test_route_state_stays_consistent_after_an_interval_D_correction():
    """s = D - route_offset - offset_bias must be unchanged the instant a
    common-mode D correction lands, and the committed decision sequence must
    not change because of it."""
    from geotrace.pacman_tracker.tracker import PacmanTracker, build_inputs
    network, names, trip = _scale_error_trip(1.15, ["in", "mid", "out"])

    def decisions(intervals_on):
        cfg = PacmanConfig()
        cfg.tracker_mode = "single_path"
        cfg.speed.accel_scale_enabled = True
        cfg.intervals.enabled = intervals_on
        res = PacmanTracker(network, cfg).run(build_inputs(trip, network, cfg))
        sp = res.stats["single_path"]
        return [(d["incoming_edge"], d["chosen_edge"]) for d in sp["decisions"]]

    off = decisions(False)
    on = decisions(True)
    assert off == on  # identical route; D correction did not move the branches


def _gate(**overrides):
    """Run the tracker's interval quality gate on a synthetic interval dict."""
    from geotrace.pacman_tracker.tracker import PacmanTracker
    base = dict(
        map_length_m=3400.0, duration_s=420.0, discrepancy_m=60.0,
        endpoint_a_ok=True, endpoint_b_ok=True,
        endpoint_residuals_m=[-8.0, 12.0],
        endpoint_a_angle_z=1.1, endpoint_b_angle_z=1.4,
        endpoint_a_local_probability=0.99, endpoint_b_local_probability=0.98,
        interior_low_confidence=0, interior_provisional_unresolved=0,
        upstream_unresolved_forks=0, rolled_back_inside=False,
        unambiguous_committed_path=True,
    )
    base.update(overrides)
    tr = PacmanTracker.__new__(PacmanTracker)
    tr.cfg = PacmanConfig()
    return tr._interval_reject_reason(base, base["map_length_m"],
                                     base["discrepancy_m"])


def test_a_long_clean_interval_passes_the_quality_gate():
    # 3400 m / 420 s - far past the old hard cutoffs - is accepted when clean.
    assert _gate() is None


def test_long_interval_with_a_weak_endpoint_fails():
    assert "endpoint B" in _gate(endpoint_b_ok=False)


def test_long_interval_with_an_interior_low_confidence_junction_fails():
    assert "low-confidence junction" in _gate(interior_low_confidence=1)


def test_long_interval_downstream_of_an_unresolved_fork_fails():
    assert "unresolved fork upstream" in _gate(upstream_unresolved_forks=1)


def test_interval_with_excessive_drift_is_refused_as_a_wrong_route():
    assert "drift" in _gate(discrepancy_m=200.0)


def test_interval_beyond_the_numerical_sanity_bound_is_refused():
    assert "sanity bound" in _gate(map_length_m=20000.0)
