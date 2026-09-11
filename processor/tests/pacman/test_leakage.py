"""Hidden GPS must not reach the reconstruction.

The reference stream exists to score the run and to say where the truth died.
If any of it leaked into the filter, every number this project produces would be
worthless, and the leak would be invisible - a tracker fed its own answer looks
excellent. So it is tested directly: corrupt the withheld GPS beyond recognition
and demand that the reconstruction come out bit-identical.
"""

import copy
import json

import numpy as np
import pytest

from geotrace.pacman_tracker.config import PacmanConfig
from geotrace.pacman_tracker.diagnostics import map_match_reference, survival_metrics
from geotrace.pacman_tracker.synthetic import (
    arc,
    build_network,
    simulate_trip,
    straight,
)
from geotrace.pacman_tracker.tracker import PacmanTracker, build_inputs


@pytest.fixture(scope="module")
def scenario():
    network, _ = build_network(
        [
            ("A", straight(300.0), {}),
            ("Bend", arc(60.0, np.pi / 2, start=(300.0, 0.0)), {}),
            ("C", straight(300.0, start=(360.0, 60.0), heading_rad=np.pi / 2), {}),
        ]
    )
    route = [e.index for e in network.edges]
    trip = simulate_trip(network, route, lambda t: 9.0, duration_s=90.0,
                         gps_visible_s=20.0, noise=0.12, seed=7)
    return network, trip


def _run(network, trip, cfg=None):
    cfg = cfg or PacmanConfig()
    inputs = build_inputs(trip, network, cfg)
    result = PacmanTracker(network, cfg).run(inputs)
    return result, inputs


def _fingerprint(result) -> str:
    return json.dumps([f.to_json() for f in result.frames], sort_keys=True)


def test_the_scenario_actually_exercises_the_tracker(scenario):
    """Guard the guard: a test that compares two no-ops proves nothing."""
    network, trip = scenario
    result, inputs = _run(network, trip)
    assert len(result.frames) > 30
    assert inputs.notes["gps_visible_until_s"] < 25.0
    assert len(trip.reference_locations) > 100
    assert result.stats["gyro_updates"] > 100
    assert result.stats["speed"]["predict"] > 100


def test_mutating_hidden_gps_does_not_change_the_reconstruction(scenario):
    network, trip = scenario
    baseline, _ = _run(network, trip)

    corrupted = copy.deepcopy(trip)
    rng = np.random.default_rng(11)
    for fix in corrupted.reference_locations:
        fix.latitude += float(rng.normal(0.0, 0.05))     # kilometres away
        fix.longitude += float(rng.normal(0.0, 0.05))
        fix.speed = float(abs(rng.normal(30.0, 5.0)))
        fix.course = float(rng.uniform(0.0, 360.0))
        fix.horizontal_accuracy = 3.0
    after, _ = _run(network, corrupted)

    assert _fingerprint(after) == _fingerprint(baseline)


def test_deleting_hidden_gps_entirely_does_not_change_the_reconstruction(scenario):
    network, trip = scenario
    baseline, _ = _run(network, trip)
    stripped = copy.deepcopy(trip)
    stripped.reference_locations = []
    after, _ = _run(network, stripped)
    assert _fingerprint(after) == _fingerprint(baseline)


def test_build_inputs_ignores_fixes_after_the_cutoff(scenario):
    """A reference fix wrongly filed under `locations` is still excluded."""
    network, trip = scenario
    cfg = PacmanConfig()
    honest = build_inputs(trip, network, cfg)
    smuggled = copy.deepcopy(trip)
    smuggled.locations = smuggled.locations + smuggled.reference_locations
    guarded = build_inputs(smuggled, network, cfg, gps_cutoff_t=honest.t_start)
    assert guarded.t_start == honest.t_start
    assert guarded.init_edges == honest.init_edges
    assert guarded.speed0 == honest.speed0


def test_the_speed_filter_never_sees_the_reference(scenario):
    """The speed filter is where the withheld GPS would do most damage, since
    a leaked speed would fix the distance and with it the whole answer."""
    network, trip = scenario
    baseline, inputs = _run(network, trip)
    assert inputs.oracle_speed is None and inputs.oracle_distance is None
    corrupted = copy.deepcopy(trip)
    for fix in corrupted.reference_locations:
        fix.speed = 30.0
    after, _ = _run(network, corrupted)
    assert _fingerprint(after) == _fingerprint(baseline)


def test_hidden_gps_does_change_the_metrics(scenario):
    """The other half of the argument: the reference is not inert everywhere.

    If corrupting it changed nothing at all - metrics included - the leakage
    test above would be passing for the wrong reason.
    """
    network, trip = scenario
    result, _ = _run(network, trip)
    truth = map_match_reference(trip.reference_locations, network, network.frame)
    good = survival_metrics(result, truth, network)

    corrupted = copy.deepcopy(trip)
    for fix in corrupted.reference_locations:
        fix.latitude += 0.02
    bad_truth = map_match_reference(corrupted.reference_locations, network, network.frame)
    bad = survival_metrics(result, bad_truth, network)
    assert good != bad


def _full_stack() -> PacmanConfig:
    """Everything this pass added, switched on at once."""
    cfg = PacmanConfig()
    cfg.attitude.enabled = True
    cfg.intervals.enabled = True
    return cfg


def test_the_new_subsystems_are_actually_running(scenario):
    """Guard the guard again: the leakage tests below must exercise the new
    code, not silently run the same default pipeline twice."""
    network, trip = scenario
    result, _ = _run(network, trip, _full_stack())
    assert result.stats["attitude"]["enabled"] is True
    assert result.stats["attitude"]["steps"] > 100
    assert "drift" in result.stats


def test_attitude_and_interval_paths_never_see_the_reference(scenario):
    """The gravity filter and the map-distance constraint are the two new ways
    the withheld GPS could reach the estimate: one reads the raw IMU, the other
    reads the map and the hypothesis weights. Neither may read the answer."""
    network, trip = scenario
    cfg = _full_stack()
    baseline, _ = _run(network, trip, cfg)

    corrupted = copy.deepcopy(trip)
    rng = np.random.default_rng(23)
    for fix in corrupted.reference_locations:
        fix.latitude += float(rng.normal(0.0, 0.05))
        fix.longitude += float(rng.normal(0.0, 0.05))
        fix.speed = float(abs(rng.normal(30.0, 5.0)))
        fix.course = float(rng.uniform(0.0, 360.0))
    after, _ = _run(network, corrupted, cfg)
    assert _fingerprint(after) == _fingerprint(baseline)

    stripped = copy.deepcopy(trip)
    stripped.reference_locations = []
    assert _fingerprint(_run(network, stripped, cfg)[0]) == _fingerprint(baseline)
