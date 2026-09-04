"""End-to-end reconstruction on the fork graph from the specification::

              branch A
             /
    start -- junction
             \\
              branch B

The car turns onto branch A. GPS is switched off *before* the junction, so at
the moment of the decision the only evidence available is the gyro. The filter
must come out preferring branch A - and must still be holding branch B as a
live hypothesis while the two are indistinguishable, rather than committing
early.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from geotrace.config import Config
from geotrace.fault_injection import FaultSpec, inject_faults
from geotrace.pipeline import build_metrics, run_reconstruction
from geotrace.polygons import build_uncertainty_set
from geotrace.road_graph import RoadNetwork
from geotrace.simulate import SimulationSpec, simulate_trip

from conftest import edge_named

TURN_ONTO = "Branch A"
OTHER = "Branch B"


def _fork_route(network: RoadNetwork, branch: str) -> list[int]:
    return [
        edge_named(network, "Stem", (0.0, 0.0)),
        edge_named(network, branch, (500.0, 0.0)),
    ]


def _run(fork_network: RoadNetwork, branch: str = TURN_ONTO, seed: int = 42):
    """Drive onto ``branch``, kill GPS across the junction, reconstruct."""
    spec = SimulationSpec(duration_s=130.0, cruise_speed_ms=9.0, warmup_still_s=5.0)
    trip = simulate_trip(
        fork_network, spec, seed=seed, route=_fork_route(fork_network, branch)
    )
    # The stem is 500 m; at ~9 m/s the junction is crossed around t = 60 s.
    # Cut GPS from 40 s to 100 s so the turn happens entirely in the dark.
    broken, _ = inject_faults(
        trip,
        [FaultSpec(kind="dropout", start_s=40.0, duration_s=60.0)],
        seed=seed,
        frame=fork_network.frame,
    )
    cfg = Config()
    cfg.seed = seed
    cfg.pf.n_particles = 3000
    result = run_reconstruction(broken, fork_network, cfg)
    return broken, cfg, result


def _branch_weight(network: RoadNetwork, snapshot, name: str) -> float:
    mask = np.array(
        [str(network.edges[int(i)].name) == name for i in snapshot.edge_idx]
    )
    return float(np.asarray(snapshot.weights, dtype=float)[mask].sum())


def test_the_outage_is_detected(fork_network: RoadNetwork) -> None:
    broken, _cfg, result = _run(fork_network)
    assert result.outage_windows, "a 60 s dropout must be detected"
    window = result.outage_windows[0]
    assert 35.0 <= window["start_s"] <= 55.0
    assert window["end_s"] - window["start_s"] > 30.0


def test_gps_outage_keeps_a_causal_road_hypothesis(
    fork_network: RoadNetwork,
) -> None:
    """The map prior propagates during the outage but receives no GPS update."""
    _broken, _cfg, result = _run(fork_network)
    pf = result.particle_filter
    assert any(s.gps_state != "TRUSTED" for s in pf.result.snapshots)

    inertial = [u for u in result.uncertainty if u.gps_state != "TRUSTED"]
    assert inertial
    assert any(u.n_particles > 0 for u in inertial)


def test_imu_uncertainty_keeps_road_branches_during_an_outage(fork_network: RoadNetwork) -> None:
    _broken, cfg, result = _run(fork_network)
    inertial = [
        u for u in result.uncertainty if u.gps_state != "TRUSTED" and u.n_particles > 0
    ]
    assert inertial[-1].total_area_m2 > 0
    assert all(u.n_particles == cfg.pf.n_particles for u in inertial)


def test_reanchor_starts_a_new_visible_route_segment(grid_network: RoadNetwork) -> None:
    _broken, _cfg, result = _composite_run(grid_network)
    assert result.diagnostics["reanchors"]
    assert result.primary.to_geojson(result.frame)["geometry"]["type"] == "MultiLineString"


def test_recovery_candidates_do_not_draw_a_cross_city_jump(grid_network: RoadNetwork) -> None:
    """Only TRUSTED GPS may correct a track; recovery starts a new segment."""
    _broken, _cfg, result = _composite_run(grid_network)
    track = result.primary
    starts = [*track.segment_starts, len(track.xy)]
    for start, end in zip(starts, starts[1:]):
        points = track.array[start:end]
        if len(points) > 1:
            steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
            assert float(steps.max()) < 100.0


def test_the_reconstructed_route_ends_on_the_driven_branch(
    fork_network: RoadNetwork,
) -> None:
    _broken, _cfg, result = _run(fork_network)
    end = result.primary.array[-1]
    assert end[1] > 50.0, "Branch A runs north-east, so N must be clearly positive"
    assert fork_network.distance_to_road(end) < 5.0


def test_the_polygons_cover_the_true_position(fork_network: RoadNetwork) -> None:
    broken, cfg, result = _run(fork_network)
    metrics = build_metrics(broken, result, cfg)
    assert metrics.polygons["coverage_95"] is not None
    assert metrics.polygons["coverage_95"] > 0.85


def test_the_road_filter_beats_holding_the_last_known_position(
    fork_network: RoadNetwork,
) -> None:
    broken, cfg, result = _run(fork_network)
    metrics = build_metrics(broken, result, cfg)
    assert metrics.position_error["mean_m"] < metrics.baselines["last_known_position"]["mean_m"]


def test_outage_estimate_is_not_snapped_to_a_road(fork_network: RoadNetwork) -> None:
    """The IMU route may be off-road; the graph is not a trajectory source."""
    _broken, _cfg, result = _run(fork_network)
    assert any(fork_network.distance_to_road(point) >= 5.0 for point in result.primary.array)


def test_the_run_is_reproducible(fork_network: RoadNetwork) -> None:
    a = _run(fork_network, seed=7)[2].primary.array
    b = _run(fork_network, seed=7)[2].primary.array
    assert np.array_equal(a, b)


def test_map_assistance_is_strictly_bounded(
    fork_network: RoadNetwork,
) -> None:
    """A road prior can nudge, but can never create a cross-city route jump."""
    _broken, cfg, result = _run(fork_network, seed=7)
    delta = np.linalg.norm(result.primary.array - result.tracks["ekf_dead_reckoning"].array, axis=1)
    assert float(delta.max()) <= (
        cfg.pf.outage_map_assist_gain * cfg.pf.outage_map_assist_max_offset_m + 1e-6
    )


def test_a_different_seed_changes_the_particle_realisation(
    fork_network: RoadNetwork,
) -> None:
    """The map prior remains stochastic, but it is no longer the route sensor."""
    a = _run(fork_network, seed=7)[2].particle_filter.snapshot(0.0).weights
    b = _run(fork_network, seed=8)[2].particle_filter.snapshot(0.0).weights
    assert not np.array_equal(a, b)


def test_metrics_json_is_fully_populated(fork_network: RoadNetwork) -> None:
    broken, cfg, result = _run(fork_network)
    payload = build_metrics(broken, result, cfg).to_json()
    assert payload["has_reference_track"] is True
    for key in ("mean_m", "median_m", "p95_m", "max_m"):
        assert isinstance(payload["position_error"][key], float)
    assert payload["polygons"]["coverage_95"] is not None
    assert payload["polygons"]["mean_area_m2"] > 0
    assert payload["branches"]["top1_accuracy"] is not None
    assert payload["branches"]["top3_recall"] is not None
    assert payload["gps_gates"]["fixes_tested"] > 0
    assert set(payload["baselines"]) == {"last_known_position", "ekf_dead_reckoning"}


def test_error_is_undefined_without_a_reference_track(fork_network: RoadNetwork) -> None:
    """A real outage has no ground truth and the metrics must say so."""
    spec = SimulationSpec(duration_s=90.0, cruise_speed_ms=9.0)
    trip = simulate_trip(fork_network, spec, seed=3, route=_fork_route(fork_network, TURN_ONTO))
    cfg = Config()
    cfg.pf.n_particles = 800
    result = run_reconstruction(trip, fork_network, cfg)
    metrics = build_metrics(trip, result, cfg)
    assert metrics.has_reference is False
    assert metrics.position_error == {"available": False}
    assert metrics.polygons["coverage_95"] is None
    assert any("no ground truth" in note.lower() for note in metrics.notes)


# ---------------------------------------------------------------------------
# The composite failure from the specification: GPS first drifts off by a
# constant offset, then disappears, then comes back with several false points.
# ---------------------------------------------------------------------------


def _composite_run(grid_network: RoadNetwork, seed: int = 42):
    """Offset, then dropout, then a return with several false points.

    The offset is large enough (3.2 km) to leave the street grid entirely, so a
    displaced fix cannot land on a real road by coincidence.
    """
    from geotrace.fault_injection import scenario_offset_dropout_recovery

    spec = SimulationSpec(duration_s=220.0, cruise_speed_ms=9.0, warmup_still_s=5.0)
    trip = simulate_trip(grid_network, spec, seed=seed)
    faults = scenario_offset_dropout_recovery(
        start_s=40.0, offset_duration_s=25.0, dropout_duration_s=40.0,
        east=2600.0, north=-1900.0, false_points=4,
    )
    broken, _ = inject_faults(trip, faults, seed=seed, frame=grid_network.frame)
    cfg = Config()
    cfg.seed = seed
    cfg.pf.n_particles = 2500
    return broken, cfg, run_reconstruction(broken, grid_network, cfg)


def test_the_composite_failure_is_detected(grid_network: RoadNetwork) -> None:
    _broken, _cfg, result = _composite_run(grid_network)
    assert result.outage_windows
    assert result.outage_windows[0]["start_s"] < 45.0, "the offset must be caught early"


def test_the_offset_fixes_are_rejected(grid_network: RoadNetwork) -> None:
    """A 3.2 km displacement reported with 14 m accuracy is the case that proves
    horizontalAccuracy alone cannot be trusted."""
    _broken, _cfg, result = _composite_run(grid_network)
    reasons = result.diagnostics["gps"]["rejection_reasons"]
    assert reasons.get("physical_gate", 0) > 0


def test_trust_comes_back_after_the_composite_failure(grid_network: RoadNetwork) -> None:
    """The bug this guards: gating returning fixes against a drifted filter made
    the monitor reject every good fix for the rest of the trip."""
    broken, cfg, result = _composite_run(grid_network)
    assert result.diagnostics["gps"]["final_state"] == "TRUSTED"
    metrics = build_metrics(broken, result, cfg)
    assert metrics.trust_recovery["recovery_events"] >= 1
    assert metrics.gps_gates["rejected_good_fraction"] < 0.15


def test_the_filters_are_reanchored_when_trust_returns(grid_network: RoadNetwork) -> None:
    _broken, _cfg, result = _composite_run(grid_network)
    assert result.diagnostics["reanchors"], "a recovery must re-anchor the filters"
    assert result.particle_filter.result.reinitializations >= 1


def test_the_opening_bootstrap_is_not_mistaken_for_a_recovery(
    fork_network: RoadNetwork,
) -> None:
    """Every trip starts untrusted; that is not an outage."""
    spec = SimulationSpec(duration_s=90.0, cruise_speed_ms=9.0)
    trip = simulate_trip(fork_network, spec, seed=5, route=_fork_route(fork_network, TURN_ONTO))
    cfg = Config()
    cfg.pf.n_particles = 800
    result = run_reconstruction(trip, fork_network, cfg)
    assert result.diagnostics["reanchors"] == []
    assert result.outage_windows == []


def test_the_road_filter_survives_the_composite_failure_far_better(
    grid_network: RoadNetwork,
) -> None:
    broken, cfg, result = _composite_run(grid_network)
    metrics = build_metrics(broken, result, cfg)
    assert metrics.position_error["mean_m"] < 0.5 * metrics.baselines["last_known_position"]["mean_m"]
    assert metrics.polygons["coverage_95"] > 0.8


# ---------------------------------------------------------------------------
# Using a road graph that does not cover the roads actually driven produces a
# plausible-looking but wrong reconstruction. It must not fail silently.
# ---------------------------------------------------------------------------


def test_a_matching_graph_reports_a_tight_track_to_road_fit(
    fork_network: RoadNetwork,
) -> None:
    spec = SimulationSpec(duration_s=90.0, cruise_speed_ms=9.0)
    trip = simulate_trip(fork_network, spec, seed=4, route=_fork_route(fork_network, TURN_ONTO))
    cfg = Config()
    cfg.pf.n_particles = 600
    result = run_reconstruction(trip, fork_network, cfg)
    fit = result.diagnostics["road_graph"]
    assert fit["median_track_to_road_m"] < cfg.gps.max_median_track_to_road_m
    assert "warning" not in fit


def test_a_graph_for_the_wrong_area_is_warned_about(fork_network: RoadNetwork) -> None:
    """The trip is driven on the fork graph but reconstructed against a graph
    whose streets are a kilometre away - the "wrong map" mistake, which would
    otherwise produce a confident reconstruction on the wrong roads."""
    from geotrace.road_graph import RoadNetwork, build_graph_from_segments

    from conftest import ORIGIN_LAT, ORIGIN_LON

    elsewhere_segments = [
        ("Far street", [(1500.0, 1800.0), (2400.0, 1900.0)], {"highway": "residential"}),
        ("Far lane", [(2400.0, 1900.0), (2600.0, 2600.0)], {"highway": "residential"}),
    ]
    graph, frame = build_graph_from_segments(elsewhere_segments, ORIGIN_LAT, ORIGIN_LON)
    elsewhere = RoadNetwork(graph, frame)

    spec = SimulationSpec(duration_s=90.0, cruise_speed_ms=9.0)
    trip = simulate_trip(fork_network, spec, seed=4, route=_fork_route(fork_network, TURN_ONTO))
    cfg = Config()
    cfg.pf.n_particles = 600
    result = run_reconstruction(trip, elsewhere, cfg)
    fit = result.diagnostics["road_graph"]
    assert fit["median_track_to_road_m"] > cfg.gps.max_median_track_to_road_m
    assert "warning" in fit
    assert "does not cover" in fit["warning"]
