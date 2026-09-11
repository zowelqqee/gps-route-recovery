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
    result = run_reconstruction(broken, fork_network, cfg, algorithm="road_particle_filter")
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


def test_noncompact_road_hypotheses_fall_back_to_an_imu_disc(fork_network: RoadNetwork) -> None:
    _broken, cfg, result = _run(fork_network)
    inertial = [
        u for u in result.uncertainty if u.gps_state != "TRUSTED"
    ]
    assert inertial[-1].total_area_m2 > 0
    # The PF still runs internally, but once its road corridors are too broad
    # the report must not label their merged OSM geometry as a 95% branch.
    assert any(u.n_particles == cfg.pf.n_particles for u in inertial)
    assert any(u.n_particles == 0 for u in inertial)
    graph_corridors = [u for u in inertial if u.n_particles > 0]
    assert all(u.total_area_m2 <= cfg.pf.outage_map_assist_max_area_m2 for u in graph_corridors)


def test_reanchor_starts_a_new_visible_route_segment(grid_network: RoadNetwork) -> None:
    _broken, _cfg, result = _composite_run(grid_network)
    assert result.diagnostics["reanchors"]
    assert result.primary.to_geojson(result.frame)["geometry"]["type"] == "MultiLineString"


def test_reanchor_exposes_a_connected_route_without_claiming_imu_timing(
    grid_network: RoadNetwork,
) -> None:
    """A weak odometer must not hide the road between two known endpoints."""
    _broken, _cfg, result = _composite_run(grid_network)
    reconciled = result.road_reconciliations
    assert reconciled
    assert all(item.length_m > 0 for item in reconciled)
    assert all(item.edge_count > 0 for item in reconciled)
    assert result.diagnostics["road_reconciliations"]


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


def test_the_inertial_baseline_is_never_snapped_to_a_road(
    fork_network: RoadNetwork,
) -> None:
    """The graph must never be fed back into the inertial solution.

    The *displayed* route is walked along the graph during an outage, but
    `ekf_dead_reckoning` is the independent reference that the displayed
    route's own sigma test is measured against (`display_route_ekf_sigma_k`)
    and that `GPSQualityMonitor` gates every returning fix against. The moment
    the graph touches it, both of those become self-agreement.
    """
    _broken, _cfg, result = _run(fork_network)
    baseline = result.tracks["ekf_dead_reckoning"].array
    assert any(fork_network.distance_to_road(point) >= 5.0 for point in baseline)


def test_the_run_is_reproducible(fork_network: RoadNetwork) -> None:
    a = _run(fork_network, seed=7)[2].primary.array
    b = _run(fork_network, seed=7)[2].primary.array
    assert np.array_equal(a, b)


def _tick_gps_states(result) -> list[str]:
    """One uncertainty set is appended per output tick, in step with the track."""
    assert len(result.uncertainty) == len(result.primary.xy)
    return [u.gps_state for u in result.uncertainty]


def test_the_displayed_outage_route_stays_on_the_graph(
    fork_network: RoadNetwork,
) -> None:
    """The failure this replaced the old 54 m envelope for: the free inertial
    estimate is bounded by nothing and has been seen leaving a bridge
    sideways into a river."""
    broken, _cfg, result = _run(fork_network, seed=7)
    states = _tick_gps_states(result)
    # The dropout runs 40-100 s; before the first trusted fix there is no
    # particle cloud to walk yet, and those bootstrap ticks are legitimately
    # the free inertial estimate (see the `elif` bootstrap branch in the
    # pipeline), so ask only about the injected outage itself.
    in_outage = [
        point
        for point, state, item in zip(result.primary.array, states, result.uncertainty)
        if state != "TRUSTED" and 45.0 <= (item.t - broken.t0) <= 95.0
    ]
    assert in_outage, "the injected dropout must produce untrusted output ticks"
    assert max(fork_network.distance_to_road(p) for p in in_outage) < 1.0


def test_the_displayed_route_never_teleports_between_streets(
    fork_network: RoadNetwork,
) -> None:
    """Constraint A: the display walks connected edges, so every step inside a
    rendered segment is bounded by what the car could physically drive - it
    can never appear on a street it had no way of reaching."""
    _broken, cfg, result = _run(fork_network, seed=7)
    step_limit = (
        cfg.motion.max_speed_ms * 1.0 + cfg.pf.display_route_max_snap_m
    )
    starts = [*result.primary.segment_starts, len(result.primary.xy)]
    for start, end in zip(starts, starts[1:]):
        points = result.primary.array[start:end]
        if len(points) > 1:
            steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
            assert float(steps.max()) <= step_limit


def test_the_display_matches_the_inertial_estimate_while_gps_is_trusted(
    fork_network: RoadNetwork,
) -> None:
    """The walker is scoped to outages only. While GPS is trusted the
    GPS-corrected inertial estimate is genuinely the better answer, and
    snapping there would also undo the off-graph parking case that
    `PolygonConfig.off_road_distance_m` exists for."""
    _broken, _cfg, result = _run(fork_network, seed=7)
    baseline = result.tracks["ekf_dead_reckoning"].array
    states = _tick_gps_states(result)
    trusted = [i for i, state in enumerate(states) if state == "TRUSTED"]
    assert trusted, "the fixture must contain trusted ticks"
    for i in trusted:
        assert result.primary.array[i] == pytest.approx(baseline[i])


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
    assert set(payload["baselines"]) == {"last_known_position", "ekf_dead_reckoning", "road_posterior"}


def test_error_is_undefined_without_a_reference_track(fork_network: RoadNetwork) -> None:
    """A real outage has no ground truth and the metrics must say so."""
    spec = SimulationSpec(duration_s=90.0, cruise_speed_ms=9.0)
    trip = simulate_trip(fork_network, spec, seed=3, route=_fork_route(fork_network, TURN_ONTO))
    cfg = Config()
    cfg.pf.n_particles = 800
    result = run_reconstruction(trip, fork_network, cfg, algorithm="road_particle_filter")
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
    return broken, cfg, run_reconstruction(broken, grid_network, cfg, algorithm="road_particle_filter")


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
    result = run_reconstruction(trip, fork_network, cfg, algorithm="road_particle_filter")
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
    result = run_reconstruction(trip, fork_network, cfg, algorithm="road_particle_filter")
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
    result = run_reconstruction(trip, elsewhere, cfg, algorithm="road_particle_filter")
    fit = result.diagnostics["road_graph"]
    assert fit["median_track_to_road_m"] > cfg.gps.max_median_track_to_road_m
    assert "warning" in fit
    assert "does not cover" in fit["warning"]


def test_trusted_gps_far_from_every_edge_is_not_reported_as_a_road_corridor(
    fork_network: RoadNetwork,
) -> None:
    """A car parked in a courtyard, or driving down a private lane the graph
    was never given, still produces a perfectly self-consistent, trusted GPS
    stream - just one nowhere near a known edge. The particle filter is still
    forced onto whichever real edge is nearest, which can be hundreds of
    metres away; the output must not dress that up as a 95%-confident
    corridor on a street the car was never on."""
    spec = SimulationSpec(duration_s=70.0, cruise_speed_ms=5.0, warmup_still_s=5.0)
    trip = simulate_trip(
        fork_network, spec, seed=3, route=[edge_named(fork_network, "Stem", (0.0, 0.0))]
    )
    offset_north_m = 150.0
    broken, _ = inject_faults(
        trip,
        [FaultSpec(kind="offset", start_s=25.0, duration_s=25.0, north=offset_north_m)],
        seed=3,
        frame=fork_network.frame,
    )
    cfg = Config()
    cfg.seed = 3
    cfg.pf.n_particles = 2000
    result = run_reconstruction(broken, fork_network, cfg, algorithm="road_particle_filter")

    # Give the trust monitor a few seconds after the jump to reanchor and
    # settle before judging its output.
    window = (broken.t0 + 32.0, broken.t0 + 45.0)
    off_road = [
        u for u in result.uncertainty
        if u.gps_state == "TRUSTED" and window[0] <= u.t <= window[1]
    ]
    assert off_road, "expected TRUSTED output inside the offset window"
    for u in off_road:
        best = u.best
        assert best is not None
        assert best.street_names == [], (
            "a trusted fix >60 m from every edge must fall back to a plain "
            "disc, not a named-road corridor"
        )


def test_an_outage_the_trip_ends_inside_is_still_a_window(
    fork_network: RoadNetwork,
) -> None:
    """GPS that goes away and never returns leaves one history entry, not two.

    Trust is only re-evaluated when a fix arrives, so an outage still open when
    the recording stops has a start and no end. Reported as a zero-length
    window it is discarded as noise, and every metric keyed to "at the end of
    the outage" silently becomes null - on exactly the recordings where the
    outage is the whole point.
    """
    spec = SimulationSpec(duration_s=130.0, cruise_speed_ms=9.0, warmup_still_s=5.0)
    trip = simulate_trip(
        fork_network, spec, seed=42, route=_fork_route(fork_network, TURN_ONTO)
    )
    broken, _ = inject_faults(
        trip,
        [FaultSpec(kind="dropout", start_s=40.0, duration_s=200.0)],
        seed=42,
        frame=fork_network.frame,
    )
    cfg = Config()
    cfg.seed = 42
    cfg.pf.n_particles = 1500
    result = run_reconstruction(broken, fork_network, cfg, algorithm="road_particle_filter")

    assert result.outage_windows, "an outage that never ends is still an outage"
    last = result.outage_windows[-1]
    assert last["end_s"] == pytest.approx(broken.duration_s, abs=1.0)
    assert last["end_s"] - last["start_s"] > 60.0

    metrics = build_metrics(broken, result, cfg).to_json()
    assert metrics["position_error"]["error_at_outage_end_m"] is not None


def test_a_window_measured_heading_is_preferred_to_the_first_fix_s_course(
    fork_network: RoadNetwork,
) -> None:
    """A recorder that averaged a heading over many fixes is not second-guessed.

    `heading_source` distinguishes the two. "gps_course" is one fix's course,
    which is what the search would find anyway; "gps_course_window" is a
    measurement across a window and is taken as given. Older trips say the
    former, so their behaviour does not move.
    """
    from geotrace.coordinates import course_to_heading
    from geotrace.pipeline import initial_heading

    spec = SimulationSpec(duration_s=60.0, cruise_speed_ms=9.0, warmup_still_s=5.0)
    trip = simulate_trip(
        fork_network, spec, seed=42, route=_fork_route(fork_network, TURN_ONTO)
    )
    cfg = Config()
    fixes = trip.usable_locations
    calibration = trip.metadata.calibration
    assert calibration is not None and calibration.initial_heading_deg is not None

    from_fixes = initial_heading(fixes, fork_network.frame, cfg)
    calibration.initial_heading_deg = 123.0

    calibration.heading_source = "gps_course"
    assert initial_heading(fixes, fork_network.frame, cfg, calibration) == pytest.approx(
        from_fixes
    )

    calibration.heading_source = "gps_course_window"
    assert initial_heading(fixes, fork_network.frame, cfg, calibration) == pytest.approx(
        course_to_heading(123.0)
    )
