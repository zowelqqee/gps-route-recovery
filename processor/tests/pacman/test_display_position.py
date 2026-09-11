"""The zero-latency corrected DISPLAY position branch (Phase 32).

Two things are being guarded:

* it is a *marker overlay* - it must never change the committed route, junction
  timing, branch weights, ``speed_trace`` or anything else the tracker decides;
* when it does run, it is the frozen Phase 32 iso-binary estimator - a positive,
  gated saturation correction that reproduces the 07-26 diagnostic result.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from geotrace.coordinates import LocalFrame
from geotrace.pacman_tracker.config import PacmanConfig
from geotrace.pacman_tracker.display_position import (
    DisplayPositionBranch, DisplayResidualModel)
from geotrace.pacman_tracker.roadmap import RoadGeometry
from geotrace.pacman_tracker.synthetic import arc, build_network, simulate_trip, straight
from geotrace.pacman_tracker.tracker import PacmanTracker, build_inputs


# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def scenario():
    network, _ = build_network([
        ("A", straight(400.0), {}),
        ("Bend", arc(80.0, np.pi / 2, start=(400.0, 0.0)), {}),
        ("C", straight(600.0, start=(480.0, 80.0), heading_rad=np.pi / 2), {}),
        ("D", straight(600.0, start=(480.0, 680.0), heading_rad=np.pi / 2), {}),
    ])
    route = [e.index for e in network.edges]
    # a speed that actually varies, so the spectral model has something to fit
    trip = simulate_trip(network, route,
                         lambda t: 6.0 + 7.0 * (1.0 + np.sin(t / 12.0)),
                         duration_s=150.0, gps_visible_s=28.0, noise=0.08, seed=13)
    return network, trip


def _run(network, trip, *, display=False, l0=False, cfg=None):
    cfg = cfg or PacmanConfig()
    cfg.tracker_mode = "single_path"
    cfg.display.position_branch_enabled = display
    cfg.display.leave_0726_out = l0
    inputs = build_inputs(trip, network, cfg)
    result = PacmanTracker(network, cfg, geometry=RoadGeometry(network, cfg.geometry)).run(inputs)
    return result


def _route_fingerprint(result) -> str:
    return json.dumps({
        "frames": [f.to_json() for f in result.frames],
        "speed": [s.to_json() for s in result.speed_trace],
        "route": result.final.routes[0].edges(),
        "decisions": result.stats.get("single_path", {}).get("decisions", []),
    }, sort_keys=True)


# ---- 1-4, 12  the branch must not touch topology / route / speed --------
def test_route_and_speed_byte_identical_with_branch_on(scenario):
    network, trip = scenario
    off = _run(network, trip, display=False)
    on = _run(network, trip, display=True)
    assert _route_fingerprint(on) == _route_fingerprint(off)


def test_branch_disabled_means_no_position_trace(scenario):
    network, trip = scenario
    assert _run(network, trip, display=False).position_trace is None


def test_disabled_flag_is_a_true_no_op(scenario):
    """D_position must equal D_route to the metre when the feature is off - it is
    off by default, so a plain run and an explicitly-disabled run must match."""
    network, trip = scenario
    default = _run(network, trip)                       # position_branch_enabled defaults False
    disabled = _run(network, trip, display=False)
    assert _route_fingerprint(default) == _route_fingerprint(disabled)


def test_perturbing_v_position_cannot_change_the_committed_edge_sequence(scenario, monkeypatch):
    """Inject an enormous display correction straight into the branch and show
    route / junctions / speed are unmoved - the branch has no feedback path."""
    network, trip = scenario
    off = _run(network, trip, display=False)

    real_step = DisplayPositionBranch.step

    def runaway_step(self, t, dt, v_route, v_prior, v_spectral, d_route, stationary):
        real_step(self, t, dt, v_route, v_prior, v_spectral, d_route, stationary)
        self._delta += 5.0 * dt          # +5 m/s of pure invented display speed
        self._gate = True

    monkeypatch.setattr(DisplayPositionBranch, "step", runaway_step)
    hard = _run(network, trip, display=True)

    assert hard.stats["display_position"]["max_abs_delta_m"] > 200.0     # correction really fired
    assert hard.position_trace[-1].distance_m != pytest.approx(
        hard.position_trace[-1].route_distance_m)                        # D_position != D_route
    assert _route_fingerprint(hard) == _route_fingerprint(off)          # ...yet nothing moved


# ---- 5-6  gate behaviour --------------------------------------------------
def test_saturation_gate_off_means_no_correction():
    m = DisplayResidualModel.load()
    b = DisplayPositionBranch(m, d0=0.0, frame=_toy_frame(), edges=_toy_edges())
    # v_spectral well below every gate threshold, for longer than any window
    for i in range(400):
        b.step(t=i * 0.1, dt=0.1, v_route=6.0, v_prior=7.0, v_spectral=6.0, d_route=6.0 * i * 0.1,
               stationary=False)
    assert b._delta == pytest.approx(0.0)
    assert b._gate is False


def test_gate_on_with_saturation_gives_a_positive_speed_correction():
    m = DisplayResidualModel.load()
    b = DisplayPositionBranch(m, d0=0.0, frame=_toy_frame(), edges=_toy_edges())
    d = 0.0
    for i in range(600):
        # sustained fast driving: the spectral speedometer saturates near 13-14
        # but spikes above 15 (its noisy upper tail), while the IMU prior runs
        # ahead - exactly the 07-26 saturation signature the gate keys on.
        vspec = 18.5 if (i % 20) < 4 else 13.0
        b.step(t=i * 0.1, dt=0.1, v_route=15.0, v_prior=19.0, v_spectral=vspec,
               d_route=d, stationary=False)
        d += 15.0 * 0.1
    assert b._gate is True
    assert b._delta > 0.0                        # only ever adds speed
    assert b._v_position >= b._v_route


def test_correction_is_one_directional_never_slows_the_marker():
    m = DisplayResidualModel.load()
    b = DisplayPositionBranch(m, d0=0.0, frame=_toy_frame(), edges=_toy_edges())
    d = 0.0
    for i in range(600):
        # gate forced on by history, but v_route already well above v_ml
        vspec = 20.0 if i < 250 else 8.0
        b.step(t=i * 0.1, dt=0.1, v_route=25.0, v_prior=25.0, v_spectral=vspec,
               d_route=d, stationary=False)
        d += 25.0 * 0.1
        assert b._v_position >= b._v_route - 1e-9


def test_gain_is_bounded_by_the_cross_trip_distance_cap():
    m = DisplayResidualModel.load()
    b = DisplayPositionBranch(m, d0=0.0, frame=_toy_frame(), edges=_toy_edges(),
                              correction_gain=2.0, max_correction_m=10.0)
    for i in range(600):
        vspec = 18.5 if (i % 20) < 4 else 13.0
        b.step(t=i * 0.1, dt=0.1, v_route=10.0, v_prior=20.0,
               v_spectral=vspec, d_route=i, stationary=False)
    assert b._delta == pytest.approx(10.0)
    assert b._v_position == pytest.approx(b._v_route)


# ---- 7  polyline interpolation -----------------------------------------
def test_position_interpolates_along_the_polyline_not_the_endpoints():
    # an L-shaped edge: (0,0)->(100,0)->(100,100), length 200
    edges = [_PolyEdge(np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]]))]
    frame = LocalFrame(59.9, 30.3)
    m = DisplayResidualModel.load()
    b = DisplayPositionBranch(m, d0=0.0, frame=frame, edges=edges)
    b._delta = 150.0                                     # push the marker 150 m along
    s = b.locate(t=1.0, route_edges=[0], edge_index=0, route_offset=0.0, offset_bias=0.0)
    # 150 m along the L is 50 m up the second leg: local xy (100, 50)
    assert s.edge == 0
    assert s.s_m == pytest.approx(150.0, abs=1e-6)
    x, y = edges[0].position(150.0)
    assert (x, y) == pytest.approx((100.0, 50.0))


# ---- 8-9  committed frontier / excess ---------------------------------
def test_excess_distance_is_held_at_the_frontier_not_used_to_pick_an_edge():
    edges = [_PolyEdge(np.array([[0.0, 0.0], [300.0, 0.0]])),
             _PolyEdge(np.array([[300.0, 0.0], [600.0, 0.0]]))]
    m = DisplayResidualModel.load()
    b = DisplayPositionBranch(m, d0=0.0, frame=LocalFrame(59.9, 30.3), edges=edges)
    b._d_route_seen = 100.0          # route odometer: car at s=100 on edge 0
    b._delta = 500.0                 # display correction wants it 500 m further
    # route has committed only edge 0 (length 300)
    s = b.locate(t=1.0, route_edges=[0], edge_index=0, route_offset=0.0, offset_bias=0.0)
    assert s.at_frontier is True
    assert s.edge == 0 and s.s_m == pytest.approx(300.0)      # clamped to edge-0 end
    assert s.excess_position_distance_m == pytest.approx(300.0)  # 100 + 500 - 300


def test_excess_is_released_once_the_route_commits_the_next_edge():
    edges = [_PolyEdge(np.array([[0.0, 0.0], [300.0, 0.0]])),
             _PolyEdge(np.array([[300.0, 0.0], [600.0, 0.0]]))]
    m = DisplayResidualModel.load()
    b = DisplayPositionBranch(m, d0=0.0, frame=LocalFrame(59.9, 30.3), edges=edges)
    b._d_route_seen = 100.0
    b._delta = 250.0
    held = b.locate(t=1.0, route_edges=[0], edge_index=0, route_offset=0.0, offset_bias=0.0)
    assert held.at_frontier is True and held.excess_position_distance_m == pytest.approx(50.0)
    # now the route independently commits edge 1 (offset 300); odometer has moved
    # to s=20 on edge 1, i.e. route arc 320; the same +250 delta now lands well
    # inside the committed geometry (320 + 250 = 570 < frontier 600).
    b._d_route_seen = 320.0
    released = b.locate(t=2.0, route_edges=[0, 1], edge_index=1, route_offset=300.0, offset_bias=0.0)
    assert released.edge == 1
    assert released.at_frontier is False
    assert released.excess_position_distance_m == pytest.approx(0.0)
    assert released.s_m == pytest.approx(270.0)     # 320 + 250 - 300 (offset of edge 1)


# ---- 10  no hidden GPS in the live estimator --------------------------
def test_display_position_source_has_no_hidden_gps_token():
    src = Path(__file__).resolve().parents[2] / "src/geotrace/pacman_tracker/display_position.py"
    text = src.read_text()
    for token in ("reference_locations", "reference-samples", "withheld", "oracle",
                  "ground_truth", "d_true", "D_true"):
        assert token not in text, f"{token!r} must not appear in the live display estimator"


def test_step_signature_takes_no_truth_argument():
    import inspect
    params = set(inspect.signature(DisplayPositionBranch.step).parameters)
    assert params == {"self", "t", "dt", "v_route", "v_prior", "v_spectral",
                      "d_route", "stationary"}


# ---- 13  rf-07-26 reproduces the Phase 32 result ---------------------
_RF = Path(__file__).resolve().parents[3] / "runs/review-final/2026-07-26/trip"
_GRAPH = Path(__file__).resolve().parents[3] / "runs/review-map.graphml"


@pytest.mark.skipif(not (_RF.exists() and _GRAPH.exists()),
                    reason="rf-07-26 review data not present")
def test_rf_07_26_reproduces_phase32_and_leaves_topology_identical():
    from geotrace.loader import load_trip
    from geotrace.road_graph import RoadNetwork, clip_graph, load_graph
    from geotrace.pacman_tracker.diagnostics import map_match_reference, GroundTruthObserver
    from geotrace.pacman_tracker.benchmark import _single_path_evaluation

    trip, _ = load_trip(_RF)
    first = trip.usable_locations[0]
    graph = load_graph(_GRAPH)
    net = RoadNetwork(clip_graph(graph, first.latitude, first.longitude, 11000.0),
                      LocalFrame(first.latitude, first.longitude))

    def go(display, l0=False):
        cfg = PacmanConfig(); cfg.tracker_mode = "single_path"
        cfg.display.position_branch_enabled = display
        cfg.display.leave_0726_out = l0
        # This test reproduces the frozen Phase-32 gamma=1 diagnostic. The
        # production wrapper applies a separately tested gain and safety cap.
        cfg.display.correction_gain = 1.0
        cfg.display.max_correction_m = float("inf")
        inp = build_inputs(trip, net, cfg)
        obs = GroundTruthObserver(map_match_reference(trip.reference_locations, net, net.frame))
        res = PacmanTracker(net, cfg, geometry=RoadGeometry(net, cfg.geometry)).run(inp, observer=obs)
        return cfg, inp, res

    _, inp0, base = go(False)
    _, inp, disp = go(True, l0=True)

    # topology byte-identical
    assert [s.distance_m for s in base.speed_trace] == [s.distance_m for s in disp.speed_trace]
    assert [f.top[0].route for f in base.frames] == [f.top[0].route for f in disp.frames]
    assert base.final.routes[0].edges() == disp.final.routes[0].edges()
    truth = map_match_reference(trip.reference_locations, net, net.frame)
    d0 = [(d["t_decision"], d["chosen_edge"])
          for d in _single_path_evaluation(base, truth, trip, inp0.t_start, net)["decisions"]]
    d1 = [(d["t_decision"], d["chosen_edge"])
          for d in _single_path_evaluation(disp, truth, trip, inp.t_start, net)["decisions"]]
    assert d0 == d1

    # displayed-point error reproduces the Phase 32 diagnostic (59 / 283 / 296)
    pt = disp.position_trace
    ref = [(f.monotonic_time, float(f.speed)) for f in trip.usable_locations if f.has_valid_speed]
    ref += [(f.monotonic_time, float(f.speed)) for f in trip.reference_locations
            if f.is_usable and f.has_valid_speed]
    ref.sort()
    rt = np.array([x[0] for x in ref]); rv = np.array([x[1] for x in ref])
    cum = np.cumsum(rv * float(np.median(np.diff(rt))))
    cum -= float(np.interp(inp.t_start, rt, cum))
    times = np.array([s.t for s in pt])
    d_pos = np.array([s.distance_m for s in pt]) - pt[0].distance_m
    err = np.abs(d_pos - np.interp(times, rt, cum))
    assert np.median(err) < 90.0                      # Phase 32: ~59 m
    assert np.percentile(err, 95) < 340.0             # Phase 32: ~283 m
    assert err.max() < 360.0                          # Phase 32: ~296 m
    assert np.median(err) < 0.5 * np.median(np.abs(
        np.array([s.route_distance_m for s in pt]) - pt[0].route_distance_m - np.interp(times, rt, cum)))


# --------------------------------------------------------------------------
def _artifact_path() -> str:
    return str(Path(__file__).resolve().parents[2]
               / "src/geotrace/pacman_tracker/data/display_residual_iso.json")


def _scratch() -> str:
    import tempfile
    return tempfile.mkdtemp()


class _PolyEdge:
    def __init__(self, coords: np.ndarray) -> None:
        self.coords = coords
        self.cumulative = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))])
        self.length = float(self.cumulative[-1])

    def position(self, s: float) -> tuple[float, float]:
        s = min(max(s, 0.0), self.length)
        i = int(np.searchsorted(self.cumulative, s, side="right")) - 1
        i = min(max(i, 0), len(self.coords) - 2)
        seg = self.cumulative[i + 1] - self.cumulative[i]
        frac = 0.0 if seg <= 0 else (s - self.cumulative[i]) / seg
        p0, p1 = self.coords[i], self.coords[i + 1]
        return float(p0[0] + frac * (p1[0] - p0[0])), float(p0[1] + frac * (p1[1] - p0[1]))


def _toy_edges():
    return [_PolyEdge(np.array([[0.0, 0.0], [5000.0, 0.0]]))]


def _toy_frame():
    return LocalFrame(59.9, 30.3)
