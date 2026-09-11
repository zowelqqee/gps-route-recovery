"""Road-curvature bend anchors wired into single_path (Phase 17).

Everything here is gated by ``single_path.bend_*`` / ``speed.bend_scale_enabled``
flags that are OFF by default, so the production baseline is unchanged.
"""
import inspect
import math

import numpy as np
import pytest

from geotrace.pacman_tracker.config import PacmanConfig
from geotrace.pacman_tracker.roadmap import RoadGeometry
from geotrace.pacman_tracker.single_path import SinglePathManager
from geotrace.pacman_tracker.speed import GlobalSpeedTracker, SpeedConfig, V
from geotrace.pacman_tracker.state import RouteNode, make_set
from geotrace.pacman_tracker.synthetic import arc, build_network, straight
from geotrace.pacman_tracker.turns import HeadingIntegrator


CURVE_TURN = math.radians(75.0)
CURVE_BEND_LEN = 260.0                                # arc length of the bend
CURVE_LEAD = 110.0


def _eased_turn(total_turn, length, start, heading0, step=4.0):
    """Points for a smooth (raised-cosine) heading change - a real curvature
    peak in the middle, unlike a constant-radius arc."""
    n = max(4, int(length / step) + 1)
    s = np.linspace(0.0, length, n)
    u = s / length
    psi = heading0 + total_turn * (u - np.sin(2 * np.pi * u) / (2 * np.pi))
    x = start[0] + np.concatenate([[0.0], np.cumsum(np.cos(psi[:-1]) * np.diff(s))])
    y = start[1] + np.concatenate([[0.0], np.cumsum(np.sin(psi[:-1]) * np.diff(s))])
    return list(zip(x.tolist(), y.tolist()))


def _bend_network():
    """in --> [curve edge: straight lead-in, smooth 75 deg bend, straight
    lead-out] --> out. The bend is inside one edge's polyline - no node at it."""
    lead = straight(CURVE_LEAD, start=(160.0, 0.0), heading_rad=0.0)
    bend = _eased_turn(CURVE_TURN, CURVE_BEND_LEN, lead[-1], 0.0)
    tail = straight(110.0, start=bend[-1], heading_rad=CURVE_TURN)
    curve_pts = lead + bend[1:] + tail[1:]
    ways = [
        ("in", straight(160.0, start=(0.0, 0.0), heading_rad=0.0), {}),
        ("curve", curve_pts, {}),
        ("out", straight(120.0, start=curve_pts[-1], heading_rad=CURVE_TURN), {}),
    ]
    return build_network(ways)


def _run(cfg, speed=15.0, dt=0.1, dur=70.0):
    net, _ = _bend_network()
    geo = RoadGeometry(net, cfg.geometry)
    names = {e.name: e.index for e in net.edges}
    times = np.arange(0.0, dur, dt)
    # curve edge's bend is CURVE_LEAD into the edge and spans CURVE_BEND_LEN.
    # Follow the eased (raised-cosine) heading so the yaw rate has a real peak.
    t_bend0 = (160.0 + CURVE_LEAD) / speed
    t_bend1 = t_bend0 + CURVE_BEND_LEN / speed
    tt = np.clip((times - t_bend0) / (t_bend1 - t_bend0), 0.0, 1.0)
    psi = CURVE_TURN * (tt - np.sin(2 * np.pi * tt) / (2 * np.pi))
    rates = np.gradient(psi, dt)
    rates[(times < t_bend0) | (times > t_bend1)] = 0.0
    m = SinglePathManager(geo, cfg.beam, cfg.single_path,
                          HeadingIntegrator(times, rates))
    hs = make_set([names["in"]], [0.0], [0.0],
                  [RouteNode.root(names["in"], 0.0, 0.0)], 0.0)
    for t in times:
        hs = m.advance(hs, speed * float(t), float(t), speed, 8.0)
    return m, names, geo, hs


def test_curve_inside_an_edge_is_classified_as_a_bend_not_a_junction():
    cfg = PacmanConfig()
    cfg.single_path.bend_classification_enabled = True
    m, names, geo, hs = _run(cfg)
    assert m.diagnostics()["bend_events"] >= 1
    bends = m.bend_anchors
    assert len(bends) == 1
    assert bends[0]["edge"] == names["curve"]
    assert bends[0]["uniqueness_margin"] >= cfg.single_path.bend_min_uniqueness_margin
    assert abs(bends[0]["v_bar_ms"] - 15.0) < 3.0
    # the physical turn is consumed -> it drove no junction decision, and no
    # turn-to-turn interval is anchored on a curvature event (Phase 22: this is
    # what stops a fake short interval like the old 07-26 3367 m candidate).
    bend_turn = next(t for t in m.turns if t.is_bend)
    assert bend_turn.consumed
    assert all(d.event_id != bend_turn.id for d in m.decisions)
    assert all(iv["event_a_id"] != bend_turn.id and iv["event_b_id"] != bend_turn.id
               for iv in m.turn_intervals)


def test_classification_off_by_default_leaves_the_event_for_the_junction_matcher():
    cfg = PacmanConfig()               # bend_classification_enabled defaults False
    m, names, geo, hs = _run(cfg)
    assert m.diagnostics()["bend_events"] == 0
    assert m.bend_anchors == []


def test_bend_match_in_the_manager_uses_no_distance_prior():
    src = inspect.getsource(SinglePathManager._classify_bends)
    # the only inputs to the matcher are geometry + gyro + t_peak
    assert "match_curvature_event(" in src
    assert "distance" not in src.split("match_curvature_event(")[1].split(")")[0]


def test_bend_speed_anchor_calibrates_k_high_not_k_s_or_k_a():
    cfg = SpeedConfig(bend_scale_enabled=True, spectral_scale_enabled=True,
                      accel_scale_enabled=True)
    tr = GlobalSpeedTracker(cfg, v0=15.0)
    ks0, ka0 = tr.spectral_scale, tr.x[4]
    ok = tr.bend_speed_anchor(v_bend=17.5, sigma_v_bend=0.7,
                              v_spectral_now=14.0, t=100.0)
    assert ok
    assert 1.15 < tr.bend_scale < 1.30                 # toward 17.5/14 = 1.25
    assert tr.bend_scale_var < cfg.initial_bend_scale_sigma ** 2
    assert tr.spectral_scale == pytest.approx(ks0)     # k_s untouched
    assert tr.x[4] == pytest.approx(ka0)               # k_a untouched


def test_k_high_stays_wide_and_at_one_without_a_bend_anchor():
    cfg = SpeedConfig(bend_scale_enabled=True, bend_scale_rw=0.002)
    tr = GlobalSpeedTracker(cfg, v0=15.0)
    for _ in range(4000):
        tr.predict(0.0, 0.1)
        tr.spectral_update(13.0, 2.0, 0.1)            # ordinary high-speed obs
    assert tr.bend_scale == pytest.approx(1.0)
    assert tr.sigma_bend_scale > 0.29                  # never collapsed


def test_ordinary_spectral_update_cannot_move_k_high():
    src = inspect.getsource(GlobalSpeedTracker.spectral_update)
    assert "bend_speed_anchor" not in src
    assert "bend_scale =" not in src                   # never assigns the state


def test_k_high_persists_through_a_stop_and_a_long_quiet_stretch():
    """Phase 20: a measured spectral calibration is CALIBRATION state, not speed
    state. Neither a ZUPT nor elapsed time may reset it toward 1 - only its
    uncertainty grows. On the review trips a stop at a light does not end the
    high-speed spectral regime, and resetting there costs D/D_true."""
    cfg = SpeedConfig(bend_scale_enabled=True, spectral_scale_enabled=True,
                      bend_scale_rw=0.002)
    tr = GlobalSpeedTracker(cfg, v0=17.0)
    tr.bend_speed_anchor(17.5, 0.7, 14.0, t=100.0)
    k_anchored = tr.bend_scale
    s_anchored = tr.bend_scale_var
    assert k_anchored > 1.1
    for i in range(12000):                              # 1200 s
        tr.predict(0.0, 0.1)
        tr.spectral_update(13.0, 2.0, 0.1)
        if 6000 <= i < 6300:                            # a 30 s stop in the middle
            tr.zero_velocity(0.0, 0.1, 0.0, run_s=1e9)
    assert tr.bend_scale == pytest.approx(k_anchored)   # value untouched
    assert tr.bend_scale_var > s_anchored               # only sigma grew
    assert tr.sigma_bend_scale < cfg.initial_bend_scale_sigma  # slowly, not reset


def test_zupt_does_not_touch_the_calibration_state():
    src = inspect.getsource(GlobalSpeedTracker.zero_velocity)
    assert "bend_scale" not in src


def test_lateral_anchor_cannot_collapse_k_high_uncertainty():
    cfg = SpeedConfig(bend_scale_enabled=True, spectral_scale_enabled=True,
                      lateral_enabled=True)
    tr = GlobalSpeedTracker(cfg, v0=8.0)
    v0 = tr.bend_scale_var
    for _ in range(50):
        tr.predict(0.0, 0.1)
        tr.lateral_anchor(2.0, 0.30, 0.1, spectral_speed=9.0)   # ~6.7 m/s turn
    assert tr.bend_scale_var >= v0 * 0.999             # only grows (random walk)


def test_regime_weight_uses_only_the_spectral_output():
    cfg = SpeedConfig(bend_scale_enabled=True,
                      bend_scale_regime_lo_ms=9.0, bend_scale_regime_hi_ms=14.0)
    tr = GlobalSpeedTracker(cfg, v0=0.0)
    assert tr._regime_weight(6.0) == 0.0
    assert tr._regime_weight(14.0) == 1.0
    assert tr._regime_weight(11.5) == pytest.approx(0.5)


def test_spectral_update_blends_k_low_and_k_high_by_regime():
    cfg = SpeedConfig(bend_scale_enabled=True, spectral_scale_enabled=True,
                      spectral_correlation_s=0.0)
    tr = GlobalSpeedTracker(cfg, v0=0.0)
    tr.spectral_scale = 1.0
    tr.bend_scale = 1.30
    # low-speed obs: ~all k_low -> corrected ~= raw
    tr.x[V] = 0.0
    tr.spectral_update(5.0, 0.1, 0.1)
    assert tr.x[V] < 6.0
    # high-speed obs: ~all k_high -> corrected ~= 1.30 * raw
    tr2 = GlobalSpeedTracker(cfg, v0=0.0)
    tr2.spectral_scale = 1.0
    tr2.bend_scale = 1.30
    tr2.x[V] = 20.0
    tr2.spectral_update(20.0, 0.1, 0.1)
    assert tr2.x[V] > 20.0                              # pulled up toward 26


def test_no_hidden_gps_tokens_in_the_bend_path():
    for obj in (SinglePathManager._classify_bends,
                SinglePathManager._junction_gap_deg,
                GlobalSpeedTracker.bend_speed_anchor,
                GlobalSpeedTracker.spectral_update):
        src = inspect.getsource(obj)
        for banned in ("reference_location", "withheld", "oracle", "truth",
                       "map_match"):
            assert banned not in src


# ----------------------------------------------------------- Phase 23: censoring

def _sat_tracker(censor=True):
    cfg = SpeedConfig(bend_scale_enabled=True, spectral_censor_enabled=censor,
                      spectral_saturation_guard_enabled=False,
                      spectral_censor_factor=15.0, spectral_correlation_s=0.0)
    return GlobalSpeedTracker(cfg, v0=17.0)


def _pump(tr, v_spec, n=40):
    for _ in range(n):
        tr.predict(0.0, 0.1)
        tr.spectral_update(v_spec, 2.0, 0.1)


def test_saturation_not_confirmed_without_a_faster_than_spectral_bend():
    tr = _sat_tracker()
    tr.bend_speed_anchor(v_bend=12.0, sigma_v_bend=0.7, v_spectral_now=13.0, t=1.0)
    assert not tr.spectral_saturation_known          # bend slower than spectral: no


def test_bend_faster_than_spectral_confirms_saturation():
    tr = _sat_tracker()
    tr.bend_speed_anchor(v_bend=17.5, sigma_v_bend=0.7, v_spectral_now=13.0, t=1.0)
    assert tr.spectral_saturation_known
    assert tr.spectral_plateau_ms == pytest.approx(13.0)


def test_downward_spectral_pull_near_the_plateau_is_down_weighted():
    tr = _sat_tracker()
    tr.bend_speed_anchor(17.5, 0.7, 13.0, t=1.0)
    tr.bend_scale = 1.0                        # isolate the censoring effect
    tr.x[V] = 19.0
    n0 = tr.spectral_censored_updates
    _pump(tr, 13.0)                            # 13 near plateau, would drag down
    assert tr.spectral_censored_updates > n0
    assert tr.x[V] > 16.0                      # mostly held near 19

    ref = _sat_tracker(censor=False)
    ref.bend_speed_anchor(17.5, 0.7, 13.0, t=1.0)
    ref.bend_scale = 1.0
    ref.x[V] = 19.0
    _pump(ref, 13.0)
    assert ref.x[V] < 14.5                     # uncensored: dragged well down
    assert tr.x[V] - ref.x[V] > 1.5            # censoring clearly holds v higher


def test_upward_spectral_pull_is_not_censored():
    tr = _sat_tracker()
    tr.bend_speed_anchor(17.5, 0.7, 13.0, t=1.0)
    tr.bend_scale = 1.0
    tr.x[V] = 9.0
    _pump(tr, 13.0)                            # 13 > 9: lifts v toward plateau
    assert tr.spectral_censored_updates == 0
    assert tr.x[V] > 11.5


def test_reading_above_the_plateau_band_is_trusted_in_full():
    # the right-skewed noise tail carries real high-speed info, so a high
    # reading above the band updates normally (does pull v toward it).
    tr = _sat_tracker()
    tr.bend_speed_anchor(17.5, 0.7, 13.0, t=1.0)
    tr.bend_scale = 1.0
    tr.x[V] = 22.0
    _pump(tr, 18.0)                            # 18 > plateau + band(1.5)
    assert tr.spectral_censored_updates == 0
    assert tr.x[V] < 20.5                      # pulled down toward the reading


def test_censoring_is_off_by_default():
    cfg = SpeedConfig(bend_scale_enabled=True, spectral_correlation_s=0.0,
                      spectral_saturation_guard_enabled=False)
    assert cfg.spectral_censor_enabled is False
    tr = GlobalSpeedTracker(cfg, v0=17.0)
    tr.bend_speed_anchor(17.5, 0.7, 13.0, t=1.0)
    tr.bend_scale = 1.0
    tr.x[V] = 19.0
    _pump(tr, 13.0)
    assert tr.spectral_censored_updates == 0
    assert tr.x[V] < 15.0                      # ordinary update drags v down


def test_censoring_needs_a_confirmed_saturation():
    tr = _sat_tracker()                        # enabled, but no bend anchor yet
    tr.bend_scale = 1.0
    tr.x[V] = 19.0
    _pump(tr, 13.0)
    assert tr.spectral_censored_updates == 0
    assert tr.x[V] < 15.0
