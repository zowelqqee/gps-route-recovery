"""Road-curvature intra-edge position anchor (prototype, not wired into the
tracker). See docs/SPECTRAL_CALIBRATION_FORENSICS.md Phase 17."""
import math

import numpy as np
import pytest

from geotrace.pacman_tracker.bend_anchor import (
    BendMatch, classify_event, match_curvature_event)


def _arc_edge(length, total_turn_rad, bend_lo, bend_hi, ds=2.0):
    """Straight, then a smooth (raised-cosine) turn of `total_turn_rad` between
    bend_lo..bend_hi so the curvature has a real peak at the centre, then
    straight again."""
    s = np.arange(0.0, length + ds, ds)
    psi = np.full_like(s, 0.0)
    seg = (s >= bend_lo) & (s <= bend_hi)
    u = (s[seg] - bend_lo) / (bend_hi - bend_lo)
    psi[seg] = total_turn_rad * (u - np.sin(2 * np.pi * u) / (2 * np.pi))
    psi[s > bend_hi] = total_turn_rad
    # real OSM polylines are never dead straight; a few tenths of a degree of
    # wiggle elsewhere is what lets the shape match localise (a perfectly
    # straight edge leaves the along-edge position on a broad plateau).
    psi = psi + math.radians(0.6) * np.sin(s / 47.0)
    return s, psi


def _gyro_from_drive(s_of_t, edge_s, edge_psi, dt=0.1, noise=0.0, seed=0):
    """Yaw rate a driver would record following edge_psi at along-edge path
    s_of_t (array, one per time step)."""
    t = np.arange(len(s_of_t)) * dt
    psi_t = np.interp(s_of_t, edge_s, edge_psi)
    rate = np.gradient(psi_t, t)
    if noise:
        rate = rate + np.random.default_rng(seed).normal(0, noise, rate.size)
    return t, rate


def test_smooth_bend_on_one_edge_is_matched_at_its_true_position():
    L, turn = 1000.0, math.radians(45.0)
    es, epsi = _arc_edge(L, turn, 650.0, 790.0)
    v = 17.0
    s_of_t = np.arange(610.0, 850.0, v * 0.1)          # drives 560 -> 900 m
    gt, gr = _gyro_from_drive(s_of_t, es, epsi)
    t_peak = gt[int(np.argmax(np.abs(gr)))]
    m = match_curvature_event(es, epsi, L, gt, gr, t_peak)
    assert m is not None
    assert 640.0 < m.s_peak < 820.0                   # inside the bend span
    assert abs(m.s_peak - 725.0) < 60.0               # near the centre ~725
    assert m.angle_resid_rad < math.radians(3.0)
    assert m.shape_corr > 0.9
    assert abs(m.v_bar - v) < 3.0                     # recovers the drive speed
    assert abs(m.map_delta_psi - turn) < math.radians(6.0)


def test_no_distance_prior_is_used():
    import inspect
    sig = inspect.signature(match_curvature_event)
    assert not any("d" == p or "prior" in p or "s_est" in p for p in sig.parameters)


def test_repeated_bends_on_one_edge_are_rejected_as_not_unique():
    # an edge with three identical evenly-spaced left arcs: the event shape fits
    # any of them, so no single along-edge position is recoverable.
    L = 1800.0
    es = np.arange(0.0, L + 2.0, 2.0)
    epsi = np.zeros_like(es)
    for lo, hi in ((250.0, 400.0), (800.0, 950.0), (1350.0, 1500.0)):
        seg = (es >= lo) & (es <= hi)
        u = (es[seg] - lo) / (hi - lo)
        epsi[seg] += math.radians(40.0) * (u - np.sin(2 * np.pi * u) / (2 * np.pi))
        epsi[es > hi] += math.radians(40.0)
    v = 15.0
    s_of_t = np.arange(210.0, 440.0, v * 0.1)          # drives through arc 1
    gt, gr = _gyro_from_drive(s_of_t, es, epsi)
    m = match_curvature_event(es, epsi, L, gt, gr, gt[len(gt) // 2])
    assert m is None                                   # ambiguous -> no anchor


def test_high_min_margin_rejects_any_match():
    L, turn = 1000.0, math.radians(45.0)
    es, epsi = _arc_edge(L, turn, 650.0, 790.0)
    s_of_t = np.arange(610.0, 850.0, 1.7)
    gt, gr = _gyro_from_drive(s_of_t, es, epsi)
    assert match_curvature_event(es, epsi, L, gt, gr, gt[len(gt) // 2],
                                 min_margin=999.0) is None


def test_one_event_yields_at_most_one_match():
    L, turn = 900.0, math.radians(50.0)
    es, epsi = _arc_edge(L, turn, 400.0, 650.0)
    s_of_t = np.arange(360.0, 700.0, 1.6)
    gt, gr = _gyro_from_drive(s_of_t, es, epsi)
    m = match_curvature_event(es, epsi, L, gt, gr, gt[len(gt) // 2])
    assert m is None or isinstance(m, BendMatch)


def test_sharp_compact_turn_on_a_straight_edge_is_not_a_bend():
    L = 800.0
    es = np.arange(0.0, L + 2.0, 2.0)
    epsi = np.zeros_like(es)                            # perfectly straight edge
    dt = 0.1
    t = np.arange(0, 5.0, dt)
    rate = np.exp(-((t - 2.5) ** 2) / (2 * 0.5 ** 2)) * 0.6   # peak 0.6 rad/s
    m = match_curvature_event(es, epsi, L, t, rate, 2.5)
    assert m is None                                   # straight edge, no bend
    cls = classify_event(delta_psi_rad=float(np.sum(rate[:-1] * np.diff(t))),
                         peak_rate_rads=0.6, duration_s=5.0, bend=m,
                         junction_angle_gap_deg=5.0)
    assert cls == "junction"


def test_classifier_calls_the_spread_low_rate_event_a_bend():
    fake = BendMatch(s_start=600, s_peak=730, s_end=900, v_bar=17.0,
                     map_delta_psi=math.radians(48), gyro_delta_psi=math.radians(47),
                     angle_resid_rad=math.radians(1.0), shape_corr=0.8,
                     margin=1.5, n_local_maxima=1)
    cls = classify_event(delta_psi_rad=math.radians(42), peak_rate_rads=0.10,
                         duration_s=11.0, bend=fake, junction_angle_gap_deg=40.0)
    assert cls == "bend"


def test_classifier_reject_when_neither_model_fits():
    cls = classify_event(delta_psi_rad=math.radians(90), peak_rate_rads=0.12,
                         duration_s=4.0, bend=None, junction_angle_gap_deg=35.0)
    assert cls == "reject"


def test_bend_match_carries_a_uniqueness_margin():
    L, turn = 1000.0, math.radians(45.0)
    es, epsi = _arc_edge(L, turn, 650.0, 790.0)
    s_of_t = np.arange(610.0, 850.0, 1.7)
    gt, gr = _gyro_from_drive(s_of_t, es, epsi)
    m = match_curvature_event(es, epsi, L, gt, gr, gt[len(gt) // 2])
    assert m is not None and m.margin >= 0.7 and m.n_local_maxima == 1
