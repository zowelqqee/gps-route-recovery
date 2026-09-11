#!/usr/bin/env python3
"""Phase 35 - timestamp-correct delayed lateral anchor: A/B on the baseline
route odometer D_route (topology / branch logic / route decisions / hidden GPS
/ v_lat formula / gate / spectral update / display iso correction untouched).

    A: current production predict-then-correct (lateral_delayed_correction_enabled=False)
    B: delayed-measurement rewind/replay via the existing fixed-lag-smoother
       machinery (lateral_delayed_correction_enabled=True, lateral_anchor_delay_s=0.5)

No ``D += (v_post - v_pre) * dt`` patch anywhere: B goes through the same
cross-covariance machinery ``apply_interval`` already uses for delayed
turn-to-turn map constraints (remember / _history_at / _smooth_delayed_history).

Reports, per trip: median/p95/max |D_route-D_true|, endpoint, the four Phase 34
episode windows (rf-07-26 only), the Phase 34 sample-level nonkinematic_residual
(recomputed on D_route), lateral update counts, route decision equality,
real_wrong.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

from geotrace.coordinates import LocalFrame
from geotrace.loader import load_trip
from geotrace.road_graph import RoadNetwork, clip_graph, load_graph
from geotrace.pacman_tracker.config import PacmanConfig
from geotrace.pacman_tracker.roadmap import RoadGeometry
from geotrace.pacman_tracker.tracker import build_inputs, PacmanTracker
from geotrace.pacman_tracker.diagnostics import map_match_reference, GroundTruthObserver
from geotrace.pacman_tracker.benchmark import _single_path_evaluation

ROOT = Path("/Users/arseniyabramidze/delimobil/gps-route-recovery")
OUT = ROOT / "docs/plots/phase32"
GRAPHS = [ROOT / "runs/review-map.graphml", ROOT / "cache/spb.graphml",
          ROOT / "cache/spb-parkgolovo.graphml", ROOT / "cache/spb-center.graphml"]

EPISODES_RF0726 = [(27, 115), (212, 304), (316, 390), (436, 455)]


def d_true_series(trip, t_start, t):
    ref = [(f.monotonic_time, float(f.speed)) for f in trip.usable_locations if f.has_valid_speed]
    ref += [(f.monotonic_time, float(f.speed)) for f in trip.reference_locations
            if f.is_usable and f.has_valid_speed]
    ref.sort()
    rt = np.array([x[0] for x in ref]); rv = np.array([x[1] for x in ref])
    dt = float(np.median(np.diff(rt)))
    cum = np.cumsum(rv * dt); cum -= float(np.interp(t_start, rt, cum))
    return np.interp(t, rt, cum), np.interp(t, rt, rv)


def load_graph_for(trip_dir):
    trip, _ = load_trip(trip_dir)
    first = trip.usable_locations[0]
    for gp in GRAPHS:
        if not gp.exists():
            continue
        g = load_graph(gp)
        try:
            net = RoadNetwork(clip_graph(g, first.latitude, first.longitude, 11000.0),
                              LocalFrame(first.latitude, first.longitude))
            build_inputs(trip, net, PacmanConfig())
            return g, gp.name
        except Exception:
            continue
    return None, None


def run(trip_dir, graph, delayed: bool, delay_s: float = 0.5):
    trip, _ = load_trip(trip_dir)
    first = trip.usable_locations[0]
    net = RoadNetwork(clip_graph(graph, first.latitude, first.longitude, 11000.0),
                      LocalFrame(first.latitude, first.longitude))
    cfg = PacmanConfig()
    cfg.tracker_mode = "single_path"
    cfg.speed.lateral_delayed_correction_enabled = delayed
    cfg.speed.lateral_anchor_delay_s = delay_s
    geom = RoadGeometry(net, cfg.geometry)
    inp = build_inputs(trip, net, cfg)
    truth = map_match_reference(trip.reference_locations, net, net.frame)
    obs = GroundTruthObserver(truth)
    res = PacmanTracker(net, cfg, geometry=geom).run(inp, observer=obs)
    spev = _single_path_evaluation(res, truth, trip, inp.t_start, net)

    st = res.speed_trace
    t = np.array([s.t for s in st])
    D_route = np.array([s.distance_m for s in st]) - st[0].distance_m
    v_route = np.array([s.speed_ms for s in st])
    D_true, v_true = d_true_series(trip, inp.t_start, t)
    D_true -= D_true[0]
    e = D_route - D_true
    dt = np.diff(t, prepend=t[0] - (t[1] - t[0]) if len(t) > 1 else 1.0)
    de = np.diff(e, prepend=e[0])
    expected_de = (v_route - v_true) * dt
    nk = de - expected_de

    decisions = [(d["t_decision"], d["chosen_edge"]) for d in spev["decisions"]]
    return dict(
        trip=trip_dir.parent.name, t=t, e=e,
        median=float(np.median(np.abs(e))), p95=float(np.percentile(np.abs(e), 95)),
        mx=float(np.abs(e).max()), endpoint=float(e[-1]),
        distance_ratio=float(D_route[-1] / D_true[-1]) if D_true[-1] > 1 else None,
        speed_mae=float(np.mean(np.abs(v_route - v_true))),
        nk_sum_abs=float(np.abs(nk).sum()), nk_max_abs=float(np.abs(nk).max()),
        lateral_count=res.stats["speed"].get("lateral", 0),
        lateral_delayed_count=res.stats["speed"].get("lateral_delayed", 0),
        lateral_rejected=res.stats["speed"].get("lateral_rejected", 0),
        real_wrong=spev["real_wrong_decision_count"], n_decisions=len(spev["decisions"]),
        decisions=decisions, route_edges=res.final.routes[0].edges(),
        speed_trace_distances=[round(s.distance_m, 4) for s in st],
    )


def episode_errors(t, e, t0, windows):
    out = []
    el = t - t0
    for a, b in windows:
        m = (el >= a) & (el <= b)
        if m.sum() < 2:
            out.append((a, b, None, None)); continue
        out.append((a, b, round(float(e[m][0]), 1), round(float(e[m][-1]), 1)))
    return out


def compare(tag, trip_dir, episodes=None, check_correctness=False):
    graph, gname = load_graph_for(trip_dir)
    if graph is None:
        print(f"{tag}: no graph coverage, skipping"); return None
    A = run(trip_dir, graph, delayed=False)
    B = run(trip_dir, graph, delayed=True, delay_s=0.5)
    same_as_A = None
    print(f"\n===== {tag}  (graph {gname}) =====")
    if check_correctness:
        B0 = run(trip_dir, graph, delayed=True, delay_s=0.0)   # must equal A exactly
        same_as_A = (A["speed_trace_distances"] == B0["speed_trace_distances"])
        print(f"  correctness check (delay=0.0 behind the flag == baseline exactly): {same_as_A}")
        if not same_as_A:
            diffs = [abs(a - b) for a, b in zip(A["speed_trace_distances"], B0["speed_trace_distances"])]
            print(f"    max |diff| = {max(diffs):.6f} m  <-- should be 0.0, investigate if not")

    print(f"  {'run':>10}{'median':>9}{'p95':>9}{'max':>9}{'endpoint':>10}{'D/Dt':>8}"
          f"{'vMAE':>8}{'lat':>6}{'lat_delay':>10}{'nk_sum':>9}{'nk_max':>8}"
          f"{'decisions':>10}{'real_wrong':>11}")
    for name, r in [("A baseline", A), ("B delayed", B)]:
        dr = r["distance_ratio"] if r["distance_ratio"] is not None else float("nan")
        print(f"  {name:>10}{r['median']:>9.1f}{r['p95']:>9.1f}{r['mx']:>9.1f}{r['endpoint']:>10.1f}"
              f"{dr:>8.3f}{r['speed_mae']:>8.2f}"
              f"{r['lateral_count']:>6}{r['lateral_delayed_count']:>10}"
              f"{r['nk_sum_abs']:>9.1f}{r['nk_max_abs']:>8.2f}"
              f"{r['n_decisions']:>10}{r['real_wrong']:>11}")

    # topology = which edges get chosen (the thing that must never silently
    # break); timing = the exact tick each junction was decided at (legitimate
    # to shift, since D_route itself changed - report, don't require equal).
    edge_equal = ([x[1] for x in A["decisions"]] == [x[1] for x in B["decisions"]]
                  and A["route_edges"] == B["route_edges"])
    timing_equal = A["decisions"] == B["decisions"]
    shifts = ([abs(a[0] - b[0]) for a, b in zip(A["decisions"], B["decisions"])]
             if len(A["decisions"]) == len(B["decisions"]) else [])
    print(f"  chosen edges identical A vs B: {edge_equal}   "
          f"decision timing identical: {timing_equal}   "
          f"max commit-time shift: {max(shifts, default=float('nan')):.3f}s   "
          f"(A {len(A['decisions'])} decisions, B {len(B['decisions'])})")

    if episodes:
        epA = episode_errors(A["t"], A["e"], A["t"][0], episodes)
        epB = episode_errors(B["t"], B["e"], B["t"][0], episodes)
        print(f"  {'episode':>14}{'A start':>9}{'A end':>9}{'B start':>9}{'B end':>9}")
        for (a, b, sa, ea), (_, _, sb, eb) in zip(epA, epB):
            print(f"  el {a:>4}-{b:<5}{sa!s:>9}{ea!s:>9}{sb!s:>9}{eb!s:>9}")

    return dict(tag=tag, correctness_ok=same_as_A, edge_equal=edge_equal,
               timing_equal=timing_equal,
               max_commit_shift_s=max(shifts, default=None),
               A={k: v for k, v in A.items() if k not in ("t", "e", "decisions", "route_edges",
                                                           "speed_trace_distances")},
               B={k: v for k, v in B.items() if k not in ("t", "e", "decisions", "route_edges",
                                                           "speed_trace_distances")})


def main():
    results = []
    r = compare("rf-07-26", ROOT / "runs/review-final/2026-07-26/trip", EPISODES_RF0726,
               check_correctness=True)
    if r: results.append(r)
    for tag, path in [("07-22", ROOT / "runs/review-final/2026-07-22/trip"),
                      ("07-23", ROOT / "runs/review-final/2026-07-23/trip"),
                      ("07-24", ROOT / "runs/review-final/2026-07-24/trip"),
                      ("07-25", ROOT / "runs/review-final/2026-07-25/trip"),
                      ("07-22-s0", ROOT / "runs/phase31/07-22-s0/trip"),
                      ("07-26-s1", ROOT / "runs/phase31/07-26-s1/trip"),
                      ("07-26-s3", ROOT / "runs/phase31/07-26-s3/trip"),
                      ("07-24-s7", ROOT / "runs/phase31/07-24-s7/trip")]:
        r = compare(tag, path)
        if r: results.append(r)

    print("\n\n================ SUMMARY ================")
    print(f"{'trip':>10}{'A max':>8}{'B max':>8}{'dMax':>8}{'A p95':>8}{'B p95':>8}"
          f"{'A end':>8}{'B end':>8}{'edges=':>8}{'A rw':>6}{'B rw':>6}")
    for r in results:
        A, B = r["A"], r["B"]
        print(f"{r['tag']:>10}{A['mx']:>8.0f}{B['mx']:>8.0f}{B['mx']-A['mx']:>+8.0f}"
              f"{A['p95']:>8.0f}{B['p95']:>8.0f}{A['endpoint']:>8.0f}{B['endpoint']:>8.0f}"
              f"{str(r['edge_equal']):>8}{A['real_wrong']:>6}{B['real_wrong']:>6}")

    (OUT / "phase35_results.json").write_text(json.dumps(results, indent=1, default=str))
    print(f"\nwrote {OUT/'phase35_results.json'}")


if __name__ == "__main__":
    sys.exit(main())
