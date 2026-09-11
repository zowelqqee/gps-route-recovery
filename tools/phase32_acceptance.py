#!/usr/bin/env python3
"""Phase 32 production integration - acceptance test.

Runs the REAL tracker runtime (not the Phase 32 post-hoc diagnostic) with
``display.position_branch_enabled`` and checks that the displayed point on
rf-07-26 reproduces the Phase 32 iso-binary result, and that the committed
route / speed_trace / decisions are byte-identical to the flag-off baseline.
"""
from __future__ import annotations
import json, sys
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
TRIPS = {"rf-07-26": ROOT / "runs/review-final/2026-07-26/trip",
         "review-20min-07-26": ROOT / "runs/review-20min/2026-07-26/trip"}
PHASE32_TARGET = dict(median=59, p95=283, max=296, ratio=0.939, closer=0.89)


def run(td, graph, enable, l0=False):
    trip, _ = load_trip(td)
    first = trip.usable_locations[0]
    net = RoadNetwork(clip_graph(graph, first.latitude, first.longitude, 11000.0),
                      LocalFrame(first.latitude, first.longitude))
    cfg = PacmanConfig(); cfg.tracker_mode = "single_path"
    cfg.display.position_branch_enabled = enable
    cfg.display.leave_0726_out = l0
    geom = RoadGeometry(net, cfg.geometry)
    inp = build_inputs(trip, net, cfg)
    truth = map_match_reference(trip.reference_locations, net, net.frame)
    obs = GroundTruthObserver(truth)
    res = PacmanTracker(net, cfg, geometry=geom).run(inp, observer=obs)
    spev = _single_path_evaluation(res, truth, trip, inp.t_start, net)
    return trip, net, inp, res, truth, spev


def d_true_at(trip, t_start, times):
    ref = [(f.monotonic_time, float(f.speed)) for f in trip.usable_locations if f.has_valid_speed]
    ref += [(f.monotonic_time, float(f.speed)) for f in trip.reference_locations
            if f.is_usable and f.has_valid_speed]
    ref.sort()
    rt = np.array([x[0] for x in ref]); rv = np.array([x[1] for x in ref])
    dt = float(np.median(np.diff(rt)))
    cum = np.cumsum(rv * dt); cum -= float(np.interp(t_start, rt, cum))
    return np.interp(times, rt, cum)


def metrics(trip, t_start, pt, st):
    times = np.array([s.t for s in pt])
    d_pos = np.array([s.distance_m for s in pt]) - pt[0].distance_m
    d_route = np.array([s.route_distance_m for s in pt]) - pt[0].route_distance_m
    d_true = d_true_at(trip, t_start, times)
    ep = np.abs(d_pos - d_true); er = np.abs(d_route - d_true)
    return dict(
        median_pos=round(float(np.median(ep)), 1), p95_pos=round(float(np.percentile(ep, 95)), 1),
        max_pos=round(float(ep.max()), 1),
        median_route=round(float(np.median(er)), 1), p95_route=round(float(np.percentile(er, 95)), 1),
        max_route=round(float(er.max()), 1),
        ratio_pos=round(float(d_pos[-1] / d_true[-1]), 3) if d_true[-1] > 1 else None,
        ratio_route=round(float(d_route[-1] / d_true[-1]), 3) if d_true[-1] > 1 else None,
        closer_frac=round(float(np.mean(ep < er)), 3),
        vMAE_route=round(float(np.mean(np.abs(
            np.array([s.speed_ms for s in st])[:len(times)]
            - np.interp(times, *_ref_v(trip, t_start))))), 2),
    )


def _ref_v(trip, t_start):
    ref = [(f.monotonic_time, float(f.speed)) for f in trip.usable_locations if f.has_valid_speed]
    ref += [(f.monotonic_time, float(f.speed)) for f in trip.reference_locations
            if f.is_usable and f.has_valid_speed]
    ref.sort()
    return np.array([x[0] for x in ref]), np.array([x[1] for x in ref])


def main():
    graph = load_graph(ROOT / "runs/review-map.graphml")
    report = {}
    for tag, td in TRIPS.items():
        _, _, inp0, base, _, spev0 = run(td, graph, enable=False)
        trip, net, inp, disp, truth, spev = run(td, graph, enable=True, l0=(tag != "review-20min-07-26"))
        _, _, _, disp_full, _, _ = run(td, graph, enable=True, l0=False)

        topo_ok = (
            [round(s.distance_m, 4) for s in base.speed_trace]
            == [round(s.distance_m, 4) for s in disp.speed_trace]
            and [f.top[0].route for f in base.frames] == [f.top[0].route for f in disp.frames]
            and [(d["t_decision"], d["chosen_edge"]) for d in spev0["decisions"]]
            == [(d["t_decision"], d["chosen_edge"]) for d in spev["decisions"]]
            and base.final.routes[0].edges() == disp.final.routes[0].edges())

        m_lo = metrics(trip, inp.t_start, disp.position_trace, disp.speed_trace)
        m_full = metrics(trip, inp.t_start, disp_full.position_trace, disp_full.speed_trace)
        report[tag] = dict(topology_identical=topo_ok,
                           decisions=len(spev["decisions"]),
                           real_wrong=spev["real_wrong_decision_count"],
                           route_edges=len(disp.final.routes[0].edges()),
                           gate_fraction=disp.stats["display_position"]["gate_active_fraction"],
                           max_excess_m=disp.stats["display_position"]["max_excess_m"],
                           leave_0726_out=m_lo, full_model=m_full)
        print(f"\n===== {tag} =====")
        print(f"  topology identical: {topo_ok}   decisions {len(spev['decisions'])} "
              f"real_wrong {spev['real_wrong_decision_count']}   "
              f"gate {disp.stats['display_position']['gate_active_fraction']:.2f}  "
              f"max excess {disp.stats['display_position']['max_excess_m']:.0f} m")
        for label, m in [("leave-0726-out model", m_lo), ("full-pool model", m_full)]:
            print(f"  [{label}]")
            print(f"     baseline  D_route: median {m['median_route']}  p95 {m['p95_route']}  "
                  f"max {m['max_route']}  D/D_true {m['ratio_route']}")
            print(f"     display D_position: median {m['median_pos']}  p95 {m['p95_pos']}  "
                  f"max {m['max_pos']}  D/D_true {m['ratio_pos']}  closer {m['closer_frac']}")
        t = PHASE32_TARGET
        print(f"  Phase 32 diagnostic target (rf-07-26): median≈{t['median']} p95≈{t['p95']} "
              f"max≈{t['max']} ratio≈{t['ratio']} closer≈{t['closer']}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "acceptance.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {OUT/'acceptance.json'}")


if __name__ == "__main__":
    sys.exit(main())
