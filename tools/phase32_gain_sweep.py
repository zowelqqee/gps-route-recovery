#!/usr/bin/env python3
"""rf-07-26 pure display-correction GAIN sweep.

Byte-identical to production Phase 32 in every respect - topology, v_route, gate
timings, the frozen isotonic curve, gate thresholds, frontier logic - EXCEPT the
per-step display velocity correction is scaled by a global gain gamma:

    delta_v = gamma * gate * max(0, v_ml - v_route)

gamma in {1.0, 1.25, 1.5, 1.75, 2.0}. Diagnostic only; nothing is retrained or
tuned. The tracker still never sees D_position.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geotrace.coordinates import LocalFrame
from geotrace.loader import load_trip
from geotrace.road_graph import RoadNetwork, clip_graph, load_graph
from geotrace.pacman_tracker.config import PacmanConfig
from geotrace.pacman_tracker.roadmap import RoadGeometry
from geotrace.pacman_tracker.tracker import build_inputs, PacmanTracker
from geotrace.pacman_tracker.diagnostics import map_match_reference, GroundTruthObserver
from geotrace.pacman_tracker.benchmark import _single_path_evaluation
import geotrace.pacman_tracker.display_position as dpm

ROOT = Path("/Users/arseniyabramidze/delimobil/gps-route-recovery")
OUT = ROOT / "docs/plots/phase32"
TRIP = ROOT / "runs/review-final/2026-07-26/trip"
GRAPH = ROOT / "runs/review-map.graphml"
GAMMAS = [1.0, 1.25, 1.5, 1.75, 2.0]


def d_true_series(trip, t_start, t):
    ref = [(f.monotonic_time, float(f.speed)) for f in trip.usable_locations if f.has_valid_speed]
    ref += [(f.monotonic_time, float(f.speed)) for f in trip.reference_locations
            if f.is_usable and f.has_valid_speed]
    ref.sort()
    rt = np.array([x[0] for x in ref]); rv = np.array([x[1] for x in ref])
    dt = float(np.median(np.diff(rt)))
    cum = np.cumsum(rv * dt); cum -= float(np.interp(t_start, rt, cum))
    return np.interp(t, rt, cum), np.interp(t, rt, rv)


def run(gamma: float, leave_0726_out: bool = True):
    trip, _ = load_trip(TRIP)
    first = trip.usable_locations[0]
    net = RoadNetwork(clip_graph(load_graph(GRAPH), first.latitude, first.longitude, 11000.0),
                      LocalFrame(first.latitude, first.longitude))
    cfg = PacmanConfig()
    cfg.tracker_mode = "single_path"
    cfg.display.position_branch_enabled = True
    cfg.display.leave_0726_out = leave_0726_out

    real_step = dpm.DisplayPositionBranch.step
    steplog: list[tuple[float, bool]] = []

    def scaled_step(self, t, dt, v_route, v_prior, v_spectral, d_route, stationary):
        d0 = self._delta
        real_step(self, t, dt, v_route, v_prior, v_spectral, d_route, stationary)
        inc = self._delta - d0                       # dv*dt applied at gamma = 1
        self._delta = d0 + gamma * inc               # scale ONLY the correction increment
        self._v_position = float(v_route) + gamma * (self._v_position - float(v_route))
        steplog.append((float(t), bool(self._gate)))

    dpm.DisplayPositionBranch.step = scaled_step
    try:
        inp = build_inputs(trip, net, cfg)
        obs = GroundTruthObserver(map_match_reference(trip.reference_locations, net, net.frame))
        res = PacmanTracker(net, cfg, geometry=RoadGeometry(net, cfg.geometry)).run(inp, observer=obs)
    finally:
        dpm.DisplayPositionBranch.step = real_step

    truth = map_match_reference(trip.reference_locations, net, net.frame)
    spev = _single_path_evaluation(res, truth, trip, inp.t_start, net)

    pt = res.position_trace
    t = np.array([s.t for s in pt])
    D_pos = np.array([s.distance_m for s in pt])
    D_route = np.array([s.route_distance_m for s in pt])
    D_true, _ = d_true_series(trip, inp.t_start, t)
    D_pos -= D_pos[0]; D_route -= D_route[0]; D_true -= D_true[0]
    e = D_pos - D_true

    # gate-ON episodes on the raw step timeline, mapped to tick series
    st = np.array([x[0] for x in steplog]); sg = np.array([x[1] for x in steplog])
    episodes = []
    i = 0
    while i < len(sg):
        if not sg[i]:
            i += 1; continue
        j = i
        while j < len(sg) and sg[j]:
            j += 1
        a, b = st[i], st[j - 1]
        m = (t >= a) & (t <= b)
        if m.sum() >= 2:
            ee = e[m]
            episodes.append((round(a - inp.t_start, 1), round(b - inp.t_start, 1),
                             round(float(ee[0]), 1), round(float(ee.min()), 1), round(float(ee[-1]), 1)))
        i = j

    return dict(
        gamma=gamma,
        median=float(np.median(np.abs(e))), p95=float(np.percentile(np.abs(e), 95)),
        mx=float(np.abs(e).max()), max_lead=float(e.max()), max_lag=float(e.min()),
        endpoint=float(e[-1]),
        Dratio=float(D_pos[-1] / D_true[-1]) if D_true[-1] > 1 else None,
        episodes=episodes, real_wrong=spev["real_wrong_decision_count"],
        decisions=len(spev["decisions"]),
        t=t - t[0], e=e, D_route=D_route, D_true=D_true, gate_steps=(st - inp.t_start, sg),
    )


def main():
    rows = [run(g) for g in GAMMAS]
    base = rows[0]
    assert all(r["real_wrong"] == base["real_wrong"] and r["decisions"] == base["decisions"]
               for r in rows), "topology changed - impossible, aborting"

    print("=== rf-07-26 display-correction gain sweep (leave_0726_out model) ===")
    print("topology identical for every gamma: decisions "
          f"{base['decisions']}, real_wrong {base['real_wrong']}\n")
    print(f"{'gamma':>6}{'median':>9}{'p95':>9}{'max':>9}{'max_lead':>10}{'max_lag':>10}"
          f"{'endpoint':>10}{'D/D_true':>10}")
    for r in rows:
        print(f"{r['gamma']:>6.2f}{r['median']:>9.1f}{r['p95']:>9.1f}{r['mx']:>9.1f}"
              f"{r['max_lead']:>10.1f}{r['max_lag']:>10.1f}{r['endpoint']:>10.1f}"
              f"{(r['Dratio'] or 0):>10.3f}")

    n_ep = max(len(r["episodes"]) for r in rows)
    print("\ngate-ON episodes:  D_position - D_true  as  start / min / end  [m]   (episode window shared across gamma)")
    for k in range(n_ep):
        w = base["episodes"][k] if k < len(base["episodes"]) else None
        span = f"el {w[0]:.0f}-{w[1]:.0f}s" if w else f"episode {k+1}"
        print(f"\n  {span}")
        print(f"    {'gamma':>6}{'start':>9}{'min':>9}{'end':>9}")
        for r in rows:
            ep = r["episodes"][k] if k < len(r["episodes"]) else None
            if ep:
                print(f"    {r['gamma']:>6.2f}{ep[2]:>9.1f}{ep[3]:>9.1f}{ep[4]:>9.1f}")

    json.dump({str(r["gamma"]): {k: r[k] for k in
                                 ("median", "p95", "mx", "max_lead", "max_lag", "endpoint",
                                  "Dratio", "episodes", "real_wrong", "decisions")}
               for r in rows}, open(OUT / "gain_sweep.json", "w"), indent=1)

    # plot
    fig, ax = plt.subplots(figsize=(15, 6.2))
    st, sg = base["gate_steps"]
    i = 0; lab = True
    while i < len(sg):
        if sg[i]:
            j = i
            while j < len(sg) and sg[j]:
                j += 1
            ax.axvspan(st[i], st[j - 1], color="#2ca02c", alpha=0.10,
                       label="gate ON" if lab else None); lab = False
            i = j
        else:
            i += 1
    ax.axhline(0, color="k", lw=0.8)
    ax.plot(base["t"], base["D_route"] - base["D_true"], "#d62728", lw=1.4,
            label="gamma 0  (D_route - D_true, no correction)")
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(rows)))
    for r, c in zip(rows, cmap):
        ax.plot(r["t"], r["e"], lw=1.6, color=c, label=f"gamma {r['gamma']:.2f}")
    ax.set_xlabel("outage elapsed [s]"); ax.set_ylabel("D_position - D_true  [m]")
    ax.set_title("rf-07-26 display-correction gain sweep - only the correction magnitude scales")
    ax.legend(loc="lower left", ncol=2); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "gain_sweep.png", dpi=130)
    print(f"\nplot -> {OUT/'gain_sweep.png'}")


if __name__ == "__main__":
    sys.exit(main())
