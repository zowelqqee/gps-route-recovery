#!/usr/bin/env python3
"""Phase 33 stage 1 - per-sample observable saturation features during gate=ON
on rf-07-26, grouped by the four gate-ON episodes, next to the 'needed gamma'
each episode showed in the Phase 32 gain sweep.

Nothing tuned. Hidden GPS only labels episodes for this analysis.
"""
from __future__ import annotations
import sys
from collections import deque
from pathlib import Path

import numpy as np

from geotrace.coordinates import LocalFrame
from geotrace.loader import load_trip
from geotrace.road_graph import RoadNetwork, clip_graph, load_graph
from geotrace.pacman_tracker.config import PacmanConfig
from geotrace.pacman_tracker.roadmap import RoadGeometry
from geotrace.pacman_tracker.tracker import build_inputs, PacmanTracker
from geotrace.pacman_tracker.diagnostics import map_match_reference, GroundTruthObserver
import geotrace.pacman_tracker.display_position as dpm

ROOT = Path("/Users/arseniyabramidze/delimobil/gps-route-recovery")
TRIP = ROOT / "runs/review-final/2026-07-26/trip"
GRAPH = ROOT / "runs/review-map.graphml"

# 'needed gamma' per gate-ON episode, read off the Phase 32 gain sweep (endpoint -> 0)
NEEDED_GAMMA = {1: 1.05, 2: 1.98, 3: 1.53, 4: 1.62}


def collect():
    trip, _ = load_trip(TRIP)
    first = trip.usable_locations[0]
    net = RoadNetwork(clip_graph(load_graph(GRAPH), first.latitude, first.longitude, 11000.0),
                      LocalFrame(first.latitude, first.longitude))
    cfg = PacmanConfig(); cfg.tracker_mode = "single_path"
    cfg.display.position_branch_enabled = True; cfg.display.leave_0726_out = True

    real = dpm.DisplayPositionBranch.step
    rows = []
    gate_run = [0.0]

    def spy(self, t, dt, v_route, v_prior, v_spectral, d_route, stationary):
        real(self, t, dt, v_route, v_prior, v_spectral, d_route, stationary)
        if not self._gate or not self._buf:
            gate_run[0] = 0.0
            return
        gate_run[0] += dt
        ts = np.fromiter((s[0] for s in self._buf), float)
        diff = np.fromiter((s[1] for s in self._buf), float)
        spec = np.fromiter((s[2] for s in self._buf), float)

        def w(sec, arr, fn):
            m = ts >= t - sec
            return float(fn(arr[m])) if m.sum() else 0.0
        rows.append(dict(
            t=float(t),
            frac15_30s=w(30.0, spec, lambda a: np.mean(a > 15.0)),
            vmax_20s=w(20.0, spec, np.max),
            vprior_minus_vspec=w(3.0, diff, np.mean),
            v_spectral=w(3.0, spec, np.mean),
            gate_dur=gate_run[0],
            dvs_5s=w(5.0, diff, np.mean), dvs_10s=w(10.0, diff, np.mean),
            dvs_20s=w(20.0, diff, np.mean),
            vspec_std_10s=w(10.0, spec, np.std), vspec_std_20s=w(20.0, spec, np.std),
        ))
    dpm.DisplayPositionBranch.step = spy
    try:
        inp = build_inputs(trip, net, cfg)
        PacmanTracker(net, cfg, geometry=RoadGeometry(net, cfg.geometry)).run(
            inp, observer=GroundTruthObserver(
                map_match_reference(trip.reference_locations, net, net.frame)))
    finally:
        dpm.DisplayPositionBranch.step = real
    t0 = inp.t_start
    for r in rows:
        r["el"] = r["t"] - t0
    return rows


EP_WIN = {1: (27, 115), 2: (211, 305), 3: (316, 390), 4: (435, 456)}


def main():
    rows = collect()
    feats = ["vprior_minus_vspec", "dvs_5s", "dvs_10s", "dvs_20s",
             "frac15_30s", "vmax_20s", "v_spectral", "gate_dur",
             "vspec_std_10s", "vspec_std_20s"]
    print(f"gate=ON samples: {len(rows)}\n")
    print(f"{'feature':>20} " + "".join(f"{'ep'+str(k):>10}" for k in EP_WIN) + f"{'sep(2vs1)':>11}")
    stats = {}
    for f in feats:
        line = f"{f:>20} "
        vals = {}
        for k, (a, b) in EP_WIN.items():
            v = np.array([r[f] for r in rows if a <= r["el"] <= b])
            vals[k] = v
            line += f"{np.median(v):>10.2f}"
        # separation: |median_ep2 - median_ep1| / pooled std
        pooled = np.sqrt(0.5 * (vals[1].std() ** 2 + vals[2].std() ** 2)) + 1e-9
        sep = abs(np.median(vals[2]) - np.median(vals[1])) / pooled
        stats[f] = sep
        line += f"{sep:>11.2f}"
        print(line)
    print("\nneeded gamma per episode:", NEEDED_GAMMA)
    print("\nbest separators (ep2 strong vs ep1 weak), by standardized median gap:")
    for f, s in sorted(stats.items(), key=lambda kv: -kv[1])[:5]:
        e1 = np.array([r[f] for r in rows if EP_WIN[1][0] <= r["el"] <= EP_WIN[1][1]])
        e2 = np.array([r[f] for r in rows if EP_WIN[2][0] <= r["el"] <= EP_WIN[2][1]])
        e3 = np.array([r[f] for r in rows if EP_WIN[3][0] <= r["el"] <= EP_WIN[3][1]])
        e4 = np.array([r[f] for r in rows if EP_WIN[4][0] <= r["el"] <= EP_WIN[4][1]])
        print(f"  {f:>20}  sep {s:.2f}   ep medians  "
              f"1:{np.median(e1):.2f}  2:{np.median(e2):.2f}  3:{np.median(e3):.2f}  4:{np.median(e4):.2f}   "
              f"p10-p90 all-ON [{np.percentile([r[f] for r in rows],10):.2f}, "
              f"{np.percentile([r[f] for r in rows],90):.2f}]")
    # dump for the rule builder
    import json
    (ROOT / "docs/plots/phase32/phase33_features.json").write_text(json.dumps(rows))
    print("\nrows -> docs/plots/phase32/phase33_features.json")


if __name__ == "__main__":
    sys.exit(main())
