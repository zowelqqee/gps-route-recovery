#!/usr/bin/env python3
"""Phase 30 TASK 10 - OFFLINE EKF replay with the learned residual correction.

For a target trip, train CatBoost on the OTHER four trips, predict the spectral
residual on a rolling grid, form  v_ml = v_spec + residual_hat , inject it as the
spectral measurement (replacing the raw model output) and re-run the FULL pacman
tracker. Compare against the untouched baseline. R_spec is left unchanged.

Nothing in production is modified. This is a diagnostic replay only.
"""
from __future__ import annotations
import sys, warnings, json
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from catboost import CatBoostRegressor
from geotrace.coordinates import LocalFrame
from geotrace.loader import load_trip
from geotrace.road_graph import RoadNetwork, clip_graph, load_graph
from geotrace.pacman_tracker.config import PacmanConfig
from geotrace.pacman_tracker.roadmap import RoadGeometry
from geotrace.pacman_tracker.tracker import build_inputs, PacmanTracker
from geotrace.pacman_tracker.diagnostics import (
    survival_metrics, speed_metrics, distance_calibration, map_match_reference,
    GroundTruthObserver)

sys.path.insert(0, str(Path(__file__).parent))
from phase30_dataset import replay_trip, make_windows, OUT, TRIPS as TRIP_DIRS

ROOT = Path("/Users/arseniyabramidze/delimobil/gps-route-recovery")
GRAPHS = [ROOT / "runs/review-map.graphml", ROOT / "cache/spb.graphml",
          ROOT / "cache/spb-parkgolovo.graphml"]
WIN_S, HOP_S = 3.0, 0.5


def cb():
    return CatBoostRegressor(iterations=500, depth=4, learning_rate=0.03,
                             loss_function="MAE", l2_leaf_reg=6.0,
                             random_seed=0, verbose=False)


def rolling_windows(grids):
    parts = []
    for tag, g in grids.items():
        w = make_windows(g, WIN_S, HOP_S)
        parts.append(w)
    return pd.concat(parts, ignore_index=True)


def bench(trip_dir, cfg, graph, inject=None, sigma_scale=1.0, sigma_fixed=None):
    trip, _ = load_trip(trip_dir)
    visible = trip.usable_locations
    first = visible[0]
    net = RoadNetwork(clip_graph(graph, first.latitude, first.longitude, 12000.0),
                      LocalFrame(first.latitude, first.longitude))
    geom = RoadGeometry(net, cfg.geometry)
    inputs = build_inputs(trip, net, cfg)
    if inject is not None:
        ti, vi = inject
        for s in inputs.samples:
            if s.t > inputs.t_start and np.isfinite(s.spectral_speed):
                s.spectral_speed = float(np.interp(s.t, ti, vi))
                if sigma_fixed is not None:
                    s.spectral_sigma = float(sigma_fixed)
                else:
                    s.spectral_sigma = float(s.spectral_sigma * sigma_scale)
    truth = map_match_reference(trip.reference_locations, net, net.frame)
    obs = GroundTruthObserver(truth)
    res = PacmanTracker(net, cfg, geometry=geom).run(inputs, observer=obs)
    sm = survival_metrics(res, truth, net)
    dc = distance_calibration(res, trip)
    spm = speed_metrics(res, trip)
    spe = res.stats.get("single_path", {})
    ev = None
    from geotrace.pacman_tracker.benchmark import _single_path_evaluation
    spev = _single_path_evaluation(res, truth, trip, inputs.t_start, net)
    return dict(
        D_ratio=dc.get("distance_ratio"),
        D_final=dc.get("final_distance_m"), D_true=dc.get("final_true_distance_m"),
        maxDerr=dc.get("max_abs_error_m"), medDerr=dc.get("median_abs_error_m"),
        d30=spm.get("distance_error_30s_m"), d60=spm.get("distance_error_60s_m"),
        maxd60=spm.get("max_distance_error_60s_m"),
        vMAE=spm.get("mae_ms"), vbias=spm.get("bias_ms"),
        vbias_moving=spm.get("bias_while_moving_ms"),
        surv_any=sm.get("ground_truth_edge_survival_rate"),
        surv_top1=sm.get("survival_top1"),
        real_wrong=spev.get("real_wrong_decision_count"),
        decisions=spev.get("decision_count"),
        first_wrong=(spev.get("first_wrong_junction") or {}).get("t_decision")
        if spev.get("first_wrong_junction") else None,
    )


def load_graph_for(trip_dir):
    trip, _ = load_trip(trip_dir)
    first = trip.usable_locations[0]
    for gp in GRAPHS:
        if not gp.exists():
            continue
        g = load_graph(gp)
        try:
            net = RoadNetwork(clip_graph(g, first.latitude, first.longitude, 12000.0),
                              LocalFrame(first.latitude, first.longitude))
            _ = build_inputs(trip, net, PacmanConfig())
            return g, gp.name
        except Exception:
            continue
    return None, None


def main():
    grids = {}
    for tag, d in TRIP_DIRS.items():
        p = OUT / f"grid_{tag}.csv"
        grids[tag] = pd.read_csv(p) if p.exists() else replay_trip(tag, d)
    allw = rolling_windows(grids)
    feats = [c for c in allw.columns if c.startswith("feat_")]

    targets = sys.argv[1:] or ["07-26", "07-22", "07-25", "07-24"]
    out = []
    for tag in targets:
        g, gname = load_graph_for(TRIP_DIRS[tag])
        if g is None:
            print(f"{tag}: no graph covers it, skipping offline replay")
            continue
        tr = allw[(allw.trip != tag) & (allw.regime == "outage") & allw._moving.astype(bool)
                  & allw._reliable_vspec.astype(bool)]
        m = cb(); m.fit(tr[feats].to_numpy(), tr.residual.to_numpy())
        gt = grids[tag]
        wt = make_windows(gt, WIN_S, HOP_S)
        wt = wt[wt.regime == "outage"].copy()
        rhat = m.predict(wt[feats].to_numpy())
        # clamp correction to a sane band; only correct upward-ish saturation
        rhat = np.clip(rhat, -3.0, 10.0)
        ti = wt.t_end.to_numpy()
        v_spec_w = wt.feat_v_spec.to_numpy()
        v_ml = np.clip(v_spec_w + rhat, 0.0, 33.0)

        cfg = PacmanConfig(); cfg.tracker_mode = "single_path"
        base = bench(TRIP_DIRS[tag], cfg, g)
        variants = {"+v_ml (R_spec unchanged)": dict(),
                    "+v_ml sigma x0.5": dict(sigma_scale=0.5),
                    "+v_ml sigma=2.0": dict(sigma_fixed=2.0),
                    "+v_ml sigma=1.0": dict(sigma_fixed=1.0)}
        runs = {}
        for vname, kw in variants.items():
            c = PacmanConfig(); c.tracker_mode = "single_path"
            runs[vname] = bench(TRIP_DIRS[tag], c, g, inject=(ti, v_ml), **kw)
        print(f"\n===== {tag}  (graph {gname}, {len(tr)} train windows) =====")
        keys = ["D_ratio", "maxDerr", "medDerr", "d30", "d60", "maxd60", "vMAE",
                "vbias_moving", "surv_any", "real_wrong", "first_wrong"]
        cols = ["baseline"] + list(variants)
        print(f"  {'metric':14} " + " ".join(f"{c:>22}" for c in cols))
        for k in keys:
            vals = [base.get(k)] + [runs[v].get(k) for v in variants]
            print(f"  {k:14} " + " ".join(f"{str(x):>22}" for x in vals))
        out.append(dict(trip=tag, **{f"base_{k}": base.get(k) for k in keys},
                        **{f"ml_{k}": runs['+v_ml (R_spec unchanged)'].get(k) for k in keys}))
    if out:
        pd.DataFrame(out).to_csv(OUT / "task10_offline_replay.csv", index=False)
        print(f"\nwrote {OUT/'task10_offline_replay.csv'}")


if __name__ == "__main__":
    sys.exit(main())
