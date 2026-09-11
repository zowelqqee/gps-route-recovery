#!/usr/bin/env python3
"""Phase 31 TASK 10-12: offline EKF replay at scale with gated / adaptive-sigma
learned correction.

For each covered held-out trip, train CatBoost + GBR on ALL OTHER trips (also
excluding the same day), predict a rolling residual, and feed v_ml into the full
single-path tracker under four regimes:

    A  baseline (no ML)
    B  ML always on, fixed sigma
    C  binary saturation gate (observable only)
    D  soft gate * adaptive sigma  (authority scaled by saturation strength and
       CatBoost/GBR agreement)

R_spec is otherwise unchanged; EKF structure / topology code untouched.
"""
from __future__ import annotations
import sys, warnings, json
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

from geotrace.coordinates import LocalFrame
from geotrace.loader import load_trip
from geotrace.road_graph import RoadNetwork, clip_graph, load_graph
from geotrace.pacman_tracker.config import PacmanConfig
from geotrace.pacman_tracker.roadmap import RoadGeometry
from geotrace.pacman_tracker.tracker import build_inputs, PacmanTracker
from geotrace.pacman_tracker.diagnostics import (
    survival_metrics, speed_metrics, distance_calibration, map_match_reference,
    GroundTruthObserver)
from geotrace.pacman_tracker.benchmark import _single_path_evaluation

sys.path.insert(0, str(Path(__file__).parent))
from phase30_dataset import make_windows

ROOT = Path("/Users/arseniyabramidze/delimobil/gps-route-recovery")
OUT = ROOT / "docs" / "plots" / "phase31"
GRAPHS = ["cache/spb", "cache/spb-parkgolovo", "cache/spb-center", "runs/review-map"]
TRIP_CONST = ["feat_dep_sigma_spec", "feat_spec_gap", "feat_sigma_spec"]
WIN_S, HOP_S = 3.0, 0.5


def cb():
    return CatBoostRegressor(iterations=500, depth=4, learning_rate=0.03,
                             loss_function="MAE", l2_leaf_reg=6.0, random_seed=0, verbose=False)

def gbr():
    return GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.03,
                                     loss="absolute_error", subsample=0.8, random_state=0)


def rolling_all(grids):
    return pd.concat([make_windows(g, WIN_S, HOP_S) for g in grids.values()],
                     ignore_index=True)


def load_graph_for(trip_dir, cache):
    trip, _ = load_trip(trip_dir)
    first = trip.usable_locations[0]
    for name in GRAPHS:
        if name not in cache:
            p = ROOT / f"{name}.graphml"
            cache[name] = load_graph(p) if p.exists() else None
        g = cache[name]
        if g is None:
            continue
        try:
            net = RoadNetwork(clip_graph(g, first.latitude, first.longitude, 11000.0),
                              LocalFrame(first.latitude, first.longitude))
            build_inputs(trip, net, PacmanConfig())
            return g, name
        except Exception:
            continue
    return None, None


def bench(trip_dir, graph, inject=None):
    trip, _ = load_trip(trip_dir)
    first = trip.usable_locations[0]
    cfg = PacmanConfig(); cfg.tracker_mode = "single_path"
    net = RoadNetwork(clip_graph(graph, first.latitude, first.longitude, 11000.0),
                      LocalFrame(first.latitude, first.longitude))
    geom = RoadGeometry(net, cfg.geometry)
    inputs = build_inputs(trip, net, cfg)
    if inject is not None:
        ti, vml, sig = inject
        for s in inputs.samples:
            if s.t > inputs.t_start and np.isfinite(s.spectral_speed):
                s.spectral_speed = float(np.interp(s.t, ti, vml))
                s.spectral_sigma = float(np.interp(s.t, ti, sig))
    truth = map_match_reference(trip.reference_locations, net, net.frame)
    obs = GroundTruthObserver(truth)
    res = PacmanTracker(net, cfg, geometry=geom).run(inputs, observer=obs)
    sm = survival_metrics(res, truth, net)
    dc = distance_calibration(res, trip)
    spm = speed_metrics(res, trip)
    spe = _single_path_evaluation(res, truth, trip, inputs.t_start, net)
    fw = spe.get("first_wrong_junction")
    return dict(
        D_ratio=dc.get("distance_ratio"), medDerr=dc.get("median_abs_error_m"),
        maxDerr=dc.get("max_abs_error_m"), finalDerr=dc.get("final_error_m"),
        d60=spm.get("distance_error_60s_m"), maxd60=spm.get("max_distance_error_60s_m"),
        vMAE=spm.get("mae_ms"), vbias_mov=spm.get("bias_while_moving_ms"),
        surv=sm.get("ground_truth_edge_survival_rate"),
        real_wrong=spe.get("real_wrong_decision_count"),
        decisions=spe.get("decision_count"),
        first_wrong=(fw or {}).get("t_decision") if fw else None,
        pos_err=sm.get("position_error_m"))


def main():
    targets = sys.argv[1:] or ["07-26-s1", "07-26-s3", "07-23-s0", "07-25-s1",
                               "07-28-s1", "07-31-s1", "07-22-s0", "07-24-s7"]
    grids = {p.stem.replace("grid_", ""): pd.read_csv(p)
             for p in sorted(OUT.glob("grid_*.csv"))}
    allw = rolling_all(grids)
    allw["day"] = allw.trip.str.slice(0, 5)
    F = [c for c in allw.columns if c.startswith("feat_") and c not in TRIP_CONST]

    gcache = {}
    rows = []
    for tag in targets:
        td = ROOT / f"runs/phase31/{tag}/trip"
        if not td.exists():
            print(f"{tag}: missing"); continue
        graph, gname = load_graph_for(td, gcache)
        if graph is None:
            print(f"{tag}: no graph coverage, skip"); continue
        day = tag[:5]
        tr = allw[(allw.trip != tag) & (allw.day != day) & (allw.regime == "outage")
                  & allw._moving.astype(bool) & allw._reliable_vspec.astype(bool)]
        tr = tr[np.isfinite(tr.residual)]
        mc = cb(); mc.fit(tr[F].to_numpy(), tr.residual.to_numpy())
        mg = gbr(); mg.fit(tr[F].to_numpy(), tr.residual.to_numpy())

        g = grids[tag]
        w = make_windows(g, WIN_S, HOP_S)
        w = w[w.regime == "outage"].copy()
        rc = mc.predict(w[F].to_numpy())
        rg = mg.predict(w[F].to_numpy())
        rhat = 0.5 * (rc + rg)
        disagree = np.abs(rc - rg)
        ti = w.t_end.to_numpy()
        vs = w.feat_v_spec.to_numpy()

        # observable saturation strength in [0,1]
        sat = np.clip(
            0.5 * np.clip(w.feat_vspec_frac_gt15_30s.to_numpy() / 0.15, 0, 1)
            + 0.5 * np.clip((w.feat_vspec_max_20s.to_numpy() - 13.0) / 4.0, 0, 1), 0, 1)
        # binary gate
        gate_bin = ((w.feat_vspec_frac_gt15_30s.to_numpy() > 0.02)
                    | (w.feat_vspec_max_20s.to_numpy() > 15.0)).astype(float)
        # soft alpha & adaptive sigma
        alpha = sat * np.clip(1.0 - disagree / 3.0, 0.0, 1.0)
        sig_native = w.feat_sigma_spec.to_numpy()
        sig_B = np.full_like(vs, 2.0)
        sig_C = np.where(gate_bin > 0, 2.0, sig_native)
        sig_D = np.clip(1.5 + 2.0 * disagree + 3.0 * (1.0 - sat), 1.5, sig_native.max())

        variants = {
            "B_always_s2": (np.clip(vs + rhat, 0, 33), sig_B),
            "C_bin_gate": (np.clip(vs + gate_bin * np.clip(rhat, -3, 12), 0, 33), sig_C),
            "D_soft_adaptive": (np.clip(vs + alpha * np.clip(rhat, -3, 12), 0, 33), sig_D),
        }
        base = bench(td, graph)
        res = {"A_baseline": base}
        for vn, (vml, sig) in variants.items():
            res[vn] = bench(td, graph, inject=(ti, vml, sig))
        print(f"\n===== {tag}  (graph {gname}, {len(tr)} train windows, "
              f"sat_mean {sat.mean():.2f}) =====")
        keys = ["D_ratio", "medDerr", "maxDerr", "d60", "vMAE", "vbias_mov",
                "surv", "real_wrong", "first_wrong", "pos_err"]
        print(f"  {'metric':11} " + " ".join(f"{k:>16}" for k in res))
        for k in keys:
            print(f"  {k:11} " + " ".join(f"{str(res[v].get(k)):>16}" for v in res))
        rows.append(dict(trip=tag, sat_mean=round(float(sat.mean()), 3),
                         **{f"{v}_{k}": res[v].get(k) for v in res for k in keys}))
    pd.DataFrame(rows).to_csv(OUT / "task12_replay.csv", index=False)
    print(f"\nwrote {OUT/'task12_replay.csv'}")


if __name__ == "__main__":
    sys.exit(main())
