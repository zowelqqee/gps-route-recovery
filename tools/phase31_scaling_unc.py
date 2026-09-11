#!/usr/bin/env python3
"""Phase 31 TASK 8 (scaling curve) + TASK 11 (uncertainty) on the clean
in-regime set (21 trips, excluding the 2 that extrapolate 2x / bad mount)."""
from __future__ import annotations
import sys, warnings, json
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

OUT = Path("/Users/arseniyabramidze/delimobil/gps-route-recovery/docs/plots/phase31")
TRIP_CONST = ["feat_dep_sigma_spec", "feat_spec_gap", "feat_sigma_spec"]
EXTRAP = {"07-24-s6", "07-26-s0"}


def cb():
    return CatBoostRegressor(iterations=500, depth=4, learning_rate=0.03,
                             loss_function="MAE", l2_leaf_reg=6.0, random_seed=0, verbose=False)

def gbr():
    return GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.03,
                                     loss="absolute_error", subsample=0.8, random_state=0)


def load(which):
    df = pd.read_csv(OUT / f"windows_{which}.csv")
    for c in ["_steady", "_moving", "_reliable_vspec"]:
        df[c] = df[c].astype(bool)
    df = df[~df.trip.str.startswith("rf-") & ~df.trip.isin(EXTRAP)].copy()
    d = df[(df.regime == "outage") & df._moving & df._reliable_vspec]
    if which == "5s":
        d = d[d._steady]
    return d[np.isfinite(d.residual)].reset_index(drop=True)


def main():
    d = load("2s")
    F = [c for c in d.columns if c.startswith("feat_") and c not in TRIP_CONST]
    trips = sorted(d.trip.unique())
    inv = pd.read_csv(OUT / "inventory.csv").set_index("trip")
    hs = [t for t in trips if inv.loc[t, "hi_speed_frac"] > 0.06]
    rng = np.random.default_rng(1)
    ho = sorted(rng.choice(hs, 4, replace=False).tolist()
                + rng.choice([t for t in trips if t not in hs], 3, replace=False).tolist())
    pool = [t for t in trips if t not in ho]
    te = d[d.trip.isin(ho)]
    vt = te.v_true_mean.to_numpy(); vs = te.feat_v_spec.to_numpy()
    hi = (vt >= 15) & (vt < 25)
    print(f"TASK 8 scaling  held-out {ho}  (pool {len(pool)} trips)")
    print(f"  baseline held-out csMAE_spec {np.mean(np.abs(vs-vt)):.3f}  "
          f"[15-25] {np.mean(np.abs(vs[hi]-vt[hi])):.3f}  n_hi={hi.sum()}")
    rows = []
    for ns in [2, 3, 5, 8, 11, 14, len(pool)]:
        mm, hh = [], []
        for rep in range(8):
            sub = rng.choice(pool, min(ns, len(pool)), replace=False)
            tr = d[d.trip.isin(sub)]
            m = cb(); m.fit(tr[F].to_numpy(), tr.residual.to_numpy())
            p = m.predict(te[F].to_numpy())
            mm.append(np.mean(np.abs(vs + p - vt)))
            hh.append(np.mean(np.abs((vs + p)[hi] - vt[hi])))
        rows.append(dict(n_train=ns, csMAE=round(np.mean(mm), 3), csMAE_sd=round(np.std(mm), 3),
                         csMAE_15_25=round(np.mean(hh), 3), csMAE_15_25_sd=round(np.std(hh), 3)))
    sc = pd.DataFrame(rows); sc.to_csv(OUT / "task8_scaling_clean.csv", index=False)
    print(sc.to_string(index=False))

    # TASK 11 uncertainty: LOTO OOF error vs disagreement / sat / regime
    print("\nTASK 11 uncertainty (LOTO OOF, clean set)")
    oofc = np.full(len(d), np.nan); oofg = np.full(len(d), np.nan)
    gv = d.trip.to_numpy()
    for t in np.unique(gv):
        tr, tei = d[gv != t], d[gv == t]
        if len(tei) < 5:
            continue
        oofc[gv == t] = cb().fit(tr[F].to_numpy(), tr.residual.to_numpy()).predict(tei[F].to_numpy())
        oofg[gv == t] = gbr().fit(tr[F].to_numpy(), tr.residual.to_numpy()).predict(tei[F].to_numpy())
    ok = np.isfinite(oofc) & np.isfinite(oofg)
    d = d[ok].reset_index(drop=True); oofc, oofg = oofc[ok], oofg[ok]
    resid = d.residual.to_numpy()
    err = np.abs(0.5 * (oofc + oofg) - resid)
    disag = np.abs(oofc - oofg)
    sat = d.feat_vspec_frac_gt15_30s.to_numpy()
    print("  corr(|pred err|, model disagreement) =", round(float(np.corrcoef(err, disag)[0, 1]), 3))
    for lo, hi_ in [(0, .5), (.5, 1), (1, 2), (2, 4), (4, 20)]:
        m = (disag >= lo) & (disag < hi_)
        if m.sum() < 10:
            continue
        print(f"    disagree {lo}-{hi_}: n={m.sum():4d}  mean|err| {err[m].mean():.2f}  "
              f"RMSE {np.sqrt(np.mean(err[m]**2)):.2f}")
    # empirical sigma_ml model: sigma = a + b*disagree + c*(1-sat_norm)
    from numpy.linalg import lstsq
    satn = np.clip(sat / 0.15, 0, 1)
    X = np.column_stack([np.ones_like(err), disag, 1 - satn, np.abs(0.5 * (oofc + oofg))])
    coef, *_ = lstsq(X, err, rcond=None)
    pred_sig = X @ coef
    print(f"  sigma_ml ~ {coef[0]:.2f} + {coef[1]:.2f}*disagree + {coef[2]:.2f}*(1-sat) + {coef[3]:.2f}*|rhat|")
    # calibration: does high predicted sigma actually flag high error?
    q = pd.qcut(pred_sig, 4, labels=["lo", "med", "hi", "vhi"])
    for lab in ["lo", "med", "hi", "vhi"]:
        m = (q == lab).to_numpy()
        print(f"    pred-sigma {lab}: n={m.sum():4d}  actual mean|err| {err[m].mean():.2f}")
    json.dump(dict(sigma_coef=list(map(float, coef)),
                   corr_err_disagree=float(np.corrcoef(err, disag)[0, 1])),
              open(OUT / "task11_uncertainty.json", "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
