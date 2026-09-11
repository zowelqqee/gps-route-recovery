#!/usr/bin/env python3
"""Phase 30 - per-held-out-trip breakdown. The decisive check: does the learned
correction beat the baselines ON EACH held-out trip, especially 07-26 / 07-25
(saturation) - or is the pooled win carried by the easy trips?"""
from __future__ import annotations
import warnings, sys
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from catboost import CatBoostRegressor

OUT = Path("/Users/arseniyabramidze/delimobil/gps-route-recovery/docs/plots/phase30")
TRIPS = ["07-22", "07-23", "07-24", "07-25", "07-26"]


def aset(df, dense):
    d = df[(df.regime == "outage") & df._moving.astype(bool) & df._reliable_vspec.astype(bool)].copy()
    if not dense:
        d = d[df._steady.astype(bool)]
    return d[np.isfinite(d.residual) & np.isfinite(d.k)].reset_index(drop=True)


def cb():
    return CatBoostRegressor(iterations=400, depth=4, learning_rate=0.03,
                             loss_function="MAE", l2_leaf_reg=6.0, random_seed=0, verbose=False)


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def run(df, dense, tag):
    d = aset(df, dense)
    feats = [c for c in d.columns if c.startswith("feat_")]
    print(f"\n===== {tag}  (n={len(d)}) =====")
    hdr = (f"{'test':>7} {'n':>4} {'vt_rng':>10} {'resid_mu':>9} | "
           f"{'MAE_spec':>8} {'MAE_ekf':>8} {'MAE_const':>9} {'MAE_CB':>7} {'MAE_CBk':>7} | "
           f"{'hi_spec':>7} {'hi_CB':>7}  (hi = true>=15)")
    print(hdr)
    agg = {k: [] for k in ["spec", "ekf", "const", "cb", "cbk"]}
    aggw = []
    for t in TRIPS:
        tr, te = d[d.trip != t], d[d.trip == t]
        if len(te) < 5:
            print(f"{t:>7} {len(te):>4}  (skip)")
            continue
        vt = te.v_true_mean.to_numpy(); vs = te.feat_v_spec.to_numpy(); ve = te.feat_v_ekf.to_numpy()
        m_res = cb(); m_res.fit(tr[feats].to_numpy(), tr.residual.to_numpy())
        m_k = cb(); m_k.fit(tr[feats].to_numpy(), tr.k.to_numpy())
        v_cb = vs + m_res.predict(te[feats].to_numpy())
        v_cbk = vs * m_k.predict(te[feats].to_numpy())
        v_const = vs + tr.residual.mean()
        hi = vt >= 15.0
        row = dict(spec=mae(vs, vt), ekf=mae(ve, vt), const=mae(v_const, vt),
                   cb=mae(v_cb, vt), cbk=mae(v_cbk, vt))
        for k in agg:
            agg[k].append((row[k], len(te)))
        hs = mae(vs[hi], vt[hi]) if hi.sum() >= 3 else np.nan
        hc = mae(v_cb[hi], vt[hi]) if hi.sum() >= 3 else np.nan
        print(f"{t:>7} {len(te):>4} {vt.min():>4.1f}-{vt.max():>4.1f} {te.residual.mean():>+9.2f} | "
              f"{row['spec']:>8.2f} {row['ekf']:>8.2f} {row['const']:>9.2f} {row['cb']:>7.2f} {row['cbk']:>7.2f} | "
              f"{hs:>7.2f} {hc:>7.2f}")
    def wm(key):
        v = agg[key]
        n = sum(x[1] for x in v)
        return sum(x[0]*x[1] for x in v)/n
    print(f"{'POOLED':>7} {'':>4} {'':>10} {'':>9} | "
          f"{wm('spec'):>8.2f} {wm('ekf'):>8.2f} {wm('const'):>9.2f} {wm('cb'):>7.2f} {wm('cbk'):>7.2f}")


def main():
    df5 = pd.read_csv(OUT / "windows_5s.csv")
    df2 = pd.read_csv(OUT / "windows_2s.csv")
    run(df5, False, "5s steady")
    run(df2, True, "2s dense")
    # high-speed only pooled
    for tag, df, dense in [("5s", df5, False), ("2s", df2, True)]:
        d = aset(df, dense)
        feats = [c for c in d.columns if c.startswith("feat_")]
        oof = np.full(len(d), np.nan)
        for t in TRIPS:
            tr, te = d[d.trip != t], d[d.trip == t]
            if len(te) < 5: continue
            m = cb(); m.fit(tr[feats].to_numpy(), tr.residual.to_numpy())
            oof[d.trip.to_numpy() == t] = te.feat_v_spec.to_numpy() + m.predict(te[feats].to_numpy())
        ok = np.isfinite(oof)
        vt = d.v_true_mean.to_numpy()
        for lo, hi in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 99)]:
            m = ok & (vt >= lo) & (vt < hi)
            if m.sum() < 3: continue
            print(f"  [{tag}] true {lo:2d}-{hi:2d}  n={m.sum():3d}  "
                  f"MAE_spec {mae(d.feat_v_spec.to_numpy()[m], vt[m]):.2f}  "
                  f"MAE_CB {mae(oof[m], vt[m]):.2f}  "
                  f"MAE_ekf {mae(d.feat_v_ekf.to_numpy()[m], vt[m]):.2f}")


if __name__ == "__main__":
    sys.exit(main())
