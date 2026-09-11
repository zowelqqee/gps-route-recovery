#!/usr/bin/env python3
"""Phase 31 follow-up: strip the pure per-trip-constant features (the trip
fingerprint the TASK 9 audit flagged), re-test cross-trip, segment out the
2x-extrapolation trips, and compare to the simple isotonic correction."""
from __future__ import annotations
import sys, warnings, json
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.isotonic import IsotonicRegression
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).parent))
OUT = Path("/Users/arseniyabramidze/delimobil/gps-route-recovery/docs/plots/phase31")

# per-trip CONSTANTS - one value for the whole trip => pure domain fingerprint
TRIP_CONST = ["feat_dep_sigma_spec", "feat_spec_gap", "feat_sigma_spec"]
SPEED_BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 99)]

# trips where the spectral model extrapolates >~1.6x its warm-up range, or the
# mount / GPS truth is known bad (inventory: dep_sigma>=12 AND p95>22, or mount)
EXTRAPOLATION = {"07-24-s6", "07-26-s0"}


def load(which):
    df = pd.read_csv(OUT / f"windows_{which}.csv")
    for c in ["_steady", "_moving", "_reliable_vspec"]:
        df[c] = df[c].astype(bool)
    df = df[~df.trip.str.startswith("rf-")].copy()
    df["day"] = df.trip.str.slice(0, 5)
    return df


def aset(df, dense):
    d = df[(df.regime == "outage") & df._moving & df._reliable_vspec].copy()
    if not dense:
        d = d[d._steady]
    return d[np.isfinite(d.residual)].reset_index(drop=True)


def cb():
    return CatBoostRegressor(iterations=500, depth=4, learning_rate=0.03,
                             loss_function="MAE", l2_leaf_reg=6.0, random_seed=0, verbose=False)


def loto(d, F, target, predfn):
    oof = np.full(len(d), np.nan)
    gv = d.trip.to_numpy()
    for t in np.unique(gv):
        tr, te = d[gv != t], d[gv == t]
        if len(te) < 5 or len(tr) < 30:
            continue
        oof[gv == t] = predfn(tr, te, F)
    return oof


def regime_table(d, oof, label):
    ok = np.isfinite(oof)
    vt = d.v_true_mean.to_numpy()[ok]; vs = d.feat_v_spec.to_numpy()[ok]
    vml = vs + oof[ok]
    rows = []
    for lo, hi in SPEED_BINS:
        m = (vt >= lo) & (vt < hi)
        if m.sum() < 5:
            continue
        a = float(np.mean(np.abs(vs[m] - vt[m])))
        c = float(np.mean(np.abs(vml[m] - vt[m])))
        rows.append(dict(regime=f"{lo}-{hi}", n=int(m.sum()),
                         MAE_spec=round(a, 2), MAE_ml=round(c, 2),
                         impr_ms=round(a - c, 2), impr_pct=round(100 * (a - c) / a, 1)))
    r = pd.DataFrame(rows)
    print(f"\n--- {label} ---")
    print(r.to_string(index=False))
    return r


def per_trip_delta(d, oof):
    ok = np.isfinite(oof)
    vt = d.v_true_mean.to_numpy(); vs = d.feat_v_spec.to_numpy()
    out = []
    for t in d.trip.unique():
        m = (d.trip.to_numpy() == t) & ok
        if m.sum() < 5:
            continue
        dspec = np.mean(np.abs(vs[m] - vt[m]))
        dml = np.mean(np.abs((vs + oof)[m] - vt[m]))
        out.append((t, dml - dspec, m.sum()))
    return pd.DataFrame(out, columns=["trip", "d_csMAE", "n"]).sort_values("d_csMAE")


CB_PRED = lambda tr, te, F: cb().fit(
    tr[F].to_numpy(), tr.residual.to_numpy()).predict(te[F].to_numpy())


def iso_pred(col, inc=True):
    def f(tr, te, F):
        io = IsotonicRegression(out_of_bounds="clip", increasing=inc)
        io.fit(tr[col].to_numpy(), tr.residual.to_numpy())
        return io.predict(te[col].to_numpy())
    return f


def main():
    for which, dense in [("5s", False), ("2s", True)]:
        df = load(which)
        d = aset(df, dense)
        Fall = [c for c in d.columns if c.startswith("feat_")]
        Fhon = [c for c in Fall if c not in TRIP_CONST]
        print(f"\n================ {which}  n={len(d)}  {d.trip.nunique()} trips ================")

        # 1. all features vs honest features
        for lab, F in [("CatBoost ALL feats", Fall), ("CatBoost honest (no trip-const)", Fhon)]:
            oof = loto(d, F, "residual", CB_PRED)
            ok = np.isfinite(oof)
            e = oof[ok] - d.residual.to_numpy()[ok]
            r2 = 1 - np.sum(e**2) / np.sum((d.residual.to_numpy()[ok] - d.residual.to_numpy()[ok].mean())**2)
            vt = d.v_true_mean.to_numpy()[ok]; vs = d.feat_v_spec.to_numpy()[ok]
            hi = (vt >= 15) & (vt < 25)
            pt = per_trip_delta(d, oof)
            print(f"\n[{lab}] resid MAE {np.mean(np.abs(e)):.2f}  R2 {r2:+.2f}  "
                  f"csMAE {np.mean(np.abs(vs+oof[ok]-vt)):.2f}  "
                  f"csMAE[15-25] {np.mean(np.abs((vs+oof[ok])[hi]-vt[hi])):.2f} (spec {np.mean(np.abs(vs[hi]-vt[hi])):.2f})")
            print(f"    per-trip d_csMAE: median {pt.d_csMAE.median():+.2f}  "
                  f"p25 {pt.d_csMAE.quantile(.25):+.2f}  p75 {pt.d_csMAE.quantile(.75):+.2f}  "
                  f"worst {pt.d_csMAE.max():+.2f} ({pt.iloc[-1].trip})  "
                  f"helped {int((pt.d_csMAE<-0.05).sum())}/{len(pt)}")

        # 2. honest CatBoost, regime table, full set vs no-extrapolation set
        oof = loto(d, Fhon, "residual", CB_PRED)
        regime_table(d, oof, f"{which} honest CatBoost - ALL {d.trip.nunique()} trips")
        keep = ~d.trip.isin(EXTRAPOLATION)
        d2 = d[keep].reset_index(drop=True)
        oof2 = loto(d2, Fhon, "residual", CB_PRED)
        regime_table(d2, oof2, f"{which} honest CatBoost - {d2.trip.nunique()} trips (no 2x-extrapolation)")
        pt = per_trip_delta(d2, oof2)
        print("  per-trip d_csMAE (no-extrap set):")
        print("   " + pt.to_string(index=False).replace("\n", "\n   "))

        # 3. simple isotonic vs CatBoost on the no-extrapolation set
        print(f"\n  --- {which} simple corrections vs CatBoost (no-extrap, {d2.trip.nunique()} trips) ---")
        for name, pf in [("isotonic residual(vprior-vspec)", iso_pred("feat_vprior_minus_vspec")),
                         ("isotonic residual(v_prior)", iso_pred("feat_v_prior")),
                         ("CatBoost honest", CB_PRED)]:
            o = loto(d2, Fhon, "residual", pf)
            ok = np.isfinite(o)
            vt = d2.v_true_mean.to_numpy()[ok]; vs = d2.feat_v_spec.to_numpy()[ok]
            hi = (vt >= 15) & (vt < 25); lo = vt < 10
            pt = per_trip_delta(d2, o)
            print(f"    {name:34s}  csMAE {np.mean(np.abs(vs+o[ok]-vt)):.2f}  "
                  f"[15-25] {np.mean(np.abs((vs+o[ok])[hi]-vt[hi])):.2f}  "
                  f"[<10] {np.mean(np.abs((vs+o[ok])[lo]-vt[lo])):.2f} (spec {np.mean(np.abs(vs[lo]-vt[lo])):.2f})  "
                  f"worst_trip {pt.d_csMAE.max():+.2f}")


if __name__ == "__main__":
    sys.exit(main())
