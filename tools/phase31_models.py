#!/usr/bin/env python3
"""Phase 31 - large-scale cross-trip validation: models, regimes, identifiability,
feature stability, scaling curve, trip-ID leakage, simple-baseline comparison.

Primary split: leave-one-trip-out over the 23 distinct logger sessions.
Leakage-controlled check: leave-one-DAY-out (sessions from the same day grouped).
Random-window metrics are NOT reported as generalisation.
"""
from __future__ import annotations
import sys, warnings, json
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
from catboost import CatBoostRegressor

OUT = Path("/Users/arseniyabramidze/delimobil/gps-route-recovery/docs/plots/phase31")
SPEED_BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 99)]


def load(which="5s"):
    df = pd.read_csv(OUT / f"windows_{which}.csv")
    for c in ["_steady", "_moving", "_reliable_vspec"]:
        df[c] = df[c].astype(bool)
    df = df[~df.trip.str.startswith("rf-")].copy()          # drop review-final prefixes
    df["day"] = df.trip.str.slice(0, 5)
    return df


def aset(df, dense=False):
    d = df[(df.regime == "outage") & df._moving & df._reliable_vspec].copy()
    if not dense:
        d = d[d._steady]
    d = d[np.isfinite(d.k) & np.isfinite(d.residual)]
    return d.reset_index(drop=True)


FEATS = None
def feats(d):
    return [c for c in d.columns if c.startswith("feat_")]


def cb():
    return CatBoostRegressor(iterations=500, depth=4, learning_rate=0.03,
                             loss_function="MAE", l2_leaf_reg=6.0, random_seed=0, verbose=False)

def gbr():
    return GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.03,
                                     loss="absolute_error", subsample=0.8, random_state=0)

def ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def mrep(y, yh):
    e = yh - y
    ss = np.sum((y - y.mean()) ** 2)
    return dict(mae=float(np.mean(np.abs(e))), rmse=float(np.sqrt(np.mean(e**2))),
                bias=float(np.mean(e)), r2=float(1 - np.sum(e**2)/ss) if ss > 1e-9 else np.nan)


def group_oof(d, F, make, target, groups):
    oof = np.full(len(d), np.nan)
    gv = d[groups].to_numpy()
    for g in np.unique(gv):
        tr, te = d[gv != g], d[gv == g]
        if len(te) < 5 or len(tr) < 30:
            continue
        m = make(); m.fit(tr[F].to_numpy(), tr[target].to_numpy())
        oof[gv == g] = m.predict(te[F].to_numpy())
    return oof


# ---------------------------------------------------------------- TASK 3/4
def task34(d, tag):
    F = feats(d)
    print(f"\n=== TASK 3/4  models, grouped OOF  [{tag}, n={len(d)}, "
          f"{d.trip.nunique()} trips / {d.day.nunique()} days] ===")
    rows = []
    for split in ["trip", "day"]:
        for mname, make in [("zero", None), ("const", "const"),
                            ("ridge", ridge), ("gbr", gbr), ("catboost", cb)]:
            if mname == "zero":
                yh = np.zeros(len(d))
            elif mname == "const":
                yh = np.full(len(d), np.nan)
                gv = d[split].to_numpy()
                for g in np.unique(gv):
                    yh[gv == g] = d[d[split] != g].residual.mean()
            else:
                yh = group_oof(d, F, make, "residual", split)
            ok = np.isfinite(yh)
            m = mrep(d.residual.to_numpy()[ok], yh[ok])
            vt = d.v_true_mean.to_numpy()[ok]; vs = d.feat_v_spec.to_numpy()[ok]
            vml = vs + yh[ok]
            cs_all = float(np.mean(np.abs(vml - vt)))
            hi = (vt >= 15) & (vt < 25)
            cs_hi = float(np.mean(np.abs(vml[hi] - vt[hi]))) if hi.sum() > 5 else np.nan
            cs_hi_spec = float(np.mean(np.abs(vs[hi] - vt[hi]))) if hi.sum() > 5 else np.nan
            # per-held-out-trip csMAE distribution
            per = []
            gv = d[split].to_numpy()[ok]
            for g in np.unique(gv):
                mm = gv == g
                if mm.sum() < 5:
                    continue
                per.append(float(np.mean(np.abs(vml[mm]-vt[mm])) - np.mean(np.abs(vs[mm]-vt[mm]))))
            per = np.array(per)
            rows.append(dict(split=split, model=mname, n=int(ok.sum()),
                             resid_mae=round(m["mae"], 3), resid_rmse=round(m["rmse"], 3),
                             resid_bias=round(m["bias"], 3), resid_r2=round(m["r2"], 3),
                             csMAE=round(cs_all, 3),
                             csMAE_15_25=round(cs_hi, 3), csMAE_15_25_spec=round(cs_hi_spec, 3),
                             d_csMAE_median=round(float(np.median(per)), 3) if len(per) else None,
                             d_csMAE_p25=round(float(np.percentile(per, 25)), 3) if len(per) else None,
                             d_csMAE_p75=round(float(np.percentile(per, 75)), 3) if len(per) else None,
                             d_csMAE_worst=round(float(per.max()), 3) if len(per) else None,
                             d_csMAE_best=round(float(per.min()), 3) if len(per) else None,
                             n_trips_helped=int((per < -0.05).sum()), n_trips_hurt=int((per > 0.05).sum())))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / f"task34_{tag}.csv", index=False)
    print(r.to_string(index=False))
    return r


# ---------------------------------------------------------------- TASK 5
def task5(d, tag):
    F = feats(d)
    yh = group_oof(d, F, cb, "residual", "trip")
    ok = np.isfinite(yh)
    vt = d.v_true_mean.to_numpy()[ok]; vs = d.feat_v_spec.to_numpy()[ok]
    ve = d.feat_v_ekf.to_numpy()[ok]; vml = vs + yh[ok]
    rows = []
    for lo, hi in SPEED_BINS:
        m = (vt >= lo) & (vt < hi)
        if m.sum() < 5:
            continue
        a, b, c = (float(np.mean(np.abs(x[m] - vt[m]))) for x in (vs, ve, vml))
        rows.append(dict(regime=f"{lo}-{hi}", n=int(m.sum()),
                         MAE_spec=round(a, 3), MAE_ekf=round(b, 3), MAE_ml=round(c, 3),
                         impr_ms=round(a - c, 3), impr_pct=round(100*(a-c)/a, 1)))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / f"task5_{tag}.csv", index=False)
    print(f"\n=== TASK 5  corrected-speed MAE by regime (trip-LOTO CatBoost) [{tag}] ===")
    print(r.to_string(index=False))
    return r


# ---------------------------------------------------------------- TASK 6
def task6(df):
    d = df[(df.regime == "outage") & df._moving].copy()
    band = d[(d.feat_v_spec >= 12) & (d.feat_v_spec <= 15)].copy()
    print(f"\n=== TASK 6  identifiability at scale  (12<=v_spec<=15, n={len(band)}, "
          f"{band.trip.nunique()} trips) ===")
    band["grp"] = pd.cut(band.v_true_mean, [0, 15, 18, 21, 99],
                         labels=["t12_15", "t15_18", "t18_21", "t21p"])
    g = band.groupby("grp", observed=True).agg(
        n=("v_true_mean", "size"), v_true=("v_true_mean", "mean"),
        v_prior_m_vspec=("feat_vprior_minus_vspec", "mean"),
        v_prior=("feat_v_prior", "mean"),
        dv_imu_20s=("feat_dv_imu_20s", "mean"),
        band_tilt=("feat_band_tilt", "mean"), vib_high=("feat_vib_high", "mean"),
        vspec_slope5=("feat_vspec_slope_5s", "mean"),
        vspec_frac15=("feat_vspec_frac_gt15_30s", "mean")).round(3)
    print(g.to_string())
    band["hi"] = (band.v_true_mean >= 17.0).astype(int)
    F = feats(band)
    aucs = []
    gv = band.trip.to_numpy()
    for t in np.unique(gv):
        tr, te = band[gv != t], band[gv == t]
        if te.hi.nunique() < 2 or tr.hi.nunique() < 2 or len(te) < 8:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.3))
        clf.fit(tr[F].to_numpy(), tr.hi.to_numpy())
        aucs.append(roc_auc_score(te.hi.to_numpy(), clf.predict_proba(te[F].to_numpy())[:, 1]))
    print(f"  LOTO logistic AUC (true>=17 | v_spec 12-15): "
          f"median {np.median(aucs):.2f}  p25 {np.percentile(aucs,25):.2f}  "
          f"p75 {np.percentile(aucs,75):.2f}  n_folds {len(aucs)}")
    json.dump(dict(auc_median=float(np.median(aucs)), auc=list(map(float, aucs))),
              open(OUT / "task6_auc.json", "w"), indent=1)


# ---------------------------------------------------------------- TASK 7
def task7(d):
    F = feats(d)
    m = cb(); m.fit(d[F].to_numpy(), d.residual.to_numpy())
    imp = pd.Series(m.get_feature_importance(), index=F).sort_values(ascending=False)
    top = list(imp.head(12).index)
    print("\n=== TASK 7  feature stability across trips ===")
    print("  pooled CatBoost top-12 importance:")
    print(imp.head(12).round(2).to_string())
    rows = []
    for f in top:
        per = []
        for t in d.trip.unique():
            b = d[d.trip == t]
            if len(b) < 15 or b[f].std() < 1e-9:
                continue
            per.append(np.corrcoef(b[f], b.residual)[0, 1])
        per = np.array(per)
        rows.append(dict(feature=f, pooled_imp=round(imp[f], 2),
                         per_trip_corr_median=round(float(np.median(per)), 2),
                         frac_same_sign=round(float(np.mean(np.sign(per) == np.sign(np.median(per)))), 2),
                         n_trips=len(per)))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "task7_stability.csv", index=False)
    print("\n  per-trip sign consistency of corr(feature, residual):")
    print(r.to_string(index=False))


# ---------------------------------------------------------------- TASK 8
def task8(d, tag):
    F = feats(d)
    trips = sorted(d.trip.unique())
    rng = np.random.default_rng(0)
    # fixed held-out: 6 trips spanning speed range
    inv = pd.read_csv(OUT / "inventory.csv").set_index("trip")
    hs = [t for t in trips if inv.loc[t, "hi_speed_frac"] > 0.08]
    ho = (rng.choice(hs, 3, replace=False).tolist()
          + rng.choice([t for t in trips if t not in hs], 3, replace=False).tolist())
    pool = [t for t in trips if t not in ho]
    te = d[d.trip.isin(ho)]
    sizes = [2, 3, 5, 8, 12, len(pool)]
    rows = []
    for ns in sizes:
        maes, hmaes = [], []
        for rep in range(6):
            sub = rng.choice(pool, min(ns, len(pool)), replace=False)
            tr = d[d.trip.isin(sub)]
            m = cb(); m.fit(tr[F].to_numpy(), tr.residual.to_numpy())
            p = m.predict(te[F].to_numpy())
            vt = te.v_true_mean.to_numpy(); vs = te.feat_v_spec.to_numpy()
            maes.append(np.mean(np.abs(vs + p - vt)))
            hi = (vt >= 15) & (vt < 25)
            if hi.sum() > 5:
                hmaes.append(np.mean(np.abs((vs + p)[hi] - vt[hi])))
        rows.append(dict(n_train_trips=ns, csMAE=round(np.mean(maes), 3),
                         csMAE_sd=round(np.std(maes), 3),
                         csMAE_15_25=round(np.mean(hmaes), 3) if hmaes else None))
    # baseline on held-out
    vt = te.v_true_mean.to_numpy(); vs = te.feat_v_spec.to_numpy()
    hi = (vt >= 15) & (vt < 25)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / f"task8_scaling_{tag}.csv", index=False)
    print(f"\n=== TASK 8  train-size scaling  [{tag}, held-out {ho}] ===")
    print(f"  held-out baseline: csMAE_spec {np.mean(np.abs(vs-vt)):.3f}   "
          f"csMAE_spec[15-25] {np.mean(np.abs(vs[hi]-vt[hi])):.3f}")
    print(r.to_string(index=False))


# ---------------------------------------------------------------- TASK 9
def task9(d):
    F = feats(d)
    print("\n=== TASK 9  trip-ID / domain leakage audit ===")
    # can we classify trip identity from the features?
    from sklearn.model_selection import cross_val_predict
    y = d.trip.astype("category").cat.codes.to_numpy()
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=0, n_jobs=-1)
    pred = cross_val_predict(clf, d[F].to_numpy(), y, cv=5)
    acc = float(np.mean(pred == y))
    print(f"  trip-ID 5-fold accuracy from features: {acc:.3f}  (chance = {1/d.trip.nunique():.3f})")
    clf.fit(d[F].to_numpy(), y)
    imp = pd.Series(clf.feature_importances_, index=F).sort_values(ascending=False)
    print("  features most predictive of trip identity (domain signature):")
    print(imp.head(10).round(3).to_string())
    # retrain residual model without the top domain-signature features
    drop = list(imp.head(8).index)
    keep = [f for f in F if f not in drop]
    for lab, FF in [("all features", F), (f"minus {len(drop)} domain feats", keep)]:
        oof = group_oof(d, FF, cb, "residual", "trip")
        ok = np.isfinite(oof)
        m = mrep(d.residual.to_numpy()[ok], oof[ok])
        vt = d.v_true_mean.to_numpy()[ok]; vs = d.feat_v_spec.to_numpy()[ok]
        hi = (vt >= 15) & (vt < 25)
        cshi = np.mean(np.abs((vs+oof[ok])[hi]-vt[hi]))
        print(f"  residual LOTO [{lab}]: MAE {m['mae']:.3f} R2 {m['r2']:+.2f}  csMAE[15-25] {cshi:.3f}")
    json.dump(dict(tripid_acc=acc, top_domain=list(imp.head(10).index)),
              open(OUT / "task9_leakage.json", "w"), indent=1)


# ---------------------------------------------------------------- TASK 14
def task14(d, tag):
    F = feats(d)
    print(f"\n=== TASK 14  ML vs simple corrections (trip-LOTO) [{tag}] ===")
    vt_all = d.v_true_mean.to_numpy(); vs_all = d.feat_v_spec.to_numpy()
    rows = []
    gv = d.trip.to_numpy()

    def eval_pred(name, predfn):
        oof = np.full(len(d), np.nan)
        for t in np.unique(gv):
            tr, te = d[gv != t], d[gv == t]
            if len(te) < 5 or len(tr) < 30:
                continue
            oof[gv == t] = predfn(tr, te)
        ok = np.isfinite(oof)
        vt, vs = vt_all[ok], vs_all[ok]
        hi = (vt >= 15) & (vt < 25)
        rows.append(dict(model=name,
                         csMAE=round(float(np.mean(np.abs(oof[ok]-vt))), 3),
                         csMAE_15_25=round(float(np.mean(np.abs(oof[ok][hi]-vt[hi]))), 3),
                         resid_mae=round(float(np.mean(np.abs((oof[ok]-vs)-(vt-vs)))), 3)))

    eval_pred("baseline v_spec", lambda tr, te: te.feat_v_spec.to_numpy())
    # monotonic k(v_spec)
    def mono_kv(col):
        def f(tr, te):
            iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
            iso.fit(tr[col].to_numpy(), (tr.v_true_mean / tr.feat_v_spec).to_numpy())
            return te.feat_v_spec.to_numpy() * np.clip(iso.predict(te[col].to_numpy()), 0.5, 2.5)
        return f
    eval_pred("isotonic k(v_spec)", mono_kv("feat_v_spec"))
    eval_pred("isotonic k(v_prior-v_spec)", mono_kv("feat_vprior_minus_vspec"))
    # isotonic residual on v_prior
    def iso_resid(col, inc=True):
        def f(tr, te):
            iso = IsotonicRegression(out_of_bounds="clip", increasing=inc)
            iso.fit(tr[col].to_numpy(), tr.residual.to_numpy())
            return te.feat_v_spec.to_numpy() + iso.predict(te[col].to_numpy())
        return f
    eval_pred("isotonic residual(v_prior)", iso_resid("feat_v_prior"))
    eval_pred("isotonic residual(vprior-vspec)", iso_resid("feat_vprior_minus_vspec"))
    # 2D lookup: bin v_prior x band_tilt, median residual
    def lut2d(tr, te):
        import pandas as pd
        a = pd.qcut(tr.feat_v_prior, 6, duplicates="drop")
        b = pd.qcut(tr.feat_band_tilt, 4, duplicates="drop")
        tbl = tr.groupby([a, b], observed=True).residual.median()
        ea = pd.IntervalIndex(a.cat.categories); eb = pd.IntervalIndex(b.cat.categories)
        def look(x, y):
            ia = min(max(ea.get_indexer([x])[0], 0), len(ea)-1)
            ib = min(max(eb.get_indexer([y])[0], 0), len(eb)-1)
            try:
                v = tbl.get((ea[ia], eb[ib]), np.nan)
            except Exception:
                v = np.nan
            return v if np.isfinite(v) else tr.residual.median()
        return te.feat_v_spec.to_numpy() + np.array(
            [look(x, y) for x, y in zip(te.feat_v_prior, te.feat_band_tilt)])
    eval_pred("2D LUT residual(v_prior,tilt)", lut2d)
    eval_pred("CatBoost residual", lambda tr, te: te.feat_v_spec.to_numpy()
              + cb().fit(tr[F].to_numpy(), tr.residual.to_numpy()).predict(te[F].to_numpy()))
    eval_pred("GBR residual", lambda tr, te: te.feat_v_spec.to_numpy()
              + gbr().fit(tr[F].to_numpy(), tr.residual.to_numpy()).predict(te[F].to_numpy()))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / f"task14_{tag}.csv", index=False)
    print(r.to_string(index=False))


def main():
    d5 = load("5s"); d2 = load("2s")
    s5, s2 = aset(d5), aset(d2, dense=True)
    print(f"analysis set: 5s steady n={len(s5)} ({s5.trip.nunique()} trips), "
          f"2s dense n={len(s2)} ({s2.trip.nunique()} trips)")
    task34(s5, "5s"); task34(s2, "2s")
    task5(s5, "5s"); task5(s2, "2s")
    task6(d2)
    task7(s2)
    task8(s2, "2s")
    task9(s2)
    task14(s5, "5s"); task14(s2, "2s")


if __name__ == "__main__":
    sys.exit(main())
