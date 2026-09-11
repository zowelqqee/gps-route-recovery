#!/usr/bin/env python3
"""Phase 30 - learned spectral residual correction: modelling & verdict.

Cross-trip (leave-one-trip-out) is the ONLY generalisation test. Random-window
and contiguous-block splits are reported strictly as leakage diagnostics.

Reads docs/plots/phase30/windows_5s.csv (+ _2s). Writes docs/plots/phase30/*.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from catboost import CatBoostRegressor

ROOT = Path("/Users/arseniyabramidze/delimobil/gps-route-recovery")
OUT = ROOT / "docs" / "plots" / "phase30"

TRIPS = ["07-22", "07-23", "07-24", "07-25", "07-26"]
SPEED_BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 99)]

FEATURE_STAGES = {
    "1_vspec": ["feat_v_spec"],
    "2_+vprior": ["feat_v_spec", "feat_v_prior", "feat_vprior_minus_vspec",
                  "feat_v_ekf", "feat_vekf_minus_vspec", "feat_pvv", "feat_sqrt_pvv",
                  "feat_sigma_spec", "feat_dep_sigma_spec", "feat_spec_gap"],
    "3_+imu_hist": None,   # + IMU history block
    "4_+spectral_pow": None,
    "5_+temporal_ctx": None,   # everything
}
IMU_HIST = ["feat_a_long", "feat_b_a", "feat_a_long_debias", "feat_a_long_debias_std",
            "feat_a_long_debias_rms", "feat_pos_accel_frac", "feat_brake_frac",
            "feat_dv_imu_2s", "feat_dv_imu_5s", "feat_dv_imu_10s", "feat_dv_imu_20s",
            "feat_vprior_slope_5s", "feat_vprior_slope_10s", "feat_vprior_slope_20s",
            "feat_along_mean_5s", "feat_along_rms_10s", "feat_along_rms_20s"]
SPEC_POW = ["feat_vib_low", "feat_vib_b1", "feat_vib_b2", "feat_vib_mid", "feat_vib_b4",
            "feat_vib_high", "feat_vib_total", "feat_gyro_power", "feat_band_tilt",
            "feat_vspec_slope_5s", "feat_vspec_slope_10s", "feat_vspec_slope_20s",
            "feat_vspec_std_10s", "feat_vspec_std_20s", "feat_vspec_max_20s",
            "feat_vspec_frac_gt15_30s", "feat_vspec_frac_gt13_30s", "feat_sat_indicator",
            "feat_v_spec_std"]
CTX = ["feat_gyro_mag_mean", "feat_gyro_mag_std_10s", "feat_yaw_abs_mean",
       "feat_yaw_abs_max_20s", "feat_a_mag_mean", "feat_stationary_frac",
       "feat_dv_imu_20s"]


def all_features(df):
    return [c for c in df.columns if c.startswith("feat_")]


def build_stage_features(df):
    s = dict(FEATURE_STAGES)
    s["3_+imu_hist"] = s["2_+vprior"] + IMU_HIST
    s["4_+spectral_pow"] = s["3_+imu_hist"] + SPEC_POW
    s["5_+temporal_ctx"] = sorted(set(s["4_+spectral_pow"] + CTX + all_features(df)))
    return s


# ----------------------------------------------------------------------------
def models(target_kind):
    """Fresh model dict. target_kind only tunes CatBoost iters slightly."""
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "catboost": CatBoostRegressor(
            iterations=400, depth=4, learning_rate=0.03, loss_function="MAE",
            l2_leaf_reg=6.0, random_seed=0, verbose=False),
        "gbr": GradientBoostingRegressor(
            n_estimators=300, max_depth=2, learning_rate=0.03, loss="absolute_error",
            subsample=0.8, random_state=0),
        "mlp": make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=2000,
            early_stopping=True, random_state=0)),
    }


def metrics_resid(y, yhat):
    e = yhat - y
    ss = np.sum((y - y.mean()) ** 2)
    return dict(
        mae=float(np.mean(np.abs(e))), rmse=float(np.sqrt(np.mean(e ** 2))),
        bias=float(np.mean(e)),
        r2=float(1 - np.sum(e ** 2) / ss) if ss > 1e-9 else float("nan"))


def corrected_speed_mae(df, resid_hat, target_kind):
    """MAE(v_ml, v_true) and the two baselines, overall + by true-speed regime."""
    vt = df.v_true_mean.to_numpy()
    vs = df.feat_v_spec.to_numpy()
    ve = df.feat_v_ekf.to_numpy()
    if target_kind == "residual":
        v_ml = vs + resid_hat
    else:  # k
        v_ml = vs * resid_hat
    rows = []
    for lo, hi in [(-1, 99)] + SPEED_BINS:
        m = (vt >= lo) & (vt < hi) if lo != -1 else np.ones(len(vt), bool)
        if m.sum() < 3:
            rows.append(dict(regime=f"{lo}-{hi}" if lo != -1 else "all", n=int(m.sum())))
            continue
        rows.append(dict(
            regime=f"{lo}-{hi}" if lo != -1 else "all", n=int(m.sum()),
            mae_v_ml=round(float(np.mean(np.abs(v_ml[m] - vt[m]))), 3),
            mae_v_spec=round(float(np.mean(np.abs(vs[m] - vt[m]))), 3),
            mae_v_ekf=round(float(np.mean(np.abs(ve[m] - vt[m]))), 3),
        ))
    return pd.DataFrame(rows)


def loto_predict(df, feats, model_name, target_col, target_kind):
    """Leave-one-trip-out out-of-fold predictions for the whole df."""
    oof = np.full(len(df), np.nan)
    for test in TRIPS:
        tr = df[df.trip != test]
        te = df[df.trip == test]
        if len(te) == 0 or len(tr) < 20:
            continue
        mdl = models(target_kind)[model_name]
        mdl.fit(tr[feats].to_numpy(), tr[target_col].to_numpy())
        oof[df.trip.to_numpy() == test] = mdl.predict(te[feats].to_numpy())
    return oof


# ----------------------------------------------------------------------------
def analysis_set(df, dense=False):
    d = df[(df.regime == "outage") & df._moving & df._reliable_vspec].copy()
    if not dense:
        d = d[df._steady]
    d = d[np.isfinite(d.k) & np.isfinite(d.residual)]
    return d.reset_index(drop=True)


def task_45_8(df5, df2):
    """Cross-trip residual prediction + corrected speed, residual vs k."""
    report = []
    for label, d in [("5s_steady", analysis_set(df5)),
                     ("2s_dense", analysis_set(df2, dense=True))]:
        feats = build_stage_features(d)["5_+temporal_ctx"]
        for target_kind, tcol in [("residual", "residual"), ("k", "k")]:
            # baselines
            if target_kind == "residual":
                base_const = np.full(len(d), d.residual.mean())  # in-sample mean (optimistic)
                base_zero = np.zeros(len(d))
            else:
                base_const = np.full(len(d), d.k.mean())
                base_zero = np.ones(len(d))
            for name in ["ZERO/1", "CONST", "ridge", "catboost", "gbr", "mlp"]:
                if name == "ZERO/1":
                    yhat = base_zero
                elif name == "CONST":
                    # honest: leave-one-trip-out constant
                    yhat = np.full(len(d), np.nan)
                    for t in TRIPS:
                        msk = d.trip.to_numpy() == t
                        if msk.sum():
                            yhat[msk] = d[~(d.trip == t)][tcol].mean()
                else:
                    yhat = loto_predict(d, feats, name, tcol, target_kind)
                ok = np.isfinite(yhat)
                if ok.sum() < 10:
                    continue
                m = metrics_resid(d[tcol].to_numpy()[ok], yhat[ok])
                cs = corrected_speed_mae(d[ok], yhat[ok], target_kind)
                allrow = cs[cs.regime == "all"].iloc[0]
                hi = cs[cs.regime == "15-20"]
                vhi = cs[cs.regime == "20-99"]
                report.append(dict(
                    dataset=label, target=target_kind, model=name,
                    n=int(ok.sum()),
                    tgt_mae=round(m["mae"], 3), tgt_rmse=round(m["rmse"], 3),
                    tgt_bias=round(m["bias"], 3), tgt_r2=round(m["r2"], 3),
                    csMAE_all=allrow.get("mae_v_ml"),
                    csMAE_spec=allrow.get("mae_v_spec"),
                    csMAE_ekf=allrow.get("mae_v_ekf"),
                    csMAE_15_20=hi.mae_v_ml.iloc[0] if len(hi) and "mae_v_ml" in hi else None,
                    csMAE_15_20_spec=hi.mae_v_spec.iloc[0] if len(hi) and "mae_v_spec" in hi else None,
                    csMAE_20p=vhi.mae_v_ml.iloc[0] if len(vhi) and "mae_v_ml" in vhi else None,
                    csMAE_20p_spec=vhi.mae_v_spec.iloc[0] if len(vhi) and "mae_v_spec" in vhi else None,
                ))
    rep = pd.DataFrame(report)
    rep.to_csv(OUT / "task45_crosstrip.csv", index=False)
    print("\n=== TASK 4/5/8 : cross-trip (LOTO) ===")
    print(rep.to_string(index=False))
    return rep


def task_7_ablation(df5, df2):
    rows = []
    for label, d in [("5s_steady", analysis_set(df5)),
                     ("2s_dense", analysis_set(df2, dense=True))]:
        stages = build_stage_features(d)
        for sname, feats in stages.items():
            feats = [f for f in feats if f in d.columns]
            for mname in ["ridge", "catboost"]:
                yhat = loto_predict(d, feats, mname, "residual", "residual")
                ok = np.isfinite(yhat)
                m = metrics_resid(d.residual.to_numpy()[ok], yhat[ok])
                cs = corrected_speed_mae(d[ok], yhat[ok], "residual")
                rows.append(dict(dataset=label, stage=sname, n_feat=len(feats),
                                 model=mname, mae=round(m["mae"], 3),
                                 rmse=round(m["rmse"], 3), bias=round(m["bias"], 3),
                                 r2=round(m["r2"], 3),
                                 csMAE=cs[cs.regime == "all"].mae_v_ml.iloc[0],
                                 csMAE_spec=cs[cs.regime == "all"].mae_v_spec.iloc[0]))
    rep = pd.DataFrame(rows)
    rep.to_csv(OUT / "task7_ablation.csv", index=False)
    print("\n=== TASK 7 : feature ablation (LOTO, residual target) ===")
    print(rep.to_string(index=False))
    return rep


def task_6_identifiability(df2):
    d = df2[(df2.regime == "outage") & df2._moving].copy()
    band = d[(d.feat_v_spec >= 12.0) & (d.feat_v_spec <= 15.0)].copy()
    print(f"\n=== TASK 6 : identifiability  (12<=v_spec<=15, n={len(band)}) ===")
    if len(band) < 20:
        print("  too few windows in band")
        return
    band["grp"] = pd.cut(band.v_true_mean, [0, 15, 18, 99],
                         labels=["true_12_15", "true_15_18", "true_18p"])
    print(band.groupby("grp", observed=True).agg(
        n=("v_true_mean", "size"),
        v_true=("v_true_mean", "mean"),
        v_spec=("feat_v_spec", "mean"),
        v_prior=("feat_v_prior", "mean"),
        vprior_m_vspec=("feat_vprior_minus_vspec", "mean"),
        dv_imu_20s=("feat_dv_imu_20s", "mean"),
        band_tilt=("feat_band_tilt", "mean"),
        vib_high=("feat_vib_high", "mean"),
        vspec_frac_gt15_30s=("feat_vspec_frac_gt15_30s", "mean"),
    ).round(3).to_string())

    # binary separability: true>=16 vs true<16 inside the band, LOTO AUC
    band["hi"] = (band.v_true_mean >= 16.0).astype(int)
    feats = [c for c in band.columns if c.startswith("feat_")]
    aucs = []
    for t in TRIPS:
        tr, te = band[band.trip != t], band[band.trip == t]
        if te.hi.nunique() < 2 or tr.hi.nunique() < 2 or len(te) < 8:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=0.3))
        clf.fit(tr[feats].to_numpy(), tr.hi.to_numpy())
        p = clf.predict_proba(te[feats].to_numpy())[:, 1]
        aucs.append((t, roc_auc_score(te.hi.to_numpy(), p), len(te)))
    print("  LOTO logistic separability (true>=16 vs <16 | v_spec 12-15):")
    for t, a, n in aucs:
        print(f"    test {t}: AUC {a:.2f}  (n={n})")
    if aucs:
        print(f"    mean AUC {np.mean([a for _, a, _ in aucs]):.2f}")

    # CatBoost feature importance on the full steady set
    ds = analysis_set(df2, dense=True)
    fe = [c for c in ds.columns if c.startswith("feat_")]
    cb = CatBoostRegressor(iterations=400, depth=4, learning_rate=0.03,
                           loss_function="MAE", random_seed=0, verbose=False)
    cb.fit(ds[fe].to_numpy(), ds.residual.to_numpy())
    imp = pd.Series(cb.get_feature_importance(), index=fe).sort_values(ascending=False)
    print("\n  CatBoost residual-model feature importance (top 15, full-fit):")
    print(imp.head(15).round(2).to_string())
    imp.to_csv(OUT / "task6_importance.csv")


def task_9_12_leakage(df5, df2):
    """Random-window and contiguous-block splits - leakage diagnostics only."""
    from sklearn.model_selection import KFold, GroupKFold
    print("\n=== TASK 9/12 : leakage diagnostics (NOT generalisation) ===")
    for label, d in [("5s_steady", analysis_set(df5)),
                     ("2s_dense", analysis_set(df2, dense=True))]:
        feats = [c for c in d.columns if c.startswith("feat_")]
        y = d.residual.to_numpy()
        X = d[feats].to_numpy()
        # random 5-fold
        for mname in ["ridge", "catboost"]:
            oof = np.full(len(d), np.nan)
            for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
                m = models("residual")[mname]
                m.fit(X[tr], y[tr]); oof[te] = m.predict(X[te])
            mr = metrics_resid(y, oof)
            # contiguous block: 10 time-blocks per trip, grouped
            d = d.sort_values(["trip", "t_start"]).reset_index(drop=True)
            grp = (d.groupby("trip").cumcount() // max(1, len(d) // 40)).to_numpy()
            grp = grp + d.trip.map({t: i * 100 for i, t in enumerate(TRIPS)}).to_numpy()
            oofb = np.full(len(d), np.nan)
            Xb, yb = d[feats].to_numpy(), d.residual.to_numpy()
            gkf = GroupKFold(5)
            for tr, te in gkf.split(Xb, yb, grp):
                m = models("residual")[mname]
                m.fit(Xb[tr], yb[tr]); oofb[te] = m.predict(Xb[te])
            mb = metrics_resid(yb, oofb)
            print(f"  {label:10s} {mname:9s}  random-KFold MAE {mr['mae']:.3f} R2 {mr['r2']:+.2f}  |  "
                  f"contig-block MAE {mb['mae']:.3f} R2 {mb['r2']:+.2f}")


def main():
    df5 = pd.read_csv(OUT / "windows_5s.csv")
    df2 = pd.read_csv(OUT / "windows_2s.csv")
    # fix _steady bool coming back as object
    for df in (df5, df2):
        for c in ["_steady", "_moving", "_reliable_vspec"]:
            df[c] = df[c].astype(bool)
    print(f"loaded 5s={len(df5)}  2s={len(df2)}")
    print("outage steady&moving&reliable per trip (5s):",
          analysis_set(df5).trip.value_counts().to_dict())

    task_45_8(df5, df2)
    task_7_ablation(df5, df2)
    task_6_identifiability(df2)
    task_9_12_leakage(df5, df2)


if __name__ == "__main__":
    sys.exit(main())
