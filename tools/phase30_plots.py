#!/usr/bin/env python3
"""Phase 30 plots."""
from __future__ import annotations
import warnings, sys
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from phase30_dataset import make_windows, OUT, TRIPS as TD
from catboost import CatBoostRegressor

TRIPS = list(TD)
COL = {"07-22": "#1f77b4", "07-23": "#2ca02c", "07-24": "#ff7f0e",
       "07-25": "#9467bd", "07-26": "#d62728"}


def cb():
    return CatBoostRegressor(iterations=500, depth=4, learning_rate=0.03,
                             loss_function="MAE", l2_leaf_reg=6.0, random_seed=0, verbose=False)


def main():
    grids = {t: pd.read_csv(OUT / f"grid_{t}.csv") for t in TRIPS}
    df = pd.concat([make_windows(g, 5.0) for g in grids.values()], ignore_index=True)
    d = df[(df.regime == "outage") & df._moving.astype(bool) & df._reliable_vspec.astype(bool)
           & df._steady.astype(bool)].copy()
    d = d[np.isfinite(d.residual)].reset_index(drop=True)
    feats = [c for c in d.columns if c.startswith("feat_")]
    oof = np.full(len(d), np.nan)
    for t in TRIPS:
        tr, te = d[d.trip != t], d[d.trip == t]
        if len(te) < 5:
            continue
        m = cb(); m.fit(tr[feats].to_numpy(), tr.residual.to_numpy())
        oof[d.trip.to_numpy() == t] = m.predict(te[feats].to_numpy())
    d["rhat"] = oof

    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))

    # (a) residual vs v_true, per trip - shows trip offset / non-transfer
    for t in TRIPS:
        b = d[d.trip == t]
        ax[0].scatter(b.v_true_mean, b.residual, s=24, alpha=.7, color=COL[t], label=t)
    ax[0].axhline(0, color="grey", lw=.8, ls="--")
    ax[0].set_xlabel("true speed [m/s]"); ax[0].set_ylabel("residual = v_true - v_spectral [m/s]")
    ax[0].set_title("(a) spectral residual vs true speed - per-trip offset")
    ax[0].legend(); ax[0].grid(alpha=.25)

    # (b) LOTO predicted vs actual residual
    ok = np.isfinite(d.rhat)
    for t in TRIPS:
        b = d[(d.trip == t) & ok]
        ax[1].scatter(b.residual, b.rhat, s=24, alpha=.7, color=COL[t], label=t)
    lim = [-6, 10]
    ax[1].plot(lim, lim, "k--", lw=1)
    ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_xlabel("actual residual [m/s]"); ax[1].set_ylabel("CatBoost LOTO prediction [m/s]")
    mae = float(np.mean(np.abs(d.rhat[ok] - d.residual[ok])))
    r2 = 1 - np.sum((d.rhat[ok]-d.residual[ok])**2)/np.sum((d.residual[ok]-d.residual[ok].mean())**2)
    ax[1].set_title(f"(b) held-out-trip residual prediction  MAE={mae:.2f}  R2={r2:+.2f}")
    ax[1].legend(); ax[1].grid(alpha=.25)

    # (c) corrected-speed MAE by true-speed regime
    bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30)]
    xs = np.arange(len(bins))
    vt = d.v_true_mean.to_numpy()[ok]
    vs = d.feat_v_spec.to_numpy()[ok]
    ve = d.feat_v_ekf.to_numpy()[ok]
    vml = vs + d.rhat.to_numpy()[ok]
    for lab, arr, c in [("v_spectral (baseline)", vs, "#888"),
                        ("v_ekf (baseline)", ve, "#000"),
                        ("v_ml = v_spec + CatBoost", vml, "#d62728")]:
        y = []
        for lo, hi in bins:
            m = (vt >= lo) & (vt < hi)
            y.append(np.mean(np.abs(arr[m] - vt[m])) if m.sum() >= 3 else np.nan)
        ax[2].plot(xs, y, "-o", label=lab, color=c, lw=2)
    ax[2].set_xticks(xs); ax[2].set_xticklabels([f"{lo}-{hi}" for lo, hi in bins])
    ax[2].set_xlabel("true-speed regime [m/s]"); ax[2].set_ylabel("corrected-speed MAE [m/s]")
    ax[2].set_title("(c) corrected-speed MAE by regime (leave-one-trip-out)")
    ax[2].legend(); ax[2].grid(alpha=.25)

    fig.suptitle("Phase 30 - learned spectral residual correction (5 trips, leave-one-trip-out)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "phase30_summary.png", dpi=130)
    print("wrote", OUT / "phase30_summary.png")


if __name__ == "__main__":
    sys.exit(main())
