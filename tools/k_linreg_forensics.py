#!/usr/bin/env python3
"""Can a LINEAR REGRESSION recover the spectral correction?  Diagnostic only.

Follow-up to k_spectral_forensics.py. Tries linear models in two framings:

  Framing A -- predict k_required  (steady 5 s windows)
    A1  k ~ v_true                     (needs the answer; baseline shape)
    A2  k ~ v_spectral                 (observable; the deployable-in-principle one)
    A3  k ~ v_spectral + 5 vib/gyro    (observable, hand features)
    A4  k ~ 42 log band powers         (observable, ridge)

  Framing B -- predict v_true directly (the recalibration framing)
    B1  v_true ~ a + b*v_spectral      (affine rescale == speed-dependent k)
    B2  v_true ~ 42 log band powers, ridge, LEAVE-ONE-TRIP-OUT
    B3  v_true ~ 42 log band powers, ridge, in-trip outage refit  (ceiling, not deployable)

Every model is scored pooled AND cross-trip (fit 07-22 -> test 07-26 and vice
versa) because cross-trip transfer is the only generalisation that matters here.

Hidden GPS used for evaluation only. No production change.
Outputs: docs/plots/k_linreg_models.csv , docs/plots/k_linreg.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geotrace.loader import load_trip
from geotrace.pacman_tracker.config import PacmanConfig
from geotrace.pacman_tracker.tracker import _fit_spectral, _lateral_channel
from geotrace.pacman_tracker.spectral import extract_features, BANDS

ROOT = Path("/Users/arseniyabramidze/delimobil/gps-route-recovery")
OUT = ROOT / "docs" / "plots"
TRIPS = {"07-22": ROOT / "runs/review-20min/2026-07-22/trip",
         "07-26": ROOT / "runs/review-20min/2026-07-26/trip"}
NB = len(BANDS) + 1


def trip_frame(tag: str, d: Path) -> pd.DataFrame:
    """Per spectral feature-window row: true speed, model speed, all 42 powers."""
    trip, _ = load_trip(d)
    cfg = PacmanConfig()
    visible = trip.usable_locations
    vis_end = visible[-1].monotonic_time
    model = _fit_spectral(trip, visible, cfg)
    rt, a_lat, omega, accel, gyro = _lateral_channel(trip)
    f = extract_features(rt, accel, gyro, cfg.spectral_window_s, cfg.spectral_hop_s)
    v_spec = model.predict_many(f.values)

    tv = [(x.monotonic_time, float(x.speed)) for x in visible if x.has_valid_speed]
    tv += [(x.monotonic_time, float(x.speed)) for x in trip.reference_locations
           if x.is_usable and x.has_valid_speed]
    tv.sort()
    tt = np.array([a for a, _ in tv]); tvv = np.array([b for _, b in tv])
    v_true = np.interp(f.times, tt, tvv)

    cols = {f"p{c}_{b}": f.values[:, c * NB + b] for c in range(6) for b in range(NB)}
    df = pd.DataFrame(cols)
    df.insert(0, "trip", tag)
    df.insert(1, "t", f.times)
    df.insert(2, "v_true", v_true)
    df.insert(3, "v_spec", v_spec)
    df.insert(4, "regime", np.where(f.times <= vis_end, "visible", "outage"))
    # hand features
    df["vib_low"] = df[[f"p{c}_0" for c in (0, 1, 2)]].mean(axis=1)
    df["vib_mid"] = df[[f"p{c}_3" for c in (0, 1, 2)]].mean(axis=1)
    df["vib_high"] = df[[f"p{c}_5" for c in (0, 1, 2)]].mean(axis=1)
    df["accel_tot"] = df[[f"p{c}_6" for c in (0, 1, 2)]].mean(axis=1)
    df["gyro_pow"] = df[[f"p{c}_6" for c in (3, 4, 5)]].mean(axis=1)
    return df


def windows(df: pd.DataFrame, w=5.0, dt=0.5, steady=2.0):
    """Non-overlapping steady windows -> mean of every numeric column + k."""
    n = int(round(w / dt))
    df = df.sort_values("t").reset_index(drop=True)
    rows = []
    for i0 in range(0, len(df) - n + 1, n):
        s = df.iloc[i0:i0 + n]
        vt, vs = s.v_true.to_numpy(), s.v_spec.to_numpy()
        if vt.min() < 1.0 or vs.mean() < 2.0:
            continue
        if vt.max() - vt.min() >= steady:
            continue
        if s.regime.iloc[-1] == "visible":
            continue
        r = s.select_dtypes(np.number).mean().to_dict()
        r["trip"] = s.trip.iloc[0]
        r["k"] = r["v_true"] / r["v_spec"]
        rows.append(r)
    return pd.DataFrame(rows)


# -- linear algebra helpers --------------------------------------------------
def fit_ridge(X, y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    A = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
    w = np.linalg.solve(A, Xs.T @ (y - y.mean()))
    b = y.mean()
    return lambda Z: ((Z - mu) / sd) @ w + b


def scores(pred, X, y, hi_mask=None):
    r = pred(X) - y
    out = dict(rmse=float(np.sqrt(np.mean(r ** 2))), mae=float(np.mean(np.abs(r))),
               bias=float(np.mean(r)))
    ss = np.sum((y - y.mean()) ** 2)
    out["r2"] = float(1 - np.sum(r ** 2) / ss) if ss > 0 else float("nan")
    if hi_mask is not None and hi_mask.sum() > 3:
        rh = r[hi_mask]
        out["hi_rmse"] = float(np.sqrt(np.mean(rh ** 2)))
        out["hi_bias"] = float(np.mean(rh))
    return out


def evaluate(name, feats, target, W, lam=0.0, hi_speed=15.0):
    """Pooled in-sample + leave-one-trip-out, on the windowed set W."""
    X = W[feats].to_numpy(float)
    y = W[target].to_numpy(float)
    tr = W["trip"].to_numpy()
    vt = W["v_true"].to_numpy()
    rows = []

    p = fit_ridge(X, y, lam)
    rows.append(dict(model=name, target=target, n_feat=len(feats), eval="pooled_insample",
                     n=len(y), **scores(p, X, y, vt >= hi_speed)))

    for hold in ("07-22", "07-26"):
        m_tr = tr != hold
        m_te = tr == hold
        if m_tr.sum() < 8 or m_te.sum() < 5:
            continue
        p = fit_ridge(X[m_tr], y[m_tr], lam)
        rows.append(dict(model=name, target=target, n_feat=len(feats),
                         eval=f"fit_{'07-26' if hold=='07-22' else '07-22'}_test_{hold}",
                         n=int(m_te.sum()),
                         **scores(p, X[m_te], y[m_te], vt[m_te] >= hi_speed)))
    return rows


def main() -> int:
    frames = {t: trip_frame(t, d) for t, d in TRIPS.items()}
    raw = pd.concat(frames.values(), ignore_index=True)
    W = pd.concat([windows(frames[t]) for t in frames], ignore_index=True)
    print(f"steady 5 s windows: {len(W)}  "
          f"({', '.join(f'{t}:{(W.trip==t).sum()}' for t in frames)})")

    P42 = [f"p{c}_{b}" for c in range(6) for b in range(NB)]
    HAND = ["v_spec", "vib_low", "vib_mid", "vib_high", "accel_tot", "gyro_pow"]

    rows = []
    # ---- Framing A: predict k ------------------------------------------
    rows += evaluate("A1_k~v_true", ["v_true"], "k", W)
    rows += evaluate("A2_k~v_spec", ["v_spec"], "k", W)
    rows += evaluate("A3_k~v_spec+hand", HAND, "k", W, lam=1.0)
    rows += evaluate("A4_k~42powers", P42, "k", W, lam=10.0)
    # ---- Framing B: predict v_true ------------------------------------
    rows += evaluate("B1_vtrue~affine_vspec", ["v_spec"], "v_true", W)
    rows += evaluate("B2_vtrue~42powers_win", P42, "v_true", W, lam=10.0)

    # B3: full-resolution in-trip outage refit (ceiling; needs outage GPS)
    for t, fr in frames.items():
        o = fr[fr.regime == "outage"]
        X = o[P42].to_numpy(float); y = o.v_true.to_numpy(float)
        # 5-fold within the trip's own outage
        idx = np.arange(len(y)); rng = np.random.default_rng(0); rng.shuffle(idx)
        preds = np.zeros(len(y))
        for k in range(5):
            te = idx[k::5]; trn = np.setdiff1d(idx, te)
            p = fit_ridge(X[trn], y[trn], 10.0)
            preds[te] = p(X[te])
        r = preds - y
        hi = y >= 15.0
        rows.append(dict(model="B3_vtrue~42powers_intrip_ceiling", target="v_true",
                         n_feat=42, eval=f"{t}_5fold_within_outage", n=len(y),
                         rmse=float(np.sqrt(np.mean(r ** 2))), mae=float(np.mean(np.abs(r))),
                         bias=float(np.mean(r)),
                         r2=float(1 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2)),
                         hi_rmse=float(np.sqrt(np.mean(r[hi] ** 2))) if hi.sum() else np.nan,
                         hi_bias=float(np.mean(r[hi])) if hi.sum() else np.nan))

    res = pd.DataFrame(rows)
    for c in ("rmse", "mae", "bias", "r2", "hi_rmse", "hi_bias"):
        res[c] = res[c].round(3)
    res.to_csv(OUT / "k_linreg_models.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(res.to_string(index=False))

    _plot(W, frames, P42, HAND)
    print("\nwrote", OUT / "k_linreg_models.csv", "and", OUT / "k_linreg.png")
    return 0


def _plot(W, frames, P42, HAND):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
    col = {"07-22": "#1f77b4", "07-26": "#d62728"}

    # (1) B1 affine v_true~v_spec: cross-trip
    for hold in ("07-22", "07-26"):
        tr = W[W.trip != hold]; te = W[W.trip == hold]
        p = fit_ridge(tr[["v_spec"]].to_numpy(float), tr.v_true.to_numpy(float), 0.0)
        xx = np.linspace(2, 20, 50)
        ax[0].plot(xx, p(xx[:, None]), color=col[hold], lw=2,
                   label=f"fit on other, test {hold}")
        ax[0].scatter(te.v_spec, te.v_true, s=16, alpha=0.5, color=col[hold])
    ax[0].plot([0, 22], [0, 22], "k--", lw=1, label="y=x")
    ax[0].set_xlabel("v_spectral [m/s]"); ax[0].set_ylabel("v_true [m/s]")
    ax[0].set_title("B1: affine recalibration v_true ~ a+b·v_spec (cross-trip)")
    ax[0].legend(); ax[0].grid(alpha=0.25)

    # (2) A4 k~42powers leave-one-trip-out: predicted vs actual k
    X = W[P42].to_numpy(float); y = W.k.to_numpy(float); tr = W.trip.to_numpy()
    for hold in ("07-22", "07-26"):
        m = tr != hold
        p = fit_ridge(X[m], y[m], 10.0)
        ax[1].scatter(y[~m], p(X[~m]), s=18, alpha=0.6, color=col[hold], label=f"test {hold}")
    ax[1].plot([0.5, 2.2], [0.5, 2.2], "k--", lw=1)
    ax[1].set_xlabel("actual k"); ax[1].set_ylabel("predicted k")
    ax[1].set_title("A4: k ~ 42 log band powers, leave-one-trip-out")
    ax[1].legend(); ax[1].grid(alpha=0.25)

    # (3) B2 residual vs speed, leave-one-trip-out
    X = W[P42].to_numpy(float); y = W.v_true.to_numpy(float)
    for hold in ("07-22", "07-26"):
        m = tr != hold
        p = fit_ridge(X[m], y[m], 10.0)
        ax[2].scatter(W.v_true[~m], p(X[~m]) - y[~m], s=18, alpha=0.6,
                      color=col[hold], label=f"test {hold}")
    ax[2].axhline(0, color="k", lw=1, ls="--")
    ax[2].set_xlabel("true speed [m/s]"); ax[2].set_ylabel("v_pred - v_true [m/s]")
    ax[2].set_title("B2: v_true ~ 42 powers, cross-trip residual vs speed")
    ax[2].legend(); ax[2].grid(alpha=0.25)

    fig.suptitle("Linear-regression attempts at the spectral correction  "
                 "(diagnostic, hidden GPS)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "k_linreg.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
