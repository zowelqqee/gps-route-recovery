#!/usr/bin/env python3
"""Diagnostic-only forensics of the spectral speed correction coefficient.

    k_required(t) = v_true(t) / v_spectral(t)

Hidden GPS is used HERE ONLY, for this analysis. Nothing in production changes.
Reproduces the pipeline's own spectral speed model (fit on the visible GPS
window, predicted over the whole trip) and compares it, in robust time windows,
to the withheld GPS speed.

Outputs (docs/plots/):
    k_windows.csv                  every windowed sample, all window sizes
    k_vs_speed_bins.csv            binned k distribution, per trip + pooled
    k_feature_correlations.csv     Spearman(k, feature) on steady 5 s windows
    k_cross_trip.csv               overlapping-bin transfer 07-22 vs 07-26
    k_fit_candidates.csv           descriptive fits + cross-trip errors
    k_vs_true_speed_scatter.png
    k_vs_true_speed_binned.png
    vspec_vs_vtrue.png
    k_vs_vspec.png
    k_fits.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geotrace.loader import load_trip
from geotrace.pacman_tracker.config import PacmanConfig
from geotrace.pacman_tracker.tracker import _fit_spectral, _lateral_channel
from geotrace.pacman_tracker.spectral import extract_features, BANDS

ROOT = Path("/Users/arseniyabramidze/delimobil/gps-route-recovery")
OUT = ROOT / "docs" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

TRIPS = {
    "07-22": ROOT / "runs/review-20min/2026-07-22/trip",
    "07-26": ROOT / "runs/review-20min/2026-07-26/trip",
}
TRIP_COLOR = {"07-22": "#1f77b4", "07-26": "#d62728", "pooled": "#333333"}

GRID_DT = 0.5
WINDOWS = (3.0, 5.0, 10.0)
MAIN_WINDOW = 5.0
V_SPEC_FLOOR = 2.0          # v_spec_mean below this -> ratio unreliable, flagged
STOP_V = 1.0               # either speed mean below this -> stop window
BINS = [0, 3, 6, 9, 12, 14, 16, 18, 20, 22, 25, np.inf]
BIN_LABELS = ["0-3", "3-6", "6-9", "9-12", "12-14", "14-16",
              "16-18", "18-20", "20-22", "22-25", "25+"]


# ----------------------------------------------------------------------------
# per-trip signal reconstruction
# ----------------------------------------------------------------------------
def trip_signals(tag: str, trip_dir: Path) -> dict:
    trip, _ = load_trip(trip_dir)
    cfg = PacmanConfig()

    visible = trip.usable_locations
    visible_end = visible[-1].monotonic_time

    # --- v_spectral: the pipeline's own model, fit on visible window only -----
    model = _fit_spectral(trip, visible, cfg)
    if not model.fitted:
        raise SystemExit(f"{tag}: spectral model did not fit: {model.reason}")
    rt, a_lat, omega, accel, gyro = _lateral_channel(trip)
    feats = extract_features(rt, accel, gyro, cfg.spectral_window_s, cfg.spectral_hop_s)
    v_spec_pred = model.predict_many(feats.values)          # clipped 0..33, as prod
    fs_t = feats.times

    # --- v_true: withheld GPS speed (+ visible-window GPS speed) --------------
    tv = [(f.monotonic_time, float(f.speed)) for f in visible if f.has_valid_speed]
    tv += [(f.monotonic_time, float(f.speed)) for f in trip.reference_locations
           if f.is_usable and f.has_valid_speed]
    tv.sort()
    vt_t = np.array([x[0] for x in tv])
    vt_v = np.array([x[1] for x in tv])

    # --- auxiliary channels at raw motion rate ------------------------------
    a_mag = np.linalg.norm(accel, axis=1)                   # user accel magnitude
    gyro_mag = np.linalg.norm(gyro, axis=1)
    yaw_abs = np.abs(omega)
    lat_abs = np.abs(a_lat)

    # band-power summaries per feature window (log powers, cols c*7+b, total c*7+6)
    def band_col(ch, b):
        return feats.values[:, ch * (len(BANDS) + 1) + b]
    accel_low = np.mean([band_col(c, 0) for c in (0, 1, 2)], axis=0)   # 0.5-2 Hz
    accel_mid = np.mean([band_col(c, 3) for c in (0, 1, 2)], axis=0)   # 8-12 Hz
    accel_high = np.mean([band_col(c, 5) for c in (0, 1, 2)], axis=0)  # 18-25 Hz
    accel_tot = np.mean([band_col(c, 6) for c in (0, 1, 2)], axis=0)
    gyro_tot = np.mean([band_col(c, 6) for c in (3, 4, 5)], axis=0)

    # --- common uniform analysis grid --------------------------------------
    t_lo = max(fs_t[0], vt_t[0])
    t_hi = min(fs_t[-1], vt_t[-1])
    grid = np.arange(t_lo, t_hi, GRID_DT)

    def rs_raw(vals):        # resample a raw-motion-rate channel to the grid
        return np.interp(grid, rt, vals)

    def rs_feat(vals):      # resample a feature-window channel to the grid
        return np.interp(grid, fs_t, vals)

    g = dict(
        tag=tag,
        visible_end=visible_end,
        grid=grid,
        v_spec=rs_feat(v_spec_pred),
        v_true=np.interp(grid, vt_t, vt_v),
        a_mag=rs_raw(a_mag),
        gyro_mag=rs_raw(gyro_mag),
        yaw_abs=rs_raw(yaw_abs),
        lat_abs=rs_raw(lat_abs),
        vib_low=rs_feat(accel_low),
        vib_mid=rs_feat(accel_mid),
        vib_high=rs_feat(accel_high),
        vib_total=rs_feat(accel_tot),
        gyro_power=rs_feat(gyro_tot),
        model=model,
    )
    return g


# ----------------------------------------------------------------------------
# windowing
# ----------------------------------------------------------------------------
def make_windows(g: dict, window_s: float) -> list[dict]:
    grid = g["grid"]
    n = int(round(window_s / GRID_DT))
    rows = []
    for i0 in range(0, len(grid) - n + 1, n):            # non-overlapping
        sl = slice(i0, i0 + n)
        t = grid[sl]
        vt = g["v_true"][sl]
        vs = g["v_spec"][sl]
        if not (np.all(np.isfinite(vt)) and np.all(np.isfinite(vs))):
            continue
        dur = float(t[-1] - t[0] + GRID_DT)
        vt_mean, vs_mean = float(vt.mean()), float(vs.mean())
        # linear slope of true speed across the window
        slope = float(np.polyfit(t - t[0], vt, 1)[0])
        dv = slope * dur
        vt_range = float(vt.max() - vt.min())

        stop = vt_mean < STOP_V or vs_mean < STOP_V
        low_vspec = vs_mean < V_SPEC_FLOOR
        if dv > 1.0:
            dyn = "accel"
        elif dv < -1.0:
            dyn = "decel"
        else:
            dyn = "steady"

        k_mom = vt_mean / vs_mean if vs_mean > 1e-6 else np.nan
        pw = vt / np.clip(vs, 1e-6, None)
        k_med = float(np.median(pw))

        rows.append(dict(
            trip=g["tag"],
            window_s=window_s,
            t_start=round(float(t[0]), 2),
            t_end=round(float(t[-1] + GRID_DT), 2),
            duration_s=round(dur, 2),
            regime="visible" if (t[-1] + GRID_DT) <= g["visible_end"] else "outage",
            v_true_mean=round(vt_mean, 4),
            v_true_median=round(float(np.median(vt)), 4),
            v_true_std=round(float(vt.std()), 4),
            v_true_min=round(float(vt.min()), 4),
            v_true_max=round(float(vt.max()), 4),
            v_true_range=round(vt_range, 4),
            v_spec_mean=round(vs_mean, 4),
            v_spec_median=round(float(np.median(vs)), 4),
            v_spec_std=round(float(vs.std()), 4),
            k_required=round(k_mom, 5),          # mean(v_true)/mean(v_spec) -- primary
            k_median_ratio=round(k_med, 5),      # median(v_true/v_spec)
            delta_v_true=round(dv, 4),
            accel_ms2=round(dv / dur, 5),
            true_speed_std=round(float(vt.std()), 4),
            dyn_class=dyn,
            stop_window=stop,
            low_vspec=low_vspec,
            steady_1ms=vt_range < 1.0,
            steady_2ms=vt_range < 2.0,
            a_mag_mean=round(float(g["a_mag"][sl].mean()), 5),
            accel_mag_abs=round(abs(dv / dur), 5),
            gyro_mag_mean=round(float(g["gyro_mag"][sl].mean()), 5),
            yaw_abs_mean=round(float(g["yaw_abs"][sl].mean()), 5),
            lat_abs_mean=round(float(g["lat_abs"][sl].mean()), 5),
            vib_low=round(float(g["vib_low"][sl].mean()), 4),
            vib_mid=round(float(g["vib_mid"][sl].mean()), 4),
            vib_high=round(float(g["vib_high"][sl].mean()), 4),
            vib_total=round(float(g["vib_total"][sl].mean()), 4),
            gyro_power=round(float(g["gyro_power"][sl].mean()), 4),
        ))
    return rows


# ----------------------------------------------------------------------------
def steady_analysis_set(df: pd.DataFrame, window_s=MAIN_WINDOW, steady_col="steady_2ms"):
    """Outage, steady, moving, reliable-v_spec 5 s windows -- the main k(v) set."""
    return df[(df.window_s == window_s)
             & (df.regime == "outage")
             & (df[steady_col])
             & (~df.stop_window)
             & (~df.low_vspec)].copy()


def binned_stats(sub: pd.DataFrame, label: str) -> pd.DataFrame:
    out = []
    cats = pd.cut(sub.v_true_mean, BINS, labels=BIN_LABELS, right=False)
    for bl in BIN_LABELS:
        b = sub[cats == bl]
        k = b.k_required.to_numpy()
        k = k[np.isfinite(k)]
        if len(k) == 0:
            out.append(dict(set=label, speed_bin=bl, n=0))
            continue
        out.append(dict(
            set=label, speed_bin=bl, n=int(len(k)),
            v_true_mean=round(float(b.v_true_mean.mean()), 2),
            k_median=round(float(np.median(k)), 4),
            k_mean=round(float(np.mean(k)), 4),
            k_std=round(float(np.std(k)), 4),
            k_p10=round(float(np.percentile(k, 10)), 4),
            k_p25=round(float(np.percentile(k, 25)), 4),
            k_p75=round(float(np.percentile(k, 75)), 4),
            k_p90=round(float(np.percentile(k, 90)), 4),
        ))
    return pd.DataFrame(out)


def bin_center(bl: str) -> float:
    if bl == "25+":
        return 26.0
    a, b = bl.split("-")
    return (float(a) + float(b)) / 2.0


# ----------------------------------------------------------------------------
# fits (diagnostic only)
# ----------------------------------------------------------------------------
def fit_piecewise_linear(x, y, knots=np.arange(6, 20.1, 0.5)):
    best = None
    for v0 in knots:
        b = np.maximum(x - v0, 0.0)
        A = np.column_stack([np.ones_like(x), x, b])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ coef
        sse = float(r @ r)
        if best is None or sse < best[0]:
            best = (sse, v0, coef)
    _, v0, coef = best
    return dict(kind="piecewise_linear", v0=float(v0),
                predict=lambda xx, c=coef, v=v0: c[0] + c[1] * xx + c[2] * np.maximum(xx - v, 0.0))


def fit_linear_after_threshold(x, y, knots=np.arange(6, 20.1, 0.5)):
    # k = 1 for v<=v0 ; k = 1 + a (v-v0) for v>v0
    best = None
    for v0 in knots:
        m = x > v0
        if m.sum() < 5:
            continue
        d = x[m] - v0
        a = float(np.sum(d * (y[m] - 1.0)) / np.sum(d * d))
        pred = np.where(x > v0, 1.0 + a * np.maximum(x - v0, 0.0), 1.0)
        r = y - pred
        sse = float(r @ r)
        if best is None or sse < best[0]:
            best = (sse, v0, a)
    _, v0, a = best
    return dict(kind="linear_after_threshold", v0=float(v0), a=float(a),
                predict=lambda xx, a=a, v=v0: np.where(xx > v, 1.0 + a * np.maximum(xx - v, 0.0), 1.0))


def fit_quadratic_after_threshold(x, y, knots=np.arange(6, 18.1, 0.5)):
    best = None
    for v0 in knots:
        m = x > v0
        if m.sum() < 6:
            continue
        d = x[m] - v0
        A = np.column_stack([d, d ** 2])
        coef, *_ = np.linalg.lstsq(A, y[m] - 1.0, rcond=None)
        pred = np.where(x > v0, 1.0 + coef[0] * np.maximum(x - v0, 0.0)
                        + coef[1] * np.maximum(x - v0, 0.0) ** 2, 1.0)
        r = y - pred
        sse = float(r @ r)
        if best is None or sse < best[0]:
            best = (sse, v0, coef)
    _, v0, coef = best
    return dict(kind="quadratic_after_threshold", v0=float(v0), a=float(coef[0]), b=float(coef[1]),
                predict=lambda xx, c=coef, v=v0: np.where(
                    xx > v, 1.0 + c[0] * np.maximum(xx - v, 0.0) + c[1] * np.maximum(xx - v, 0.0) ** 2, 1.0))


def isotonic(x, y):
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    # pool adjacent violators
    w = np.ones_like(ys)
    v = ys.copy()
    i = 0
    blocks = [[v[k], w[k], 1] for k in range(len(v))]
    merged = True
    while merged:
        merged = False
        out = []
        for blk in blocks:
            if out and out[-1][0] > blk[0] + 1e-12:
                tv, tw, tc = out[-1]
                nv = (tv * tw + blk[0] * blk[1]) / (tw + blk[1])
                out[-1] = [nv, tw + blk[1], tc + blk[2]]
                merged = True
            else:
                out.append(blk)
        blocks = out
    fitted = np.concatenate([[b[0]] * b[2] for b in blocks])
    inv = np.empty_like(fitted)
    inv[order] = fitted
    return dict(kind="isotonic", predict=lambda xx, xs=xs, f=fitted: np.interp(xx, xs, f))


def err(pred, x, y):
    r = pred(x) - y
    return dict(rmse=float(np.sqrt(np.mean(r ** 2))), mae=float(np.mean(np.abs(r))))


# ----------------------------------------------------------------------------
_SIGNALS: dict = {}


def main() -> int:
    global _SIGNALS
    signals = {tag: trip_signals(tag, d) for tag, d in TRIPS.items()}
    _SIGNALS = signals

    all_rows = []
    for tag, g in signals.items():
        for w in WINDOWS:
            all_rows += make_windows(g, w)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "k_windows.csv", index=False)
    print(f"k_windows.csv: {len(df)} rows "
          f"({', '.join(f'{w}s:{(df.window_s==w).sum()}' for w in WINDOWS)})")

    # ---- main steady 5 s set (both steady defs) -----------------------------
    s2 = steady_analysis_set(df, steady_col="steady_2ms")
    s1 = steady_analysis_set(df, steady_col="steady_1ms")
    print(f"steady 5s windows: <2 m/s range n={len(s2)} "
          f"(07-22 {sum(s2.trip=='07-22')}, 07-26 {sum(s2.trip=='07-26')}); "
          f"<1 m/s range n={len(s1)}")

    # sanity: k in the visible (in-sample) window should sit near 1
    vis = df[(df.window_s == MAIN_WINDOW) & (df.regime == "visible")
             & df.steady_2ms & ~df.stop_window & ~df.low_vspec]
    if len(vis):
        print(f"visible-window steady k (in-sample sanity): median "
              f"{vis.k_required.median():.3f} n={len(vis)}")

    # ---- binned CSV ------------------------------------------------------
    parts = [binned_stats(s2[s2.trip == "07-22"], "07-22"),
             binned_stats(s2[s2.trip == "07-26"], "07-26"),
             binned_stats(s2, "pooled")]
    bins_df = pd.concat(parts, ignore_index=True)
    bins_df.to_csv(OUT / "k_vs_speed_bins.csv", index=False)
    print("k_vs_speed_bins.csv written")

    # ================= PLOTS =================
    _plot_scatter(s2, df)
    _plot_binned(bins_df)
    _plot_vspec_vs_vtrue(s2)
    _plot_k_vs_vspec(s2, df)

    # ---- forensic correlations (section 8) --------------------------------
    _correlations(s2)

    # ---- cross-trip transfer (section 9) ---------------------------------
    _cross_trip(bins_df, s2)

    # ---- fits (section 10) ----------------------------------------------
    _fits(s2)

    print("\nAll outputs in", OUT)
    return 0


def _plot_scatter(s2: pd.DataFrame, df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    # (a) RAW pointwise k(t) = v_true / v_spec on the 0.5 s grid, outage only
    npw = 0
    for tag, g in _SIGNALS.items():
        m = g["grid"] > g["visible_end"]
        vt, vs = g["v_true"][m], g["v_spec"][m]
        ok = (vs > V_SPEC_FLOOR) & (vt > STOP_V)
        axes[0].scatter(vt[ok], vt[ok] / vs[ok], s=6, alpha=0.2,
                        color=TRIP_COLOR[tag], label=tag)
        npw += int(ok.sum())
    axes[0].set_title(f"(a) RAW pointwise k(t) on 0.5 s grid, outage  n={npw}")
    axes[0].legend()

    # (b) steady, by trip
    for tag in ("07-22", "07-26"):
        b = s2[s2.trip == tag]
        axes[1].scatter(b.v_true_mean, b.k_required, s=20, alpha=0.7,
                        color=TRIP_COLOR[tag], label=f"{tag}  (n={len(b)})")
    axes[1].set_title("(b) steady 5 s windows (<2 m/s range), by trip")
    axes[1].legend()

    # (c) steady, pooled
    axes[2].scatter(s2.v_true_mean, s2.k_required, s=20, alpha=0.6,
                    color=TRIP_COLOR["pooled"])
    axes[2].set_title(f"(c) steady 5 s windows, pooled  n={len(s2)}")

    for ax in axes:
        ax.axhline(1.0, color="grey", lw=0.8, ls="--")
        ax.set_xlim(0, 25)
        ax.set_ylim(0.5, 2.2)
        ax.set_xlabel("true speed  v_true_mean  [m/s]")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("k_required = mean(v_true) / mean(v_spectral)")
    fig.suptitle("Spectral correction coefficient vs true speed  "
                 "(diagnostic, hidden GPS)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "k_vs_true_speed_scatter.png", dpi=130)
    plt.close(fig)
    print("k_vs_true_speed_scatter.png written")


def _plot_binned(bins_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 7))
    for label in ("07-22", "07-26", "pooled"):
        d = bins_df[(bins_df.set == label) & (bins_df.n >= 3)].copy()
        if d.empty:
            continue
        x = d.speed_bin.map(bin_center).to_numpy()
        med = d.k_median.to_numpy()
        c = TRIP_COLOR[label]
        ax.plot(x, med, "-o", color=c, label=f"{label} median", lw=2, zorder=3)
        if label == "pooled":
            ax.fill_between(x, d.k_p25, d.k_p75, color=c, alpha=0.20,
                            label="pooled p25-p75")
            ax.fill_between(x, d.k_p10, d.k_p90, color=c, alpha=0.10,
                            label="pooled p10-p90")
        else:
            ax.fill_between(x, d.k_p10, d.k_p90, color=c, alpha=0.12)
        for _, row in d.iterrows():
            ax.annotate(f"n={row.n}", (bin_center(row.speed_bin), row.k_median),
                        textcoords="offset points", xytext=(0, 7),
                        fontsize=7, ha="center", color=c)
    ax.axhline(1.0, color="grey", lw=0.8, ls="--")
    ax.set_xlim(0, 25)
    ax.set_ylim(0.5, 2.2)
    ax.set_xlabel("true-speed bin centre  [m/s]")
    ax.set_ylabel("k_required")
    ax.set_title("k_required by true-speed bin: median, p25-p75, p10-p90  "
                 "(steady 5 s windows; no fit overlaid)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "k_vs_true_speed_binned.png", dpi=130)
    plt.close(fig)
    print("k_vs_true_speed_binned.png written")


def _plot_vspec_vs_vtrue(s2: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 8))
    for tag in ("07-22", "07-26"):
        b = s2[s2.trip == tag]
        ax.scatter(b.v_true_mean, b.v_spec_mean, s=22, alpha=0.55,
                   color=TRIP_COLOR[tag], label=tag)
        # binned-mean v_spec vs v_true -- shows the saturation knee per trip
        edges = np.arange(0, 26, 2.0)
        cx, cy = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mm = (b.v_true_mean >= lo) & (b.v_true_mean < hi)
            if mm.sum() >= 3:
                cx.append(float(b.v_true_mean[mm].mean()))
                cy.append(float(b.v_spec_mean[mm].mean()))
        ax.plot(cx, cy, "-o", color=TRIP_COLOR[tag], lw=2.5, ms=5)
    lim = 26
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y = x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("true speed  [m/s]")
    ax.set_ylabel("v_spectral (model output)  [m/s]")
    ax.set_title("Spectral speed vs true speed -- where the model saturates  "
                 "(steady 5 s windows)")
    ax.legend()
    ax.grid(alpha=0.25)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT / "vspec_vs_vtrue.png", dpi=130)
    plt.close(fig)
    print("vspec_vs_vtrue.png written")


def _plot_k_vs_vspec(s2: pd.DataFrame, df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 7))
    for tag in ("07-22", "07-26"):
        b = s2[s2.trip == tag]
        ax.scatter(b.v_spec_mean, b.k_required, s=22, alpha=0.7,
                   color=TRIP_COLOR[tag], label=tag)
    ax.axhline(1.0, color="grey", lw=0.8, ls="--")
    ax.axvline(13.0, color="green", lw=0.8, ls=":", label="v_spec ~ 13 (saturation)")
    ax.set_xlim(0, 25)
    ax.set_ylim(0.5, 2.2)
    ax.set_xlabel("v_spectral (model output)  [m/s]")
    ax.set_ylabel("k_required = mean(v_true) / mean(v_spectral)")
    ax.set_title("k_required vs the OBSERVABLE v_spectral -- identifiability check  "
                 "(steady 5 s windows)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "k_vs_vspec.png", dpi=130)
    plt.close(fig)
    print("k_vs_vspec.png written")


def _correlations(s2: pd.DataFrame):
    feats = ["v_true_mean", "v_spec_mean", "accel_mag_abs", "a_mag_mean",
             "vib_low", "vib_mid", "vib_high", "vib_total", "gyro_power",
             "gyro_mag_mean", "yaw_abs_mean", "lat_abs_mean", "t_start"]
    rows = []
    k = s2.k_required.to_numpy()
    for f in feats:
        x = s2[f].to_numpy()
        good = np.isfinite(x) & np.isfinite(k)
        if good.sum() < 10:
            continue
        rho, p = stats.spearmanr(x[good], k[good])
        rows.append(dict(feature=f, spearman_rho=round(float(rho), 3),
                         p_value=round(float(p), 4), n=int(good.sum())))

    # trip effect after conditioning on true speed:
    # residual of k after a smooth (binned-median) speed model, then compare trips
    order = np.argsort(s2.v_true_mean.to_numpy())
    xs = s2.v_true_mean.to_numpy()
    # LOESS-ish: rolling median in speed
    resid = np.full(len(s2), np.nan)
    xv = s2.v_true_mean.to_numpy()
    for i in range(len(s2)):
        near = np.abs(xv - xv[i]) <= 1.5
        if near.sum() >= 5:
            resid[i] = k[i] - np.median(k[near])
    s2 = s2.assign(k_resid_speed=resid)
    r22 = s2[s2.trip == "07-22"].k_resid_speed.dropna()
    r26 = s2[s2.trip == "07-26"].k_resid_speed.dropna()
    trip_row = None
    if len(r22) > 5 and len(r26) > 5:
        u, pu = stats.mannwhitneyu(r22, r26, alternative="two-sided")
        trip_row = dict(feature="trip (after conditioning on v_true)",
                        median_resid_07_22=round(float(r22.median()), 3),
                        median_resid_07_26=round(float(r26.median()), 3),
                        mannwhitney_p=round(float(pu), 4),
                        n=int(len(r22) + len(r26)))
    cdf = pd.DataFrame(rows)
    cdf.to_csv(OUT / "k_feature_correlations.csv", index=False)
    print("\nk_feature_correlations.csv (Spearman rho with k, steady 5 s):")
    print(cdf.to_string(index=False))
    if trip_row:
        print("\nresidual trip effect after conditioning on v_true:")
        for kk, vv in trip_row.items():
            print(f"   {kk}: {vv}")
        pd.DataFrame([trip_row]).to_csv(OUT / "k_trip_effect.csv", index=False)


def _cross_trip(bins_df: pd.DataFrame, s2: pd.DataFrame):
    rows = []
    for bl in BIN_LABELS:
        a = bins_df[(bins_df.set == "07-22") & (bins_df.speed_bin == bl)]
        b = bins_df[(bins_df.set == "07-26") & (bins_df.speed_bin == bl)]
        if a.empty or b.empty or int(a.n.iloc[0]) < 3 or int(b.n.iloc[0]) < 3:
            continue
        ka = s2[(s2.trip == "07-22")].pipe(
            lambda d: d[pd.cut(d.v_true_mean, BINS, labels=BIN_LABELS, right=False) == bl]
        ).k_required.to_numpy()
        kb = s2[(s2.trip == "07-26")].pipe(
            lambda d: d[pd.cut(d.v_true_mean, BINS, labels=BIN_LABELS, right=False) == bl]
        ).k_required.to_numpy()
        # bootstrap CI on the median difference
        rng = np.random.default_rng(0)
        diffs = [np.median(rng.choice(kb, len(kb))) - np.median(rng.choice(ka, len(ka)))
                 for _ in range(2000)]
        rows.append(dict(
            speed_bin=bl,
            n_07_22=len(ka), n_07_26=len(kb),
            median_k_07_22=round(float(np.median(ka)), 3),
            median_k_07_26=round(float(np.median(kb)), 3),
            median_diff=round(float(np.median(kb) - np.median(ka)), 3),
            diff_ci95_lo=round(float(np.percentile(diffs, 2.5)), 3),
            diff_ci95_hi=round(float(np.percentile(diffs, 97.5)), 3),
        ))
    cdf = pd.DataFrame(rows)
    cdf.to_csv(OUT / "k_cross_trip.csv", index=False)
    print("\nk_cross_trip.csv (overlapping speed bins):")
    print(cdf.to_string(index=False) if len(cdf) else "  (no bin has n>=3 on BOTH trips)")


def _fits(s2: pd.DataFrame):
    x = s2.v_true_mean.to_numpy()
    y = s2.k_required.to_numpy()
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    tr = s2.trip.to_numpy()[good]

    builders = {
        "piecewise_linear": fit_piecewise_linear,
        "linear_after_threshold": fit_linear_after_threshold,
        "quadratic_after_threshold": fit_quadratic_after_threshold,
        "isotonic": isotonic,
    }
    rows = []
    fitted_models = {}
    for name, fn in builders.items():
        m = fn(x, y)
        fitted_models[name] = m
        e = err(m["predict"], x, y)
        row = dict(fit=name, pooled_rmse=round(e["rmse"], 4), pooled_mae=round(e["mae"], 4))
        for p in ("v0", "a", "b"):
            if p in m:
                row[p] = round(m[p], 4)
        # cross-trip
        x22, y22 = x[tr == "07-22"], y[tr == "07-22"]
        x26, y26 = x[tr == "07-26"], y[tr == "07-26"]
        if len(x22) > 8 and len(x26) > 8:
            m22 = fn(x22, y22)
            m26 = fn(x26, y26)
            e_22to26 = err(m22["predict"], x26, y26)
            e_26to22 = err(m26["predict"], x22, y22)
            row["fit22_test26_rmse"] = round(e_22to26["rmse"], 4)
            row["fit22_test26_mae"] = round(e_22to26["mae"], 4)
            row["fit26_test22_rmse"] = round(e_26to22["rmse"], 4)
            row["fit26_test22_mae"] = round(e_26to22["mae"], 4)
        rows.append(row)
    fdf = pd.DataFrame(rows)
    fdf.to_csv(OUT / "k_fit_candidates.csv", index=False)
    print("\nk_fit_candidates.csv:")
    print(fdf.to_string(index=False))

    # plot fits over pooled scatter + residuals
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].scatter(x, y, s=16, alpha=0.4, color="#888")
    xx = np.linspace(0, 25, 300)
    for name, m in fitted_models.items():
        axes[0].plot(xx, m["predict"](xx), lw=2, label=name)
    axes[0].axhline(1.0, color="grey", lw=0.8, ls="--")
    axes[0].set_xlim(0, 25)
    axes[0].set_ylim(0.5, 2.2)
    axes[0].set_xlabel("true speed [m/s]")
    axes[0].set_ylabel("k_required")
    axes[0].set_title("Descriptive fits over pooled steady 5 s windows")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    for name, m in fitted_models.items():
        axes[1].scatter(x, m["predict"](x) - y, s=12, alpha=0.4, label=name)
    axes[1].axhline(0.0, color="grey", lw=0.8, ls="--")
    axes[1].set_xlabel("true speed [m/s]")
    axes[1].set_ylabel("residual  (fit - k_required)")
    axes[1].set_title("Fit residual vs speed")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "k_fits.png", dpi=130)
    plt.close(fig)
    print("k_fits.png written")


if __name__ == "__main__":
    sys.exit(main())
