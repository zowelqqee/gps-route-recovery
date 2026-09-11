#!/usr/bin/env python3
"""Freeze the Phase 32 iso-binary display-residual model into a shipped artifact.

Fits exactly the estimator that Phase 32 (`tools/phase32_alltrip.py`) used for
the iso-binary display branch:

    residual_hat = clip( isotonic_increasing( v_prior - v_spectral ), -2, 12 )
    gate on iff  rolling_30s_fraction(v_spectral > 15) > 0.02
             or  rolling_20s_max(v_spectral) > 15

then  v_position = v_route + gate * max(0, (v_spectral + residual_hat) - v_route).

The isotonic curve is fit on the pooled 2 s->3 s rolling windows of the 21
in-regime sessions (all Phase 31 sessions minus the two 2x-extrapolation /
broken-mount trips). A leave-07-26-out variant is also written for the
acceptance-test comparison. NO retraining happens at tracker runtime.

Output: processor/src/geotrace/pacman_tracker/data/display_residual_iso.json
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parent))
from phase30_dataset import make_windows

ROOT = Path("/Users/arseniyabramidze/delimobil/gps-route-recovery")
P31 = ROOT / "docs/plots/phase31"
DEST = ROOT / "processor/src/geotrace/pacman_tracker/data/display_residual_iso.json"
EXTRAP = {"07-24-s6", "07-26-s0"}
WIN_S, HOP_S = 3.0, 0.5
RESID_CLIP = (-2.0, 12.0)
GATE_FRAC15_THR = 0.02
GATE_VMAX20_THR = 15.0
GATE_FRAC_LEVEL = 15.0          # v_spectral level the "fraction above" gate counts
GATE_FRAC_WINDOW_S = 30.0
GATE_MAX_WINDOW_S = 20.0
RESID_FEATURE_WINDOW_S = 3.0


def pool(exclude_day: str | None):
    parts = []
    for p in sorted(P31.glob("grid_*.csv")):
        tag = p.stem.replace("grid_", "")
        if tag.startswith("rf-") or tag in EXTRAP:
            continue
        if exclude_day and tag[:5] == exclude_day:
            continue
        w = make_windows(pd.read_csv(p), WIN_S, HOP_S)
        w = w[(w.regime == "outage") & w._moving.astype(bool) & w._reliable_vspec.astype(bool)]
        parts.append(w)
    d = pd.concat(parts, ignore_index=True)
    return d[np.isfinite(d.residual)].reset_index(drop=True)


def fit_curve(d, n_grid=201):
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(d.feat_vprior_minus_vspec.to_numpy(), d.residual.to_numpy())
    x = d.feat_vprior_minus_vspec.to_numpy()
    xg = np.linspace(np.percentile(x, 0.5), np.percentile(x, 99.5), n_grid)
    yg = np.clip(iso.predict(xg), *RESID_CLIP)
    return xg.round(4).tolist(), yg.round(4).tolist(), int(len(d))


def main():
    xg, yg, n_all = fit_curve(pool(None))
    xg26, yg26, n26 = fit_curve(pool("07-26"))
    art = {
        "_comment": "Phase 32 iso-binary display-residual estimator. Frozen. "
                    "Do not retrain at runtime. See docs/DISPLAY_ODOMETRY_SPLIT.md.",
        "schema": 1,
        "estimator": "iso-binary",
        "residual_feature": "mean(v_prior - v_spectral) over trailing 3 s",
        "residual_feature_window_s": RESID_FEATURE_WINDOW_S,
        "residual_clip_ms": list(RESID_CLIP),
        "isotonic_grid_x": xg,          # v_prior - v_spectral  [m/s]
        "isotonic_grid_y": yg,          # residual_hat = v_true - v_spectral  [m/s]
        "isotonic_grid_x_leave_0726_out": xg26,
        "isotonic_grid_y_leave_0726_out": yg26,
        "gate": {
            "kind": "binary_or",
            "frac_above_level_ms": GATE_FRAC_LEVEL,
            "frac_window_s": GATE_FRAC_WINDOW_S,
            "frac_threshold": GATE_FRAC15_THR,
            "max_window_s": GATE_MAX_WINDOW_S,
            "max_threshold_ms": GATE_VMAX20_THR,
        },
        "apply": "v_position = v_route + gate * max(0, (v_spectral + residual_hat) - v_route)",
        "train": {
            "sessions": "Phase 31 in-regime set (23 minus 07-24-s6, 07-26-s0)",
            "n_windows_all": n_all,
            "n_windows_leave_0726_out": n26,
            "window_s": WIN_S, "hop_s": HOP_S,
        },
    }
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(art, indent=1), encoding="utf-8")
    print(f"wrote {DEST}")
    print(f"  isotonic curve: x {xg[0]:.1f}..{xg[-1]:.1f}  y {yg[0]:.2f}..{yg[-1]:.2f}  n={n_all}")
    # sanity: curve at a few points
    for xv in (-4, -2, 0, 2, 4, 6, 8):
        print(f"    residual_hat(v_prior-v_spec={xv:+d}) = {np.interp(xv, xg, yg):+.2f}   "
              f"(leave-0726-out {np.interp(xv, xg26, yg26):+.2f})")


if __name__ == "__main__":
    main()
