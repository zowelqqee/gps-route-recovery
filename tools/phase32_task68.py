#!/usr/bin/env python3
"""Phase 32 TASK 6 (07-26 full table incl. bounded) + TASK 8 (parallel
longitudinal hypotheses, oracle-choose-at-anchor) + TASK 11 (coverage)."""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from phase32_alltrip import corrected, stats, trip_series, GRIDS, day_of, fits, win_of

OUT = Path("/Users/arseniyabramidze/delimobil/gps-route-recovery/docs/plots/phase32")


def task6_full(tag="rf-07-26"):
    print(f"\n### TASK 6 - {tag} full table ###")
    rows = []
    for name, mode, gate, B in [
            ("A_baseline", "none", "soft", None),
            ("iso-soft unbounded", "iso", "soft", None),
            ("iso-binary unbounded", "iso", "binary", None),
            ("cb-soft unbounded", "cb", "soft", None),
            ("iso-soft B=150", "iso", "soft", 150),
            ("iso-soft B=250", "iso", "soft", 250),
            ("cb-soft B=250", "cb", "soft", 250),
            ("iso-binary B=250", "iso", "binary", 250)]:
        b = corrected(tag, mode, gate, bound=B)
        s = stats(b)
        rows.append(dict(run=name, **{k: s[k] for k in
                    ["Dr_pos", "med_pos", "p95_pos", "max_pos", "vbias_pos", "vMAE_pos",
                     "max_lead", "max_lag", "gate", "max_delta", "closer",
                     "f_lt100", "f_lt200", "f_gt300", "dur_gt200"]}))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"task6_full_{tag}.csv", index=False)
    print(df.to_string(index=False))
    s0 = stats(corrected(tag, "none", "soft"))
    print(f"  baseline: max {s0['max_route']} p95 {s0['p95_route']} med {s0['med_route']} "
          f"D/D_true {s0['Dr_route']} vMAE {s0['vMAE_route']}")


def task8_parallel():
    """Two longitudinal states on ONE topology; at a D-independent anchor
    (here: the withheld-GPS position at fixed elapsed times, used ONLY to score
    which hypothesis was closer - never to pick one online) compare residuals."""
    print("\n### TASK 8 - parallel longitudinal hypotheses (oracle-at-anchor, diagnostic) ###")
    tags = [t for t in GRIDS if not t.startswith("rf-")] + ["rf-07-26"]
    rows = []
    for tag in tags:
        o, t, dt, D_route, v_route, D_true = trip_series(tag)
        b = corrected(tag, "cb", "soft")            # D_corr
        D_base, D_corr = D_route, b["D_pos"]
        # "anchors" = elapsed-time checkpoints; residual = D_true - D_hyp
        base_wins = corr_wins = 0
        oracle_err = base_err = 0.0
        checks = np.arange(60, t[-1] - t[0], 60)
        sep = []
        for el in checks:
            i = int(np.argmin(np.abs((t - t[0]) - el)))
            rb = abs(D_true[i] - D_base[i]); rc = abs(D_true[i] - D_corr[i])
            sep.append(abs(D_corr[i] - D_base[i]))
            base_err += rb; oracle_err += min(rb, rc)
            if rc < rb - 5:
                corr_wins += 1
            elif rb < rc - 5:
                base_wins += 1
        n = len(checks)
        rows.append(dict(trip=tag, n_anchors=n, base_wins=base_wins, corr_wins=corr_wins,
                         tie=n - base_wins - corr_wins,
                         oracle_gain_pct=round(100 * (1 - oracle_err / max(base_err, 1e-6)), 1),
                         mean_separation=round(float(np.mean(sep)), 0),
                         Dr_base=round(float(D_base[-1] / D_true[-1]), 3) if D_true[-1] > 1 else None,
                         Dr_corr=round(float(D_corr[-1] / D_true[-1]), 3) if D_true[-1] > 1 else None))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "task8_parallel.csv", index=False)
    print(df.to_string(index=False))
    print(f"\n  totals: corr wins {df.corr_wins.sum()}  base wins {df.base_wins.sum()}  "
          f"tie {df.tie.sum()}  | median oracle gain {df.oracle_gain_pct.median():.1f}%")
    # does the separation between hypotheses predict which is right? (proxy for uncertainty)
    print("  -> if a small bank of longitudinal hypotheses were kept, an anchor could "
          "pick the better one; online (no hidden GPS) there is still no selector.")


def task11_coverage():
    print("\n### TASK 11 - uncertainty coverage (D_position vs hidden truth) ###")
    tags = [t for t in GRIDS if not t.startswith("rf-")] + ["rf-07-26"]
    zr, zp = [], []
    for tag in tags:
        o, t, dt, D_route, v_route, D_true = trip_series(tag)
        sd = o.sigma_v.to_numpy()  # not a distance sigma; use grid's D sigma proxy
        # distance sigma: integrate speed sigma crudely (route branch already has one
        # in speed_trace; here approximate with grid pvv growth)
        sigmaD = np.sqrt(np.maximum(o.pvv.to_numpy(), 0)) * np.sqrt(np.arange(1, len(t) + 1) * dt[0])
        sigmaD = np.maximum(sigmaD, 20.0)
        b = corrected(tag, "cb", "soft", bound=250)
        er = D_route - D_true
        ep = b["D_pos"] - D_true
        zr.append(er / sigmaD); zp.append(ep / sigmaD)
    zr = np.concatenate(zr); zp = np.concatenate(zp)
    for lab, z in [("route", zr), ("position(B=250)", z if False else zp)]:
        print(f"  {lab:16s}  |z|<=1 {np.mean(np.abs(z) <= 1):.2f}  <=2 {np.mean(np.abs(z) <= 2):.2f}  "
              f"<=3 {np.mean(np.abs(z) <= 3):.2f}   (target 0.68 / 0.95 / 0.997)")
    print("  NOTE the two branches share the accel channel, ZUPT and the spectral model - "
          "their errors are correlated, sigma_D_position is NOT independent of sigma_D_route.")


if __name__ == "__main__":
    task6_full("rf-07-26")
    task6_full("07-26-s1")
    task8_parallel()
    task11_coverage()
