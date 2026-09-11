#!/usr/bin/env python3
"""Phase 32 TASK 7/9/10 - all-trip 1-D display-branch evaluation.

For every usable trip: baseline D_route (from the speed-only BASELINE EKF replay,
which is what the tracker's odometer is when bend/censor are off), a separate
D_position that adds gated saturation correction, and D_true from withheld GPS.
Topology is untouched by construction (post-hoc, no feedback path).

Reports, per trip: baseline vs corrected {median, p95, max} |D_err|, D/D_true,
v-bias, gate-fired fraction, whether it helped/hurt, and a bounded-correction
(|delta| <= B) sweep. Covered trips additionally get a full-tracker topology
equality check.
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.isotonic import IsotonicRegression
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).parent))
from phase30_dataset import make_windows, replay_trip

ROOT = Path("/Users/arseniyabramidze/delimobil/gps-route-recovery")
P31 = ROOT / "docs/plots/phase31"
OUT = ROOT / "docs/plots/phase32"
OUT.mkdir(parents=True, exist_ok=True)
TRIP_CONST = ["feat_dep_sigma_spec", "feat_spec_gap", "feat_sigma_spec"]
EXTRAP = {"07-24-s6", "07-26-s0"}
BROKEN_BASELINE = {"07-24-s6", "07-28-s2", "07-30-s1"}   # D/D_true way off, route already lost
WIN_S, HOP_S = 3.0, 0.5
BOUNDS = [50, 100, 150, 200, 300, 500, None]


def cb():
    return CatBoostRegressor(iterations=500, depth=4, learning_rate=0.03,
                             loss_function="MAE", l2_leaf_reg=6.0, random_seed=0, verbose=False)


GRIDS = {p.stem.replace("grid_", ""): p for p in sorted(P31.glob("grid_*.csv"))}
POOL = None
FITC = {}
_GRID = {}
_WIN = {}
_SER = {}
_RHAT = {}


def grid_of(tag):
    if tag not in _GRID:
        _GRID[tag] = (pd.read_csv(GRIDS[tag]) if tag in GRIDS
                      else replay_trip(tag, ROOT / f"runs/phase31/{tag}/trip"))
    return _GRID[tag]


def win_of(tag):
    if tag not in _WIN:
        w = make_windows(grid_of(tag), WIN_S, HOP_S)
        _WIN[tag] = w[w.regime == "outage"].reset_index(drop=True)
    return _WIN[tag]


def get_pool():
    global POOL
    if POOL is None:
        parts = []
        for tag, p in GRIDS.items():
            if tag.startswith("rf-"):
                continue
            w = make_windows(pd.read_csv(p), WIN_S, HOP_S)
            w["trip"] = tag
            parts.append(w)
        POOL = pd.concat(parts, ignore_index=True)
        POOL["day"] = POOL.trip.str.slice(0, 5)
        POOL = POOL[(POOL.regime == "outage") & POOL._moving.astype(bool)
                    & POOL._reliable_vspec.astype(bool)]
        POOL = POOL[np.isfinite(POOL.residual)].reset_index(drop=True)
    return POOL


def fits(day):
    if day in FITC:
        return FITC[day]
    tr = get_pool()
    tr = tr[(tr.day != day) & ~tr.trip.isin(EXTRAP)]
    F = [c for c in tr.columns if c.startswith("feat_") and c not in TRIP_CONST]
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(tr.feat_vprior_minus_vspec.to_numpy(), tr.residual.to_numpy())
    m = cb(); m.fit(tr[F].to_numpy(), tr.residual.to_numpy())
    FITC[day] = (iso, m, F)
    return FITC[day]


def day_of(tag):
    return tag[3:8] if tag.startswith("rf-") else tag[:5]


def trip_series(tag):
    if tag not in _SER:
        g = grid_of(tag)
        o = g[g.outage].reset_index(drop=True)
        t = o.t.to_numpy()
        dt = np.full(len(t), float(np.median(np.diff(t))))
        D_route = o.D.to_numpy() - o.D.to_numpy()[0]
        v_route = o.v_ekf.to_numpy()
        D_true = np.cumsum(o.v_true.to_numpy() * dt)
        D_true = D_true - D_true[0]
        _SER[tag] = (o, t, dt, D_route, v_route, D_true)
    return _SER[tag]


def corrected(tag, mode, gate, bound=None):
    o, t, dt, D_route, v_route, D_true = trip_series(tag)
    w = win_of(tag)
    wt = w.t_end.to_numpy()
    ck = (tag, mode)
    if ck not in _RHAT:
        iso, m, F = fits(day_of(tag))
        if mode == "iso":
            _RHAT[ck] = np.clip(iso.predict(w.feat_vprior_minus_vspec.to_numpy()), -2, 12)
        elif mode == "cb":
            _RHAT[ck] = np.clip(m.predict(w[F].to_numpy()), -2, 12)
        else:
            _RHAT[ck] = np.zeros(len(w))
    rhat = _RHAT[ck]
    frac15 = w.feat_vspec_frac_gt15_30s.to_numpy()
    vmax20 = w.feat_vspec_max_20s.to_numpy()
    sat = np.clip(0.5 * np.clip(frac15 / 0.15, 0, 1)
                  + 0.5 * np.clip((vmax20 - 13.0) / 4.0, 0, 1), 0, 1)
    alpha_w = ((frac15 > 0.02) | (vmax20 > 15.0)).astype(float) if gate == "binary" else sat
    vml_w = np.clip(w.feat_v_spec.to_numpy() + rhat, 0, 33)

    def itp(a):
        return np.interp(t, wt, a, left=a[0], right=a[-1])
    alpha = np.clip(itp(alpha_w), 0, 1)
    vml = itp(vml_w)
    if mode == "none":
        alpha = np.zeros_like(alpha)
    v_pos = v_route + alpha * np.maximum(0.0, vml - v_route)
    delta = np.cumsum((v_pos - v_route) * dt)
    if bound is not None:
        delta = np.clip(delta, -bound, bound)
    D_pos = D_route + delta
    return dict(t=t, D_route=D_route, D_pos=D_pos, D_true=D_true, v_route=v_route,
                v_pos=v_pos, v_true=o.v_true.to_numpy(), alpha=alpha, delta=delta)


def stats(b):
    er = np.abs(b["D_route"] - b["D_true"])
    ep = np.abs(b["D_pos"] - b["D_true"])
    sp = b["D_pos"] - b["D_true"]
    T = b["t"] - b["t"][0]
    dur = T[-1] - T[0]
    def band(a, lo, hi):
        m = (a >= lo) & (a < hi)
        return float(np.mean(m))
    def longest(mask):
        best = cur = 0
        for x in mask:
            cur = cur + 1 if x else 0
            best = max(best, cur)
        return best * float(np.median(np.diff(b["t"])))
    return dict(
        Dr_route=round(float(b["D_route"][-1] / b["D_true"][-1]), 3) if b["D_true"][-1] > 1 else None,
        Dr_pos=round(float(b["D_pos"][-1] / b["D_true"][-1]), 3) if b["D_true"][-1] > 1 else None,
        med_route=round(float(np.median(er)), 0), med_pos=round(float(np.median(ep)), 0),
        p95_route=round(float(np.percentile(er, 95)), 0), p95_pos=round(float(np.percentile(ep, 95)), 0),
        max_route=round(float(er.max()), 0), max_pos=round(float(ep.max()), 0),
        vbias_route=round(float(np.mean(b["v_route"] - b["v_true"])), 2),
        vbias_pos=round(float(np.mean(b["v_pos"] - b["v_true"])), 2),
        vMAE_route=round(float(np.mean(np.abs(b["v_route"] - b["v_true"]))), 2),
        vMAE_pos=round(float(np.mean(np.abs(b["v_pos"] - b["v_true"]))), 2),
        max_lead=round(float(sp.max()), 0), max_lag=round(float(sp.min()), 0),
        gate=round(float(np.mean(b["alpha"] > 0.05)), 2),
        max_delta=round(float(np.max(np.abs(b["delta"]))), 0),
        f_lt25=round(band(ep, 0, 25), 3), f_lt50=round(band(ep, 0, 50), 3),
        f_lt100=round(band(ep, 0, 100), 3), f_lt200=round(band(ep, 0, 200), 3),
        f_gt300=round(band(ep, 300, 1e9), 3), f_gt500=round(band(ep, 500, 1e9), 3),
        f_lt100_route=round(band(er, 0, 100), 3), f_gt300_route=round(band(er, 300, 1e9), 3),
        dur_gt200=round(longest(ep > 200), 0), dur_gt500=round(longest(ep > 500), 0),
        closer=round(float(np.mean(ep < er)), 3),
    )


def main():
    tags = sorted([t for t in GRIDS if not t.startswith("rf-")]) + ["rf-07-26"]
    rows, bnd_rows, wc_rows = [], [], []
    for tag in tags:
        base = stats(corrected(tag, "none", "binary"))
        for mode in ["iso", "cb"]:
            for gate in ["soft", "binary"]:
                s = stats(corrected(tag, mode, gate))
                rows.append(dict(trip=tag, variant=f"{mode}-{gate}",
                                 broken=tag in BROKEN_BASELINE, extrap=tag in EXTRAP, **s))
        # bound sweep on iso-binary + iso-soft
        for gate in ["soft", "binary"]:
            for B in BOUNDS:
                s = stats(corrected(tag, "iso", gate, bound=B))
                bnd_rows.append(dict(trip=tag, gate=gate, B=(B or 9999),
                                     med_pos=s["med_pos"], p95_pos=s["p95_pos"],
                                     max_pos=s["max_pos"], max_route=s["max_route"],
                                     p95_route=s["p95_route"], Dr_pos=s["Dr_pos"],
                                     d_max=round(s["max_pos"] - s["max_route"], 0)))
        # worst-case band stats for the chosen recommended variant (iso-soft, B=200)
        s = stats(corrected(tag, "iso", "soft", bound=200))
        wc_rows.append(dict(trip=tag, broken=tag in BROKEN_BASELINE, **{
            k: s[k] for k in ["f_lt25", "f_lt50", "f_lt100", "f_lt200", "f_gt300", "f_gt500",
                              "f_lt100_route", "f_gt300_route", "max_lead", "max_lag",
                              "dur_gt200", "dur_gt500", "closer", "max_pos", "max_route"]}))

    df = pd.DataFrame(rows); df.to_csv(OUT / "task7_1d_alltrip.csv", index=False)
    bd = pd.DataFrame(bnd_rows); bd.to_csv(OUT / "task10_bound_sweep.csv", index=False)
    wc = pd.DataFrame(wc_rows); wc.to_csv(OUT / "task9_worstcase.csv", index=False)

    print("### TASK 7 - 1-D display branch, all trips (unbounded) ###")
    show = ["trip", "variant", "Dr_route", "Dr_pos", "med_route", "med_pos",
            "p95_route", "p95_pos", "max_route", "max_pos", "vbias_route", "vbias_pos",
            "gate", "max_delta", "closer"]
    for tag in tags:
        sub = df[df.trip == tag]
        print(sub[show].to_string(index=False, header=(tag == tags[0])))
    fair = df[~df.broken & ~df.extrap]
    print("\n### summary (fair set: exclude broken-baseline + extrapolation) ###")
    for v in fair.variant.unique():
        s = fair[fair.variant == v]
        dmax = s.max_pos - s.max_route
        dp95 = s.p95_pos - s.p95_route
        helped = int((dp95 < -15).sum()); hurt = int((dp95 > 15).sum())
        print(f"  {v:11s}  n={len(s)}  helped {helped}  hurt {hurt}  "
              f"median Δp95 {dp95.median():+.0f}  median Δmax {dmax.median():+.0f}  "
              f"worst Δmax {dmax.max():+.0f} ({s.loc[dmax.idxmax(),'trip']})  "
              f"best Δmax {dmax.min():+.0f} ({s.loc[dmax.idxmin(),'trip']})")

    print("\n### TASK 10 - bounded correction sweep (iso-soft), fair set ###")
    bdf = bd[bd.gate == "soft"].merge(df[["trip", "broken", "extrap"]].drop_duplicates(), on="trip")
    bdf = bdf[~bdf.broken & ~bdf.extrap]
    for B in [50, 100, 150, 200, 300, 500, 9999]:
        s = bdf[bdf.B == B]
        dmax = s.max_pos - s.max_route
        print(f"  B={B if B<9999 else 'inf':>4}  median Δmax {dmax.median():+.0f}  "
              f"worst Δmax {dmax.max():+.0f}  best Δmax {dmax.min():+.0f}  "
              f"n_helped(Δmax<-15) {(dmax<-15).sum()}  n_hurt(Δmax>15) {(dmax>15).sum()}  "
              f"rf-07-26 max {s[s.trip=='rf-07-26'].max_pos.iloc[0]:.0f}")

    print("\n### TASK 9 - worst-case position quality (iso-soft, B=200), per trip ###")
    print(wc[["trip", "broken", "f_lt50", "f_lt100", "f_lt200", "f_gt300", "f_gt500",
              "max_lead", "max_lag", "dur_gt200", "max_pos", "max_route"]].to_string(index=False))


if __name__ == "__main__":
    main()
