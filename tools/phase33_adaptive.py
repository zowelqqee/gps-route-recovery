#!/usr/bin/env python3
"""Phase 33 - adaptive saturation authority.

Base formula unchanged:  delta_v = gamma(t) * gate * max(0, v_ml - v_route)
gamma(t) is picked from CAUSAL OBSERVABLE features only (no hidden GPS, no ML):

  A : gamma by  mean_3s(v_prior - v_spectral)          weak<-1.5 / mid / strong>+1.8
  B : gamma by  frac_30s(v_spectral > 15)              weak<0.10 / mid / strong>0.18
  C : gamma by  continuous gate-on duration [s]        weak<24  / mid / strong>54
  best : gamma by mean_20s(v_prior - v_spectral)       weak<-1.8 / mid / strong>+0.7
  D : best, but clamp gamma up one tier when vmax_20s > 18   (two-feature)

gamma tiers: weak=1.0, mid=1.5, strong=2.0. Thresholds from the ON-sample
feature distribution, NOT from 07-26 hidden error.

Topology / v_route / isotonic curve / gate timings / thresholds / anchors /
route decisions are byte-identical to production Phase 32. Diagnostic only.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geotrace.coordinates import LocalFrame
from geotrace.loader import load_trip
from geotrace.road_graph import RoadNetwork, clip_graph, load_graph
from geotrace.pacman_tracker.config import PacmanConfig
from geotrace.pacman_tracker.roadmap import RoadGeometry
from geotrace.pacman_tracker.tracker import build_inputs, PacmanTracker
from geotrace.pacman_tracker.diagnostics import map_match_reference, GroundTruthObserver
from geotrace.pacman_tracker.benchmark import _single_path_evaluation
import geotrace.pacman_tracker.display_position as dpm

sys.path.insert(0, str(Path(__file__).parent))
from phase30_dataset import make_windows

ROOT = Path("/Users/arseniyabramidze/delimobil/gps-route-recovery")
OUT = ROOT / "docs/plots/phase32"
P31 = ROOT / "docs/plots/phase31"
GRAPH = ROOT / "runs/review-map.graphml"
RF = ROOT / "runs/review-final/2026-07-26/trip"
TIERS = (1.0, 1.5, 2.0)


# ---------- gamma rules (observable only) --------------------------------
def _tier(x, lo, hi):
    return TIERS[0] if x < lo else (TIERS[2] if x > hi else TIERS[1])


def g_const(val):
    return lambda f: val


def g_A(f):    return _tier(f["vprior_minus_vspec"], -1.5, 1.8)
def g_B(f):    return _tier(f["frac15_30s"], 0.10, 0.18)
def g_C(f):    return _tier(f["gate_dur"], 24.0, 54.0)
def g_best(f): return _tier(f["dvs_20s"], -1.8, 0.7)
def g_dvs10(f): return _tier(f["dvs_10s"], -1.2, 0.9)
def g_E(f):
    # two-tier by dvs_10s sign, one threshold pair
    x = f["dvs_10s"]
    return 1.0 if x < -1.0 else (2.0 if x > 1.0 else 1.5)


def g_D(f):
    g = g_best(f)
    if f["vmax_20s"] > 18.0 and g < TIERS[2]:
        g = TIERS[TIERS.index(g) + 1]
    return g


METHODS = {
    "const 1.0": g_const(1.0), "const 1.5": g_const(1.5),
    "adaptive A": g_A, "adaptive B": g_B, "adaptive C": g_C,
    "best (dvs20)": g_best, "adaptive D": g_D,
    "dvs10 3tier": g_dvs10, "dvs10 E": g_E,
}


# ---------- feature computation shared by runtime + grid ----------------
def _feats(ts, diff, spec, t, gate_dur):
    def w(sec, arr, fn):
        m = ts >= t - sec
        return float(fn(arr[m])) if m.sum() else 0.0
    return dict(
        vprior_minus_vspec=w(3.0, diff, np.mean),
        dvs_10s=w(10.0, diff, np.mean),
        dvs_20s=w(20.0, diff, np.mean),
        frac15_30s=w(30.0, spec, lambda a: np.mean(a > 15.0)),
        vmax_20s=w(20.0, spec, np.max),
        gate_dur=gate_dur,
    )


# ---------- runtime run on rf-07-26 ------------------------------------
def run_runtime(gamma_fn, leave_0726_out=True):
    trip, _ = load_trip(RF)
    first = trip.usable_locations[0]
    net = RoadNetwork(clip_graph(load_graph(GRAPH), first.latitude, first.longitude, 11000.0),
                      LocalFrame(first.latitude, first.longitude))
    cfg = PacmanConfig(); cfg.tracker_mode = "single_path"
    cfg.display.position_branch_enabled = True; cfg.display.leave_0726_out = leave_0726_out

    real = dpm.DisplayPositionBranch.step
    gate_dur = [0.0]
    glog = []

    def spy(self, t, dt, v_route, v_prior, v_spectral, d_route, stationary):
        d0 = self._delta
        real(self, t, dt, v_route, v_prior, v_spectral, d_route, stationary)
        inc = self._delta - d0
        if self._gate and self._buf:
            gate_dur[0] += dt
            ts = np.fromiter((s[0] for s in self._buf), float)
            diff = np.fromiter((s[1] for s in self._buf), float)
            spec = np.fromiter((s[2] for s in self._buf), float)
            g = float(gamma_fn(_feats(ts, diff, spec, t, gate_dur[0])))
        else:
            gate_dur[0] = 0.0
            g = 1.0
        self._delta = d0 + g * inc
        self._v_position = float(v_route) + g * (self._v_position - float(v_route))
        glog.append((float(t), bool(self._gate), g))

    dpm.DisplayPositionBranch.step = spy
    try:
        inp = build_inputs(trip, net, cfg)
        res = PacmanTracker(net, cfg, geometry=RoadGeometry(net, cfg.geometry)).run(
            inp, observer=GroundTruthObserver(
                map_match_reference(trip.reference_locations, net, net.frame)))
    finally:
        dpm.DisplayPositionBranch.step = real

    truth = map_match_reference(trip.reference_locations, net, net.frame)
    spev = _single_path_evaluation(res, truth, trip, inp.t_start, net)
    pt = res.position_trace
    t = np.array([s.t for s in pt])
    D_pos = np.array([s.distance_m for s in pt]) - pt[0].distance_m
    D_true, _ = _dtrue(trip, inp.t_start, t)
    D_true -= D_true[0]
    e = D_pos - D_true
    gl = np.array(glog)
    return dict(t=t - t[0], e=e, D_pos=D_pos, D_true=D_true,
                glog=(gl[:, 0] - inp.t_start, gl[:, 1].astype(bool), gl[:, 2]),
                spev=spev, real_wrong=spev["real_wrong_decision_count"],
                decisions=len(spev["decisions"]))


def _dtrue(trip, t_start, t):
    ref = [(f.monotonic_time, float(f.speed)) for f in trip.usable_locations if f.has_valid_speed]
    ref += [(f.monotonic_time, float(f.speed)) for f in trip.reference_locations
            if f.is_usable and f.has_valid_speed]
    ref.sort()
    rt = np.array([x[0] for x in ref]); rv = np.array([x[1] for x in ref])
    dt = float(np.median(np.diff(rt)))
    cum = np.cumsum(rv * dt); cum -= float(np.interp(t_start, rt, cum))
    return np.interp(t, rt, cum), np.interp(t, rt, rv)


def _metrics(t, e):
    return dict(median=float(np.median(np.abs(e))), p95=float(np.percentile(np.abs(e), 95)),
                mx=float(np.abs(e).max()), lead=float(e.max()), lag=float(e.min()),
                endpoint=float(e[-1]),
                ratio=float((e[-1] + _LASTDTRUE[0]) / _LASTDTRUE[0]) if _LASTDTRUE[0] else None)


_LASTDTRUE = [0.0]


def _episodes(t, e, glog):
    st, sg, sgam = glog
    eps = []
    i = 0
    while i < len(sg):
        if not sg[i]:
            i += 1; continue
        j = i
        while j < len(sg) and sg[j]:
            j += 1
        a, b = st[i], st[j - 1]
        m = (t >= a) & (t <= b)
        gm = (st >= a) & (st <= b)
        if m.sum() >= 2:
            ee = e[m]
            eps.append(dict(a=round(a, 0), b=round(b, 0),
                            gmean=round(float(sgam[gm].mean()), 2),
                            gmin=round(float(sgam[gm].min()), 2), gmax=round(float(sgam[gm].max()), 2),
                            e0=round(float(ee[0]), 1), emin=round(float(ee.min()), 1),
                            emax=round(float(ee.max()), 1), e1=round(float(ee[-1]), 1)))
        i = j
    return eps


# ---------- grid-based 1-D run for the cross-trip fair set -------------
EXTRAP = {"07-24-s6", "07-26-s0"}
BROKEN = {"07-24-s6", "07-28-s2", "07-30-s1"}
WIN_S, HOP_S = 3.0, 0.5
TRIP_CONST = ["feat_dep_sigma_spec", "feat_spec_gap", "feat_sigma_spec"]
from sklearn.isotonic import IsotonicRegression


def _pool(exclude_day):
    parts = []
    for p in sorted(P31.glob("grid_*.csv")):
        tag = p.stem.replace("grid_", "")
        if tag.startswith("rf-") or tag in EXTRAP or tag[:5] == exclude_day:
            continue
        w = make_windows(__import__("pandas").read_csv(p), WIN_S, HOP_S)
        w = w[(w.regime == "outage") & w._moving.astype(bool) & w._reliable_vspec.astype(bool)]
        parts.append(w)
    import pandas as pd
    d = pd.concat(parts, ignore_index=True)
    return d[np.isfinite(d.residual)].reset_index(drop=True)


_ISO = {}


def _iso(day):
    if day not in _ISO:
        d = _pool(day)
        io = IsotonicRegression(out_of_bounds="clip", increasing=True)
        io.fit(d.feat_vprior_minus_vspec.to_numpy(), d.residual.to_numpy())
        _ISO[day] = io
    return _ISO[day]


def run_grid(tag, gamma_fn):
    import pandas as pd
    g = pd.read_csv(P31 / f"grid_{tag}.csv")
    o = g[g.outage].reset_index(drop=True)
    t = o.t.to_numpy(); dt = float(np.median(np.diff(t)))
    D_route = o.D.to_numpy() - o.D.to_numpy()[0]
    v_route = o.v_ekf.to_numpy()
    v_true = o.v_true.to_numpy()
    D_true = np.cumsum(v_true * dt); D_true -= D_true[0]
    io = _iso(tag[:5])

    vs_all = o.v_spec.to_numpy()
    diff_all = o.v_prior.to_numpy() - vs_all           # v_prior - v_spectral
    n = len(t)
    ok = np.isfinite(vs_all)
    # rolling residual_hat once (isotonic on the trailing-3s mean of diff)
    def roll_mean(arr, sec):
        lo = np.searchsorted(t, t - sec, side="left")
        c = np.concatenate([[0.0], np.cumsum(np.where(ok, arr, 0.0))])
        cn = np.concatenate([[0.0], np.cumsum(ok.astype(float))])
        num = c[np.arange(1, n + 1)] - c[lo]
        den = np.maximum(cn[np.arange(1, n + 1)] - cn[lo], 1)
        return num / den

    def roll_frac_gt(arr, sec, thr):
        lo = np.searchsorted(t, t - sec, side="left")
        c = np.concatenate([[0.0], np.cumsum(np.where(ok, arr > thr, 0.0))])
        cn = np.concatenate([[0.0], np.cumsum(ok.astype(float))])
        return (c[np.arange(1, n + 1)] - c[lo]) / np.maximum(cn[np.arange(1, n + 1)] - cn[lo], 1)

    def roll_max(arr, sec):
        lo = np.searchsorted(t, t - sec, side="left")
        out = np.zeros(n)
        for i in range(n):
            seg = arr[lo[i]:i + 1][ok[lo[i]:i + 1]]
            out[i] = seg.max() if seg.size else 0.0
        return out

    diff3 = roll_mean(diff_all, 3.0)
    diff10 = roll_mean(diff_all, 10.0)
    diff20 = roll_mean(diff_all, 20.0)
    vspec3 = roll_mean(np.where(ok, vs_all, 0.0), 3.0)
    frac15 = roll_frac_gt(vs_all, 30.0, 15.0)
    vmax20 = roll_max(vs_all, 20.0)
    rhat = np.clip(io.predict(diff3), -2.0, 12.0)
    v_ml = np.clip(vspec3 + rhat, 0.0, 33.0)
    gate = ok & ((frac15 > 0.02) | (vmax20 > 15.0))

    gate_dur = np.zeros(n)
    run = 0.0
    for i in range(n):
        run = run + dt if gate[i] else 0.0
        gate_dur[i] = run

    delta = 0.0; D_pos = np.zeros(n)
    for i in range(n):
        if gate[i]:
            F = dict(vprior_minus_vspec=diff3[i], dvs_10s=diff10[i], dvs_20s=diff20[i],
                     frac15_30s=frac15[i], vmax_20s=vmax20[i], gate_dur=gate_dur[i])
            dv = float(gamma_fn(F)) * max(0.0, v_ml[i] - v_route[i])
        else:
            dv = 0.0
        delta += dv * dt
        D_pos[i] = D_route[i] + delta
    e = D_pos - D_true
    er = D_route - D_true
    return dict(median=float(np.median(np.abs(e))), p95=float(np.percentile(np.abs(e), 95)),
                mx=float(np.abs(e).max()), lead=float(e.max()), lag=float(e.min()),
                endpoint=float(e[-1]),
                base_median=float(np.median(np.abs(er))), base_p95=float(np.percentile(np.abs(er), 95)),
                base_mx=float(np.abs(er).max()))


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"

    if what in ("all", "rf"):
        print("=" * 78)
        print("rf-07-26  (runtime tracker, leave-07-26-out model)")
        print("=" * 78)
        rf = {}
        for name, fn in METHODS.items():
            r = run_runtime(fn)
            _LASTDTRUE[0] = r["D_true"][-1]
            m = _metrics(r["t"], r["e"])
            eps = _episodes(r["t"], r["e"], r["glog"])
            rf[name] = dict(m=m, eps=eps, real_wrong=r["real_wrong"], decisions=r["decisions"],
                            t=r["t"].tolist(), e=r["e"].tolist(),
                            glog=[r["glog"][0].tolist(), r["glog"][1].tolist(), r["glog"][2].tolist()])
        print(f"\n{'method':>14}{'median':>9}{'p95':>9}{'max':>9}{'lead':>9}{'lag':>10}"
              f"{'endpoint':>10}{'D/Dtrue':>9}  topo")
        for name, r in rf.items():
            m = r["m"]
            print(f"{name:>14}{m['median']:>9.1f}{m['p95']:>9.1f}{m['mx']:>9.1f}{m['lead']:>9.1f}"
                  f"{m['lag']:>10.1f}{m['endpoint']:>10.1f}{(m['ratio'] or 0):>9.3f}"
                  f"  dec {r['decisions']} rw {r['real_wrong']}")
        for name, r in rf.items():
            print(f"\n  --- {name} : gate episodes ---")
            print(f"  {'el':>10}{'gamma m/mn/mx':>16}{'e0':>8}{'emin':>8}{'emax':>8}{'e1':>8}")
            for ep in r["eps"]:
                print(f"  {ep['a']:>4.0f}-{ep['b']:>4.0f}s"
                      f"{ep['gmean']:>7.2f}/{ep['gmin']:.2f}/{ep['gmax']:.2f}"
                      f"{ep['e0']:>8.1f}{ep['emin']:>8.1f}{ep['emax']:>8.1f}{ep['e1']:>8.1f}")
        (OUT / "phase33_rf.json").write_text(json.dumps(rf))
        _plot_rf(rf)

    if what in ("all", "cross"):
        print("\n" + "=" * 78)
        print("cross-trip fair set (grid 1-D; adaptive = best (dvs20))")
        print("=" * 78)
        tags = [p.stem.replace("grid_", "") for p in sorted(P31.glob("grid_*.csv"))
                if not p.stem.endswith("rf-07-26") and "rf-" not in p.stem]
        rows = []
        adname = sys.argv[2] if len(sys.argv) > 2 else "C"
        adfn = {"C": g_C, "best": g_best, "A": g_A, "B": g_B, "D": g_D}[adname]
        print(f"adaptive rule = {adname}")
        for tag in tags:
            base = run_grid(tag, g_const(1.0))
            g15 = run_grid(tag, g_const(1.5))
            ad = run_grid(tag, adfn)
            rows.append(dict(trip=tag, broken=tag in BROKEN, extrap=tag in EXTRAP,
                             base=base, g15=g15, ad=ad))
        print(f"\n{'trip':>10}{'baseMx':>8} | {'g1.0 med/p95/mx':>18} | {'g1.5 med/p95/mx':>18} | "
              f"{'adapt med/p95/mx':>18} | {'ad lead':>8}{'ad lag':>8}  vs g1.5")
        for r in rows:
            b, g, a = r["base"], r["g15"], r["ad"]
            tag_flag = " *broken*" if r["broken"] else (" *extrap*" if r["extrap"] else "")
            verdict = "help" if a["mx"] < g["mx"] - 10 else ("hurt" if a["mx"] > g["mx"] + 10 else "~")
            print(f"{r['trip']:>10}{b['base_mx']:>8.0f} | "
                  f"{b['median']:>5.0f}/{b['p95']:>4.0f}/{b['mx']:>5.0f} | "
                  f"{g['median']:>5.0f}/{g['p95']:>4.0f}/{g['mx']:>5.0f} | "
                  f"{a['median']:>5.0f}/{a['p95']:>4.0f}/{a['mx']:>5.0f} | "
                  f"{a['lead']:>8.0f}{a['lag']:>8.0f}  {verdict}{tag_flag}")
        fair = [r for r in rows if not r["broken"] and not r["extrap"]]
        d15 = np.array([r["ad"]["mx"] - r["g15"]["mx"] for r in fair])
        d10 = np.array([r["ad"]["mx"] - r["base"]["mx"] for r in fair])
        print(f"\nfair set n={len(fair)}   adaptive vs const-1.5 :  helped(<-10) {int((d15<-10).sum())}  "
              f"hurt(>+10) {int((d15>10).sum())}  median dMax {np.median(d15):+.0f}  worst dMax {d15.max():+.0f}")
        print(f"                          adaptive vs const-1.0 :  worst dMax {d10.max():+.0f}  "
              f"(the Phase 32 'no single gain' regressors)")
        for t in ("07-27-s0", "07-28-s1"):
            r = next(x for x in rows if x["trip"] == t)
            print(f"  {t}: base max {r['base']['base_mx']:.0f} | g1.0 max {r['base']['mx']:.0f} "
                  f"lead {r['base']['lead']:.0f} | g1.5 max {r['g15']['mx']:.0f} lead {r['g15']['lead']:.0f} "
                  f"| adaptive max {r['ad']['mx']:.0f} lead {r['ad']['lead']:.0f}")
        (OUT / "phase33_cross.json").write_text(json.dumps(
            [{k: (v if not isinstance(v, dict) else v) for k, v in r.items()} for r in rows], default=float))


def _plot_rf(rf):
    fig, ax = plt.subplots(figsize=(15, 6))
    st, sg, _ = rf["const 1.0"]["glog"]
    st = np.array(st); sg = np.array(sg)
    i = 0; lab = True
    while i < len(sg):
        if sg[i]:
            j = i
            while j < len(sg) and sg[j]:
                j += 1
            ax.axvspan(st[i], st[j - 1], color="#2ca02c", alpha=.09, label="gate ON" if lab else None)
            lab = False; i = j
        else:
            i += 1
    ax.axhline(0, color="k", lw=.8)
    for name in ["const 1.0", "const 1.5", "adaptive A", "best (dvs20)", "adaptive D"]:
        r = rf[name]
        ax.plot(r["t"], r["e"], lw=1.6, label=name)
    ax.set_xlabel("outage elapsed [s]"); ax.set_ylabel("D_position - D_true [m]")
    ax.set_title("Phase 33 - adaptive authority on rf-07-26")
    ax.legend(loc="lower left", ncol=2); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / "phase33_rf.png", dpi=130)
    print(f"\nplot -> {OUT/'phase33_rf.png'}")


if __name__ == "__main__":
    sys.exit(main())
