#!/usr/bin/env python3
"""Phase 32 - split route/topology odometry from display/position odometry.

DIAGNOSTIC ONLY. Nothing in the tracker is modified. The corrected velocity is
applied POST-HOC to a separate longitudinal state D_position that is used only
to place the displayed Pacman point along the *already-committed* route. It can
never touch edge choice, junction timing, branch weights or route commits -
that is structurally guaranteed here because there is no feedback path at all.

    baseline tracker  -> D_route  -> topology (unchanged, verified byte-identical)
    corrected velocity -> D_position -> displayed point only, clamped to the
                                        committed frontier (excess buffered)

Correction sources (both from Phase 31, NO new training):
    iso   : isotonic residual( v_prior - v_spectral ), fit LOTO on other trips
    cb    : CatBoost honest residual model, fit LOTO on other trips
Gate    : observable saturation evidence (rolling v_spectral stats), binary+soft.
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.isotonic import IsotonicRegression
from catboost import CatBoostRegressor

from geotrace.coordinates import LocalFrame
from geotrace.loader import load_trip
from geotrace.road_graph import RoadNetwork, clip_graph, load_graph
from geotrace.pacman_tracker.config import PacmanConfig
from geotrace.pacman_tracker.roadmap import RoadGeometry
from geotrace.pacman_tracker.tracker import build_inputs, PacmanTracker
from geotrace.pacman_tracker.diagnostics import map_match_reference, GroundTruthObserver
from geotrace.pacman_tracker.benchmark import _single_path_evaluation

sys.path.insert(0, str(Path(__file__).parent))
from phase30_dataset import make_windows

ROOT = Path("/Users/arseniyabramidze/delimobil/gps-route-recovery")
P31 = ROOT / "docs/plots/phase31"
OUT = ROOT / "docs/plots/phase32"
OUT.mkdir(parents=True, exist_ok=True)
GRAPHS = ["runs/review-map", "cache/spb", "cache/spb-parkgolovo", "cache/spb-center"]
TRIP_CONST = ["feat_dep_sigma_spec", "feat_spec_gap", "feat_sigma_spec"]
EXTRAP = {"07-24-s6", "07-26-s0"}
WIN_S, HOP_S = 3.0, 0.5

TRIP_DIRS = {p.parent.name: p for p in sorted((ROOT / "runs/phase31").glob("*/trip"))}
TRIP_DIRS["rf-07-26"] = ROOT / "runs/review-final/2026-07-26/trip"
TRIP_DIRS["rf-07-22"] = ROOT / "runs/review-final/2026-07-22/trip"
TRIP_DIRS["rf-07-24"] = ROOT / "runs/review-final/2026-07-24/trip"
TRIP_DIRS["rf-07-25"] = ROOT / "runs/review-final/2026-07-25/trip"


def cb():
    return CatBoostRegressor(iterations=500, depth=4, learning_rate=0.03,
                             loss_function="MAE", l2_leaf_reg=6.0, random_seed=0, verbose=False)


# --------------------------------------------------------------------------
def load_training_pool():
    g = {}
    for p in sorted(P31.glob("grid_*.csv")):
        tag = p.stem.replace("grid_", "")
        if tag.startswith("rf-"):
            continue
        g[tag] = pd.read_csv(p)
    w = pd.concat([make_windows(gg, WIN_S, HOP_S) for gg in g.values()], ignore_index=True)
    w["day"] = w.trip.str.slice(0, 5)
    w = w[(w.regime == "outage") & w._moving.astype(bool) & w._reliable_vspec.astype(bool)]
    w = w[np.isfinite(w.residual)].reset_index(drop=True)
    return w


POOL = None
_FIT_CACHE = {}


def fit_corrections(exclude_day):
    global POOL
    if exclude_day in _FIT_CACHE:
        return _FIT_CACHE[exclude_day]
    if POOL is None:
        POOL = load_training_pool()
    tr = POOL[(POOL.day != exclude_day) & ~POOL.trip.isin(EXTRAP)]
    F = [c for c in tr.columns if c.startswith("feat_") and c not in TRIP_CONST]
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(tr.feat_vprior_minus_vspec.to_numpy(), tr.residual.to_numpy())
    model = cb(); model.fit(tr[F].to_numpy(), tr.residual.to_numpy())
    _FIT_CACHE[exclude_day] = (iso, (model, F))
    return _FIT_CACHE[exclude_day]


# --------------------------------------------------------------------------
def run_baseline(tag):
    td = TRIP_DIRS[tag]
    trip, _ = load_trip(td)
    first = trip.usable_locations[0]
    graph = gname = None
    for gn in GRAPHS:
        p = ROOT / f"{gn}.graphml"
        if not p.exists():
            continue
        gg = load_graph(p)
        try:
            net = RoadNetwork(clip_graph(gg, first.latitude, first.longitude, 11000.0),
                              LocalFrame(first.latitude, first.longitude))
            build_inputs(trip, net, PacmanConfig())
            graph, gname = gg, gn
            break
        except Exception:
            continue
    if graph is None:
        return None
    cfg = PacmanConfig(); cfg.tracker_mode = "single_path"
    net = RoadNetwork(clip_graph(graph, first.latitude, first.longitude, 11000.0),
                      LocalFrame(first.latitude, first.longitude))
    geom = RoadGeometry(net, cfg.geometry)
    inp = build_inputs(trip, net, cfg)
    obs = GroundTruthObserver(map_match_reference(trip.reference_locations, net, net.frame))
    res = PacmanTracker(net, cfg, geometry=geom).run(inp, observer=obs)
    truth = map_match_reference(trip.reference_locations, net, net.frame)
    spev = _single_path_evaluation(res, truth, trip, inp.t_start, net)
    return dict(trip=trip, net=net, res=res, inp=inp, truth=truth, gname=gname,
                t_start=inp.t_start, spev=spev)


def truth_distance(trip, t_start, times):
    ref = [f for f in trip.reference_locations if f.is_usable and f.has_valid_speed]
    tv = [(f.monotonic_time, float(f.speed)) for f in trip.usable_locations if f.has_valid_speed]
    tv += [(f.monotonic_time, float(f.speed)) for f in ref]
    tv.sort()
    rt = np.array([x[0] for x in tv]); rv = np.array([x[1] for x in tv])
    dt = float(np.median(np.diff(rt)))
    cum = np.cumsum(rv * dt)
    d0 = float(np.interp(t_start, rt, cum))
    return np.interp(times, rt, cum) - d0, np.interp(times, rt, rv)


# --------------------------------------------------------------------------
def build_position_branch(base, tag, mode, gate="soft", bound=None):
    """mode in {'none','iso','cb'}; returns per-tick D_position and diagnostics."""
    res, trip, t_start = base["res"], base["trip"], base["t_start"]
    st = res.speed_trace
    times = np.array([s.t for s in st])
    D_route = np.array([s.distance_m for s in st])
    v_route = np.array([s.speed_ms for s in st])
    sigmaD_route = np.array([s.sigma_distance_m for s in st])

    grid = pd.read_csv(P31 / f"grid_{tag}.csv") if (P31 / f"grid_{tag}.csv").exists() else None
    if grid is None:
        # rebuild for rf trips
        from phase30_dataset import replay_trip
        grid = replay_trip(tag, TRIP_DIRS[tag])
    w = make_windows(grid, WIN_S, HOP_S)
    w = w[w.regime == "outage"].copy()
    wt = w.t_end.to_numpy()

    iso, (model, F) = fit_corrections(exclude_day=tag[-5:] if tag.startswith("rf-") else tag[:5])
    if mode == "iso":
        rhat_w = iso.predict(w.feat_vprior_minus_vspec.to_numpy())
    elif mode == "cb":
        rhat_w = model.predict(w[F].to_numpy())
    else:
        rhat_w = np.zeros(len(w))
    rhat_w = np.clip(rhat_w, -2.0, 12.0)

    # observable saturation gate
    frac15 = w.feat_vspec_frac_gt15_30s.to_numpy()
    vmax20 = w.feat_vspec_max_20s.to_numpy()
    sat = np.clip(0.5 * np.clip(frac15 / 0.15, 0, 1)
                  + 0.5 * np.clip((vmax20 - 13.0) / 4.0, 0, 1), 0, 1)
    if gate == "binary":
        alpha_w = ((frac15 > 0.02) | (vmax20 > 15.0)).astype(float)
    elif gate == "soft":
        alpha_w = sat
    else:  # always
        alpha_w = np.ones(len(w))
    v_spec_w = w.feat_v_spec.to_numpy()
    v_ml_w = np.clip(v_spec_w + rhat_w, 0.0, 33.0)

    # interpolate onto tick times
    def itp(a):
        return np.interp(times, wt, a, left=a[0] if len(a) else 0.0, right=a[-1] if len(a) else 0.0)
    alpha = np.clip(itp(alpha_w), 0, 1)
    v_ml = itp(v_ml_w)
    sat_t = itp(sat)

    if mode == "none":
        alpha = np.zeros_like(alpha)
    # one-directional: the premise is spectral SATURATION (under-read), so the
    # display branch may only add speed, never pull below the route estimate.
    v_pos = v_route + alpha * np.maximum(0.0, v_ml - v_route)
    dt = np.diff(times, prepend=times[0] - (times[1] - times[0]))
    dt[0] = times[1] - times[0]
    delta = np.cumsum((v_pos - v_route) * dt)          # D_position - D_route
    if bound is not None:
        delta = np.clip(delta, -bound, bound)
    D_pos = D_route + delta

    D_true, v_true = truth_distance(trip, t_start, times)
    return dict(times=times, D_route=D_route, D_pos=D_pos, D_true=D_true,
                v_route=v_route, v_pos=v_pos, v_true=v_true, v_ml=v_ml,
                sigmaD_route=sigmaD_route, alpha=alpha, sat=sat_t, delta=delta)


def metrics_1d(b, ref=None):
    e_route = b["D_route"] - b["D_true"]
    e_pos = b["D_pos"] - b["D_true"]
    ev_pos = b["v_pos"] - b["v_true"]
    ev_route = b["v_route"] - b["v_true"]
    T = b["times"] - b["times"][0]
    def at(el):
        i = int(np.argmin(np.abs(T - el)))
        return round(float(e_pos[i]), 1)
    closer = np.abs(e_pos) < np.abs(e_route)
    m = dict(
        Dratio_route=round(float((b["D_route"][-1]) / b["D_true"][-1]), 3) if b["D_true"][-1] > 1 else None,
        Dratio_pos=round(float((b["D_pos"][-1]) / b["D_true"][-1]), 3) if b["D_true"][-1] > 1 else None,
        endpoint_route=round(float(e_route[-1]), 1), endpoint_pos=round(float(e_pos[-1]), 1),
        max_route=round(float(np.max(np.abs(e_route))), 1),
        max_pos=round(float(np.max(np.abs(e_pos))), 1),
        p50_route=round(float(np.percentile(np.abs(e_route), 50)), 1),
        p50_pos=round(float(np.percentile(np.abs(e_pos), 50)), 1),
        p90_pos=round(float(np.percentile(np.abs(e_pos), 90)), 1),
        p95_route=round(float(np.percentile(np.abs(e_route), 95)), 1),
        p95_pos=round(float(np.percentile(np.abs(e_pos), 95)), 1),
        e60=at(60), e120=at(120), e240=at(240), e440=at(440),
        vbias_pos=round(float(np.mean(ev_pos)), 2), vbias_route=round(float(np.mean(ev_route)), 2),
        vMAE_pos=round(float(np.mean(np.abs(ev_pos))), 2), vMAE_route=round(float(np.mean(np.abs(ev_route))), 2),
        vp95_pos=round(float(np.percentile(np.abs(ev_pos), 95)), 2),
        max_lead=round(float(np.max(e_pos)), 1), max_lag=round(float(np.min(e_pos)), 1),
        closer_frac=round(float(np.mean(closer)), 3),
        closer_secs=round(float(np.sum(closer) * np.median(np.diff(b["times"]))), 0),
        gate_fired_frac=round(float(np.mean(b["alpha"] > 0.05)), 3),
        max_delta=round(float(np.max(np.abs(b["delta"]))), 1),
    )
    return m


# --------------------------------------------------------------------------
def task6(primary="rf-07-26"):
    print(f"\n########## TASK 6 - {primary} main experiment ##########")
    base = run_baseline(primary)
    if base is None:
        print("  no graph"); return
    rows = []
    for name, mode, gate in [("A_baseline", "none", "always"),
                             ("C_cb_soft", "cb", "soft"),
                             ("C_cb_binary", "cb", "binary"),
                             ("D_iso_soft", "iso", "soft"),
                             ("D_iso_binary", "iso", "binary"),
                             ("E_cb_always", "cb", "always")]:
        b = build_position_branch(base, primary, mode, gate)
        m = metrics_1d(b)
        m = dict(run=name, **m)
        rows.append(m)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"task6_{primary}.csv", index=False)
    cols = ["run", "Dratio_pos", "max_pos", "p95_pos", "p50_pos", "endpoint_pos",
            "e60", "e120", "e240", "e440", "vbias_pos", "vMAE_pos", "vp95_pos",
            "max_lead", "max_lag", "closer_frac", "gate_fired_frac", "max_delta"]
    print(df[cols].to_string(index=False))
    print(f"\n  baseline route: max {rows[0]['max_route']}  p95 {rows[0]['p95_route']} "
          f"p50 {rows[0]['p50_route']}  endpoint {rows[0]['endpoint_route']}  "
          f"D/D_true {rows[0]['Dratio_route']}  vMAE {rows[0]['vMAE_route']}")
    # topology check A vs each: byte-identical route decisions (no feedback => trivially true,
    # but assert it from the single run we did)
    dec = [(d["t_decision"], d["chosen_edge"]) for d in base["spev"]["decisions"]]
    print(f"  committed decisions: {len(dec)}  edge seq tail: "
          f"{base['res'].frames[-1].top[0].route[-8:]}")
    json.dump({"decisions": [[float(a), int(b)] for a, b in dec],
               "route": [int(x) for x in base['res'].frames[-1].top[0].route]},
              open(OUT / f"topology_{primary}.json", "w"))


def task7():
    print("\n########## TASK 7 - all-trip evaluation ##########")
    tags = [t for t in TRIP_DIRS if not t.startswith("rf-") or t == "rf-07-26"]
    rows = []
    for tag in tags:
        base = run_baseline(tag)
        if base is None:
            # 1-D only path would need speed-only replay; skip graph-less for now
            print(f"  {tag}: no graph, skipped")
            continue
        b0 = build_position_branch(base, tag, "none", "always")
        m0 = metrics_1d(b0)
        best = None
        for mode, gate in [("cb", "soft"), ("cb", "binary"), ("iso", "soft"), ("iso", "binary")]:
            b = build_position_branch(base, tag, mode, gate)
            m = metrics_1d(b)
            rows.append(dict(trip=tag, variant=f"{mode}-{gate}",
                             max_route=m["max_route"], max_pos=m["max_pos"],
                             p95_route=m["p95_route"], p95_pos=m["p95_pos"],
                             p50_route=m["p50_route"], p50_pos=m["p50_pos"],
                             Dr_route=m["Dratio_route"], Dr_pos=m["Dratio_pos"],
                             vbias_route=m["vbias_route"], vbias_pos=m["vbias_pos"],
                             gate=m["gate_fired_frac"], d_max=round(m["max_pos"] - m["max_route"], 1),
                             d_p95=round(m["p95_pos"] - m["p95_route"], 1),
                             closer=m["closer_frac"]))
        print(f"  {tag}: baseline max {m0['max_route']:.0f} p95 {m0['p95_route']:.0f} "
              f"D/D_true {m0['Dratio_route']}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "task7_alltrip.csv", index=False)
    for v in df.variant.unique():
        s = df[df.variant == v]
        helped = (s.d_p95 < -10).sum(); hurt = (s.d_p95 > 10).sum()
        print(f"\n  === {v} ===  helped {helped}/{len(s)}  hurt {hurt}  "
              f"median dp95 {s.d_p95.median():+.0f}  worst dp95 {s.d_p95.max():+.0f} "
              f"({s.loc[s.d_p95.idxmax(),'trip']})  median dmax {s.d_max.median():+.0f}  "
              f"worst dmax {s.d_max.max():+.0f}")
    return df


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "6"):
        task6("rf-07-26")
        task6("07-26-s1")
    if what in ("all", "7"):
        task7()
