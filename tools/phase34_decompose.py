#!/usr/bin/env python3
"""Phase 34 - decompose every sharp drop in D_position - D_true on rf-07-26.

Diagnostic only. Does not tune gamma / gate / isotonic model / topology; does
not modify any production file (all instrumentation is monkeypatched onto the
already-shipped runtime for the duration of this script).

For every 0.5 s output sample:
    e             = D_position_raw - D_true
    de            = e[i] - e[i-1]
    expected_de   = (v_position[i] - v_true[i]) * dt
    nonkinematic_residual = de - expected_de

A drop is classified:
    A - velocity deficit      : nonkinematic_residual ~ 0, v_position << v_true
    B - non-kinematic jump    : nonkinematic_residual large (edge remap, frontier
                                 clamp/release, anchor snap, state reset, ...)
    C - both
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
from geotrace.pacman_tracker.speed import GlobalSpeedTracker
import geotrace.pacman_tracker.tracker as trkmod
import geotrace.pacman_tracker.display_position as dpm

ROOT = Path("/Users/arseniyabramidze/delimobil/gps-route-recovery")
OUT = ROOT / "docs/plots/phase32"
RF = ROOT / "runs/review-final/2026-07-26/trip"
GRAPH = ROOT / "runs/review-map.graphml"
GRID_DT = 0.5   # analysis / output-tick grid, matches the estimator's own feature grid


# --------------------------------------------------------------------------
def run_instrumented(leave_0726_out=True):
    trip, _ = load_trip(RF)
    first = trip.usable_locations[0]
    net = RoadNetwork(clip_graph(load_graph(GRAPH), first.latitude, first.longitude, 11000.0),
                      LocalFrame(first.latitude, first.longitude))
    cfg = PacmanConfig()
    cfg.tracker_mode = "single_path"
    cfg.display.position_branch_enabled = True
    cfg.display.leave_0726_out = leave_0726_out
    cfg.output_dt_s = GRID_DT       # so position_trace / frames land exactly on the analysis grid

    steplog: list[dict] = []
    tickevents: list[dict] = []     # from lateral_anchor / zero_velocity / spectral_update
    real_step = dpm.DisplayPositionBranch.step
    real_locate = trkmod.PacmanTracker._display_locate
    real_lateral = GlobalSpeedTracker.lateral_anchor
    real_zupt = GlobalSpeedTracker.zero_velocity
    real_spectral_update = GlobalSpeedTracker.spectral_update

    ev = {"lateral_applied": False, "zupt_ran": False, "spectral_ran": False, "t_last": None}

    def spy_lateral(self, a_lat, yaw_rate_smooth, dt, shock=False, spectral_speed=float("nan")):
        r = real_lateral(self, a_lat, yaw_rate_smooth, dt, shock, spectral_speed)
        ev["lateral_applied"] = r is not None
        return r

    def spy_zupt(self, a_long, dt, yaw_rate, run_s=1e9):
        ev["zupt_ran"] = True
        return real_zupt(self, a_long, dt, yaw_rate, run_s)

    def spy_spectral(self, speed, sigma, dt):
        ev["spectral_ran"] = True
        return real_spectral_update(self, speed, sigma, dt)

    def spy_step(self, t, dt, v_route, v_prior, v_spectral, d_route, stationary):
        d0 = self._delta
        real_step(self, t, dt, v_route, v_prior, v_spectral, d_route, stationary)
        # recompute the exact same features the real step just used, to expose
        # residual_hat / v_ml / the gate features without touching production code
        feat = vspec_mean = frac_above = max_recent = float("nan")
        if self._buf:
            ts = np.fromiter((s[0] for s in self._buf), float)
            diff = np.fromiter((s[1] for s in self._buf), float)
            spec = np.fromiter((s[2] for s in self._buf), float)
            win = ts >= t - self._m.resid_feature_window_s
            feat = float(np.mean(diff[win])) if win.any() else float(diff[-1])
            vspec_mean = float(np.mean(spec[win])) if win.any() else (
                float(v_spectral) if np.isfinite(v_spectral) else float("nan"))
            f_frac = spec[ts >= t - self._m.gate_frac_window_s]
            f_max = spec[ts >= t - self._m.gate_max_window_s]
            frac_above = float(np.mean(f_frac > self._m.gate_frac_level_ms)) if f_frac.size else 0.0
            max_recent = float(np.max(f_max)) if f_max.size else 0.0
        residual_hat = self._m.residual_hat(feat) if np.isfinite(feat) else float("nan")
        v_ml = (float(np.clip(vspec_mean + residual_hat, 0.0, 33.0))
                if np.isfinite(vspec_mean) else float("nan"))
        dv = self._v_position - float(v_route)
        steplog.append(dict(
            t=float(t), dt=float(dt), gate=bool(self._gate),
            v_route=float(v_route), v_prior=float(v_prior),
            v_spectral=float(v_spectral) if np.isfinite(v_spectral) else float("nan"),
            residual_hat=residual_hat, v_ml=v_ml, correction_dv=float(dv),
            v_position=float(self._v_position), D_route=float(d_route),
            D_position_raw=float(d_route + self._delta), delta=float(self._delta),
            stationary=bool(stationary),
            lateral_applied=ev["lateral_applied"], zupt_ran=ev["zupt_ran"],
            spectral_ran=ev["spectral_ran"], frac15_30s=frac_above, vmax_20s=max_recent,
        ))
        ev["lateral_applied"] = ev["zupt_ran"] = ev["spectral_ran"] = False

    tickevents_ref = tickevents

    def spy_locate(self, hs, t):
        n_edges_before = len(hs.routes[0].edges()) if hs.routes else 0
        offset_bias_before = float(hs.offset_bias[0])
        route_offset_before = float(hs.route_offset[0])
        edge_before = int(hs.edge[0])
        sample = real_locate(self, hs, t)
        tickevents_ref.append(dict(
            t=float(t), n_committed_edges=n_edges_before,
            edge=sample.edge, s_m=sample.s_m, at_frontier=sample.at_frontier,
            excess_m=sample.excess_position_distance_m,
            offset_bias=offset_bias_before, route_offset=route_offset_before,
            edge_index=edge_before,
        ))
        return sample

    dpm.DisplayPositionBranch.step = spy_step
    trkmod.PacmanTracker._display_locate = spy_locate
    GlobalSpeedTracker.lateral_anchor = spy_lateral
    GlobalSpeedTracker.zero_velocity = spy_zupt
    GlobalSpeedTracker.spectral_update = spy_spectral
    try:
        inp = build_inputs(trip, net, cfg)
        truth = map_match_reference(trip.reference_locations, net, net.frame)
        res = PacmanTracker(net, cfg, geometry=RoadGeometry(net, cfg.geometry)).run(
            inp, observer=GroundTruthObserver(truth))
    finally:
        dpm.DisplayPositionBranch.step = real_step
        trkmod.PacmanTracker._display_locate = real_locate
        GlobalSpeedTracker.lateral_anchor = real_lateral
        GlobalSpeedTracker.zero_velocity = real_zupt
        GlobalSpeedTracker.spectral_update = real_spectral_update

    spev = _single_path_evaluation(res, truth, trip, inp.t_start, net)
    return trip, inp, res, spev, steplog, tickevents


def d_true_series(trip, t_start, t):
    ref = [(f.monotonic_time, float(f.speed)) for f in trip.usable_locations if f.has_valid_speed]
    ref += [(f.monotonic_time, float(f.speed)) for f in trip.reference_locations
            if f.is_usable and f.has_valid_speed]
    ref.sort()
    rt = np.array([x[0] for x in ref]); rv = np.array([x[1] for x in ref])
    dt = float(np.median(np.diff(rt)))
    cum = np.cumsum(rv * dt); cum -= float(np.interp(t_start, rt, cum))
    return np.interp(t, rt, cum), np.interp(t, rt, rv)


def add_exact_kinematics(g: pd.DataFrame, trip, inp, steplog):
    """The literal formula `expected_de=(v_position[i]-v_true[i])*dt` is a
    RECTANGLE-RULE approximation at the 0.5 s tick spacing: it uses only the
    endpoint velocity, not the actual sub-step (0.1 s) trajectory in between.
    During a fast acceleration ramp that approximation itself has real error,
    which would masquerade as "nonkinematic". This recomputes the same
    residual using the FULL native-resolution step log (exact integral of the
    real v_position(tau) and v_true(tau) over each 0.5 s tick interval) so a
    genuine position-mapping jump can be told apart from plain discretization
    error of this forensic method.
    """
    sl = pd.DataFrame(steplog).sort_values("t").reset_index(drop=True)
    _, v_true_fine = d_true_series(trip, inp.t_start, sl["t"].to_numpy())
    sl["v_true"] = v_true_fine
    sl["v_pos_minus_vtrue_dt"] = (sl["v_position"] - sl["v_true"]) * sl["t"].diff().fillna(0.0)
    sl["cum_exact"] = sl["v_pos_minus_vtrue_dt"].cumsum()
    # interpolate the fine cumulative kinematic integral onto the tick grid
    exact_cum_at_tick = np.interp(g["t"].to_numpy(), sl["t"].to_numpy(), sl["cum_exact"].to_numpy())
    g = g.copy()
    g["expected_de_exact"] = np.diff(exact_cum_at_tick, prepend=exact_cum_at_tick[0])
    g["nonkinematic_residual_exact"] = g["de"] - g["expected_de_exact"]
    return g


def build_grid(trip, inp, steplog, tickevents):
    """Merge the per-IMU-step log onto the 0.5 s output-tick grid (tickevents)."""
    sl = pd.DataFrame(steplog)
    tk = pd.DataFrame(tickevents)
    sl = sl.sort_values("t").reset_index(drop=True)
    tk = tk.sort_values("t").reset_index(drop=True)
    idx = np.searchsorted(sl["t"].to_numpy(), tk["t"].to_numpy(), side="right") - 1
    idx = np.clip(idx, 0, len(sl) - 1)
    merged = sl.iloc[idx].reset_index(drop=True)
    for c in tk.columns:
        merged[f"disp_{c}" if c != "t" else "t_tick"] = tk[c].to_numpy()
    merged["t"] = merged["t_tick"]

    D_true, v_true = d_true_series(trip, inp.t_start, merged["t"].to_numpy())
    merged["D_true"] = D_true
    merged["v_true"] = v_true
    # displayed_L = l_disp - excess (excess=0 unless frontier-clamped), and
    # l_disp = D_position_raw - offset_bias, so in the D_route/D_true frame:
    #   D_position_displayed = D_position_raw - excess_position_distance
    merged["D_position_displayed"] = merged["D_position_raw"] - merged["disp_excess_m"]
    return merged


def add_kinematics(g: pd.DataFrame):
    g = g.copy()
    g["dt_actual"] = g["t"].diff().fillna(g["t"].iloc[1] - g["t"].iloc[0])
    g["e"] = g["D_position_raw"] - g["D_true"]
    g["e_route"] = g["D_route"] - g["D_true"]
    g["de"] = g["e"].diff()
    g["expected_de"] = (g["v_position"] - g["v_true"]) * g["dt_actual"]
    g["nonkinematic_residual"] = g["de"] - g["expected_de"]
    # the RENDERED marker (frontier-clamped) - checked separately: this is where
    # a genuine "position-mapping" jump (excess release at an edge commit) would
    # show up even if the raw scalar D_position_raw is perfectly smooth.
    g["e_displayed"] = g["D_position_displayed"] - g["D_true"]
    g["de_displayed"] = g["e_displayed"].diff()
    g["nonkinematic_residual_displayed"] = g["de_displayed"] - g["expected_de"]
    # also decompose D_route's own step (v_position = v_route + correction_dv, so
    # nonkinematic_residual attributable to the route EKF's own predict-vs-report
    # velocity mismatch, isolated from the display branch's own bookkeeping)
    g["de_route"] = g["e_route"].diff()
    g["expected_de_route"] = (g["v_route"] - g["v_true"]) * g["dt_actual"]
    g["nonkinematic_residual_route"] = g["de_route"] - g["expected_de_route"]
    for w in (5, 10, 20, 30):
        n = max(1, int(round(w / GRID_DT)))
        g[f"slope_{w}s"] = g["e"] - g["e"].shift(n)
    g["gate_prev"] = g["gate"].shift(1).fillna(False)
    g["gate_transition"] = g["gate"] != g["gate_prev"]
    g["edge_prev"] = g["disp_edge"].shift(1)
    g["edge_transition"] = g["disp_edge"] != g["edge_prev"]
    g["ncommit_prev"] = g["disp_n_committed_edges"].shift(1)
    g["route_commit"] = g["disp_n_committed_edges"] > g["ncommit_prev"]
    g["frontier_prev"] = g["disp_at_frontier"].shift(1).fillna(False)
    g["frontier_clamp_start"] = g["disp_at_frontier"] & ~g["frontier_prev"]
    g["frontier_release"] = ~g["disp_at_frontier"] & g["frontier_prev"]
    return g


def main():
    trip, inp, res, spev, steplog, tickevents = run_instrumented(leave_0726_out=True)
    print(f"topology: decisions {len(spev['decisions'])}  real_wrong {spev['real_wrong_decision_count']}")
    g = build_grid(trip, inp, steplog, tickevents)
    g = add_kinematics(g)
    g = add_exact_kinematics(g, trip, inp, steplog)
    el = g["t"].to_numpy() - g["t"].iloc[0]
    g["el"] = el

    print(f"\n=== discretization check: naive (endpoint) vs exact (sub-step-integrated) "
          f"expected_de ===")
    print(f"  naive  nonkinematic_residual: sum|.| {g['nonkinematic_residual'].abs().sum():.1f} m   "
          f"max|.| {g['nonkinematic_residual'].abs().max():.2f}")
    print(f"  exact  nonkinematic_residual: sum|.| {g['nonkinematic_residual_exact'].abs().sum():.1f} m   "
          f"max|.| {g['nonkinematic_residual_exact'].abs().max():.2f}")
    corr = np.corrcoef(g["nonkinematic_residual"].to_numpy()[1:],
                       (g["v_position"] - g["v_true"]).diff().to_numpy()[1:])[0, 1]
    print(f"  corr(naive nonkinematic_residual, d(v_position-v_true)/dt) = {corr:.2f}  "
          f"(high => rectangle-rule artifact, not a real jump)")

    g.to_csv(OUT / "phase34_samples.csv", index=False)
    print(f"per-sample log -> phase34_samples.csv  ({len(g)} rows)")

    print(f"\nwhole trip: median|e| {g['e'].abs().median():.1f}  p95 {g['e'].abs().quantile(.95):.1f}  "
          f"max {g['e'].abs().max():.1f}  endpoint {g['e'].iloc[-1]:.1f}")
    print(f"nonkinematic_residual: mean {g['nonkinematic_residual'].mean():.4f}  "
          f"std {g['nonkinematic_residual'].std():.4f}  "
          f"sum(abs) {g['nonkinematic_residual'].abs().sum():.2f} m over whole trip  "
          f"max|.| {g['nonkinematic_residual'].abs().max():.3f}")
    print(f"nonkinematic_residual_route (D_route's own predict-vs-report mismatch): "
          f"sum(abs) {g['nonkinematic_residual_route'].abs().sum():.2f} m  "
          f"max|.| {g['nonkinematic_residual_route'].abs().max():.3f}")
    print(f"\nRENDERED MARKER check (frontier-clamped D_position_displayed vs raw scalar):")
    print(f"  median|e_displayed| {g['e_displayed'].abs().median():.1f}  "
          f"max {g['e_displayed'].abs().max():.1f}  "
          f"nonkinematic_residual_displayed sum(abs) {g['nonkinematic_residual_displayed'].abs().sum():.1f} m  "
          f"max|.| {g['nonkinematic_residual_displayed'].abs().max():.1f} m at "
          f"el={g.loc[g['nonkinematic_residual_displayed'].abs().idxmax(),'el']:.1f}s")

    print("\n=== TOP 20 most negative de ===")
    top_de = g.nsmallest(20, "de")[["el", "de", "nonkinematic_residual", "nonkinematic_residual_exact",
                                    "gate", "e", "disp_edge", "edge_transition", "route_commit",
                                    "frontier_clamp_start", "frontier_release"]]
    print(top_de.to_string(index=False))

    print("\n=== TOP 20 most negative nonkinematic_residual (naive rectangle-rule) ===")
    top_nk = g.nsmallest(20, "nonkinematic_residual")[
        ["el", "de", "nonkinematic_residual", "nonkinematic_residual_exact", "gate",
         "lateral_applied", "zupt_ran", "stationary", "edge_transition", "route_commit",
         "frontier_clamp_start", "frontier_release"]]
    print(top_nk.to_string(index=False))

    print("\n=== TOP 20 most negative nonkinematic_residual_exact (sub-step-integrated - the real check) ===")
    top_nke = g.nsmallest(20, "nonkinematic_residual_exact")[
        ["el", "de", "nonkinematic_residual_exact", "gate", "lateral_applied", "zupt_ran",
         "stationary", "edge_transition", "route_commit", "frontier_clamp_start", "frontier_release"]]
    print(top_nke.to_string(index=False))

    # self-cancellation check on the isolated lateral-anchor spikes: do they net
    # to ~0 within a couple of seconds (a timing notch) or leave a permanent step?
    print("\n=== self-cancellation check around the largest |nonkinematic_residual| spikes ===")
    spikes = g.reindex(g["nonkinematic_residual"].abs().nlargest(6).index).sort_values("el")
    for _, row in spikes.iterrows():
        i = g.index.get_loc(row.name)
        lo, hi = max(0, i - 6), min(len(g) - 1, i + 6)
        net = float(g["nonkinematic_residual"].iloc[lo:hi + 1].sum())
        print(f"  el={row['el']:.1f}s  spike={row['nonkinematic_residual']:+.1f}  "
              f"net over +-3s window = {net:+.2f} m  "
              f"(lateral_applied={row['lateral_applied']}, frontier_clamp_start={row['frontier_clamp_start']})")

    print("\n=== worst sustained slopes (most negative e[t]-e[t-window]) ===")
    for w in (5, 10, 20, 30):
        i = g[f"slope_{w}s"].idxmin()
        print(f"  {w:>2}s window: worst at el={g['el'].iloc[i]:.1f}s  slope={g[f'slope_{w}s'].iloc[i]:+.1f} m "
              f"(e {g['e'].iloc[max(0,i-int(w/GRID_DT))]:+.1f} -> {g['e'].iloc[i]:+.1f})")

    # -------- gate-ON episodes, decomposed A/B/C --------------------------
    on = g["gate"].to_numpy()
    print("\n=== per gate-ON episode: velocity-deficit vs non-kinematic decomposition ===")
    rows = []
    i = 0
    while i < len(on):
        if not on[i]:
            i += 1; continue
        j = i
        while j < len(on) and on[j]:
            j += 1
        rows.append(_decompose(g, i, j - 1))
        i = j
    epdf = pd.DataFrame(rows)
    epdf.to_csv(OUT / "phase34_episodes.csv", index=False)
    print(epdf.to_string(index=False))

    # -------- deep dive: start of episode 2, el 212-304 --------------------
    print("\n=== DEEP DIVE: episode-2 onset, el 205-235 s ===")
    win = g[(g["el"] >= 205) & (g["el"] <= 235)]
    cols = ["el", "gate", "v_true", "v_route", "v_spectral", "v_prior", "residual_hat",
            "v_ml", "correction_dv", "v_position", "D_true", "D_route", "D_position_raw",
            "e", "de", "expected_de", "nonkinematic_residual", "disp_edge",
            "disp_s_m", "disp_at_frontier", "disp_excess_m", "edge_transition",
            "route_commit", "lateral_applied", "zupt_ran"]
    print(win[cols].round(3).to_string(index=False))

    # -------- root-cause table ----------------------------------------
    print("\n=== ROOT-CAUSE TABLE ===")
    rc = pd.DataFrame(rows)[["interval", "e_start", "e_end", "D_true_delta", "D_route_delta",
                             "D_position_delta", "metres_velocity_deficit",
                             "metres_nonkinematic", "root_cause"]]
    print(rc.to_string(index=False))
    rc.to_csv(OUT / "phase34_rootcause.csv", index=False)

    total_v = sum(abs(r["metres_velocity_deficit"]) for r in rows)
    total_nk = sum(abs(r["metres_nonkinematic"]) for r in rows)
    tot = total_v + total_nk
    pct_v = 100 * total_v / tot if tot else 0
    pct_nk = 100 * total_nk / tot if tot else 0
    print(f"\n>>> sharp drops are {pct_v:.0f}% velocity underestimation and "
          f"{pct_nk:.0f}% position-mapping/state jumps  "
          f"(sum |velocity deficit| = {total_v:.0f} m, sum |nonkinematic| = {total_nk:.0f} m)")

    _plot(g)
    json.dump(dict(pct_velocity=pct_v, pct_nonkinematic=pct_nk,
                   total_velocity_m=total_v, total_nonkinematic_m=total_nk,
                   episodes=rows), open(OUT / "phase34_verdict.json", "w"), indent=1, default=float)


def _decompose(g: pd.DataFrame, i0: int, i1: int) -> dict:
    seg = g.iloc[i0:i1 + 1]
    dt = seg["dt_actual"].to_numpy()
    baseline_deficit = float(np.sum((seg["v_true"] - seg["v_route"]).to_numpy() * dt))
    correction_added = float(np.sum(seg["correction_dv"].to_numpy() * dt))
    remaining_deficit = baseline_deficit - correction_added
    # actual vs kinematic-only accumulated error change, to isolate nonkinematic jumps.
    # Use the EXACT sub-step-integrated kinematic reference, not the naive
    # endpoint*dt rectangle rule - the naive rule has real truncation error of
    # its own during a fast acceleration ramp that would masquerade as "jump".
    de_actual = float(seg["e"].iloc[-1] - g["e"].iloc[max(i0 - 1, 0)])
    de_kinematic_naive = float(np.sum(seg["expected_de"].to_numpy()))
    de_kinematic_exact = float(np.sum(seg["expected_de_exact"].to_numpy()))
    nonkinematic_naive = de_actual - de_kinematic_naive
    nonkinematic_total = de_actual - de_kinematic_exact
    e_start = float(g["e"].iloc[max(i0 - 1, 0)])
    e_end = float(seg["e"].iloc[-1])
    kind = ("B" if abs(nonkinematic_total) > 0.25 * max(abs(de_actual), 1.0) and abs(nonkinematic_total) > 15
            else "A")
    if kind == "B" and abs(remaining_deficit) > 15:
        kind = "C"
    cause = "velocity deficit (spectral saturation, correction undersized)"
    if kind in ("B", "C"):
        flags = seg[["edge_transition", "route_commit", "frontier_clamp_start",
                    "frontier_release", "lateral_applied", "zupt_ran"]].sum()
        hit = flags[flags > 0]
        cause = ("non-kinematic: " + ", ".join(f"{k}x{int(v)}" for k, v in hit.items())
                 if len(hit) else "non-kinematic (unattributed)")
        if kind == "C":
            cause = "velocity deficit + " + cause
    return dict(
        interval=f"el {g['el'].iloc[i0]:.0f}-{g['el'].iloc[i1]:.0f}s",
        e_start=round(e_start, 1), e_end=round(e_end, 1),
        D_true_delta=round(float(seg['D_true'].iloc[-1] - g['D_true'].iloc[max(i0-1,0)]), 1),
        D_route_delta=round(float(seg['D_route'].iloc[-1] - g['D_route'].iloc[max(i0-1,0)]), 1),
        D_position_delta=round(float(seg['D_position_raw'].iloc[-1] - g['D_position_raw'].iloc[max(i0-1,0)]), 1),
        metres_velocity_deficit=round(remaining_deficit, 1),
        metres_nonkinematic=round(nonkinematic_total, 1),
        metres_nonkinematic_naive_rectangle_rule=round(nonkinematic_naive, 1),
        baseline_deficit=round(baseline_deficit, 1), correction_added=round(correction_added, 1),
        classification=kind, root_cause=cause,
    )


def _plot(g: pd.DataFrame):
    fig, ax = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    el = g["el"].to_numpy()
    on = g["gate"].to_numpy()
    for a in ax:
        i = 0
        while i < len(on):
            if on[i]:
                j = i
                while j < len(on) and on[j]:
                    j += 1
                a.axvspan(el[i], el[j - 1], color="#2ca02c", alpha=.08)
                i = j
            else:
                i += 1
        a.axhline(0, color="k", lw=.7)
        a.grid(alpha=.25)
    ax[0].plot(el, g["e"], color="#1f77b4", lw=1.4, label="e = D_position_raw - D_true")
    ax[0].plot(el, g["e_route"], color="#d62728", lw=1.0, alpha=.6, label="D_route - D_true")
    ax[0].set_ylabel("position error [m]"); ax[0].legend(loc="lower left")
    ax[0].set_title("Phase 34 - rf-07-26, production iso-binary (gate ON shaded)")
    ax[1].plot(el, g["v_position"] - g["v_true"], color="#9467bd", lw=1.0)
    ax[1].set_ylabel("v_position - v_true [m/s]")
    ax[2].plot(el, g["nonkinematic_residual"], color="#8c564b", lw=.8, alpha=.5,
              label="naive (endpoint*dt rectangle rule)")
    ax[2].plot(el, g["nonkinematic_residual_exact"], color="#000000", lw=1.1,
              label="exact (sub-step integrated) - the real check")
    ax[2].set_ylabel("nonkinematic_residual [m/step]")
    ax[2].legend(loc="upper right", fontsize=8)
    ax[2].set_xlabel("outage elapsed [s]")
    fig.tight_layout()
    fig.savefig(OUT / "phase34_decompose.png", dpi=130)
    print(f"\nplot -> {OUT/'phase34_decompose.png'}")


if __name__ == "__main__":
    sys.exit(main())
