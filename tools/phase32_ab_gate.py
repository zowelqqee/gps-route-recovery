#!/usr/bin/env python3
"""Strict causal A/B on rf-07-26 - the ONLY thing that differs is the display
saturation gate.

    A : production Phase 32 iso-binary, real saturation gate
    B : identical run, gate forced OFF for the whole trip

The tracker is run ONCE. v_route, topology, edge sequence, route decisions,
anchors and timestamps are therefore literally the same object for both. The
display branch is stepped per IMU sample; branch A uses its real gate, branch B
uses gate = 0 (so v_position_B == v_route, D_position_B == D_route).

    e_A(t) = D_position_A(t) - D_true(t)
    e_B(t) = D_position_B(t) - D_true(t) = D_route(t) - D_true(t)

Nothing is tuned. estimator / topology / frontier logic untouched.
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

ROOT = Path("/Users/arseniyabramidze/delimobil/gps-route-recovery")
OUT = ROOT / "docs/plots/phase32"
TRIP = ROOT / "runs/review-final/2026-07-26/trip"
GRAPH = ROOT / "runs/review-map.graphml"


def d_true_fn(trip, t_start):
    ref = [(f.monotonic_time, float(f.speed)) for f in trip.usable_locations if f.has_valid_speed]
    ref += [(f.monotonic_time, float(f.speed)) for f in trip.reference_locations
            if f.is_usable and f.has_valid_speed]
    ref.sort()
    rt = np.array([x[0] for x in ref]); rv = np.array([x[1] for x in ref])
    dt = float(np.median(np.diff(rt)))
    cum = np.cumsum(rv * dt)
    d0 = float(np.interp(t_start, rt, cum))
    return (lambda tt: np.interp(tt, rt, cum) - d0), (lambda tt: np.interp(tt, rt, rv))


def run(leave_0726_out: bool):
    trip, _ = load_trip(TRIP)
    first = trip.usable_locations[0]
    net = RoadNetwork(clip_graph(load_graph(GRAPH), first.latitude, first.longitude, 11000.0),
                      LocalFrame(first.latitude, first.longitude))
    cfg = PacmanConfig()
    cfg.tracker_mode = "single_path"
    cfg.display.position_branch_enabled = True
    cfg.display.leave_0726_out = leave_0726_out

    log: list[dict] = []
    real_step = dpm.DisplayPositionBranch.step

    def spy(self, t, dt, v_route, v_prior, v_spectral, d_route, stationary):
        # ----- branch B : gate forced OFF -----------------------------------
        # replay the exact same rolling-buffer bookkeeping, then take dv_B = 0.
        d_pos_b = d_route + getattr(self, "_delta_b", 0.0)
        # ----- branch A : real gate ----------------------------------------
        delta_before = self._delta
        real_step(self, t, dt, v_route, v_prior, v_spectral, d_route, stationary)
        dv_a = (self._delta - delta_before) / dt if dt > 0 else 0.0
        self._delta_b = getattr(self, "_delta_b", 0.0) + 0.0        # gate OFF -> dv_B = 0
        log.append(dict(
            t=float(t), dt=float(dt), gate_A=bool(self._gate),
            v_route=float(v_route), v_prior=float(v_prior),
            v_spectral=float(v_spectral) if np.isfinite(v_spectral) else float("nan"),
            v_position_A=float(self._v_position), v_position_B=float(v_route),
            dv_A=float(dv_a), d_route=float(d_route), delta_A=float(self._delta),
            stationary=bool(stationary)))

    dpm.DisplayPositionBranch.step = spy
    try:
        inp = build_inputs(trip, net, cfg)
        obs = GroundTruthObserver(map_match_reference(trip.reference_locations, net, net.frame))
        res = PacmanTracker(net, cfg, geometry=RoadGeometry(net, cfg.geometry)).run(inp, observer=obs)
    finally:
        dpm.DisplayPositionBranch.step = real_step

    truth = map_match_reference(trip.reference_locations, net, net.frame)
    spev = _single_path_evaluation(res, truth, trip, inp.t_start, net)
    return trip, net, inp, res, spev, log


def analyse(leave_0726_out: bool, tag: str):
    trip, net, inp, res, spev, log = run(leave_0726_out)
    df = _rows_to_arrays(log)
    dtf, dvf = d_true_fn(trip, inp.t_start)

    t = df["t"]
    D_true = dtf(t)
    v_true = dvf(t)
    D_pos_A = df["d_route"] + df["delta_A"]
    D_pos_B = df["d_route"]                       # gate OFF -> D_position_B == D_route
    e_A = D_pos_A - D_true
    e_B = D_pos_B - D_true

    # ---- per-sample invariant : v_position_A >= v_position_B ---------------
    inv = df["v_position_A"] - df["v_position_B"]
    n_viol = int(np.sum(inv < -1e-9))
    print(f"\n================  {tag}  ================")
    print(f"samples {len(t)}   gate_A ON fraction {df['gate_A'].mean():.3f}")
    print(f"invariant  v_position_A >= v_position_B : "
          f"violations {n_viol} / {len(t)}   min(v_A - v_B) = {inv.min():+.4f}")

    # ---- per-sample log ---------------------------------------------------
    samp = np.column_stack([
        t, df["gate_A"].astype(int), v_true, df["v_route"], df["v_position_A"],
        df["v_position_B"], df["v_position_A"] - df["v_position_B"],
        D_true, D_pos_A, D_pos_B, e_A, e_B])
    hdr = ("t gate_A v_true v_route v_position_A v_position_B dv_A_minus_B "
           "D_true D_position_A D_position_B e_A e_B")
    np.savetxt(OUT / f"ab_gate_samples_{tag}.csv", samp, delimiter=",",
               header=hdr.replace(" ", ","), comments="", fmt="%.4f")
    print(f"per-sample log -> ab_gate_samples_{tag}.csv  ({len(t)} rows)")

    # ---- gate=ON episodes ----------------------------------------------
    on = df["gate_A"].astype(bool)
    episodes = []
    i = 0
    while i < len(on):
        if not on[i]:
            i += 1
            continue
        j = i
        while j < len(on) and on[j]:
            j += 1
        sl = slice(i, j)
        dt_s = df["dt"][sl]
        dist_true = float(np.sum(v_true[sl] * dt_s))
        dist_A = float(np.sum(df["v_position_A"][sl] * dt_s))
        dist_B = float(np.sum(df["v_position_B"][sl] * dt_s))
        episodes.append(dict(
            start=float(t[i] - t[0]), end=float(t[j - 1] - t[0]),
            dur=float(t[j - 1] - t[i]),
            eA_start=float(e_A[i]), eA_end=float(e_A[j - 1]), eA_min=float(e_A[sl].min()),
            eB_start=float(e_B[i]), eB_end=float(e_B[j - 1]), eB_min=float(e_B[sl].min()),
            dist_true=dist_true, dist_A=dist_A, dist_B=dist_B,
            corr_added=dist_A - dist_B))
        i = j
    _write_episodes_csv(OUT / f"ab_gate_episodes_{tag}.csv", episodes)
    print(f"\ngate=ON episodes ({len(episodes)}):  [times are outage-elapsed s]")
    print(f"{'start':>7} {'end':>7} {'dur':>6} | {'eA_st':>7} {'eA_end':>7} {'eA_min':>7} | "
          f"{'eB_st':>7} {'eB_end':>7} {'eB_min':>7} | {'d_true':>7} {'d_A':>7} {'d_B':>7} {'corr+':>7}")
    for e in episodes:
        print(f"{e['start']:7.1f} {e['end']:7.1f} {e['dur']:6.1f} | "
              f"{e['eA_start']:7.1f} {e['eA_end']:7.1f} {e['eA_min']:7.1f} | "
              f"{e['eB_start']:7.1f} {e['eB_end']:7.1f} {e['eB_min']:7.1f} | "
              f"{e['dist_true']:7.1f} {e['dist_A']:7.1f} {e['dist_B']:7.1f} {e['corr_added']:7.1f}")

    # ---- during-ON comparison : is e_A always >= e_B ? --------------------
    on_idx = on
    dA = e_A[on_idx]; dB = e_B[on_idx]
    a_less_neg = np.mean(np.abs(dA) <= np.abs(dB) + 1e-6)
    a_above_b = np.mean(dA >= dB - 1e-6)
    print(f"\nduring gate=ON (n={int(on_idx.sum())}):")
    print(f"  e_A >= e_B (A less negative)         : {a_above_b:.3f} of samples")
    print(f"  |e_A| <= |e_B| (A closer to truth)   : {a_less_neg:.3f} of samples")
    print(f"  mean e_A {dA.mean():+.1f}   mean e_B {dB.mean():+.1f}   "
          f"mean |e_A| {np.abs(dA).mean():.1f}   mean |e_B| {np.abs(dB).mean():.1f}")

    # ---- whole-trip summary table --------------------------------------
    def summ(e, D):
        return dict(median=float(np.median(np.abs(e))), p95=float(np.percentile(np.abs(e), 95)),
                    mx=float(np.abs(e).max()), endpoint=float(e[-1]))
    sA, sB = summ(e_A, D_pos_A), summ(e_B, D_pos_B)
    print(f"\n{'run':<12}{'median|err|':>12}{'p95|err|':>11}{'max|err|':>11}{'endpoint':>11}")
    print(f"{'gate ON':<12}{sA['median']:>12.1f}{sA['p95']:>11.1f}{sA['mx']:>11.1f}{sA['endpoint']:>11.1f}")
    print(f"{'forced OFF':<12}{sB['median']:>12.1f}{sB['p95']:>11.1f}{sB['mx']:>11.1f}{sB['endpoint']:>11.1f}")

    verdict = _verdict(dA, dB, n_viol)
    print(f"\n>>> {verdict}")

    _plot(tag, t - t[0], e_A, e_B, on, dvf)
    return dict(tag=tag, summaryA=sA, summaryB=sB, n_viol=n_viol,
                gate_on_frac=float(df["gate_A"].mean()), verdict=verdict,
                a_above_b_on=float(a_above_b), a_closer_on=float(a_less_neg),
                topology_decisions=len(spev["decisions"]),
                real_wrong=spev["real_wrong_decision_count"])


def _verdict(dA, dB, n_viol):
    if n_viol > 0:
        return "Runtime inconsistent with Phase 32 equation (v_position_A < v_position_B somewhere)"
    a_above = np.mean(dA >= dB - 1e-6)
    if a_above >= 0.999:
        return "Gate helps (e_A always >= e_B during ON) - correction is in the right direction"
    if a_above <= 0.5:
        return "Gate hurts (e_A drops below e_B during ON)"
    return (f"Gate helps mostly (e_A >= e_B on {a_above:.1%} of ON samples); "
            "elsewhere the correction overshoots past truth")


def _plot(tag, el, e_A, e_B, on, dvf):
    fig, ax = plt.subplots(figsize=(15, 6))
    # shade gate-ON intervals
    i = 0
    lab = True
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            ax.axvspan(el[i], el[j - 1], color="#2ca02c", alpha=0.10,
                       label="gate ON (branch A)" if lab else None)
            lab = False
            i = j
        else:
            i += 1
    ax.axhline(0, color="k", lw=0.8)
    ax.plot(el, e_B, "#d62728", lw=1.6, label="e_B  = D_route − D_true  (gate forced OFF)")
    ax.plot(el, e_A, "#1f77b4", lw=1.6, label="e_A  = D_position − D_true  (real gate)")
    ax.set_xlabel("outage elapsed [s]")
    ax.set_ylabel("along-route error  D_position − D_true  [m]")
    ax.set_title(f"rf-07-26 strict A/B — only the display saturation gate differs  [{tag}]")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / f"ab_gate_{tag}.png", dpi=130)
    plt.close(fig)
    print(f"plot -> ab_gate_{tag}.png")


# --------------------------------------------------------------------------
def _rows_to_arrays(log):
    keys = log[0].keys()
    return {k: np.array([r[k] for r in log], dtype=float if not isinstance(log[0][k], bool) else bool)
            for k in keys}


def _write_episodes_csv(path, episodes):
    cols = ["start", "end", "dur", "eA_start", "eA_end", "eA_min",
            "eB_start", "eB_end", "eB_min", "dist_true", "dist_A", "dist_B", "corr_added"]
    with open(path, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for e in episodes:
            fh.write(",".join(f"{e[c]:.3f}" for c in cols) + "\n")


def main():
    res = {}
    res["leave_0726_out"] = analyse(True, "leave_0726_out")
    res["full_pool"] = analyse(False, "full_pool")
    (OUT / "ab_gate_verdict.json").write_text(json.dumps(res, indent=1))
    print("\n\n=========================  FINAL  =========================")
    for k, r in res.items():
        a, b = r["summaryA"], r["summaryB"]
        print(f"\n[{k}]  (gate ON fraction {r['gate_on_frac']:.2f}, "
              f"topology decisions {r['topology_decisions']}, real_wrong {r['real_wrong']})")
        print(f"{'run':<12}{'median|err|':>12}{'p95|err|':>11}{'max|err|':>11}{'endpoint':>11}")
        print(f"{'gate ON':<12}{a['median']:>12.1f}{a['p95']:>11.1f}{a['mx']:>11.1f}{a['endpoint']:>11.1f}")
        print(f"{'forced OFF':<12}{b['median']:>12.1f}{b['p95']:>11.1f}{b['mx']:>11.1f}{b['endpoint']:>11.1f}")
        print(f">>> {r['verdict']}")


if __name__ == "__main__":
    sys.exit(main())
