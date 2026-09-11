#!/usr/bin/env python3
"""Phase 31 - large-scale cross-trip ML validation: inventory + dataset rebuild.

Same observable-only feature set and BASELINE speed-EKF replay as Phase 30
(tools/phase30_dataset.py), now over EVERY usable logger session found under
runs/phase31/ (imported from live_logs day files) plus runs/review-final/.

Outputs (docs/plots/phase31/):
    inventory.csv          one row per trip found, with usability verdict
    windows_5s.csv         non-overlapping 5 s windows, all usable trips
    windows_2s.csv         2 s windows
    grid_<trip>.csv        the 0.5 s replay grid per usable trip
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from phase30_dataset import replay_trip as _replay30, make_windows, GRID_DT  # noqa

from geotrace.loader import load_trip
from geotrace.pacman_tracker.config import PacmanConfig
from geotrace.pacman_tracker.turns import detect_turns

ROOT = Path("/Users/arseniyabramidze/delimobil/gps-route-recovery")
OUT = ROOT / "docs" / "plots" / "phase31"
OUT.mkdir(parents=True, exist_ok=True)

SPEED_EDGES = [0, 5, 10, 15, 20, 25, 999]


def discover() -> dict[str, Path]:
    trips: dict[str, Path] = {}
    for d in sorted((ROOT / "runs/phase31").glob("*/trip")):
        trips[d.parent.name] = d
    for d in sorted((ROOT / "runs/review-final").glob("2026-*/trip")):
        tag = "rf-" + d.parent.name.replace("2026-", "")
        if tag not in trips:
            trips[tag] = d
    return trips


def inventory_row(tag: str, trip_dir: Path, grid: pd.DataFrame | None,
                  fail: str | None) -> dict:
    row = dict(trip=tag, path=str(trip_dir.relative_to(ROOT)))
    try:
        trip, _ = load_trip(trip_dir)
        md = trip.metadata
        vis = [f for f in trip.usable_locations if f.has_valid_speed]
        ref = [f for f in trip.reference_locations if f.is_usable and f.has_valid_speed]
        row["trip_id"] = md.trip_id
        row["n_visible_gps"] = len(vis)
        row["n_withheld_gps"] = len(ref)
        row["duration_s"] = round(trip.duration_s, 0)
        row["warmup_s"] = round((md.extra or {}).get("live_import", {}).get(
            "gps_visible_until_s", float("nan")), 1)
        has_mag = any(getattr(m, "magnetic_field", None) is not None for m in trip.motions[:50])
        row["has_mag"] = bool(has_mag)
    except Exception as e:
        row["load_error"] = f"{type(e).__name__}: {e}"
    if fail:
        row["usable"] = False
        row["reason"] = fail
        return row
    if grid is None:
        row["usable"] = False
        row["reason"] = "no grid"
        return row

    o = grid[grid.outage]
    vt = o.v_true.to_numpy()
    vs = o.v_spec.to_numpy()
    moving = vt > 2.0
    dt = GRID_DT
    row["outage_s"] = round(len(o) * dt, 0)
    row["moving_s"] = round(int(moving.sum()) * dt, 0)
    row["v_true_med"] = round(float(np.median(vt[moving])), 1) if moving.any() else 0.0
    row["v_true_p95"] = round(float(np.percentile(vt[moving], 95)), 1) if moving.any() else 0.0
    row["v_true_max"] = round(float(vt.max()), 1)
    for lo, hi in zip(SPEED_EDGES[:-1], SPEED_EDGES[1:]):
        frac = float(np.mean((vt >= lo) & (vt < hi)))
        row[f"f_{lo}_{hi}"] = round(frac, 3)
    # spectral saturation fraction: v_spec in 12-15 while v_true clearly higher
    sat = (vs >= 12.0) & (vs <= 15.5) & (vt - vs > 2.0)
    row["f_saturation"] = round(float(np.mean(sat)), 3)
    row["f_vspec_gt15"] = round(float(np.mean(vs > 15.0)), 3)
    row["spec_coverage"] = round(float(np.mean(np.isfinite(vs))), 3)
    row["dep_sigma_spec"] = round(float(grid.dep_sigma_spec.iloc[0]), 2)
    row["spec_gap"] = round(float(grid.spec_gap.iloc[0]), 2)
    row["zupt_frac"] = round(float(o.stationary.mean()), 3)
    # ZUPT episodes
    st = (o.stationary.to_numpy() > 0.5).astype(int)
    row["zupt_episodes"] = int(np.sum(np.diff(st) == 1))
    # bends
    try:
        trip, _ = load_trip(trip_dir)
        from phase30_dataset import _lateral_channel
        rt, _la, om, _ac, _gy = _lateral_channel(trip)
        ev = detect_turns(rt, om, 0.0)
        row["n_turns"] = len(ev)
        row["n_bends"] = sum(1 for e in ev if abs(getattr(e, "angle_deg", 0.0)) < 35.0
                             and getattr(e, "peak_rate_rads", 1.0) < 0.18)
    except Exception:
        row["n_turns"] = row["n_bends"] = -1

    hi_frac = row["f_15_20"] + row["f_20_25"] + row["f_25_999"]
    reasons = []
    if row["moving_s"] < 120:
        reasons.append("moving<120s")
    if row["outage_s"] < 200:
        reasons.append("outage<200s")
    if row["spec_coverage"] < 0.8:
        reasons.append("spectral coverage<0.8")
    if row.get("n_withheld_gps", 0) < 100:
        reasons.append("withheld GPS<100")
    row["hi_speed_frac"] = round(hi_frac, 3)
    row["usable"] = len(reasons) == 0
    row["reason"] = "; ".join(reasons) if reasons else "ok"
    return row


def main() -> int:
    trips = discover()
    print(f"discovered {len(trips)} trip dirs")
    inv, grids = [], {}
    for tag, d in trips.items():
        fail = None
        g = None
        try:
            g = _replay30(tag, d)
        except SystemExit as e:
            fail = str(e)
        except Exception as e:
            fail = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        row = inventory_row(tag, d, g, fail)
        inv.append(row)
        status = "OK  " if row["usable"] else "SKIP"
        print(f"  {status} {tag:12s}  {row.get('reason','')[:60]}")
        if row["usable"]:
            g.to_csv(OUT / f"grid_{tag}.csv", index=False)
            grids[tag] = g

    invdf = pd.DataFrame(inv)
    invdf.to_csv(OUT / "inventory.csv", index=False)
    print(f"\ntotal={len(invdf)}  usable={int(invdf.usable.sum())}  "
          f"excluded={int((~invdf.usable).sum())}")
    hi = invdf[invdf.usable & (invdf.hi_speed_frac > 0.10)]
    print(f"high-speed trips (>10% time above 15 m/s): {len(hi)}")

    for ws, name in ((5.0, "windows_5s.csv"), (2.0, "windows_2s.csv")):
        parts = [make_windows(g, ws) for g in grids.values()]
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(OUT / name, index=False)
        out = df[df.regime == "outage"]
        sm = out[out._steady & out._moving & out._reliable_vspec]
        print(f"{name}: {len(df)} rows, {len(out)} outage, {len(sm)} steady&moving&reliable "
              f"across {out.trip.nunique()} trips")
    return 0


if __name__ == "__main__":
    sys.exit(main())
