#!/usr/bin/env python3
"""Ablate causal EKF authority guards without feeding GPS to the filter."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from geotrace.coordinates import LocalFrame
from geotrace.loader import load_trip
from geotrace.pacman_tracker.config import PacmanConfig
from geotrace.pacman_tracker.speed import GlobalSpeedTracker
from geotrace.pacman_tracker.tracker import build_inputs
from geotrace.road_graph import RoadNetwork, clip_graph, load_graph


ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "runs/review-map.graphml"
OUT = ROOT / "docs/plots/phase37"
TRIPS = {
    "rf-07-22": ROOT / "runs/review-final/2026-07-22/trip",
    "rf-07-26": ROOT / "runs/review-final/2026-07-26/trip",
    "ir-07-23": ROOT / "runs/independent-review/2026-07-23/trip",
    "ir-07-24": ROOT / "runs/independent-review/2026-07-24/trip",
    "ir-07-25": ROOT / "runs/independent-review/2026-07-25/trip",
}
WINDOWS = [(27.0, 115.0), (212.0, 304.0), (316.0, 390.0), (436.0, 455.0)]


def truth(trip, t_start: float, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = [(f.monotonic_time, float(f.speed))
                 for f in trip.usable_locations if f.has_valid_speed]
    reference += [(f.monotonic_time, float(f.speed))
                  for f in trip.reference_locations
                  if f.is_usable and f.has_valid_speed]
    reference.sort()
    rt = np.asarray([p[0] for p in reference], dtype=float)
    rv = np.asarray([p[1] for p in reference], dtype=float)
    dt = float(np.median(np.diff(rt)))
    cumulative = np.cumsum(rv * dt)
    cumulative -= float(np.interp(t_start, rt, cumulative))
    return np.interp(times, rt, cumulative), np.interp(times, rt, rv)


def replay(inputs, trip, flags: dict[str, float | bool]) -> dict:
    cfg = PacmanConfig()
    cfg.speed.zupt_motion_guard_enabled = bool(flags.get("zupt", False))
    cfg.speed.lateral_consensus_guard_enabled = bool(flags.get("lateral", False))
    cfg.speed.lateral_consensus_deadband_ms = float(flags.get("lat_deadband", 2.0))
    cfg.speed.spectral_saturation_guard_enabled = bool(flags.get("spectral", False))
    cfg.speed.spectral_guard_factor = float(flags.get("factor", 8.0))
    tracker = GlobalSpeedTracker(
        cfg.speed, v0=inputs.speed0, gyro_bias0=inputs.gyro_bias0,
        accel_bias0=inputs.accel_bias0)
    ts, ds, vs = [], [], []
    for sample in (s for s in inputs.samples if s.t > inputs.t_start):
        tracker.predict(sample.a_long, sample.dt, sample.shock, sample.gap)
        stopped = False
        if sample.stationary:
            stopped = tracker.zero_velocity(
                sample.a_long, sample.dt, sample.yaw_rate,
                sample.stationary_run_s, spectral_speed=sample.spectral_speed)
        if not stopped:
            tracker.lateral_anchor(
                sample.a_lat, sample.yaw_rate_smooth, sample.dt, sample.shock,
                spectral_speed=sample.spectral_speed, t=sample.t)
            if math.isfinite(sample.spectral_speed):
                tracker.spectral_update(sample.spectral_speed,
                                        sample.spectral_sigma, sample.dt)
        ts.append(float(sample.t)); ds.append(tracker.distance); vs.append(tracker.speed)
    t = np.asarray(ts); d = np.asarray(ds); v = np.asarray(vs)
    d_true, v_true = truth(trip, inputs.t_start, t)
    e = d - d_true
    elapsed = t - inputs.t_start
    windows = []
    for start, end in WINDOWS:
        if end > elapsed[-1]:
            continue
        e0 = float(np.interp(start, elapsed, e))
        e1 = float(np.interp(end, elapsed, e))
        windows.append({"start_s": start, "end_s": end,
                        "error_change_m": e1 - e0, "error_end_m": e1})
    return {
        "median_abs_error_m": float(np.median(np.abs(e))),
        "p95_abs_error_m": float(np.percentile(np.abs(e), 95)),
        "max_abs_error_m": float(np.max(np.abs(e))),
        "endpoint_error_m": float(e[-1]),
        "speed_mae_ms": float(np.mean(np.abs(v - v_true))),
        "speed_bias_ms": float(np.mean(v - v_true)),
        "distance_ratio": float(d[-1] / d_true[-1]),
        "windows": windows,
        "counts": {k: int(tracker.counts[k]) for k in (
            "zupt", "zupt_motion_rejected", "lateral",
            "lateral_consensus_downweighted", "spectral", "spectral_guarded")},
    }


def main() -> None:
    variants = {
        "baseline": {},
        "zupt": {"zupt": True},
        "lateral": {"lateral": True},
        "lateral_db3": {"lateral": True, "lat_deadband": 3.0},
        "lateral_db4": {"lateral": True, "lat_deadband": 4.0},
        "spectral_f4": {"spectral": True, "factor": 4.0},
        "spectral_f8": {"spectral": True, "factor": 8.0},
        "spectral_f15": {"spectral": True, "factor": 15.0},
        "zupt_lateral": {"zupt": True, "lateral": True},
        "all_f4": {"zupt": True, "lateral": True, "spectral": True,
                   "factor": 4.0},
        "all_f8": {"zupt": True, "lateral": True, "spectral": True,
                   "factor": 8.0},
        "all_f8_db3": {"zupt": True, "lateral": True, "spectral": True,
                       "factor": 8.0, "lat_deadband": 3.0},
        "all_f8_db4": {"zupt": True, "lateral": True, "spectral": True,
                       "factor": 8.0, "lat_deadband": 4.0},
        "all_f15": {"zupt": True, "lateral": True, "spectral": True,
                    "factor": 15.0},
    }
    graph = load_graph(GRAPH)
    output = {}
    for tag, path in TRIPS.items():
        try:
            trip, _ = load_trip(path)
            first = trip.usable_locations[0]
            network = RoadNetwork(
                clip_graph(graph, first.latitude, first.longitude, 11000.0),
                LocalFrame(first.latitude, first.longitude))
            inputs = build_inputs(trip, network, PacmanConfig())
        except ValueError as exc:
            output[tag] = {"skipped": str(exc)}
            print(f"{tag}: skipped ({exc})")
            continue
        output[tag] = {name: replay(inputs, trip, flags)
                       for name, flags in variants.items()}
        print(tag)
        for name, result in output[tag].items():
            print(f"  {name:14s} med={result['median_abs_error_m']:7.1f} "
                  f"p95={result['p95_abs_error_m']:7.1f} "
                  f"max={result['max_abs_error_m']:7.1f} "
                  f"end={result['endpoint_error_m']:8.1f} "
                  f"vmae={result['speed_mae_ms']:5.2f} "
                  f"ratio={result['distance_ratio']:.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "authority_guard_ablation.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
