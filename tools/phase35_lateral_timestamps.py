#!/usr/bin/env python3
"""Forensic timestamp/state trace for the baseline lateral speed anchor.

This tool does not change tracker configuration, route logic, or display
position.  It reconstructs the exact raw-sample support of the centered
boxcar used by ``build_imu_samples`` and replays only the baseline speed
tracker in its production predict -> lateral -> spectral order.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from geotrace.coordinates import LocalFrame
from geotrace.loader import load_trip
from geotrace.pacman_tracker.config import PacmanConfig
from geotrace.pacman_tracker.speed import GlobalSpeedTracker
from geotrace.pacman_tracker.tracker import (
    _attitude_channel,
    _lateral_channel,
    build_inputs,
)
from geotrace.road_graph import RoadNetwork, clip_graph, load_graph


ROOT = Path(__file__).resolve().parents[1]
TRIP_DIR = ROOT / "runs/review-final/2026-07-26/trip"
GRAPH = ROOT / "runs/review-map.graphml"
OUT = ROOT / "docs/plots/phase35"


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _window_indices(raw_count: int, raw_index: int, width: int) -> np.ndarray:
    """Indices contributing to the exact edge-padded centered boxcar."""
    left = width // 2
    return np.clip(np.arange(raw_index - left, raw_index - left + width),
                   0, raw_count - 1)


def main() -> None:
    cfg = PacmanConfig()
    trip, _ = load_trip(TRIP_DIR)
    first = trip.usable_locations[0]
    network = RoadNetwork(
        clip_graph(load_graph(GRAPH), first.latitude, first.longitude, 11000.0),
        LocalFrame(first.latitude, first.longitude),
    )
    inputs = build_inputs(trip, network, cfg)

    raw_t, raw_lat, raw_omega, _accel, _gyro = _lateral_channel(trip)
    if cfg.attitude.enabled:
        att_t, _long, att_lat, att_omega, _diag = _attitude_channel(trip, cfg)
        if att_t.size:
            raw_t, raw_lat, raw_omega = att_t, att_lat, att_omega
    raw_dt = float(np.median(np.diff(raw_t)))
    width = max(1, int(round(cfg.speed.lateral_smooth_s / raw_dt)))

    tracker = GlobalSpeedTracker(
        cfg.speed,
        v0=inputs.speed0,
        gyro_bias0=inputs.gyro_bias0,
        accel_bias0=inputs.accel_bias0,
    )
    rows: list[dict[str, float | bool]] = []
    attempted = 0

    for sample in (s for s in inputs.samples if s.t > inputs.t_start):
        d_before_predict = tracker.distance
        v_before_predict = tracker.speed
        tracker.predict(sample.a_long, sample.dt, sample.shock, sample.gap)
        d_after_predict = tracker.distance
        v_pred = tracker.speed

        if sample.stationary:
            tracker.zero_velocity(sample.a_long, sample.dt, sample.yaw_rate,
                                  sample.stationary_run_s)
            continue

        attempted += 1
        d_before_lateral = tracker.distance
        v_before_lateral = tracker.speed
        result = tracker.lateral_anchor(
            sample.a_lat,
            sample.yaw_rate_smooth,
            sample.dt,
            sample.shock,
            spectral_speed=sample.spectral_speed,
            t=sample.t,
        )
        d_after_lateral = tracker.distance
        v_post = tracker.speed

        if result is not None:
            raw_index = int(np.clip(np.searchsorted(raw_t, sample.t),
                                    0, len(raw_t) - 1))
            indices = _window_indices(len(raw_t), raw_index, width)
            times = raw_t[indices]
            omega = np.abs(raw_omega[indices] - tracker.gyro_bias)
            omega_sum = float(omega.sum())
            turn_centroid = (float(np.sum(times * omega) / omega_sum)
                             if omega_sum > 1e-12 else float(np.mean(times)))
            peak_index = int(indices[int(np.argmax(omega))])
            physical_t = float(np.mean(times))
            measured, sigma = result
            rows.append({
                "anchor_received_time": float(sample.t),
                "anchor_received_elapsed_s": float(sample.t - inputs.t_start),
                # A uniform centered FIR's timestamp is its centre of mass.
                "physical_measurement_time": physical_t,
                "delay_seconds": float(sample.t - physical_t),
                "v_lat": float(measured),
                "sigma": float(sigma),
                "confidence_inv_var": float(1.0 / (sigma * sigma)),
                "window_start": float(times[0]),
                "window_end": float(times[-1]),
                "window_duration_s": float(times[-1] - times[0]),
                "turn_centroid_time": turn_centroid,
                "delay_to_turn_centroid_s": float(sample.t - turn_centroid),
                "peak_turn_time": float(raw_t[peak_index]),
                "delay_to_peak_s": float(sample.t - raw_t[peak_index]),
                "a_lat_smoothed": float(sample.a_lat),
                "omega_smoothed": float(sample.yaw_rate_smooth - tracker.gyro_bias),
                "D_before_predict": d_before_predict,
                "v_before_predict": v_before_predict,
                "D_after_predict": d_after_predict,
                "v_pred": v_pred,
                "D_before_lateral_update": d_before_lateral,
                "v_before_lateral_update": v_before_lateral,
                "D_after_lateral_update": d_after_lateral,
                "v_post": v_post,
                "delta_D_lateral": d_after_lateral - d_before_lateral,
                "delta_v_lateral": v_post - v_before_lateral,
                "shock": bool(sample.shock),
            })

        if np.isfinite(sample.spectral_speed):
            tracker.spectral_update(sample.spectral_speed,
                                    sample.spectral_sigma, sample.dt)

    frame = pd.DataFrame(rows)
    delays = frame["delay_seconds"].to_numpy()
    centroid_delays = frame["delay_to_turn_centroid_s"].to_numpy()
    d_updates = frame["delta_D_lateral"].to_numpy()
    summary = {
        "trip": "rf-07-26",
        "t_start": inputs.t_start,
        "raw_sample_dt_s": raw_dt,
        "raw_boxcar_samples": width,
        "configured_window_s": cfg.speed.lateral_smooth_s,
        "accepted_anchors": len(frame),
        "attempted_nonstationary_steps": attempted,
        "reported_tracker_count": tracker.counts["lateral"],
        "timestamp_semantics": "uniform centered-window centre of mass",
        "delay_seconds": _percentiles(delays),
        "absolute_delay_seconds": _percentiles(np.abs(delays)),
        "content_centroid_delay_seconds": _percentiles(centroid_delays),
        "absolute_content_centroid_delay_seconds": _percentiles(
            np.abs(centroid_delays)),
        "lateral_distance_state_update_m": {
            **_percentiles(np.abs(d_updates)),
            "nonzero_count": int(np.count_nonzero(np.abs(d_updates) > 1e-12)),
            "signed_sum": float(np.sum(d_updates)),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "lateral_anchors_rf-07-26.csv", index=False)
    (OUT / "lateral_timing_summary_rf-07-26.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT / 'lateral_anchors_rf-07-26.csv'}")


if __name__ == "__main__":
    main()
