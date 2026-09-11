#!/usr/bin/env python3
"""Attribute baseline D/v changes to prediction, lateral, spectral, and ZUPT.

GPS reference is used only after replay to score measurement quality and
distance error. It never enters the filter. No production object is patched.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from geotrace.coordinates import LocalFrame
from geotrace.loader import load_trip
from geotrace.pacman_tracker.config import PacmanConfig
from geotrace.pacman_tracker.speed import KA, V, GlobalSpeedTracker
from geotrace.pacman_tracker.tracker import build_inputs
from geotrace.road_graph import RoadNetwork, clip_graph, load_graph


ROOT = Path(__file__).resolve().parents[1]
TRIP_DIR = ROOT / "runs/review-final/2026-07-26/trip"
GRAPH = ROOT / "runs/review-map.graphml"
OUT = ROOT / "docs/plots/phase36"
WINDOWS = [(27.0, 115.0), (212.0, 304.0), (316.0, 390.0), (436.0, 455.0)]


def _phase36_config() -> PacmanConfig:
    """Reproduce the pre-Phase-37 baseline investigated by this report."""
    cfg = PacmanConfig()
    cfg.speed.zupt_motion_guard_enabled = False
    cfg.speed.lateral_consensus_guard_enabled = False
    cfg.speed.spectral_saturation_guard_enabled = False
    return cfg


def _truth(trip, t_start: float, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


class RecordingSpeedTracker(GlobalSpeedTracker):
    """Expose the exact scalar Kalman updates without changing their behavior."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inner_updates: list[dict[str, Any]] = []

    def _update(self, H: np.ndarray, innovation: float, R: float,
                allow_scale: bool = False) -> None:
        H = np.asarray(H, dtype=float)
        before_x = self.x.copy()
        before_p = self.P.copy()
        ph = before_p @ H
        s = float(H @ ph + R)
        gain = ph / s if math.isfinite(s) and s > 0.0 else np.zeros_like(ph)
        if self.cfg.accel_scale_enabled and not allow_scale:
            gain[KA] = 0.0
        super()._update(H, innovation, R, allow_scale=allow_scale)
        self.inner_updates.append({
            "method": "full_state",
            "H": H.tolist(),
            "z": float(H @ before_x + innovation),
            "predicted_z": float(H @ before_x),
            "innovation": float(innovation),
            "R": float(R),
            "effective_sigma": float(math.sqrt(max(R, 0.0))),
            "K_D": float(gain[0]),
            "K_v": float(gain[V]),
            "delta_D": float(self.x[0] - before_x[0]),
            "delta_v": float(self.x[V] - before_x[V]),
        })

    def _update_speed_only(self, innovation: float, R: float) -> None:
        before_x = self.x.copy()
        pv = float(self.P[V, V])
        s = pv + R
        gain_v = pv / s if math.isfinite(s) and s > 0.0 else 0.0
        super()._update_speed_only(innovation, R)
        self.inner_updates.append({
            "method": "speed_only",
            "H": [0.0, 1.0, 0.0, 0.0, 0.0],
            "z": float(before_x[V] + innovation),
            "predicted_z": float(before_x[V]),
            "innovation": float(innovation),
            "R": float(R),
            "effective_sigma": float(math.sqrt(max(R, 0.0))),
            "K_D": 0.0,
            "K_v": float(gain_v),
            "delta_D": float(self.x[0] - before_x[0]),
            "delta_v": float(self.x[V] - before_x[V]),
        })


def _record_update(tracker: RecordingSpeedTracker, kind: str, t: float,
                   nominal_sigma: float, callback: Callable[[], Any],
                   metadata: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
    before_x = tracker.x.copy()
    before_p = tracker.P.copy()
    tracker.inner_updates = []
    result = callback()
    after_x = tracker.x.copy()
    primary = next((u for u in tracker.inner_updates
                    if np.argmax(np.abs(u["H"])) == V),
                   tracker.inner_updates[0] if tracker.inner_updates else None)
    row: dict[str, Any] = {
        "t": float(t),
        "measurement_type": kind,
        "D_before": float(before_x[0]),
        "v_before": float(before_x[V]),
        "P_DD_before": float(before_p[0, 0]),
        "P_Dv_before": float(before_p[0, V]),
        "P_vv_before": float(before_p[V, V]),
        "z": float(primary["z"]) if primary else float("nan"),
        "predicted_z": float(primary["predicted_z"]) if primary else float("nan"),
        "innovation": float(primary["innovation"]) if primary else float("nan"),
        "nominal_sigma": float(nominal_sigma),
        "effective_sigma": (float(primary["effective_sigma"])
                            if primary else float("nan")),
        "K_D": float(primary["K_D"]) if primary else float("nan"),
        "K_v": float(primary["K_v"]) if primary else float("nan"),
        "D_after": float(after_x[0]),
        "v_after": float(after_x[V]),
        "delta_D": float(after_x[0] - before_x[0]),
        "delta_v": float(after_x[V] - before_x[V]),
        "inner_updates": json.dumps(tracker.inner_updates, separators=(",", ":")),
    }
    if metadata:
        row.update(metadata)
    return result, row


def _summary(group: pd.DataFrame) -> dict[str, Any]:
    if group.empty:
        return {"count": 0}
    kind = str(group["measurement_type"].iloc[0])
    # For ZUPT, raw_z=0 is the detector's physical claim. `z` is the softened
    # pseudo-target after the reachability credibility gate.
    quality_z = group["raw_z"] if kind == "zupt" else group["z"]
    error = quality_z - group["v_true"]
    normalized = error / group["nominal_sigma"].replace(0.0, np.nan)
    out = {
        "count": int(len(group)),
        "mean_z_ms": float(group["z"].mean()),
        "mean_raw_z_ms": float(group["raw_z"].mean()),
        "mean_true_speed_ms": float(group["v_true"].mean()),
        "mean_measurement_bias_ms": float(error.mean()),
        "median_measurement_bias_ms": float(error.median()),
        "measurement_mae_ms": float(error.abs().mean()),
        "measurement_rmse_ms": float(np.sqrt(np.mean(error * error))),
        "mean_nominal_sigma_ms": float(group["nominal_sigma"].mean()),
        "mean_effective_sigma_ms": float(group["effective_sigma"].mean()),
        "within_nominal_1sigma_fraction": float((normalized.abs() <= 1.0).mean()),
        "within_nominal_2sigma_fraction": float((normalized.abs() <= 2.0).mean()),
        "mean_K_D": float(group["K_D"].mean()),
        "mean_K_v": float(group["K_v"].mean()),
        "sum_delta_D_m": float(group["delta_D"].sum()),
        "sum_delta_v_ms": float(group["delta_v"].sum()),
    }
    if kind == "zupt":
        ordered = group.sort_values("t")
        runs = (ordered["t"].diff().fillna(1.0) > 0.11).cumsum()
        run_durations = ordered.groupby(runs)["t"].agg(lambda x: float(x.max() - x.min() + 0.1))
        out.update({
            "truth_speed_gt_2ms_count": int((group["v_true"] > 2.0).sum()),
            "truth_speed_gt_5ms_count": int((group["v_true"] > 5.0).sum()),
            "truth_speed_gt_10ms_count": int((group["v_true"] > 10.0).sum()),
            "longest_contiguous_run_s": float(run_durations.max()),
        })
    return out


def _interp(frame: pd.DataFrame, column: str, elapsed: float) -> float:
    return float(np.interp(elapsed, frame["elapsed_s"], frame[column]))


def _interval_summary(steps: pd.DataFrame, measurements: pd.DataFrame,
                      start: float, end: float) -> dict[str, Any]:
    d_true = _interp(steps, "D_true", end) - _interp(steps, "D_true", start)
    pred = _interp(steps, "cum_prediction_D", end) - _interp(
        steps, "cum_prediction_D", start)
    corrections = {}
    for kind in ("lateral", "spectral", "zupt"):
        col = f"cum_{kind}_D"
        corrections[kind] = _interp(steps, col, end) - _interp(steps, col, start)
    error_start = _interp(steps, "distance_error", start)
    error_end = _interp(steps, "distance_error", end)
    inside = measurements[
        (measurements["elapsed_s"] >= start) & (measurements["elapsed_s"] <= end)]
    return {
        "start_s": start,
        "end_s": end,
        "error_start_m": error_start,
        "error_end_m": error_end,
        "error_change_m": error_end - error_start,
        "true_distance_change_m": d_true,
        "ordinary_prediction_distance_m": pred,
        "prediction_minus_true_m": pred - d_true,
        "lateral_D_correction_m": corrections["lateral"],
        "spectral_D_correction_m": corrections["spectral"],
        "zupt_D_correction_m": corrections["zupt"],
        "closure_m": ((pred - d_true) + sum(corrections.values())
                      - (error_end - error_start)),
        "measurements": {
            kind: _summary(inside[inside["measurement_type"] == kind])
            for kind in ("lateral", "spectral", "zupt")
        },
    }


def _counterfactual(inputs, cfg: PacmanConfig, trip,
                    *, lateral: bool, spectral: bool, zupt: bool) -> dict[str, Any]:
    """Replay a source-removal diagnostic; truth scores only after filtering."""
    local_cfg = _phase36_config()
    local_cfg.speed.lateral_enabled = lateral
    local_cfg.speed.spectral_enabled = spectral
    tracker = GlobalSpeedTracker(
        local_cfg.speed, v0=inputs.speed0, gyro_bias0=inputs.gyro_bias0,
        accel_bias0=inputs.accel_bias0)
    times: list[float] = []
    distances: list[float] = []
    speeds: list[float] = []
    for sample in (s for s in inputs.samples if s.t > inputs.t_start):
        tracker.predict(sample.a_long, sample.dt, sample.shock, sample.gap)
        if sample.stationary:
            if zupt:
                tracker.zero_velocity(sample.a_long, sample.dt, sample.yaw_rate,
                                      sample.stationary_run_s)
        else:
            tracker.lateral_anchor(
                sample.a_lat, sample.yaw_rate_smooth, sample.dt, sample.shock,
                spectral_speed=sample.spectral_speed, t=sample.t)
            if math.isfinite(sample.spectral_speed):
                tracker.spectral_update(sample.spectral_speed,
                                        sample.spectral_sigma, sample.dt)
        times.append(float(sample.t))
        distances.append(tracker.distance)
        speeds.append(tracker.speed)
    t = np.asarray(times)
    d = np.asarray(distances)
    v = np.asarray(speeds)
    d_true, v_true = _truth(trip, inputs.t_start, t)
    elapsed = t - inputs.t_start
    error = d - d_true
    intervals = []
    for start, end in WINDOWS:
        e0 = float(np.interp(start, elapsed, error))
        e1 = float(np.interp(end, elapsed, error))
        intervals.append({"start_s": start, "end_s": end,
                          "error_start_m": e0, "error_end_m": e1,
                          "error_change_m": e1 - e0})
    return {
        "lateral_enabled": lateral,
        "spectral_enabled": spectral,
        "zupt_enabled": zupt,
        "median_abs_error_m": float(np.median(np.abs(error))),
        "p95_abs_error_m": float(np.percentile(np.abs(error), 95)),
        "max_abs_error_m": float(np.max(np.abs(error))),
        "endpoint_error_m": float(error[-1]),
        "speed_mae_ms": float(np.mean(np.abs(v - v_true))),
        "intervals": intervals,
    }


def main() -> None:
    cfg = _phase36_config()
    trip, _ = load_trip(TRIP_DIR)
    first = trip.usable_locations[0]
    network = RoadNetwork(
        clip_graph(load_graph(GRAPH), first.latitude, first.longitude, 11000.0),
        LocalFrame(first.latitude, first.longitude),
    )
    inputs = build_inputs(trip, network, cfg)
    tracker = RecordingSpeedTracker(
        cfg.speed, v0=inputs.speed0, gyro_bias0=inputs.gyro_bias0,
        accel_bias0=inputs.accel_bias0)

    measurements: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    cumulative = {"prediction": 0.0, "lateral": 0.0,
                  "spectral": 0.0, "zupt": 0.0}

    for sample in (s for s in inputs.samples if s.t > inputs.t_start):
        d0 = tracker.distance
        v0 = tracker.speed
        tracker.predict(sample.a_long, sample.dt, sample.shock, sample.gap)
        prediction_d = tracker.distance - d0
        cumulative["prediction"] += prediction_d
        lateral_d = spectral_d = zupt_d = 0.0

        if sample.stationary:
            _, row = _record_update(
                tracker, "zupt", sample.t, cfg.speed.zupt_speed_sigma_ms,
                lambda: tracker.zero_velocity(
                    sample.a_long, sample.dt, sample.yaw_rate,
                    sample.stationary_run_s),
                {"raw_z": 0.0, "input_sigma": cfg.speed.zupt_speed_sigma_ms,
                 "a_long": float(sample.a_long),
                 "yaw_rate": float(sample.yaw_rate),
                 "accel_std": float(sample.accel_std),
                 "gyro_std": float(sample.gyro_std),
                 "stationary_run_s": float(sample.stationary_run_s)})
            zupt_d = row["delta_D"]
            measurements.append(row)
        else:
            n_before = tracker.counts["lateral"]
            result, row = _record_update(
                tracker, "lateral", sample.t, float("nan"),
                lambda: tracker.lateral_anchor(
                    sample.a_lat, sample.yaw_rate_smooth, sample.dt,
                    sample.shock, spectral_speed=sample.spectral_speed,
                    t=sample.t),
                {"raw_z": float(sample.a_lat / (
                    sample.yaw_rate_smooth - tracker.gyro_bias))
                 if abs(sample.yaw_rate_smooth - tracker.gyro_bias) > 1e-12
                 else float("nan"),
                 "input_sigma": float("nan"),
                 "a_lat": float(sample.a_lat),
                 "yaw_rate_smooth": float(sample.yaw_rate_smooth),
                 "accel_std": float(sample.accel_std),
                 "gyro_std": float(sample.gyro_std)})
            if result is not None and tracker.counts["lateral"] > n_before:
                row["nominal_sigma"] = float(result[1])
                lateral_d = row["delta_D"]
                measurements.append(row)

            if math.isfinite(sample.spectral_speed):
                nominal = float(sample.spectral_sigma * cfg.speed.spectral_sigma_scale)
                _, row = _record_update(
                    tracker, "spectral", sample.t, nominal,
                    lambda: tracker.spectral_update(
                        sample.spectral_speed, sample.spectral_sigma, sample.dt),
                    {"raw_z": float(sample.spectral_speed),
                     "input_sigma": float(sample.spectral_sigma),
                     "accel_std": float(sample.accel_std),
                     "gyro_std": float(sample.gyro_std)})
                spectral_d = row["delta_D"]
                measurements.append(row)

        cumulative["lateral"] += lateral_d
        cumulative["spectral"] += spectral_d
        cumulative["zupt"] += zupt_d
        steps.append({
            "t": float(sample.t),
            "elapsed_s": float(sample.t - inputs.t_start),
            "dt": float(sample.dt),
            "D_before_predict": d0,
            "v_before_predict": v0,
            "prediction_delta_D": prediction_d,
            "D_after_all_updates": tracker.distance,
            "v_after_all_updates": tracker.speed,
            "stationary": bool(sample.stationary),
            "lateral_delta_D": lateral_d,
            "spectral_delta_D": spectral_d,
            "zupt_delta_D": zupt_d,
            **{f"cum_{key}_D": value for key, value in cumulative.items()},
        })

    step_frame = pd.DataFrame(steps)
    measurement_frame = pd.DataFrame(measurements)
    d_true, v_true = _truth(trip, inputs.t_start, step_frame["t"].to_numpy())
    step_frame["D_true"] = d_true
    step_frame["v_true"] = v_true
    step_frame["distance_error"] = step_frame["D_after_all_updates"] - d_true
    measurement_frame["elapsed_s"] = measurement_frame["t"] - inputs.t_start
    measurement_frame["v_true"] = np.interp(
        measurement_frame["t"], step_frame["t"], step_frame["v_true"])

    summary = {
        "trip": "rf-07-26",
        "t_start": inputs.t_start,
        "steps": int(len(step_frame)),
        "whole_outage": {
            kind: _summary(measurement_frame[
                measurement_frame["measurement_type"] == kind])
            for kind in ("lateral", "spectral", "zupt")
        },
        "intervals": [_interval_summary(step_frame, measurement_frame, a, b)
                      for a, b in WINDOWS],
        "source_removal_counterfactuals": {
            name: _counterfactual(inputs, cfg, trip, **flags)
            for name, flags in {
                "baseline": {"lateral": True, "spectral": True, "zupt": True},
                "no_lateral": {"lateral": False, "spectral": True, "zupt": True},
                "no_spectral": {"lateral": True, "spectral": False, "zupt": True},
                "no_zupt": {"lateral": True, "spectral": True, "zupt": False},
                "prediction_only": {"lateral": False, "spectral": False,
                                    "zupt": False},
            }.items()
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    step_frame.to_csv(OUT / "step_attribution_rf-07-26.csv", index=False)
    measurement_frame.to_csv(OUT / "measurement_updates_rf-07-26.csv", index=False)
    (OUT / "measurement_authority_summary_rf-07-26.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
