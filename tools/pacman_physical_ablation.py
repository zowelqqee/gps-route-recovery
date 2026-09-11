#!/usr/bin/env python3
"""Reproducible GPS-free physical-estimator ablations for the review trips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geotrace.pacman_tracker.benchmark import run_benchmark
from geotrace.pacman_tracker.config import PacmanConfig
from geotrace.road_graph import load_graph


VARIANTS = (
    "base", "5", "6", "7", "8", "5+7", "5+6+7", "5+6+7+8",
    "best-no-spectral", "best-with-spectral",
)


def config_for(name: str) -> PacmanConfig:
    cfg = PacmanConfig()
    if name in {"5", "5+7", "5+6+7", "5+6+7+8", "best-no-spectral",
                "best-with-spectral"}:
        cfg.intervals.enabled = True
        cfg.intervals.kind_turn = True
        cfg.intervals.kind_stop = name in {"5+6+7", "5+6+7+8",
                                           "best-no-spectral", "best-with-spectral"}
    elif name == "6":
        cfg.intervals.enabled = True
        cfg.intervals.kind_turn = False
        cfg.intervals.kind_stop = True
    if name in {"7", "5+7", "5+6+7", "5+6+7+8", "best-no-spectral",
                "best-with-spectral"}:
        cfg.speed.accel_scale_enabled = True
    if name in {"8", "5+6+7+8", "best-no-spectral", "best-with-spectral"}:
        cfg.speed.fixed_lag_s = 30.0
    if name == "best-no-spectral":
        cfg.speed.spectral_enabled = False
    return cfg


def slim(name: str, report: dict) -> dict:
    speed = report["speed_metrics"]
    distance = report["distance_calibration"]
    route = report["metrics"]
    drift = report["tracker_stats"]["drift"]
    return {
        "variant": name,
        "trip": Path(report["trip"]).parent.name,
        "distance_ratio": distance.get("distance_ratio"),
        "final_distance_error_m": distance.get("final_error_m"),
        "max_distance_error_m": distance.get("max_abs_error_m"),
        "distance_1sigma_coverage": distance.get("within_1_sigma"),
        "distance_2sigma_coverage": distance.get("within_2_sigma"),
        "speed_bias_ms": speed.get("bias_ms"),
        "speed_mae_ms": speed.get("mae_ms"),
        "speed_rmse_ms": speed.get("rmse_ms"),
        "distance_error_10s_m": speed.get("distance_error_10s_m"),
        "distance_error_30s_m": speed.get("distance_error_30s_m"),
        "distance_error_60s_m": speed.get("distance_error_60s_m"),
        "max_distance_error_30s_m": speed.get("max_distance_error_30s_m"),
        "max_distance_error_60s_m": speed.get("max_distance_error_60s_m"),
        "speed_1sigma_coverage": speed.get("speed_within_1_sigma"),
        "speed_2sigma_coverage": speed.get("speed_within_2_sigma"),
        "truth_survival": route.get("ground_truth_edge_survival_rate"),
        "top1": route.get("survival_top1"),
        "top3": route.get("survival_top3"),
        "top5": route.get("survival_top5"),
        "first_truth_loss_t": report["death_report"].get("first_ground_truth_loss_t"),
        "permanent_truth_loss_t": report["death_report"].get(
            "permanent_ground_truth_loss_t"),
        "interval_observations": drift.get("observations"),
        "interval_applied": drift.get("applied"),
        "interval_rejections": drift.get("reject_reasons"),
        "accel_scale": report["tracker_stats"]["accel_scale"],
        "runtime_s": report["tracker_stats"].get("runtime_s"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trip", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    graph = load_graph(args.graph)
    rows = []
    for name in args.variant:
        report = run_benchmark(
            args.trip, args.graph, config_for(name), graph_cache=graph,
            output=args.output / name, verbose=False)
        rows.append(slim(name, report))
        print(json.dumps(rows[-1], sort_keys=True), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
