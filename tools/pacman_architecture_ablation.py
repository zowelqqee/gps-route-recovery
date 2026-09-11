#!/usr/bin/env python3
"""Beam diagnostics and single-active-path architecture ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geotrace.pacman_tracker.benchmark import run_benchmark
from geotrace.pacman_tracker.config import PacmanConfig
from geotrace.road_graph import load_graph


RUNS = (
    ("beam-production", "beam", "none", True),
    ("beam-oracle-distance", "beam", "distance", True),
    ("single-path-production", "single_path", "none", False),
    ("single-path-oracle-distance", "single_path", "distance", False),
    ("single-path-rollback-production", "single_path_rollback", "none", False),
    ("single-path-rollback-oracle-distance", "single_path_rollback", "distance", False),
)


def slim(name: str, report: dict) -> dict:
    stats = report["tracker_stats"]
    metrics = report["metrics"]
    return {
        "run": name,
        "runtime_s": stats["runtime_s"],
        "peak_population": stats["peak_population"],
        "final_population": stats["final_population"],
        "truth_survival": metrics["ground_truth_edge_survival_rate"],
        "top1": metrics["survival_top1"],
        "distance_ratio": report["distance_calibration"].get("distance_ratio"),
        "accepted_intervals": stats["drift"]["applied"],
        "single_path": stats.get("single_path"),
        "single_path_truth": report.get("single_path_evaluation"),
        "turn_event_count": len(report.get("turn_diagnostics", [])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trip", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    graph = load_graph(args.graph)
    rows = []
    for name, mode, oracle, instrument in RUNS:
        cfg = PacmanConfig(tracker_mode=mode)
        report = run_benchmark(
            args.trip, args.graph, cfg, output=args.output / name,
            graph_cache=graph, oracle=oracle, verbose=False,
            turn_diagnostics=instrument)
        rows.append(slim(name, report))
        print(json.dumps(rows[-1], sort_keys=True), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
