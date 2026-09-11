#!/usr/bin/env python3
"""Run Phase-37 guards through the real single-path tracker and score topology."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from geotrace.coordinates import LocalFrame
from geotrace.loader import load_trip
from geotrace.pacman_tracker.benchmark import _single_path_evaluation
from geotrace.pacman_tracker.config import PacmanConfig
from geotrace.pacman_tracker.diagnostics import GroundTruthObserver, map_match_reference
from geotrace.pacman_tracker.roadmap import RoadGeometry
from geotrace.pacman_tracker.tracker import PacmanTracker, build_inputs
from geotrace.road_graph import RoadNetwork, clip_graph, load_graph

from phase37_authority_guards import TRIPS, truth


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/plots/phase37/topology_gate.json"


def run(tag: str, path: Path, graph, enabled: bool) -> dict:
    trip, _ = load_trip(path)
    first = trip.usable_locations[0]
    network = RoadNetwork(
        clip_graph(graph, first.latitude, first.longitude, 11000.0),
        LocalFrame(first.latitude, first.longitude))
    cfg = PacmanConfig(tracker_mode="single_path")
    cfg.speed.zupt_motion_guard_enabled = enabled
    cfg.speed.lateral_consensus_guard_enabled = enabled
    cfg.speed.spectral_saturation_guard_enabled = enabled
    inputs = build_inputs(trip, network, cfg)
    mapped_truth = map_match_reference(trip.reference_locations, network, network.frame)
    result = PacmanTracker(network, cfg, geometry=RoadGeometry(
        network, cfg.geometry)).run(inputs, observer=GroundTruthObserver(mapped_truth))
    evaluation = _single_path_evaluation(
        result, mapped_truth, trip, inputs.t_start, network)
    t = np.asarray([s.t for s in result.speed_trace])
    d = np.asarray([s.distance_m for s in result.speed_trace])
    v = np.asarray([s.speed_ms for s in result.speed_trace])
    d_true, v_true = truth(trip, inputs.t_start, t)
    e = d - d_true
    return {
        "trip": tag,
        "guards": enabled,
        "median_abs_error_m": float(np.median(np.abs(e))),
        "p95_abs_error_m": float(np.percentile(np.abs(e), 95)),
        "max_abs_error_m": float(np.max(np.abs(e))),
        "endpoint_error_m": float(e[-1]),
        "speed_mae_ms": float(np.mean(np.abs(v - v_true))),
        "decisions": evaluation["decision_count"],
        "chosen_edges": [int(row["chosen_edge"])
                         for row in evaluation["decisions"]],
        "real_wrong_decisions": evaluation["real_wrong_decision_count"],
        "final_edges": list(result.final.routes[0].edges()),
        "counts": result.stats["speed"],
    }


def main() -> None:
    graph = load_graph(ROOT / "runs/review-map.graphml")
    output = {}
    for tag in ("rf-07-22", "rf-07-26"):
        output[tag] = {
            "baseline": run(tag, TRIPS[tag], graph, False),
            "guards": run(tag, TRIPS[tag], graph, True),
        }
        base, guarded = output[tag]["baseline"], output[tag]["guards"]
        output[tag]["topology_identical"] = (
            base["chosen_edges"] == guarded["chosen_edges"]
            and base["final_edges"] == guarded["final_edges"])
        print(json.dumps(output[tag], indent=2), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
