#!/usr/bin/env python3
"""Run and export the first honest RFID-checkpoint experiment.

The first two complete circuits calibrate the per-device spectral/mount model.
The following complete circuit is a GPS-free outage.  Its RFID points are
withheld from Pacman and consulted only after tracking has finished.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "processor" / "src"))

from geotrace.coordinates import LocalFrame
from geotrace.pacman_tracker.config import PacmanConfig
from geotrace.pacman_tracker.tracker import PacmanTracker, build_inputs
from geotrace.rfid_logs import build_rfid_trip, find_complete_laps, read_rfid_anchors
from geotrace.road_graph import RoadNetwork, clip_graph, load_graph


def route_position(route, fraction: float) -> np.ndarray:
    coords = np.asarray(route.coords, dtype=float)
    if len(coords) == 1:
        return coords[0]
    lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    wanted = float(np.clip(fraction, 0.0, 1.0)) * cumulative[-1]
    index = int(np.clip(np.searchsorted(cumulative, wanted) - 1, 0, len(lengths) - 1))
    mix = (wanted - cumulative[index]) / max(lengths[index], 1e-9)
    return coords[index] + mix * (coords[index + 1] - coords[index])


def point_json(network: RoadNetwork, xy: np.ndarray, edge: int, s=None, hypothesis=None):
    lat, lon = network.frame.to_geo(float(xy[0]), float(xy[1]))
    out = {"lon": lon, "lat": lat, "edgeId": int(edge)}
    if s is not None:
        out["s"] = round(float(s), 2)
    if hypothesis is not None:
        out["hypothesisId"] = int(hypothesis)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imu", type=Path, default=ROOT / "real_tests/imu_2026-09-08.csv")
    parser.add_argument("--anchors", type=Path, default=ROOT / "real_tests/anchor_bindings_2026-09-08.csv")
    parser.add_argument("--graph", type=Path, default=ROOT / "cache/spb.graphml")
    parser.add_argument("--lap", type=int, default=1,
                        help="complete lap index; lap 1 follows two calibration laps")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "replay-ui/public/data/replay-09-08-rfid.json")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "runs/real-tests-rfid/2026-09-08/report.json")
    parser.add_argument("--diagnostic-rollback", action="store_true",
                        help="enable non-production dead-end rollback ablation")
    args = parser.parse_args()

    anchors = read_rfid_anchors(args.anchors)
    laps = find_complete_laps(anchors)
    if not 0 <= args.lap < len(laps):
        raise SystemExit(f"--lap must be 0..{len(laps) - 1}")
    lap = laps[args.lap]
    if lap.start_anchor < 30:
        raise SystemExit("selected lap has too little earlier data for honest calibration")

    centre_lat = float(np.mean([a.latitude for a in anchors]))
    centre_lon = float(np.mean([a.longitude for a in anchors]))
    graph = clip_graph(load_graph(args.graph), centre_lat, centre_lon, 5000.0)
    network = RoadNetwork(graph, LocalFrame(centre_lat, centre_lon))
    built = build_rfid_trip(
        args.imu, args.anchors, network,
        cutoff_anchor=lap.start_anchor, end_anchor=lap.end_anchor,
    )

    cfg = PacmanConfig()
    cfg.tracker_mode = "single_path"
    cfg.display.position_branch_enabled = False
    if args.diagnostic_rollback:
        cfg.single_path.rollback_enabled = True
        cfg.single_path.rollback_require_low_confidence = False
    inputs = build_inputs(built.trip, network, cfg)
    result = PacmanTracker(network, cfg).run(inputs)

    t0_ms = built.trip.metadata.started_at.timestamp() * 1000.0
    test_anchors = built.anchors[lap.start_anchor:lap.end_anchor + 1]
    anchor_times = np.array([(a.timestamp_ms - t0_ms) / 1000.0 for a in test_anchors])
    test_routes = built.routes[lap.start_anchor:lap.end_anchor]
    if any(route is None for route in test_routes):
        raise SystemExit("one of the test RFID intervals has no legal graph route")
    lengths = np.array([route.length_m for route in test_routes], dtype=float)
    cumulative = np.r_[0.0, np.cumsum(lengths)]

    checkpoint_errors = []
    checkpoint_rows = []
    for index, (anchor, t) in enumerate(zip(test_anchors[1:], anchor_times[1:]), 1):
        frame = min(result.frames, key=lambda item: abs(item.t - t))
        truth_xy = np.asarray(network.frame.to_local(anchor.latitude, anchor.longitude))
        error = float(np.linalg.norm(np.asarray(frame.position) - truth_xy))
        checkpoint_errors.append(error)
        checkpoint_rows.append({
            "anchorId": anchor.anchor_id,
            "elapsed": round(t - inputs.t_start, 2),
            "markerErrorM": round(error, 2),
            "estimatedDistanceM": round(frame.speed.distance_m, 2),
            "referenceDistanceM": round(cumulative[index], 2),
        })

    frames = []
    for frame in result.frames:
        if frame.t < inputs.t_start or frame.t > anchor_times[-1]:
            continue
        interval = int(np.clip(np.searchsorted(anchor_times, frame.t) - 1, 0, len(test_routes) - 1))
        dt = anchor_times[interval + 1] - anchor_times[interval]
        mix = (frame.t - anchor_times[interval]) / max(dt, 1e-9)
        route = test_routes[interval]
        truth_xy = route_position(route, mix)
        truth_edge = int(route.edge_indices[min(
            len(route.edge_indices) - 1,
            max(0, int(mix * len(route.edge_indices))),
        )])
        truth_distance = cumulative[interval] + float(np.clip(mix, 0.0, 1.0)) * lengths[interval]
        truth_speed = lengths[interval] / dt
        top = frame.top[0] if frame.top else None
        tracker_edge = int(top.edge) if top is not None else -1
        reverse = network.edges[tracker_edge].reverse_index if tracker_edge >= 0 else None
        same_edge = tracker_edge == truth_edge or reverse == truth_edge
        interval_edges = set(int(edge) for edge in route.edge_indices)
        same_route_segment = tracker_edge in interval_edges or (
            reverse is not None and int(reverse) in interval_edges
        )
        tracker_xy = np.asarray(frame.position)
        lat, lon = network.frame.to_geo(float(tracker_xy[0]), float(tracker_xy[1]))
        frames.append({
            "t": round(frame.t, 2),
            "elapsed": round(frame.t - inputs.t_start, 2),
            "timestamp": datetime.fromtimestamp(
                (t0_ms / 1000.0) + frame.t, tz=timezone.utc
            ).isoformat().replace("+00:00", "Z"),
            "tracker": {
                "lon": lon, "lat": lat, "edgeId": tracker_edge,
                "s": round(float(top.s), 2) if top is not None else 0.0,
                "hypothesisId": int(top.hypothesis_id) if top is not None else -1,
            },
            "route": point_json(network, tracker_xy, tracker_edge,
                                top.s if top is not None else 0.0,
                                top.hypothesis_id if top is not None else -1),
            "truth": point_json(network, truth_xy, truth_edge),
            "vEst": round(frame.speed.speed_ms, 3),
            "vRoute": round(frame.speed.speed_ms, 3),
            "vTrue": round(truth_speed, 3),
            "dEst": round(frame.speed.distance_m, 2),
            "dRoute": round(frame.speed.distance_m, 2),
            "dTrue": round(truth_distance, 2),
            "alongError": round(frame.speed.distance_m - truth_distance, 2),
            "routeAlongError": round(frame.speed.distance_m - truth_distance, 2),
            "gpsAvailable": False,
            "edgeMatch": same_edge,
            # This is only membership in the legal road segment connecting the
            # surrounding RFID points. Decision-level truth remains unknown.
            "topologyMatch": same_route_segment,
        })

    marker_errors = np.asarray(checkpoint_errors)
    report = {
        "meanPositionErrorM": round(float(np.mean(marker_errors)), 2),
        "maxPositionErrorM": round(float(np.max(marker_errors)), 2),
        "finalAlongErrorM": round(float(frames[-1]["alongError"]), 2),
        "decisionCount": int(result.stats.get("single_path", {}).get("decision_count", 0)),
        "realWrongDecisionCount": None,
        "twinEdgeArtifactCount": 0,
        "checkpointCount": len(checkpoint_rows),
        "checkpointFinalErrorM": round(float(marker_errors[-1]), 2),
    }
    output = {
        "schemaVersion": 3,
        "meta": {
            "tripId": built.trip.metadata.trip_id,
            "tripLabel": "09-08 RFID",
            "mode": "sparse-rfid-validation",
            "trackerMode": "single_path / production / non-oracle",
            "referenceKind": "RFID checkpoints; path between them is OSM interpolation",
            "outageStartT": round(inputs.t_start, 2),
            "outageDuration": round(anchor_times[-1] - inputs.t_start, 2),
            "source": {
                "tracker": "real_tests/imu_2026-09-08.csv",
                "truthPositions": "withheld real_tests/anchor_bindings_2026-09-08.csv",
                "roadNetwork": "cache/spb.graphml",
                "basemap": "OpenStreetMap raster tiles (openstreetmap.de)",
            },
            "report": report,
            "rfid": {
                "checkpointCount": len(checkpoint_rows),
                "checkpoints": checkpoint_rows,
                "bindingResidualMedianMsec": built.report["anchor_binding_residual_ms"]["median"],
                "physicalTimestampUsed": True,
                "note": (
                    "Errors are scored only at RFID checkpoints. The cyan line between "
                    "checkpoints is a legal-road interpolation, not continuous GPS truth."
                ),
            },
            "display": None,
        },
        "bounds": [
            [min(min(p["tracker"]["lon"], p["truth"]["lon"]) for p in frames)
             if frames else centre_lon,
             min(min(p["tracker"]["lat"], p["truth"]["lat"]) for p in frames)
             if frames else centre_lat],
            [max(max(p["tracker"]["lon"], p["truth"]["lon"]) for p in frames)
             if frames else centre_lon,
             max(max(p["tracker"]["lat"], p["truth"]["lat"]) for p in frames)
             if frames else centre_lat],
        ],
        "frames": frames,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, separators=(",", ":")), encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "experiment": output["meta"],
        "import": built.report,
        "spectral": inputs.spectral.to_json(),
        "trackerStats": result.stats,
    }, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
