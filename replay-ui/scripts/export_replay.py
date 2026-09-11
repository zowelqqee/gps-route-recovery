#!/usr/bin/env python3
"""Normalize existing Pacman benchmark artifacts for the replay UI.

This is deliberately a read-only diagnostic/export layer.  It consumes the
already-produced tracker result plus the withheld reference stream; it never
imports or executes the tracker.
"""

from __future__ import annotations

import argparse
import bisect
import json
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "runs/pacman-display/2026-07-26"
DEFAULT_TRIP = ROOT / "runs/review-final/2026-07-26/trip"
DEFAULT_GRAPH = ROOT / "runs/review-map.graphml"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "public/data/replay-07-26.json"


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_reference(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("type") == "location" and row.get("latitude") is not None:
                rows.append(row)
    return rows


def lerp(times: list[float], values: list[float], t: float) -> float:
    if t <= times[0]:
        return values[0]
    if t >= times[-1]:
        return values[-1]
    right = bisect.bisect_right(times, t)
    left = right - 1
    span = times[right] - times[left]
    mix = 0.0 if span <= 0 else (t - times[left]) / span
    return values[left] + (values[right] - values[left]) * mix


def build(run_dir: Path, trip_dir: Path, graph_path: Path) -> dict[str, Any]:
    report = load_json(run_dir / "report.json")
    tracker_frames = load_json(run_dir / "frames.json")
    truth_trace = load_json(run_dir / "ground_truth_trace.json")
    track = load_json(run_dir / "track.geojson")
    metadata = load_json(trip_dir / "metadata.json")
    trip_date = trip_dir.parent.name
    trip_label = trip_date[5:] if trip_date.startswith("2026-") else trip_date
    reference = load_reference(trip_dir / "reference-samples.jsonl")

    features = {item["properties"]["name"]: item["geometry"]["coordinates"] for item in track["features"]}
    tracker_coords = features["pacman_top1"]
    if len(tracker_frames) != len(tracker_coords):
        raise ValueError("frames.json and pacman_top1 coordinates are not aligned")

    # Phase 32 zero-latency corrected DISPLAY position, if the run produced it.
    # It is a read-only marker overlay: the committed route, junctions and
    # speed_trace are byte-identical to the flag-off baseline. When present the
    # displayed Pacman point is this corrected position; the conservative route
    # position is kept alongside under ``route``.
    pos_path = run_dir / "position_trace.json"
    position_trace = load_json(pos_path) if pos_path.exists() else None
    pos_by_t = {round(float(p["t"]), 1): p for p in position_trace} if position_trace else {}

    ref_t = [float(row["monotonic_time"]) for row in reference]
    ref_lon = [float(row["longitude"]) for row in reference]
    ref_lat = [float(row["latitude"]) for row in reference]
    ref_v = [max(0.0, float(row.get("speed") or 0.0)) for row in reference]
    dense_dt = statistics.median(b - a for a, b in zip(ref_t, ref_t[1:]))
    ref_d: list[float] = []
    running = 0.0
    for speed in ref_v:
        running += speed * dense_dt
        ref_d.append(running)

    gt_t = [float(row["t"]) for row in truth_trace]
    start_t = float(tracker_frames[0]["t"])
    start_d_true = lerp(ref_t, ref_d, start_t)
    started_at = datetime.fromisoformat(metadata["started_at"].replace("Z", "+00:00"))
    single_path_evaluation = report.get("single_path_evaluation") or {}
    first_wrong_junction = single_path_evaluation.get("first_wrong_junction")
    first_wrong_t = (
        float(first_wrong_junction["t_cross"])
        if first_wrong_junction and first_wrong_junction.get("t_cross") is not None
        else None
    )
    normalized: list[dict[str, Any]] = []

    for index, (frame, tracker_coord) in enumerate(zip(tracker_frames, tracker_coords)):
        t = float(frame["t"])
        gt_index = max(0, min(len(gt_t) - 1, bisect.bisect_right(gt_t, t) - 1))
        gt = truth_trace[gt_index]
        top = frame["top"][0] if frame.get("top") else {}
        d_route = float(frame["speed"]["distance_m"])
        d_true = lerp(ref_t, ref_d, t) - start_d_true + float(tracker_frames[0]["speed"]["distance_m"])
        route_edge = top.get("edge")
        truth_edge = gt.get("ground_truth_edge")

        pos = pos_by_t.get(round(t, 1))
        route_actor = {
            "lon": float(tracker_coord[0]),
            "lat": float(tracker_coord[1]),
            "edgeId": route_edge,
            "s": top.get("distance_along_edge_m"),
            "hypothesisId": top.get("hypothesis_id"),
        }
        if pos is not None:
            d_disp = float(pos["distance_m"])
            tracker_actor = {
                "lon": round(float(pos["lon"]), 7),
                "lat": round(float(pos["lat"]), 7),
                "edgeId": int(pos["edge"]),
                "s": round(float(pos["s_m"]), 2),
                "hypothesisId": top.get("hypothesis_id"),
            }
            v_est = round(float(pos["v_position_ms"]), 3)
            d_est = d_disp
        else:
            tracker_actor = route_actor
            v_est = round(float(frame["speed"]["speed_ms"]), 3)
            d_est = d_route

        entry = {
            "t": round(t, 3),
            "elapsed": round(t - start_t, 3),
            "timestamp": (started_at + timedelta(seconds=t)).isoformat().replace("+00:00", "Z"),
            "tracker": tracker_actor,
            "route": route_actor,
            "truth": {
                "lon": round(lerp(ref_t, ref_lon, t), 7),
                "lat": round(lerp(ref_t, ref_lat, t), 7),
                "edgeId": truth_edge,
            },
            "vEst": v_est,
            "vRoute": round(float(frame["speed"]["speed_ms"]), 3),
            "vTrue": round(lerp(ref_t, ref_v, t), 3),
            "dEst": round(d_est, 2),
            "dRoute": round(d_route, 2),
            "dTrue": round(d_true, 2),
            "alongError": round(d_est - d_true, 2),
            "routeAlongError": round(d_route - d_true, 2),
            "gpsAvailable": False,
            "edgeMatch": tracker_actor["edgeId"] == truth_edge,
            "topologyMatch": (
                t < first_wrong_t
                if first_wrong_t is not None
                else bool(single_path_evaluation)
                if single_path_evaluation
                else route_edge == truth_edge
            ),
        }
        if pos is not None:
            entry["display"] = {
                "estimator": pos.get("source", "display:iso-binary"),
                "gateActive": bool(pos["gate_active"]),
                "deltaM": round(float(pos["delta_m"]), 2),
                "excessM": round(float(pos["excess_position_distance_m"]), 2),
                "atFrontier": bool(pos["at_frontier"]),
                "displayError": round(d_est - d_true, 2),
            }
        normalized.append(entry)

    all_coords = [
        (frame[actor]["lon"], frame[actor]["lat"])
        for frame in normalized
        for actor in ("tracker", "route", "truth")
    ]
    bounds = [
        [min(x for x, _ in all_coords), min(y for _, y in all_coords)],
        [max(x for x, _ in all_coords), max(y for _, y in all_coords)],
    ]

    display_summary: dict[str, Any] | None = None
    if position_trace is not None:
        run_config = load_json(run_dir / "config.json")
        display_config = run_config.get("display", {})
        gain = float(display_config.get("correction_gain", 1.0))
        cap = float(display_config.get("max_correction_m", float("inf")))
        cap_label = f"{cap:g} m" if cap < float("inf") else "unbounded"
        disp_err = sorted(abs(f["display"]["displayError"]) for f in normalized if "display" in f)
        route_err = sorted(abs(f["routeAlongError"]) for f in normalized)
        closer = sum(
            1 for f in normalized
            if "display" in f and abs(f["display"]["displayError"]) < abs(f["routeAlongError"])
        )
        n = len(disp_err)
        display_summary = {
            "estimator": f"iso-binary ×{gain:g}, cap {cap_label}",
            "note": "Bounded corrected DISPLAY position only. Committed route, "
                    "junctions and speed_trace are byte-identical to the baseline.",
            "correctionGain": gain,
            "maxCorrectionM": cap if cap < float("inf") else None,
            "leave0726Out": bool(display_config.get("leave_0726_out", False)),
            "gateActiveFraction": round(
                sum(1 for f in normalized if f.get("display", {}).get("gateActive")) / max(n, 1), 3),
            "maxAbsDeltaM": round(max((abs(f["display"]["deltaM"]) for f in normalized
                                       if "display" in f), default=0.0), 1),
            "displayError": {
                "medianM": round(disp_err[n // 2], 1) if n else None,
                "p95M": round(disp_err[int(n * 0.95)], 1) if n else None,
                "maxM": round(disp_err[-1], 1) if n else None,
            },
            "routeError": {
                "medianM": round(route_err[len(route_err) // 2], 1),
                "p95M": round(route_err[int(len(route_err) * 0.95)], 1),
                "maxM": round(route_err[-1], 1),
            },
            "closerThanRouteFraction": round(closer / max(n, 1), 3),
        }
    return {
        "schemaVersion": 2,
        "meta": {
            "tripId": report["trip_id"],
            "tripLabel": trip_label,
            "mode": "replay",
            "trackerMode": f"{report.get('config', {}).get('tracker_mode', 'unknown')} / production / non-oracle",
            "outageStartT": start_t,
            "outageDuration": normalized[-1]["elapsed"],
            "source": {
                "tracker": str(run_dir.relative_to(ROOT)),
                "truthPositions": str((trip_dir / "reference-samples.jsonl").relative_to(ROOT)),
                "truthEdges": str((run_dir / "ground_truth_trace.json").relative_to(ROOT)),
                "roadNetwork": str(graph_path.relative_to(ROOT)),
                "basemap": "OpenStreetMap raster tiles (openstreetmap.de)",
            },
            "report": {
                "meanPositionErrorM": report["metrics"]["position_error_m"]["mean"],
                "maxPositionErrorM": report["metrics"]["position_error_m"]["max"],
                "finalAlongErrorM": report["distance_calibration"]["final_error_m"],
                "decisionCount": single_path_evaluation.get("decision_count"),
                "realWrongDecisionCount": single_path_evaluation.get("real_wrong_decision_count"),
                "twinEdgeArtifactCount": single_path_evaluation.get("twin_edge_artifact_count"),
            },
            "display": display_summary,
        },
        "bounds": bounds,
        "frames": normalized,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--trip", type=Path, default=DEFAULT_TRIP)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = build(args.run.resolve(), args.trip.resolve(), args.graph.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {len(payload['frames'])} replay frames to {args.output}")


if __name__ == "__main__":
    main()
