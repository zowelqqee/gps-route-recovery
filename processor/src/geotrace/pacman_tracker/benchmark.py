"""Benchmark the Pacman tracker against a trip with withheld GPS.

    processor/.venv/bin/python -m geotrace.pacman_tracker.benchmark \
        --trip runs/review-20min/2026-07-22/trip \
        --graph runs/review-map.graphml \
        --output runs/pacman/2026-07-22

The order of business is fixed by the brief: the first number that matters is
``ground_truth_edge_survival_rate``, then top-1/3/5 survival and where the truth
was first lost, and only then position error. A run that reports a small mean
error while the true road died at minute three has not solved anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from geotrace.coordinates import LocalFrame
from geotrace.loader import load_trip
from geotrace.pacman_tracker.config import PacmanConfig
from geotrace.pacman_tracker.beam_diagnostics import BeamTurnObserver, CompositeObserver
from geotrace.pacman_tracker.diagnostics import (
    GroundTruthObserver,
    distance_calibration,
    map_match_reference,
    speed_metrics,
    survival_metrics,
)
from geotrace.pacman_tracker.roadmap import RoadGeometry
from geotrace.pacman_tracker.tracker import PacmanTracker, TrackerResult, build_inputs
from geotrace.pacman_tracker.turns import detect_turns
from geotrace.road_graph import RoadNetwork, clip_graph, load_graph


def clip_radius_for(outage_s: float, nominal_speed_ms: float = 15.0,
                    floor_m: float = 8000.0, cap_m: float = 25000.0) -> float:
    """How much map to load, from the outage duration alone.

    Deliberately *not* from the extent of the reference track: that would size
    the tracker's world using the answer. Duration times a nominal urban speed
    is information the tracker legitimately has.
    """
    return float(min(max(3000.0 + outage_s * nominal_speed_ms, floor_m), cap_m))


ORACLES = ("none", "distance", "speed", "no-spectral")
"""Benchmark modes.

``distance`` and ``speed`` feed the tracker the *withheld* GPS on purpose, to
separate "the route manager is wrong" from "the speed is wrong". They are
diagnostic ceilings, never a result, and the plain run cannot reach them: the
oracle series are attached to the inputs here, and :func:`build_inputs` has no
code path that produces one.
"""


def _oracle_series(trip, inputs, mode: str) -> None:
    """Attach true speed / distance to the inputs. Benchmark only."""
    reference = [f for f in trip.reference_locations if f.is_usable and f.has_valid_speed]
    if not reference or mode not in ("distance", "speed"):
        return
    rt = np.array([f.monotonic_time for f in reference])
    rv = np.array([float(f.speed) for f in reference])
    steps = [s for s in inputs.samples if s.t > inputs.t_start]
    times = np.array([s.t for s in steps])
    dts = np.array([s.dt for s in steps])
    speed = np.interp(times, rt, rv)
    inputs.oracle_speed = speed
    if mode == "distance":
        inputs.oracle_distance = np.cumsum(speed * dts)


def run_benchmark(
    trip_dir: Path,
    graph_path: Path,
    cfg: Optional[PacmanConfig] = None,
    clip_radius_m: Optional[float] = None,
    output: Optional[Path] = None,
    graph_cache: Optional[Any] = None,
    verbose: bool = True,
    oracle: str = "none",
    turn_diagnostics: bool = False,
) -> dict[str, Any]:
    cfg = cfg or PacmanConfig()
    if oracle == "no-spectral":
        cfg.speed.spectral_enabled = False
    timings: dict[str, float] = {}

    mark = time.perf_counter()
    trip, load_report = load_trip(trip_dir)
    timings["load_trip_s"] = round(time.perf_counter() - mark, 2)

    visible = trip.usable_locations
    if not visible:
        raise SystemExit(f"{trip_dir}: no usable GPS at all")
    first = visible[0]

    if clip_radius_m is None:
        outage_s = trip.t0 + trip.duration_s - visible[-1].monotonic_time
        clip_radius_m = clip_radius_for(outage_s)

    mark = time.perf_counter()
    graph = graph_cache if graph_cache is not None else load_graph(graph_path)
    network = RoadNetwork(
        clip_graph(graph, first.latitude, first.longitude, clip_radius_m),
        LocalFrame(first.latitude, first.longitude),
    )
    timings["build_network_s"] = round(time.perf_counter() - mark, 2)

    mark = time.perf_counter()
    geometry = RoadGeometry(network, cfg.geometry)
    timings["build_geometry_s"] = round(time.perf_counter() - mark, 2)

    inputs = build_inputs(trip, network, cfg)
    _oracle_series(trip, inputs, oracle)

    mark = time.perf_counter()
    truth = map_match_reference(trip.reference_locations, network, network.frame)
    timings["map_match_reference_s"] = round(time.perf_counter() - mark, 2)

    observer = GroundTruthObserver(truth)
    turn_observer = None
    run_observer = observer
    if turn_diagnostics:
        steps = [s for s in inputs.samples if s.t > inputs.t_start]
        events = detect_turns(
            np.array([s.t for s in steps]), np.array([s.yaw_rate for s in steps]),
            inputs.gyro_bias0)
        turn_observer = BeamTurnObserver(events, truth)
        run_observer = CompositeObserver(observer, turn_observer)
    tracker = PacmanTracker(network, cfg, geometry=geometry)
    result = tracker.run(inputs, observer=run_observer)

    metrics = survival_metrics(result, truth, network)
    report: dict[str, Any] = {
        "trip": str(trip_dir),
        "oracle": oracle,
        "trip_id": trip.metadata.trip_id,
        "graph": str(graph_path),
        "graph_sha256": _digest(graph_path) if graph_path.exists() else None,
        "source_sha256": _source_digest(),
        "clip_radius_m": clip_radius_m,
        "network": {"edges": len(network.edges), "nodes": network.graph.number_of_nodes()},
        "outage": {
            "gps_visible_until_s": inputs.notes["gps_visible_until_s"],
            "outage_duration_s": round(trip.t0 + trip.duration_s - inputs.t_start, 1),
            "withheld_fixes": len(trip.reference_locations),
        },
        "inputs": inputs.notes,
        "ground_truth": {
            "matched_fraction": round(truth.matched_fraction, 4),
            **truth.notes,
            "distinct_edges": int(len(np.unique(truth.edges[truth.edges >= 0]))),
        },
        "metrics": metrics,
        "speed_metrics": speed_metrics(result, trip),
        "distance_calibration": distance_calibration(result, trip),
        "death_report": observer.death_report(),
        "tracker_stats": result.stats,
        "timings_s": timings,
        "loader": load_report.to_json(),
        "config": cfg.to_json(),
    }
    if turn_observer is not None:
        report["turn_diagnostics"] = turn_observer.to_json()
    if cfg.tracker_mode != "beam":
        report["single_path_evaluation"] = _single_path_evaluation(
            result, truth, trip, inputs.t_start, network)

    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (output / "frames.json").write_text(
            json.dumps([f.to_json() for f in result.frames], indent=1), encoding="utf-8"
        )
        (output / "corridors.geojson").write_text(
            json.dumps(_corridor_geojson(result, network.frame)), encoding="utf-8"
        )
        (output / "track.geojson").write_text(
            json.dumps(_track_geojson(result, truth, network.frame)), encoding="utf-8"
        )
        (output / "ground_truth_trace.json").write_text(
            json.dumps([r.to_json() for r in observer.trace], indent=1), encoding="utf-8"
        )
        (output / "speed_trace.json").write_text(
            json.dumps([s.to_json() for s in result.speed_trace], indent=1),
            encoding="utf-8")
        if result.position_trace is not None:
            (output / "position_trace.json").write_text(
                json.dumps([s.to_json() for s in result.position_trace], indent=1),
                encoding="utf-8")
        cfg.dump(output / "config.json")
    if verbose:
        print_summary(report)
    return report


def _corridor_geojson(result, frame: LocalFrame) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    for tick in result.frames[::5]:
        collection = tick.corridors.to_geojson(frame)
        for feature in collection["features"]:
            feature["properties"]["t"] = round(tick.t, 1)
            features.append(feature)
    return {"type": "FeatureCollection", "features": features}


def _track_geojson(result, truth, frame: LocalFrame) -> dict[str, Any]:
    best = np.array([f.position for f in result.frames], dtype=float)
    features = [
        {
            "type": "Feature",
            "properties": {"name": "pacman_top1"},
            "geometry": {"type": "LineString",
                         "coordinates": frame.coords_to_geojson(best)},
        }
    ]
    if len(truth.xy):
        features.append(
            {
                "type": "Feature",
                "properties": {"name": "withheld_gps_reference"},
                "geometry": {"type": "LineString",
                             "coordinates": frame.coords_to_geojson(truth.xy)},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def print_summary(report: dict[str, Any]) -> None:
    m = report["metrics"]
    d = report["death_report"]
    s = report["tracker_stats"]
    print(f"\n=== {report['trip']} ===")
    print(f"  outage {report['outage']['outage_duration_s']:.0f} s after "
          f"{report['outage']['gps_visible_until_s']:.1f} s of GPS; "
          f"{report['network']['edges']} edges in the clip")
    print(f"  ground truth matched {report['ground_truth']['matched_fraction']:.2%} of ticks, "
          f"median snap {report['ground_truth'].get('median_snap_m')} m, "
          f"{report['ground_truth']['distinct_edges']} distinct edges")
    print(f"  survival  any={_pct(m['ground_truth_edge_survival_rate'])}  "
          f"top1={_pct(m['survival_top1'])}  top3={_pct(m['survival_top3'])}  "
          f"top5={_pct(m['survival_top5'])}")
    print(f"  first ground-truth loss: {d['first_ground_truth_loss_t']} s   "
          f"permanent: {d['permanent_ground_truth_loss_t']} s")
    print(f"  position error m: {m['position_error_m']}")
    print(f"  corridor coverage={_pct(m['corridor_coverage'])} "
          f"median area={m['corridor_median_area_m2']} m^2  {m['confidence_histogram']}")
    sm = report.get("speed_metrics", {})
    dc = report.get("distance_calibration", {})
    print(f"  speed: bias={sm.get('bias_ms')} MAE={sm.get('mae_ms')} "
          f"corr={sm.get('correlation')} | D/D_true={dc.get('distance_ratio')} "
          f"30s err={sm.get('distance_error_30s_m')} 60s err={sm.get('distance_error_60s_m')}")
    print(f"  distance calibration: P(|D-D_true| <= 2 sigma) = "
          f"{_pct(dc.get('within_2_sigma'))} (target 95%)")
    print(f"  runtime {s['runtime_s']} s, population peak/final "
          f"{s.get('peak_population', 'n/a')}/{s['final_population']}, "
          f"branches {s['branches']}, merged {s['merged']}, "
          f"pruned {s['pruned_by_weight']}+{s['pruned_by_beam']}, "
          f"turns scored {s['turns_scored']}")


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_digest() -> str:
    h = hashlib.sha256()
    root = Path(__file__).parent
    for path in sorted(root.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def _truth_traversal(truth) -> list[tuple[int, float, float]]:
    """The ordered list of distinct edges the withheld GPS actually drove.

    ``(edge, t_first, t_last)`` per run of a matched edge. This is the physical
    route in travel order; the true successor of any edge is simply the next
    entry, with no reference to the tracker's (possibly lagged) crossing time.
    """
    runs: list[list[float]] = []
    for t, e in zip(truth.times, truth.edges):
        e = int(e)
        if e < 0:
            continue
        if runs and runs[-1][0] == e:
            runs[-1][2] = float(t)
        else:
            runs.append([e, float(t), float(t)])
    return [(int(a), b, c) for a, b, c in runs]


def _divergence_cause(row: dict[str, Any], earlier_diverged: bool,
                      is_first_decision: bool) -> str:
    """Why did this local decision not match the physical route?"""
    truth_edge = row["truth_edge"]
    if truth_edge is None:
        if is_first_decision:
            return "seed/init: the incoming edge is not the true start edge"
        return "active route already diverged at an earlier junction"

    event = row.get("real_turn_event")
    d_err = row.get("distance_error_at_turn_m")
    timing_gap = (abs(float(row["t_cross"]) - float(event["t_peak"]))
                  if event else None)
    misaligned = (timing_gap is not None and timing_gap > 8.0
                  and d_err is not None and abs(d_err) > 40.0)

    big_lag = d_err is not None and abs(d_err) > 60.0
    if not row["truth_was_candidate"]:
        if misaligned or big_lag or not row["truth_junction_in_crossing_tolerance"]:
            gap = f"{timing_gap:.0f} s" if timing_gap is not None else "some way"
            err = f"{d_err:+.0f} m" if d_err is not None else "the odometer error"
            return ("junction timing mismatch: the odometer error "
                    f"({err}) put the decision {gap} "
                    "after the real turn, so the true successor was never "
                    "generated at this node")
        return ("successor not generated: graph topology, turn restriction, or "
                "branching did not offer the true edge at this node")

    if misaligned:
        return ("turn/junction temporal misalignment: the true edge was a "
                f"candidate but the odometer lag ({d_err:+.0f} m) made the "
                f"decision {timing_gap:.0f} s late, so the +/-"
                f"{abs(row.get('truth_edge_map_turn_deg') or 0):.0f} deg gyro "
                "turn fell outside the scoring window and the straight "
                "successor out-scored it")
    return ("true edge present among candidates but out-scored on the turn "
            f"angle: measured d_psi={row['measured_turn_deg']:+.1f} deg vs "
            f"true-edge map turn "
            f"{row.get('truth_edge_map_turn_deg', float('nan')):+.1f} deg")


def _twin_edge_artifact(network, incoming: int, chosen: int, truth_edge: int,
                        chosen_is_successor: bool) -> Optional[str]:
    """Did the map-matched reference snap to a disconnected parallel edge?

    OSM splits a carriageway into segments at points the drivable graph does
    not join. When the HMM reference picks one representation of a stretch and
    the graph continues via its twin, the committed route is not wrong - the
    label is. The signature: the true edge is *not* a graph successor of the
    incoming edge, the edge the tracker took *is*, and the true edge starts a
    few metres from the incoming edge's end node on a same-named street.
    """
    if truth_edge is None or truth_edge in network.successors(incoming, allow_uturn=True):
        return None
    if not chosen_is_successor:
        return None
    ea, et = network.edges[incoming], network.edges[truth_edge]
    a_end = np.asarray(ea.position(ea.length), dtype=float)
    gap = float(np.linalg.norm(a_end - np.asarray(et.position(0.0), dtype=float)))
    same_street = bool(ea.name) and ea.name == et.name
    if gap <= 25.0 and (same_street or gap <= 12.0):
        return (f"ground-truth map-match snapped to a topologically "
                f"disconnected twin edge ({et.name!r}, {gap:.0f} m from the "
                f"real successor node); the committed route is not diverging here")
    return None


def _single_path_evaluation(result: TrackerResult, truth, trip,
                            t_start: float, network=None) -> dict[str, Any]:
    """Attach hidden-truth labels to already-made local decisions.

    This runs only in the benchmark after tracking has finished. Nothing here
    is reachable by branch selection.

    The true successor of a decision's incoming edge is taken from the physical
    route traversal (:func:`_truth_traversal`) walked in lockstep with the
    decisions, not from the map-matched GPS edge at the moment the tracker got
    round to making the decision - which, when the odometer lags, is already
    one or two junctions further on.
    """
    decisions = result.stats.get("single_path", {}).get("decisions", [])
    reference = [fix for fix in trip.reference_locations
                 if fix.is_usable and fix.has_valid_speed]
    if reference:
        rt = np.array([fix.monotonic_time for fix in reference])
        rv = np.array([float(fix.speed) for fix in reference])
        dt = float(np.median(np.diff(rt))) if len(rt) > 1 else 0.1
        cumulative = np.cumsum(rv * dt)
        d0 = float(np.interp(t_start, rt, cumulative))
    else:
        rt = cumulative = np.zeros(0)
        d0 = 0.0

    traversal = _truth_traversal(truth)
    seq = [e for e, _, _ in traversal]
    turn_events = detect_turns(
        np.array([s.t for s in result.inputs.samples]),
        np.array([s.yaw_rate for s in result.inputs.samples]),
        result.inputs.gyro_bias0)

    evaluated = []
    pointer = 0  # position already consumed in the physical traversal
    earlier_diverged = False
    for di, decision in enumerate(decisions):
        label_t = float(decision["t_decision"])
        candidates = [a["edge"] for a in decision["alternatives"]]
        incoming = int(decision["incoming_edge"])

        # Walk the physical route forward from where the last decision left it.
        truth_edge = None
        truth_incoming_left_t = None
        found_at = None
        for q in range(pointer, len(seq)):
            if seq[q] == incoming:
                found_at = q
                truth_incoming_left_t = traversal[q][2]
                if q + 1 < len(seq):
                    truth_edge = seq[q + 1]
                break
        if found_at is not None:
            pointer = found_at + 1
        else:
            earlier_diverged = True

        estimated_d = float(np.interp(
            decision["t_cross"],
            [sample.t for sample in result.speed_trace],
            [sample.distance_m for sample in result.speed_trace]))
        true_d = (float(np.interp(decision["t_cross"], rt, cumulative)) - d0
                  if len(rt) else float("nan"))

        # The gyro turn event that actually carried the car off `incoming`.
        real_turn_event = None
        if truth_incoming_left_t is not None and turn_events:
            near = min(turn_events,
                       key=lambda e: abs(e.t_peak - truth_incoming_left_t))
            if abs(near.t_peak - truth_incoming_left_t) <= 15.0:
                real_turn_event = near.to_json()

        row = dict(decision)
        row.update({
            "truth_edge": truth_edge,
            "truth_edge_at_decision": truth.edge_at(label_t),
            "truth_incoming_left_t": (round(truth_incoming_left_t, 2)
                                      if truth_incoming_left_t is not None else None),
            "on_truth_route": found_at is not None,
            "truth_was_candidate": truth_edge in candidates,
            "correct_choice": int(decision["chosen_edge"]) == truth_edge,
            "distance_error_at_turn_m": (round(estimated_d - true_d, 2)
                                          if math.isfinite(true_d) else None),
            "estimated_distance_at_turn_m": round(estimated_d, 2),
            "true_distance_at_turn_m": (round(true_d, 2)
                                        if math.isfinite(true_d) else None),
            "truth_junction_in_crossing_tolerance": (
                abs(float(decision["distance_to_junction_at_turn_m"]))
                <= float(decision["crossing_tolerance_m"])),
            "real_turn_event": real_turn_event,
        })
        # Turn angle and score the true edge received, if it was a candidate.
        for alt in decision["alternatives"]:
            if alt["edge"] == truth_edge:
                row["truth_edge_map_turn_deg"] = alt["map_turn_deg"]
                row["truth_edge_local_probability"] = alt["local_probability"]
                row["truth_edge_score"] = alt["score"]
                row["truth_edge_rank"] = decision["alternatives"].index(alt)
                break
        twin = None
        if not row["correct_choice"] and network is not None:
            chosen_is_succ = int(decision["chosen_edge"]) in candidates
            twin = _twin_edge_artifact(network, incoming,
                                       int(decision["chosen_edge"]), truth_edge,
                                       chosen_is_succ)
        cascade = (twin is None and not row["correct_choice"]
                   and truth_edge is None and evaluated
                   and evaluated[-1].get("map_match_twin_artifact")
                   and incoming == int(evaluated[-1]["chosen_edge"]))
        row["map_match_twin_artifact"] = twin is not None or bool(cascade)
        if twin is not None:
            pointer = found_at + 2  # step over the incoming edge and its twin
        row["divergence_cause"] = (
            None if row["correct_choice"]
            else (twin
                  or ("ground-truth map-match twin-edge cascade from the "
                      "previous decision" if cascade else None)
                  or _divergence_cause(row, earlier_diverged, di == 0)))
        evaluated.append(row)
        if not row["correct_choice"] and found_at is not None and not row["map_match_twin_artifact"]:
            earlier_diverged = True

    wrong = [row for row in evaluated if not row["correct_choice"]]
    real_wrong = [row for row in evaluated if not row["correct_choice"]
                  and not row["map_match_twin_artifact"]]
    on_route_wrong = [row for row in real_wrong if row["on_truth_route"]]
    return {
        "decision_count": len(evaluated),
        "wrong_decision_count": len(wrong),
        "real_wrong_decision_count": len(real_wrong),
        "twin_edge_artifact_count": len(wrong) - len(real_wrong),
        "first_wrong_junction": (on_route_wrong[0] if on_route_wrong
                                 else (real_wrong[0] if real_wrong else None)),
        "physical_route_edges": seq,
        "decisions": evaluated,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trip", type=Path, action="append", required=True,
                        help="trip directory; repeat for several")
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--clip-radius", type=float, default=None,
                        help="metres; default is derived from the outage duration")
    parser.add_argument("--max-hypotheses", type=int)
    parser.add_argument("--prune-margin", type=float)
    parser.add_argument("--oracle", choices=ORACLES, default="none",
                        help="diagnostic ceilings; 'distance'/'speed' use the "
                             "withheld GPS deliberately and are never a result")
    parser.add_argument("--tracker-mode",
                        choices=("beam", "single_path", "single_path_rollback"),
                        default="beam")
    parser.add_argument("--turn-diagnostics", action="store_true")
    parser.add_argument("--display-position", action="store_true",
                        help="also run the zero-latency corrected display "
                             "position branch (Phase 32 iso-binary); writes "
                             "position_trace.json. Read-only w.r.t. the tracker.")
    parser.add_argument("--display-leave-0726-out", action="store_true",
                        help="use the leave-07-26-out isotonic curve (for "
                             "reproducing the Phase 32 diagnostic on 07-26)")
    args = parser.parse_args(argv)

    cfg = PacmanConfig()
    cfg.tracker_mode = args.tracker_mode
    cfg.display.position_branch_enabled = args.display_position
    cfg.display.leave_0726_out = args.display_leave_0726_out
    if args.max_hypotheses:
        cfg.beam.max_hypotheses = args.max_hypotheses
    if args.prune_margin:
        cfg.beam.prune_log_margin = args.prune_margin

    graph = load_graph(args.graph)
    reports = []
    for trip_dir in args.trip:
        name = trip_dir.parent.name
        if args.oracle != "none":
            name = f"{name}-oracle-{args.oracle}"
        out = (args.output / name) if args.output else None
        reports.append(run_benchmark(trip_dir, args.graph, cfg, args.clip_radius, out,
                                     graph_cache=graph, oracle=args.oracle,
                                     turn_diagnostics=args.turn_diagnostics))
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "summary.json").write_text(
            json.dumps([_slim(r) for r in reports], indent=2), encoding="utf-8"
        )
    return 0


def _slim(report: dict[str, Any]) -> dict[str, Any]:
    return {k: report[k] for k in
            ("trip", "trip_id", "oracle", "outage", "ground_truth", "metrics",
             "speed_metrics", "distance_calibration", "death_report",
             "tracker_stats", "timings_s")
            if k in report} | {"death_report": {k: v for k, v in report["death_report"].items()
                                                if k not in ("pre_death_trace",
                                                             "prune_events_at_death")}}


if __name__ == "__main__":
    sys.exit(main())
