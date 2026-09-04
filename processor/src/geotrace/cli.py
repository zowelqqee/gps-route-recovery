"""``geotrace`` command line interface.

    geotrace download-map   --place "Saint Petersburg, Russia" --output cache/spb.graphml
    geotrace simulate       --graph cache/spb.graphml --output sample-data/trip-001
    geotrace inject-fault   --trip sample-data/trip-001 --fault dropout --start 60 --duration 45 \
                            --output runs/trip-001-broken
    geotrace reconstruct    --trip runs/trip-001-broken --graph cache/spb.graphml \
                            --algorithm road-particle-filter --particles 5000 \
                            --confidence 0.95 --seed 42
    geotrace report         --run runs/trip-001-broken --output runs/trip-001-broken/report.html
    geotrace inspect        --trip runs/trip-001-broken
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from geotrace import __version__
from geotrace.config import Config
from geotrace.coordinates import LocalFrame
from geotrace.fault_injection import (
    FAULT_KINDS,
    FaultError,
    FaultSpec,
    inject_faults,
    parse_fault_args,
    scenario_offset_dropout_recovery,
)
from geotrace.loader import TripLoadError, load_trip, results_dir, write_trip
from geotrace.pipeline import ALGORITHMS, ReconstructionError, build_metrics, run_reconstruction
from geotrace.road_graph import RoadGraphError, RoadNetwork, clip_graph, download_graph, download_graph_bbox, load_graph

ALGORITHM_ALIASES = {a.replace("_", "-"): a for a in ALGORITHMS}
ALGORITHM_ALIASES.update({a: a for a in ALGORITHMS})


class CLIError(RuntimeError):
    """A problem the user can fix; printed without a traceback."""


# ---------------------------------------------------------------- helpers


def _resolve_algorithm(name: str) -> str:
    key = name.strip().lower()
    if key not in ALGORITHM_ALIASES:
        raise CLIError(
            f"unknown --algorithm '{name}'.\nAvailable: "
            + ", ".join(sorted(set(ALGORITHM_ALIASES)))
        )
    return ALGORITHM_ALIASES[key]


def _build_config(args: argparse.Namespace) -> Config:
    cfg = Config.load(args.config) if getattr(args, "config", None) else Config()
    if getattr(args, "seed", None) is not None:
        cfg.seed = int(args.seed)
    if getattr(args, "particles", None) is not None:
        if args.particles < 1:
            raise CLIError("--particles must be at least 1")
        cfg.pf.n_particles = int(args.particles)
    if getattr(args, "rng", None):
        cfg.rng_mode = args.rng
    if getattr(args, "allow_simulated", False):
        cfg.gps.allow_simulated_fixes = True
    if getattr(args, "confidence", None) is not None:
        if not 0.0 < args.confidence < 1.0:
            raise CLIError("--confidence must be strictly between 0 and 1 (e.g. 0.95)")
        cfg.polygon.confidence = float(args.confidence)
    return cfg


def _load_network(
    graph_path: Optional[str], lat: float, lon: float, radius_m: float
) -> Optional[RoadNetwork]:
    if not graph_path:
        return None
    graph = load_graph(graph_path)
    try:
        clipped = clip_graph(graph, lat, lon, radius_m)
    except RoadGraphError as exc:
        raise CLIError(str(exc)) from exc
    return RoadNetwork(clipped, LocalFrame(lat, lon))


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


# ---------------------------------------------------------------- commands


def cmd_download_map(args: argparse.Namespace) -> int:
    output = Path(args.output)
    if output.exists() and not args.force:
        print(f"{output} already exists; nothing downloaded. Use --force to refresh.")
        return 0
    if args.bbox:
        try:
            north, south, east, west = (float(v) for v in args.bbox)
        except ValueError as exc:
            raise CLIError("--bbox takes four numbers: NORTH SOUTH EAST WEST") from exc
        print(f"Downloading OSM drive network for bbox {north},{south},{east},{west} ...")
        path = download_graph_bbox(north, south, east, west, output, args.network_type)
    else:
        if not args.place:
            raise CLIError("pass --place, or --bbox NORTH SOUTH EAST WEST")
        print(f"Downloading OSM drive network for {args.place!r} (this can take a few minutes) ...")
        path = download_graph(args.place, output, args.network_type)
    graph = load_graph(path)
    print(
        f"Cached {graph.number_of_nodes()} nodes / {graph.number_of_edges()} edges -> {path}\n"
        "This file is reused on every later run; the map is never re-downloaded."
    )
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    from geotrace.simulate import SimulationSpec, simulate_trip, spb_synthetic_network

    cfg = _build_config(args)
    if args.graph:
        graph = load_graph(args.graph)
        network = RoadNetwork(
            clip_graph(graph, args.lat, args.lon, args.radius), LocalFrame(args.lat, args.lon)
        )
        print(f"Simulating on the cached OSM graph: {len(network)} edges")
    else:
        network = spb_synthetic_network(args.lat, args.lon)
        print(
            f"No --graph given: simulating on the built-in synthetic street grid "
            f"({len(network)} edges). This is a toy network, not a map of Saint Petersburg."
        )
    spec = SimulationSpec(
        duration_s=args.duration,
        cruise_speed_ms=args.speed,
        stop_at_s=args.stop_at,
        gps_sigma_m=args.gps_noise,
    )
    trip = simulate_trip(network, spec, seed=cfg.seed, trip_id=args.trip_id)
    out = write_trip(trip, args.output)
    print(
        f"Wrote {out}\n"
        f"  {len(trip.locations)} GPS fixes, {len(trip.motions)} motion samples, "
        f"{trip.duration_s:.0f} s"
    )
    return 0


def cmd_inject_fault(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    trip, report = load_trip(args.trip)
    if args.scenario:
        faults = scenario_offset_dropout_recovery(
            start_s=args.start,
            offset_duration_s=args.duration or 25.0,
            dropout_duration_s=args.dropout_duration,
            east=args.east or 900.0,
            north=args.north or -600.0,
        )
        print("Applying the composite scenario: offset -> dropout -> false recovery")
    else:
        faults = [
            parse_fault_args(
                kind=args.fault,
                start=args.start,
                duration=args.duration,
                east=args.east,
                north=args.north,
                drift_east=args.drift_east,
                drift_north=args.drift_north,
                sigma=args.sigma,
                count=args.count,
                accuracy=args.accuracy,
            )
        ]
    origin = trip.usable_locations[0]
    frame = LocalFrame(origin.latitude, origin.longitude)
    broken, fault_report = inject_faults(trip, faults, seed=cfg.seed, frame=frame)
    out = write_trip(broken, args.output)
    _write_json(Path(out) / "faults.json", broken.faults)
    print(
        f"Wrote {out}\n"
        f"  removed {fault_report.removed_samples} fixes, modified {fault_report.modified_samples}\n"
        f"  clean track kept in reference-samples.jsonl for error metrics\n"
        f"  fault manifest: {Path(out) / 'faults.json'}"
    )
    return 0


def cmd_reconstruct(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    algorithm = _resolve_algorithm(args.algorithm)
    trip, load_report = load_trip(args.trip)
    if not trip.usable_locations:
        raise CLIError(
            f"{args.trip} contains no usable GPS fix. Every fix was dropped: "
            f"{load_report.rejected_reasons}"
        )
    origin = trip.usable_locations[0]
    network = _load_network(args.graph, origin.latitude, origin.longitude, args.radius)
    if network is None and algorithm == "road_particle_filter":
        raise CLIError(
            "--algorithm road-particle-filter needs --graph.\n"
            "Download the map once with:\n"
            '  geotrace download-map --place "Saint Petersburg, Russia" --output cache/spb.graphml'
        )
    if network is not None:
        print(f"Road graph: {len(network)} edges within {args.radius:.0f} m of the trip origin")

    result = run_reconstruction(trip, network, cfg, algorithm=algorithm, output_dt=args.output_dt)
    # Remembered so `geotrace report` can rebuild the exact same run.
    result.diagnostics["graph_path"] = args.graph
    result.diagnostics["graph_radius_m"] = args.radius
    result.diagnostics["output_dt"] = args.output_dt

    parking = None
    photo_diags: list[dict[str, Any]] = []
    if network is not None and result.parking_result is not None:
        parking = _parking_decision(args, result, cfg)
        photo_diags = _photo_analysis(trip, result, cfg)

    metrics = build_metrics(trip, result, cfg, parking=parking)
    out = results_dir(args.trip if not args.output else args.output)
    _write_outputs(trip, result, metrics, out, cfg, parking, photo_diags, load_report)

    warning = (result.diagnostics.get("road_graph") or {}).get("warning")
    if warning:
        print(f"\nWARNING: {warning}", file=sys.stderr)

    print(f"\nAlgorithm: {algorithm}   particles: {cfg.pf.n_particles}   seed: {cfg.seed}")
    print(f"GPS: {result.diagnostics['gps']['fixes_accepted']} accepted / "
          f"{result.diagnostics['gps']['fixes_rejected']} rejected")
    if result.outage_windows:
        for w in result.outage_windows:
            print(f"Detected outage: {w['start_s']:.1f}s -> {w['end_s']:.1f}s "
                  f"({w['end_s'] - w['start_s']:.1f}s)")
    else:
        print("No GPS outage detected.")
    error = metrics.position_error
    if error.get("mean_m") is not None:
        print(f"Position error vs reference: mean {error['mean_m']:.1f} m, "
              f"median {error['median_m']:.1f} m, p95 {error['p95_m']:.1f} m, "
              f"max {error['max_m']:.1f} m")
        print(f"95% polygon coverage: {metrics.polygons.get('coverage_95')}")
    else:
        print("No reference track: position error is undefined for a real outage.")
    print(f"\nResults written to {out}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from geotrace.visualization import ReportInputs, build_report, collect_photo_blocks

    run = Path(args.run)
    trip, _ = load_trip(run)
    out_dir = run / "results"
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        raise CLIError(
            f"{metrics_path} not found. Run `geotrace reconstruct --trip {run} ...` first."
        )
    metrics = json.loads(metrics_path.read_text("utf-8"))
    state_path = out_dir / "reconstruction-state.json"
    if not state_path.exists():
        raise CLIError(
            f"{state_path} not found. It is written by `geotrace reconstruct`; "
            "re-run that command."
        )
    saved = json.loads(state_path.read_text("utf-8"))

    cfg = Config.from_dict(saved["config"])
    origin = trip.usable_locations[0]
    network = _load_network(saved.get("graph_path"), origin.latitude, origin.longitude, saved.get("radius", 4000))
    result = run_reconstruction(
        trip, network, cfg, algorithm=saved["algorithm"], output_dt=saved.get("output_dt", 1.0)
    )
    photo_diags = json.loads((out_dir / "diagnostics.json").read_text("utf-8")).get("photos", [])
    inputs = ReportInputs(
        trip=trip,
        result=result,
        metrics=metrics,
        original_latlon=_latlon(trip.reference_locations or trip.locations),
        corrupted_latlon=_latlon(trip.locations) if trip.reference_locations else None,
        photo_blocks=collect_photo_blocks(trip, photo_diags),
        parking=metrics.get("parking") or None,
        polygon_stride=args.polygon_stride,
    )
    out = build_report(inputs, args.output or (out_dir / "report.html"))
    print(f"Wrote {out}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    trip, report = load_trip(args.trip)
    payload = {"trip": trip.summary(), "loader": report.to_json()}
    if trip.metadata.calibration:
        payload["calibration"] = trip.metadata.calibration.to_json()
    if trip.faults:
        payload["faults"] = trip.faults.get("faults")
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))
    return 0


# ------------------------------------------------------------- reconstruct bits


def _latlon(samples: Sequence[Any]) -> np.ndarray:
    usable = [s for s in samples if s.is_usable]
    if not usable:
        return np.zeros((0, 2))
    return np.array([[s.latitude, s.longitude] for s in usable], dtype=float)


def _parking_decision(args: argparse.Namespace, result: Any, cfg: Config) -> Optional[dict[str, Any]]:
    if not args.parking_zones:
        return None
    from geotrace.parking import GeoJSONParkingZoneProvider, decide_zone, zone_probabilities

    provider = GeoJSONParkingZoneProvider(args.parking_zones)
    zones = provider.load_zones()
    parking = result.parking_result
    if parking is None:
        return None
    # Zone selection deliberately samples the ParkingTracker confidence region,
    # never the OSM road particles.  This is the parked-car downstream API.
    angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    radius = max(parking.polygon_radius_m * 0.55, 1.0)
    local = np.vstack([
        np.asarray(parking.position),
        np.column_stack((
            parking.position[0] + radius * np.cos(angles),
            parking.position[1] + radius * np.sin(angles),
        )),
    ])
    ranking = zone_probabilities(zones, result.frame.to_geo_array(local), np.full(len(local), 1.0 / len(local)))
    payload = decide_zone(ranking, cfg.parking, provider.attribution).to_json()
    payload.update({
        "position_source": "parking_tracker",
        "parking_status": parking.status,
        "parking_confidence": parking.confidence,
    })
    return payload


def _photo_analysis(trip: Any, result: Any, cfg: Config) -> list[dict[str, Any]]:
    from geotrace.models import WeightedRegion
    from geotrace.photo import OCRStreetMatcher, parse_ocr, photo_diagnostics

    if not trip.photos or result.network is None:
        return []
    matcher = OCRStreetMatcher(result.network, cfg.photo)
    regions: list[WeightedRegion] = []
    if result.uncertainty:
        last = result.uncertainty[-1]
        for component in last.components:
            lat, lon = result.frame.to_geo(*component.representative_xy)
            regions.append(
                WeightedRegion(
                    component_id=component.component_id,
                    probability=component.probability,
                    latitude=lat,
                    longitude=lon,
                    radius_m=math.sqrt(max(component.area_m2, 1.0)),
                )
            )
    out: list[dict[str, Any]] = []
    pf = result.particle_filter
    for photo in trip.photos:
        parsed = parse_ocr(photo.ocr, cfg.photo)
        candidates = matcher.localize(Path(photo.image_path), regions, photo.ocr)
        if pf is not None and pf.initialized and not parsed.is_empty:
            # w'_i ~ w_i * L_OCR(p_i)^beta
            pf.apply_likelihood(matcher.particle_likelihood(parsed, pf.edge_idx))
        out.append(photo_diagnostics(photo, parsed, candidates))
    return out


def _write_outputs(
    trip: Any,
    result: Any,
    metrics: Any,
    out: Path,
    cfg: Config,
    parking: Optional[dict[str, Any]],
    photo_diags: list[dict[str, Any]],
    load_report: Any,
) -> None:
    from geotrace.polygons import uncertainty_to_geojson
    from geotrace.visualization import locations_geojson, track_geojson

    frame = result.frame
    has_reference = bool(trip.reference_locations)

    _write_json(
        out / "reconstructed-route.geojson",
        track_geojson(
            result,
            result.algorithm,
            {
                "description": (
                    "Probabilistic estimate of the driven route. Not a measured "
                    "track and not an exact path."
                ),
                "algorithm": result.algorithm,
                "seed": cfg.seed,
            },
        ),
    )
    _write_json(out / "uncertainty-polygons.geojson", uncertainty_to_geojson(result.uncertainty, frame))
    if result.parking_result is not None:
        parking = result.parking_result
        trajectory = frame.coords_to_geojson(parking.trajectory)
        lat, lon = frame.to_geo(*parking.position)
        ring = []
        for angle in np.linspace(0, 2 * np.pi, 49):
            plat, plon = frame.to_geo(
                parking.position[0] + parking.polygon_radius_m * np.cos(angle),
                parking.position[1] + parking.polygon_radius_m * np.sin(angle),
            )
            ring.append([plon, plat])
        _write_json(out / "parking-tracker.geojson", {
            "type": "FeatureCollection", "features": [
                {"type": "Feature", "properties": {"role": "trajectory"},
                 "geometry": {"type": "LineString", "coordinates": trajectory}},
                {"type": "Feature", "properties": {"role": "endpoint", "status": parking.status,
                    "confidence": parking.confidence}, "geometry": {"type": "Point", "coordinates": [lon, lat]}},
                {"type": "Feature", "properties": {"role": "confidence_polygon", "status": parking.status},
                 "geometry": {"type": "Polygon", "coordinates": [ring]}},
            ],
        })
    _write_json(
        out / "corrupted-gps.geojson",
        locations_geojson(
            trip.locations,
            frame,
            "corrupted GPS" if has_reference else "recorded GPS",
            {"synthetically_corrupted": has_reference},
        ),
    )
    if has_reference:
        _write_json(
            out / "reference-gps.geojson",
            locations_geojson(
                trip.reference_locations, frame, "reference route",
                {"note": "ground truth only because the corruption was synthetic"},
            ),
        )
    for name in result.tracks:
        if name != result.algorithm:
            _write_json(out / f"baseline-{name.replace('_', '-')}.geojson",
                        track_geojson(result, name, {"role": "baseline"}))

    _write_json(out / "metrics.json", metrics.to_json())
    diagnostics = dict(result.diagnostics)
    diagnostics["loader"] = load_report.to_json()
    diagnostics["photos"] = photo_diags
    if parking:
        diagnostics["parking"] = parking
    diagnostics["config"] = cfg.to_dict()
    _write_json(out / "diagnostics.json", diagnostics)
    parking_payload = result.parking_result.to_json(frame) if result.parking_result else None
    road_endpoint = result.primary.xy[-1] if result.primary.xy else None
    _write_json(out / "tracking-result.json", {
        "schema_version": 2,
        "tracking_architecture": "dual-tracker-v1",
        "road_result": {
            "route": result.primary.to_geojson(frame),
            "endpoint": list(road_endpoint) if road_endpoint else None,
            "confidence": cfg.polygon.confidence if result.algorithm == "road_particle_filter" else None,
        },
        "parking_result": parking_payload,
        "final_vehicle_position": parking_payload["position"] if parking_payload else None,
        "final_vehicle_position_source": "parking_tracker",
    })
    _write_json(
        out / "reconstruction-state.json",
        {
            "schema_version": 2,
            "tracking_architecture": "dual-tracker-v1",
            "algorithm": result.algorithm,
            "config": cfg.to_dict(),
            "graph_path": diagnostics.get("graph_path"),
            "output_dt": diagnostics.get("output_dt", 1.0),
            "radius": diagnostics.get("graph_radius_m", 4000.0),
        },
    )


# ------------------------------------------------------------------ parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geotrace",
        description=(
            "Probabilistic recovery of a car route from degraded GPS, using IMU "
            "dead reckoning constrained by an OpenStreetMap road graph."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"geotrace {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p = sub.add_parser("download-map", help="download and cache an OSM drive graph as GraphML")
    p.add_argument("--place", help='e.g. "Saint Petersburg, Russia"')
    p.add_argument("--bbox", nargs=4, metavar=("NORTH", "SOUTH", "EAST", "WEST"),
                   help="bounding box instead of a place name; much faster")
    p.add_argument("--output", required=True, help="destination .graphml file")
    p.add_argument("--network-type", default="drive", help="OSMnx network type (default: drive)")
    p.add_argument("--force", action="store_true", help="re-download even if the cache exists")
    p.set_defaults(func=cmd_download_map)

    p = sub.add_parser("simulate", help="generate a synthetic trip in the iPhone's format")
    p.add_argument("--graph", help="cached GraphML; omitted = built-in synthetic grid")
    p.add_argument("--output", required=True)
    p.add_argument("--duration", type=float, default=300.0, help="seconds (default: 300)")
    p.add_argument("--speed", type=float, default=11.0, help="cruise speed, m/s (default: 11)")
    p.add_argument("--stop-at", type=float, default=None, help="insert a stop at this second")
    p.add_argument("--gps-noise", type=float, default=5.0, help="GPS sigma, m (default: 5)")
    p.add_argument("--lat", type=float, default=59.9311)
    p.add_argument("--lon", type=float, default=30.3609)
    p.add_argument("--radius", type=float, default=4000.0)
    p.add_argument("--trip-id", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default=None)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("inject-fault", help="write a corrupted copy of a trip")
    p.add_argument("--trip", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fault", choices=FAULT_KINDS, default="dropout")
    p.add_argument("--scenario", action="store_true",
                   help="composite failure: offset, then dropout, then a false recovery")
    p.add_argument("--start", type=float, default=60.0, help="seconds from the start of the trip")
    p.add_argument("--duration", type=float, default=None, help="seconds")
    p.add_argument("--east", type=float, default=0.0, help="offset fault: metres east")
    p.add_argument("--north", type=float, default=0.0, help="offset fault: metres north")
    p.add_argument("--drift-east", type=float, default=0.0, help="drift fault: m/s east")
    p.add_argument("--drift-north", type=float, default=0.0, help="drift fault: m/s north")
    p.add_argument("--sigma", type=float, default=250.0, help="jumps fault: sigma in metres")
    p.add_argument("--count", type=int, default=3, help="false_recovery: number of false fixes")
    p.add_argument("--accuracy", type=float, default=None,
                   help="horizontal accuracy the corrupted fixes claim to have")
    p.add_argument("--dropout-duration", type=float, default=45.0, help="--scenario only")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default=None)
    p.set_defaults(func=cmd_inject_fault)

    p = sub.add_parser("reconstruct", help="detect the GPS failure and rebuild the route")
    p.add_argument("--trip", required=True)
    p.add_argument("--graph", default=None, help="cached GraphML road graph")
    p.add_argument("--algorithm", default="road-particle-filter",
                   help="road-particle-filter | ekf-dead-reckoning | last-known-position")
    p.add_argument("--particles", type=int, default=None, help="default: 5000")
    p.add_argument("--confidence", type=float, default=None, help="polygon mass, default 0.95")
    p.add_argument("--seed", type=int, default=None, help="default: 42")
    p.add_argument("--radius", type=float, default=4000.0,
                   help="metres of road graph to keep around the trip origin")
    p.add_argument("--output-dt", type=float, default=1.0, help="estimate output interval, s")
    p.add_argument("--parking-zones", default=None, help="GeoJSON of parking-zone polygons")
    p.add_argument(
        "--rng", choices=("numpy", "parity"), default=None,
        help=(
            "random generator for the particle filter. 'parity' uses the "
            "cross-language generator that the Swift on-device implementation "
            "reimplements exactly, so the two can be compared draw for draw."
        ),
    )
    p.add_argument(
        "--allow-simulated", action="store_true",
        help=(
            "accept fixes CoreLocation flagged as simulated by software. Needed "
            "only for trips recorded in the iOS Simulator; never for real data."
        ),
    )
    p.add_argument("--output", default=None, help="write results here instead of <trip>/results")
    p.add_argument("--config", default=None, help="JSON config overriding every threshold")
    p.set_defaults(func=cmd_reconstruct)

    p = sub.add_parser("report", help="render results/report.html for a finished run")
    p.add_argument("--run", required=True, help="the trip directory that was reconstructed")
    p.add_argument("--output", default=None)
    p.add_argument("--polygon-stride", type=int, default=10,
                   help="draw every Nth polygon set (default 10, keeps the map readable)")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("inspect", help="print a summary of a trip directory")
    p.add_argument("--trip", required=True)
    p.set_defaults(func=cmd_inspect)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    try:
        return int(args.func(args) or 0)
    except (CLIError, FaultError, TripLoadError, ReconstructionError, RoadGraphError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: file not found: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # pragma: no cover - unexpected, show the trace
        print(f"internal error: {exc}\n", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
