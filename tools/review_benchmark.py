"""Reproducible live-log audit; each variant sees exactly the same withheld split.

Run from repository root with processor/.venv/bin/python tools/review_benchmark.py.
No tuning against the reference is performed. Results include a source digest.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from geotrace.config import Config
from geotrace.coordinates import LocalFrame
from geotrace.live_logs import ImportSpec, build_trip
from geotrace.loader import write_trip, load_trip
from geotrace.pipeline import build_metrics, run_reconstruction
from geotrace.road_graph import RoadNetwork, load_graph, clip_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--days', nargs='+', default=[f'2026-07-{d}' for d in range(22, 27)])
    parser.add_argument('--duration', type=float, default=600)
    parser.add_argument('--warmup', type=float, default=120)
    parser.add_argument('--particles', type=int, default=500)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--variants', nargs='+', default=['default', 'smooth5-db2', 'combined'])
    parser.add_argument('--graph', type=Path, default=Path('cache/spb.graphml'))
    parser.add_argument('--trip-cache', type=Path)
    parser.add_argument('--sessions', nargs='*', default=[], help='Explicit day:index pairs; defaults to session zero')
    parser.add_argument('--reports', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('runs/independent-review'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sessions = {item.split(':')[0]:int(item.split(':')[1]) for item in args.sessions}
    digest = hashlib.sha256()
    for path in sorted(Path('processor/src/geotrace').glob('*.py')):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    record = {'source_sha256': digest.hexdigest(), 'graph_sha256': hashlib.sha256(args.graph.read_bytes()).hexdigest(),
              'duration_s': args.duration, 'warmup_s': args.warmup, 'particles': args.particles,
              'sessions':sessions, 'scope': 'explicit sessions, one window per day; exploratory, not held-out calibration', 'runs': []}
    graph = load_graph(args.graph)
    for day in args.days:
        print(f'Import {day}', flush=True)
        started = time.time()
        try:
            cached = args.trip_cache / day / 'trip' if args.trip_cache else None
            if cached is not None and cached.exists() and sessions.get(day, 0) == 0:
                trip,_ = load_trip(cached)
                provenance = trip.metadata.extra['live_import']
                if abs(trip.duration_s - args.duration) > 2 or provenance['gps_warmup_s'] != args.warmup:
                    raise ValueError('Cached trip does not match requested duration/warmup')
            else:
                trip, provenance = build_trip(ImportSpec(gps_dir=Path('live_logs/gps_logs'),
                    imu_dir=Path('live_logs/imu_logs'), day=day, gps_warmup_s=args.warmup,
                    max_duration_s=args.duration, session_index=sessions.get(day, 0)))
            write_trip(trip, args.output / day / 'trip')
            trip,_ = load_trip(args.output / day / 'trip')
            first = trip.usable_locations[0]
            frame = LocalFrame(first.latitude, first.longitude)
            network = RoadNetwork(clip_graph(graph, first.latitude, first.longitude, 30000), frame)
            print(f'Imported {day} in {time.time()-started:.1f}s, clock={provenance["clock_offset"]}', flush=True)
        except Exception as exc:
            record['runs'].append({'day':day, 'error':repr(exc)})
            (args.output/'summary.json').write_text(json.dumps(record, indent=2))
            print(repr(exc), flush=True)
            continue
        for variant in args.variants:
            for seed in args.seeds:
                cfg=Config(); cfg.pf.n_particles=args.particles; cfg.seed=seed
                if variant in ('smooth5-db2', 'combined'):
                    cfg.motion.accel_smooth_window_s=5.; cfg.motion.accel_deadband_ms2=2.
                if variant == 'combined':
                    cfg.motion.leveling_recovery_tau_s=20.; cfg.motion.zupt_vibration_g=.02
                    cfg.motion.zupt_min_interval_s=5.; cfg.motion.zupt_requires_gps=False
                directory=args.output/day/f'{variant}-{seed}'; directory.mkdir(parents=True, exist_ok=True)
                cfg.dump(directory/'config.json')
                print(f'Run {day} {variant} seed={seed}', flush=True)
                try:
                    result=run_reconstruction(trip, network, cfg)
                    metrics=build_metrics(trip,result,cfg).to_json()
                    (directory/'metrics.json').write_text(json.dumps(metrics,indent=2))
                    (directory/'diagnostics.json').write_text(json.dumps(result.diagnostics,indent=2))
                    (directory/'route.geojson').write_text(json.dumps(result.primary.to_geojson(frame)))
                    row={'day':day,'variant':variant,'seed':seed,'clock':provenance['clock_offset'],
                         'mount_quality':provenance['mount_estimate'].get('quality'), 'session':sessions.get(day,0),
                         'display_mean_m':metrics['position_error'].get('mean_m'),
                         'ekf_mean_m':metrics['baselines']['ekf_dead_reckoning'].get('mean_m'),
                         'road_posterior_mean_m':metrics['baselines']['road_posterior'].get('mean_m'),
                         'display_coverage':metrics['polygons'].get('coverage_95'),
                         'road_coverage':metrics['polygons']['road_posterior'].get('coverage_95'),
                         'mean_area_m2':metrics['polygons'].get('mean_area_m2'),
                         'runtime_s':metrics['runtime']['seconds'], 'reference_quality':metrics.get('reference_quality',{})}
                    record['runs'].append(row)
                    print(json.dumps(row),flush=True)
                    if args.reports:
                        from geotrace.visualization import ReportInputs, build_report
                        build_report(ReportInputs(trip,result,metrics, original_latlon=np.array([[s.latitude,s.longitude] for s in trip.reference_locations]), corrupted_latlon=np.array([[s.latitude,s.longitude] for s in trip.locations])),directory/'report.html')
                    del result
                except Exception as exc:
                    record['runs'].append({'day':day,'variant':variant,'seed':seed,'error':repr(exc)})
                    print(repr(exc),flush=True)
                (args.output/'summary.json').write_text(json.dumps(record, indent=2))
    print(args.output/'summary.json',flush=True)

if __name__=='__main__':
    main()
