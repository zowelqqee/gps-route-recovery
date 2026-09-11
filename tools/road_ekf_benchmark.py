"""Evaluate the new road-coordinate model on previously imported, withheld trips.

No importing or tuning is performed here. The reference enters only metrics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from geotrace.config import Config
from geotrace.coordinates import LocalFrame
from geotrace.loader import load_trip
from geotrace.pipeline import run_reconstruction, build_metrics
from geotrace.road_graph import RoadNetwork, load_graph, clip_graph
from geotrace.visualization import ReportInputs, build_report
from geotrace.polygons import uncertainty_to_geojson


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trips',type=Path,nargs='+',required=True)
    p.add_argument('--graph',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    cfg=Config.load(args.config) if args.config else Config()
    graph=load_graph(args.graph)
    digest=hashlib.sha256()
    for source in sorted(Path('processor/src/geotrace').glob('*.py')):
        digest.update(source.name.encode());digest.update(source.read_bytes())
    rows=[]
    for path in args.trips:
        trip,_=load_trip(path)
        first=trip.usable_locations[0]
        frame=LocalFrame(first.latitude,first.longitude)
        net=RoadNetwork(clip_graph(graph,first.latitude,first.longitude,30000.),frame)
        out=args.output/path.parent.name;out.mkdir(parents=True,exist_ok=True)
        print(f'Running {path}',flush=True)
        result=run_reconstruction(trip,net,cfg)
        metrics=build_metrics(trip,result,cfg).to_json()
        cfg.dump(out/'config.json')
        for name,value in [('metrics',metrics),('diagnostics',result.diagnostics),
                           ('uncertainty-polygons',uncertainty_to_geojson(result.uncertainty,frame,trip.t0)),
                           ('route',result.primary.to_geojson(frame))]:
            suffix='geojson' if name in ('route','uncertainty-polygons') else 'json'
            (out/f'{name}.{suffix}').write_text(json.dumps(value,indent=2))
        build_report(ReportInputs(trip,result,metrics,
            original_latlon=np.array([[s.latitude,s.longitude] for s in trip.reference_locations]),
            corrupted_latlon=np.array([[s.latitude,s.longitude] for s in trip.locations])),out/'report.html')
        row={'trip':str(path),'source_sha256':digest.hexdigest(),
             'mean_error_m':metrics['position_error'].get('mean_m'),
             'coverage':metrics['polygons'].get('coverage'),
             'max_total_area_m2':metrics['polygons'].get('max_area_m2'),
             'runtime_s':result.diagnostics['runtime_s'],
             'final_status':result.uncertainty[-1].status,
             'final_represented_mass':result.uncertainty[-1].represented_mass,
             'lost_reason':result.diagnostics['road_ekf']['lost_reason']}
        rows.append(row);print(json.dumps(row),flush=True)
        (args.output/'summary.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':main()
