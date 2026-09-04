# geotrace

The Python processor. See the [repository README](../README.md) for the full
workflow; this file is a quick map of the package.

| module | what it does |
| --- | --- |
| `config.py` | every threshold, sigma and probability limit, in one place |
| `models.py` | the on-disk sample format, shared with the iOS app |
| `loader.py` | reads and writes trip directories (and `.zip` exports) |
| `coordinates.py` | WGS84 <-> local metric frame, angle wrapping |
| `gps_quality.py` | physical gate, Mahalanobis gate, TRUSTED/SUSPECT/LOST/RECOVERING |
| `motion_model.py` | strapdown IMU handling and the state transition |
| `ekf.py` | the `ekf_dead_reckoning` baseline |
| `road_graph.py` | OSM graph, cached as GraphML, projected into the local frame |
| `particle_filter.py` | the `road_particle_filter` main algorithm |
| `polygons.py` | gamma-mass selection, branch clustering, corridor buffers |
| `parking.py` | parking-zone probability and the confidence decision |
| `photo.py` | OCR parsing, street matching, the VPR interface (stubbed) |
| `fault_injection.py` | dropout / offset / drift / jumps / false recovery |
| `metrics.py` | error, coverage, branch accuracy, gate rates, recovery time |
| `visualization.py` | `report.html` |
| `simulate.py` | synthetic trips, so nothing needs an iPhone to run |
| `pipeline.py` | ties it together |
| `cli.py` | the `geotrace` command |
