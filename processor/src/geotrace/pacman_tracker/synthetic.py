"""Synthetic road networks for tests and for the straight-road regression.

`geotrace.road_graph.build_graph_from_segments` splits every polyline into
one graph edge per vertex pair, which is right for the simulator but wrong
here: this tracker's whole subject is the shape of an edge *between* its
endpoints, so a test road has to arrive as a single edge carrying a real
``geometry``.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import networkx as nx
import numpy as np
from shapely.geometry import LineString

from geotrace.coordinates import LocalFrame, wrap_angle
from geotrace.road_graph import RoadNetwork

ORIGIN_LAT = 59.9343
ORIGIN_LON = 30.3351


def build_network(
    ways: Iterable[tuple[str, Sequence[tuple[float, float]], dict[str, Any]]],
    origin_lat: float = ORIGIN_LAT,
    origin_lon: float = ORIGIN_LON,
) -> tuple[RoadNetwork, LocalFrame]:
    """Build a network in which each way is exactly one directed edge.

    ``ways`` is (name, [(E, N), ...], attrs); ``attrs['oneway']`` defaults to
    True so a test can control the successor set exactly.
    """
    frame = LocalFrame(origin_lat, origin_lon)
    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "epsg:4326"
    node_of: dict[tuple[int, int], int] = {}

    def node_for(point: Sequence[float]) -> int:
        key = (int(round(point[0] * 10)), int(round(point[1] * 10)))
        if key not in node_of:
            node_id = len(node_of) + 1
            lat, lon = frame.to_geo(float(point[0]), float(point[1]))
            graph.add_node(node_id, x=lon, y=lat)
            node_of[key] = node_id
        return node_of[key]

    for name, points, attrs in ways:
        pts = [(float(p[0]), float(p[1])) for p in points]
        u, v = node_for(pts[0]), node_for(pts[-1])
        if u == v:
            raise ValueError(f"way {name!r} is a closed loop; not supported here")
        lonlat = [frame.to_geo(e, n)[::-1] for e, n in pts]
        length = float(sum(math.dist(a, b) for a, b in zip(pts[:-1], pts[1:])))
        payload = {
            "name": name,
            "highway": attrs.get("highway", "residential"),
            "maxspeed": attrs.get("maxspeed"),
            "oneway": True,
            "length": length,
            "geometry": LineString(lonlat),
        }
        graph.add_edge(u, v, **payload)
        if not attrs.get("oneway", True):
            back = dict(payload)
            back["geometry"] = LineString(list(reversed(lonlat)))
            graph.add_edge(v, u, **back)
    return RoadNetwork(graph, frame), frame


def straight(length_m: float, start: tuple[float, float] = (0.0, 0.0),
             heading_rad: float = 0.0, step_m: float = 10.0) -> list[tuple[float, float]]:
    n = max(2, int(round(length_m / step_m)) + 1)
    d = np.linspace(0.0, length_m, n)
    return [(start[0] + t * math.cos(heading_rad), start[1] + t * math.sin(heading_rad)) for t in d]


def arc(radius_m: float, turn_rad: float, start: tuple[float, float] = (0.0, 0.0),
        heading_rad: float = 0.0, step_m: float = 5.0) -> list[tuple[float, float]]:
    """Constant-curvature arc. ``turn_rad`` positive turns left."""
    length = abs(radius_m * turn_rad)
    n = max(3, int(round(length / step_m)) + 1)
    t = np.linspace(0.0, turn_rad, n)
    sign = 1.0 if turn_rad >= 0 else -1.0
    cx = start[0] - sign * radius_m * math.sin(heading_rad)
    cy = start[1] + sign * radius_m * math.cos(heading_rad)
    ang = math.atan2(start[1] - cy, start[0] - cx)
    return [
        (cx + radius_m * math.cos(ang + ti), cy + radius_m * math.sin(ang + ti))
        for ti in (t * sign * sign)
    ]


# --------------------------------------------------------------- synthetic trips


def _yaw_quaternion(psi: float) -> tuple[float, float, float, float]:
    return (math.cos(psi / 2.0), 0.0, 0.0, math.sin(psi / 2.0))


def simulate_trip(
    network: "RoadNetwork",
    route: Sequence[int],
    speed_profile,
    duration_s: float,
    gps_visible_s: float,
    dt: float = 0.02,
    gps_dt: float = 0.1,
    accel_bias: float = 0.0,
    gyro_bias: float = 0.0,
    noise: float = 0.0,
    seed: int = 0,
    trip_id: str = "pacman-synthetic",
):
    """Drive a known route and record it in the recorder's own wire format.

    Used by the leakage and end-to-end tests: it produces a
    :class:`~geotrace.models.Trip` whose ``locations`` stop at
    ``gps_visible_s`` and whose ``reference_locations`` hold the withheld
    remainder, exactly as the real review recordings are split.
    """
    from datetime import datetime, timedelta, timezone

    from geotrace.models import (
        LocationSample,
        MotionSample,
        MountCalibration,
        Trip,
        TripMetadata,
    )

    rng = np.random.default_rng(seed)
    frame = network.frame
    edges = [network.edges[i] for i in route]
    lengths = [e.length for e in edges]
    total = float(sum(lengths))

    motions: list[MotionSample] = []
    visible: list[LocationSample] = []
    withheld: list[LocationSample] = []
    t0 = datetime(2026, 7, 22, 12, 0, 0, tzinfo=timezone.utc)

    s = 0.0
    v = float(speed_profile(0.0))
    prev_psi = None
    next_gps = 0.0
    t = 0.0
    while t <= duration_s:
        v_next = float(speed_profile(t))
        a = (v_next - v) / dt if dt > 0 else 0.0
        v = v_next
        s = min(s + v * dt, total - 1e-6)

        cursor, remaining = 0, s
        while cursor < len(edges) - 1 and remaining > lengths[cursor]:
            remaining -= lengths[cursor]
            cursor += 1
        edge = edges[cursor]
        psi = float(edge.bearing(remaining))
        omega = 0.0 if prev_psi is None else float(wrap_angle(psi - prev_psi)) / dt
        prev_psi = psi

        # Centripetal acceleration is not optional: a car going round a bend
        # feels a_lat = v * omega, and that ratio is how the tracker measures
        # its speed. A simulator that emits zero lateral acceleration tells
        # every turn "you are stopped" and silently invalidates any test of it.
        a_lat = v * omega
        motions.append(
            MotionSample(
                monotonic_time=round(t, 6),
                user_acceleration_g=(
                    (a + accel_bias + noise * rng.normal()) / 9.80665,
                    (a_lat + noise * rng.normal()) / 9.80665,
                    0.0,
                ),
                rotation_rate=(0.0, 0.0, omega + gyro_bias + noise * rng.normal() * 0.05),
                gravity=(0.0, 0.0, -1.0),
                quaternion=_yaw_quaternion(psi),
                wall_time=t0 + timedelta(seconds=t),
            )
        )
        if t >= next_gps:
            x, y = edge.position(remaining)
            lat, lon = frame.to_geo(x, y)
            fix = LocationSample(
                monotonic_time=round(t, 6), latitude=lat, longitude=lon,
                wall_time=t0 + timedelta(seconds=t), horizontal_accuracy=5.0,
                speed=v, speed_accuracy=1.0,
                course=float(np.mod(90.0 - math.degrees(psi), 360.0)), course_accuracy=5.0,
            )
            (visible if t <= gps_visible_s else withheld).append(fix)
            next_gps += gps_dt
        t += dt

    metadata = TripMetadata(
        trip_id=trip_id, started_at=t0, ended_at=t0 + timedelta(seconds=duration_s),
        device_model="pacman synthetic",
        calibration=MountCalibration(
            forward_axis_device=(1.0, 0.0, 0.0),
            initial_heading_deg=None, heading_source="synthetic",
            attitude_source="rigid_mount_simulator_v2", still_duration_s=0.0,
        ),
        location_sample_count=len(visible), motion_sample_count=len(motions),
    )
    return Trip(metadata=metadata, locations=visible, motions=motions,
                reference_locations=withheld, root=None)
