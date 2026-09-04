"""Synthetic trip generator.

Produces a trip in exactly the format the iPhone writes, by driving a virtual
car along a real (or synthetic) road graph and rendering consistent GPS and
CMDeviceMotion streams from that motion.

It exists for two reasons:

  * the whole pipeline must be runnable and testable with no iPhone and no
    network access;
  * fault injection needs a clean track to corrupt, and error metrics need a
    reference. Only a synthetic trip can provide one honestly.

The IMU is rendered the way CoreMotion really behaves: acceleration and rotation
rate in the *device* frame, with the phone at a fixed arbitrary mount attitude,
and with the reference frame's yaw offset from true north - so the processor's
heading-alignment code is genuinely exercised rather than handed the answer.
"""

from __future__ import annotations

import math
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d

from geotrace.config import G_TO_MS2, Config
from geotrace.coordinates import heading_to_course, wrap_angle
from geotrace.models import (
    LocationSample,
    MotionSample,
    MountCalibration,
    Trip,
    TripMetadata,
)
from geotrace.motion_model import quaternion_to_matrix
from geotrace.road_graph import RoadNetwork

IMU_HZ = 50.0
GPS_HZ = 1.0


@dataclass
class SimulationSpec:
    duration_s: float = 300.0
    cruise_speed_ms: float = 11.0
    stop_at_s: Optional[float] = None
    stop_duration_s: float = 12.0
    gps_sigma_m: float = 5.0
    gps_accuracy_m: float = 9.0
    accel_noise_ms2: float = 0.22
    gyro_noise_rads: float = 0.012
    accel_bias_ms2: float = 0.06
    gyro_bias_rads: float = 0.004
    mount_yaw_deg: float = 34.0
    mount_pitch_deg: float = 68.0
    """Phone tilted back in a windscreen cradle, rotated in the holder."""

    reference_yaw_offset_deg: float = -51.0
    """Yaw offset between the CMDeviceMotion reference frame and true East."""

    warmup_still_s: float = 6.0


def _quaternion_from_euler(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def pick_route(
    network: RoadNetwork,
    rng: np.random.Generator,
    target_length_m: float,
    start_edge: Optional[int] = None,
    straightness: float = 0.85,
) -> list[int]:
    """Walk the graph, preferring to carry straight on, until long enough.

    ``straightness`` biases the choice at every junction towards the outgoing
    edge closest in bearing to the current one, which is what a car driving
    through a city actually does most of the time.
    """
    successors = network.successor_table(allow_uturn=False)
    if start_edge is None:
        long_edges = [e.index for e in network.edges if e.length > 40 and successors[e.index].size]
        if not long_edges:
            long_edges = [e.index for e in network.edges]
        start_edge = int(rng.choice(long_edges))

    route = [int(start_edge)]
    total = network.edges[start_edge].length
    # Avoid only the recent past, not everything ever visited: on a real street
    # grid a 4 km drive legitimately crosses the same avenue twice, and banning
    # every seen edge walks the route into a dead end within a few hundred
    # metres.
    recent: deque[int] = deque([int(start_edge)], maxlen=10)
    stuck = 0
    while total < target_length_m:
        current = route[-1]
        options = successors[current]
        fresh = np.array([o for o in options if int(o) not in recent], dtype=np.int64)
        if fresh.size:
            options = fresh
            stuck = 0
        else:
            stuck += 1
            if stuck > 3 or options.size == 0:
                break
        if options.size == 1:
            choice = int(options[0])
        else:
            current_bearing = network.edges[current].bearings[-1]
            bearings = np.array([network.edges[int(o)].start_bearing for o in options])
            turn = np.abs(np.asarray(wrap_angle(bearings - current_bearing), dtype=float))
            scores = np.exp(-turn / max(1e-3, 1.0 - straightness))
            scores = scores / scores.sum()
            choice = int(options[rng.choice(options.size, p=scores)])
        route.append(choice)
        recent.append(choice)
        total += network.edges[choice].length
    return route


def _route_geometry(network: RoadNetwork, route: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Dense polyline of the route, plus its cumulative length."""
    points: list[np.ndarray] = []
    for i, edge_index in enumerate(route):
        coords = network.edges[int(edge_index)].coords
        points.append(coords if i == 0 else coords[1:])
    poly = np.concatenate(points, axis=0)
    steps = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    return poly, cumulative


def _speed_profile(spec: SimulationSpec, times: np.ndarray) -> np.ndarray:
    """Realistic drive: stand still, accelerate, cruise, optional stop, stop."""
    v = np.full_like(times, spec.cruise_speed_ms)
    accel_time = spec.cruise_speed_ms / 1.6
    warm = spec.warmup_still_s
    v = np.where(times < warm, 0.0, v)
    ramp = (times >= warm) & (times < warm + accel_time)
    v = np.where(ramp, spec.cruise_speed_ms * (times - warm) / accel_time, v)

    end = times[-1]
    brake_time = spec.cruise_speed_ms / 2.0
    tail_still = 4.0
    brake = (times >= end - brake_time - tail_still) & (times < end - tail_still)
    v = np.where(
        brake,
        spec.cruise_speed_ms * np.clip((end - tail_still - times) / brake_time, 0.0, 1.0),
        v,
    )
    v = np.where(times >= end - tail_still, 0.0, v)

    if spec.stop_at_s is not None:
        s0 = spec.stop_at_s
        s1 = s0 + spec.stop_duration_s
        decel = (times >= s0 - 5) & (times < s0)
        v = np.where(decel, v * np.clip((s0 - times) / 5.0, 0.0, 1.0), v)
        v = np.where((times >= s0) & (times < s1), 0.0, v)
        accel = (times >= s1) & (times < s1 + 6)
        v = np.where(accel, spec.cruise_speed_ms * np.clip((times - s1) / 6.0, 0.0, 1.0), v)

    # Gentle traffic variation so the accelerometer is not a step function.
    v = v * (1.0 + 0.06 * np.sin(times / 11.0) + 0.04 * np.sin(times / 3.7))
    return gaussian_filter1d(np.maximum(v, 0.0), sigma=IMU_HZ * 0.6)


def simulate_trip(
    network: RoadNetwork,
    spec: Optional[SimulationSpec] = None,
    seed: int = 42,
    trip_id: Optional[str] = None,
    start_edge: Optional[int] = None,
    started_at: Optional[datetime] = None,
    route: Optional[Sequence[int]] = None,
) -> Trip:
    """Generate a complete synthetic trip on ``network``.

    ``route`` pins the exact sequence of edges the car drives. Without it the
    route is walked from ``start_edge`` with a bias towards carrying straight
    on. Tests that need to know which branch of a fork was really taken pass it
    explicitly.
    """
    spec = spec or SimulationSpec()
    rng = np.random.default_rng(seed)

    n = int(spec.duration_s * IMU_HZ)
    times = np.arange(n) / IMU_HZ
    speed = _speed_profile(spec, times)
    distance = np.cumsum(speed) / IMU_HZ

    needed = float(distance[-1]) * 1.15 + 200.0
    if route is None:
        route = pick_route(network, rng, needed, start_edge=start_edge)
    else:
        route = [int(e) for e in route]
    poly, cumulative = _route_geometry(network, route)
    if cumulative[-1] < distance[-1]:
        # The graph ran out of road before the requested duration: shorten the
        # drive rather than teleporting the car back to the start.
        keep = distance <= cumulative[-1] - 5.0
        times, speed, distance = times[keep], speed[keep], distance[keep]
        n = len(times)
        if n < int(IMU_HZ * 20):
            raise ValueError(
                "the road graph is too small for a trip of this length; "
                "reduce --duration or use a larger graph"
            )

    east = np.interp(distance, cumulative, poly[:, 0])
    north = np.interp(distance, cumulative, poly[:, 1])

    # Heading from the path, smoothed so turns have realistic yaw dynamics.
    raw_heading = np.arctan2(np.gradient(north), np.gradient(east))
    moving = speed > 0.4
    if np.any(moving):
        raw_heading[~moving] = np.interp(
            times[~moving], times[moving], np.unwrap(raw_heading[moving])
        )
    heading = gaussian_filter1d(np.unwrap(raw_heading), sigma=IMU_HZ * 0.8)
    yaw_rate = np.gradient(heading) * IMU_HZ
    a_long = np.gradient(speed) * IMU_HZ
    a_lat = speed * yaw_rate

    cos_h, sin_h = np.cos(heading), np.sin(heading)
    a_east = a_long * cos_h - a_lat * sin_h
    a_north = a_long * sin_h + a_lat * cos_h
    a_up = np.zeros(n)

    # World (E/N/U) -> CMDeviceMotion reference frame: a fixed yaw offset.
    ref_yaw = math.radians(spec.reference_yaw_offset_deg)
    c, s_ref = math.cos(-ref_yaw), math.sin(-ref_yaw)
    a_ref = np.column_stack(
        [c * a_east - s_ref * a_north, s_ref * a_east + c * a_north, a_up]
    )
    w_ref = np.column_stack([np.zeros(n), np.zeros(n), yaw_rate])

    # Reference frame -> device frame: the fixed mount attitude.
    q_mount = _quaternion_from_euler(
        math.radians(spec.mount_yaw_deg), math.radians(spec.mount_pitch_deg), 0.0
    )
    R = quaternion_to_matrix(q_mount)
    Rt = R.T
    a_device = a_ref @ Rt.T
    w_device = w_ref @ Rt.T
    gravity_device = Rt @ np.array([0.0, 0.0, -1.0])

    a_device += rng.normal(0.0, spec.accel_noise_ms2, a_device.shape) + spec.accel_bias_ms2 * Rt[:, 0]
    w_device += rng.normal(0.0, spec.gyro_noise_rads, w_device.shape) + spec.gyro_bias_rads * Rt[:, 2]

    started_at = started_at or datetime.now(timezone.utc)
    t0_mono = 13800.0

    motions: list[MotionSample] = []
    for i in range(n):
        motions.append(
            MotionSample(
                monotonic_time=t0_mono + float(times[i]),
                user_acceleration_g=tuple(a_device[i] / G_TO_MS2),  # type: ignore[arg-type]
                rotation_rate=tuple(w_device[i]),  # type: ignore[arg-type]
                gravity=tuple(gravity_device + rng.normal(0, 0.004, 3)),  # type: ignore[arg-type]
                quaternion=tuple(q_mount),  # type: ignore[arg-type]
                magnetic_field=tuple(rng.normal([22.0, -8.0, 41.0], 3.5)),  # type: ignore[arg-type]
                magnetic_accuracy=1,
                wall_time=started_at + timedelta(seconds=float(times[i])),
            )
        )

    gps_step = int(IMU_HZ / GPS_HZ)
    locations: list[LocationSample] = []
    for i in range(0, n, gps_step):
        noise = rng.normal(0.0, spec.gps_sigma_m, 2)
        lat, lon = network.frame.to_geo(float(east[i] + noise[0]), float(north[i] + noise[1]))
        moving_now = speed[i] >= 1.0
        locations.append(
            LocationSample(
                monotonic_time=t0_mono + float(times[i]),
                latitude=lat,
                longitude=lon,
                wall_time=started_at + timedelta(seconds=float(times[i])),
                altitude=float(6.0 + rng.normal(0, 1.5)),
                horizontal_accuracy=float(spec.gps_accuracy_m + abs(rng.normal(0, 2.0))),
                vertical_accuracy=float(spec.gps_accuracy_m * 1.6),
                speed=float(max(0.0, speed[i] + rng.normal(0, 0.35))),
                speed_accuracy=1.0,
                course=heading_to_course(float(heading[i])) if moving_now else -1.0,
                course_accuracy=8.0 if moving_now else -1.0,
                source_information={"is_simulated_by_software": False,
                                    "is_produced_by_accessory": False},
            )
        )

    # Vehicle forward axis at the moment the car starts moving, expressed in
    # the device frame: world E/N -> reference frame -> device frame.
    heading_index = min(n - 1, int(IMU_HZ * spec.warmup_still_s) + 50)
    psi0 = float(heading[heading_index])
    forward_world = np.array([math.cos(psi0), math.sin(psi0), 0.0])
    forward_ref = np.array(
        [
            c * forward_world[0] - s_ref * forward_world[1],
            s_ref * forward_world[0] + c * forward_world[1],
            0.0,
        ]
    )
    calibration = MountCalibration(
        reference_quaternion=tuple(q_mount),  # type: ignore[arg-type]
        gravity_device=tuple(gravity_device),  # type: ignore[arg-type]
        forward_axis_device=tuple(Rt @ forward_ref),  # type: ignore[arg-type]
        initial_heading_deg=heading_to_course(psi0),
        heading_source="gps_course",
        still_duration_s=spec.warmup_still_s,
        captured_at=started_at,
    )

    metadata = TripMetadata(
        trip_id=trip_id or f"sim-{uuid.uuid5(uuid.NAMESPACE_URL, str(seed)).hex[:12]}",
        started_at=started_at,
        ended_at=started_at + timedelta(seconds=float(times[-1])),
        device_model="synthetic",
        os_version="synthetic",
        app_version="geotrace-simulate",
        calibration=calibration,
        location_sample_count=len(locations),
        motion_sample_count=len(motions),
        notes=(
            "Synthetic trip generated by `geotrace simulate`. Not a real "
            "recording; used to exercise the pipeline and to provide a reference "
            "track for error metrics."
        ),
    )
    trip = Trip(metadata=metadata, locations=locations, motions=motions)
    trip.metadata.extra["simulation"] = {
        "seed": seed,
        "route_edges": [int(e) for e in route],
        "route_length_m": round(float(distance[-1]), 1),
        "cruise_speed_ms": spec.cruise_speed_ms,
        "reference_yaw_offset_deg": spec.reference_yaw_offset_deg,
    }
    return trip


def spb_synthetic_network(origin_lat: float = 59.9311, origin_lon: float = 30.3609):
    """A small hand-built street grid with a genuine fork.

    Used when no OSM cache is available so nothing in the pipeline is blocked on
    a network download. It is a toy, not a map of Saint Petersburg.
    """
    from geotrace.road_graph import RoadNetwork, build_graph_from_segments

    segments = [
        ("Prospekt A", [(-600, 0), (0, 0), (600, 40), (1200, 60)], {"highway": "primary", "maxspeed": "60"}),
        ("Ulitsa B", [(0, -700), (0, 0), (0, 700)], {"highway": "secondary", "maxspeed": "50"}),
        ("Naberezhnaya C", [(600, 40), (900, 500), (1100, 1000)], {"highway": "secondary"}),
        ("Pereulok D", [(600, 40), (700, -400), (900, -800)], {"highway": "residential"}),
        ("Ulitsa E", [(1200, 60), (1250, 600), (1100, 1000)], {"highway": "residential"}),
        ("Odnostoronniy F", [(0, 700), (600, 760), (1250, 600)], {"highway": "tertiary", "oneway": True}),
        ("Ulitsa G", [(-600, 0), (-620, 700), (0, 700)], {"highway": "residential"}),
        ("Ulitsa H", [(-600, 0), (-580, -700), (0, -700)], {"highway": "residential"}),
        ("Proezd I", [(900, -800), (1400, -700), (1500, 0), (1200, 60)], {"highway": "residential"}),
    ]
    graph, frame = build_graph_from_segments(segments, origin_lat, origin_lon)
    return RoadNetwork(graph, frame)
