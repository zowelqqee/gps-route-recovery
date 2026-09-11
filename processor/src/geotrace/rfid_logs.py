"""Import the sparse RFID + IMU validation logs in ``real_tests``.

Unlike :mod:`geotrace.live_logs`, this source has no continuous GPS stream.
RFID readers provide accurate, sparse road checkpoints.  They are therefore
kept as reference samples after a chosen cutoff and are never presented to the
tracker during the evaluated outage.

The hardware CSV also uses different units from the older vehicle logger:
accelerometer columns are m/s2 and gyro columns are rad/s.  Internally we map
them to the compact array used by ``live_logs`` (g and deg/s) so the existing,
tested gravity removal and quaternion conversion stay the single conversion
path.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from geotrace.config import G_TO_MS2
from geotrace.live_logs import (
    GPS_COURSE,
    GPS_LAT,
    GPS_LON,
    GPS_SPEED,
    GPS_TIME,
    IMU_ACCEL,
    IMU_GYRO,
    IMU_QUATERNION,
    IMU_TIME,
    ReadReport,
    estimate_mount,
    gps_to_location_samples,
    imu_geometry,
    imu_to_motion_samples,
    mount_calibration,
)
from geotrace.models import Trip, TripMetadata
from geotrace.road_graph import RoadNetwork


class RfidLogError(RuntimeError):
    """The real-test files cannot be interpreted safely."""


@dataclass(frozen=True)
class RfidAnchor:
    device_id: str
    anchor_id: str
    timestamp_ms: float
    latitude: float
    longitude: float
    course_deg: float
    host_time_ms: float
    bound_imu_time_ms: Optional[float]
    residual_ms: Optional[float]
    clamped: bool


@dataclass(frozen=True)
class RfidLap:
    index: int
    start_anchor: int
    end_anchor: int
    start_ms: float
    end_ms: float
    anchor_count: int

    @property
    def duration_s(self) -> float:
        return (self.end_ms - self.start_ms) / 1000.0


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _optional_number(value: Any) -> Optional[float]:
    result = _number(value)
    return result if math.isfinite(result) else None


def read_rfid_anchors(path: str | Path) -> list[RfidAnchor]:
    """Read and validate RFID checkpoints, ordered by their physical time."""
    source = Path(path)
    required = {
        "device_id", "host_time_ms", "anchor_id", "anchor_ts_ms",
        "heading", "lat", "lon",
    }
    anchors: list[RfidAnchor] = []
    with source.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise RfidLogError(f"{source} is missing: {', '.join(sorted(missing))}")
        for line, row in enumerate(reader, 2):
            t = _number(row.get("anchor_ts_ms"))
            host = _number(row.get("host_time_ms"))
            lat = _number(row.get("lat"))
            lon = _number(row.get("lon"))
            heading = _number(row.get("heading"))
            if not all(math.isfinite(v) for v in (t, host, lat, lon, heading)):
                raise RfidLogError(f"{source}:{line} has an invalid required number")
            if abs(lat) > 90.0 or abs(lon) > 180.0:
                raise RfidLogError(f"{source}:{line} has an invalid coordinate")
            anchors.append(RfidAnchor(
                device_id=str(row.get("device_id", "")),
                anchor_id=str(row.get("anchor_id", "")),
                timestamp_ms=t,
                latitude=lat,
                longitude=lon,
                course_deg=heading % 360.0,
                host_time_ms=host,
                bound_imu_time_ms=_optional_number(row.get("imu_sample_ts_ms")),
                residual_ms=_optional_number(row.get("dt_residual_ms")),
                clamped=str(row.get("clamped", "")).strip().lower() == "true",
            ))
    if not anchors:
        raise RfidLogError(f"{source} contains no RFID anchor")
    anchors.sort(key=lambda item: item.timestamp_ms)
    devices = {item.device_id for item in anchors}
    if len(devices) != 1:
        raise RfidLogError(
            f"{source} contains {len(devices)} devices; select one before importing"
        )
    return anchors


def find_complete_laps(
    anchors: list[RfidAnchor],
    start_anchor_id: str = "1",
    max_duration_s: float = 2400.0,
    min_anchors: int = 15,
) -> list[RfidLap]:
    """Find repeated route laps delimited by consecutive occurrences of RFID 1."""
    starts = [i for i, anchor in enumerate(anchors) if anchor.anchor_id == start_anchor_id]
    laps: list[RfidLap] = []
    for left, right in zip(starts, starts[1:]):
        duration = (anchors[right].timestamp_ms - anchors[left].timestamp_ms) / 1000.0
        count = right - left + 1
        if 0.0 < duration <= max_duration_s and count >= min_anchors:
            laps.append(RfidLap(
                index=len(laps), start_anchor=left, end_anchor=right,
                start_ms=anchors[left].timestamp_ms,
                end_ms=anchors[right].timestamp_ms, anchor_count=count,
            ))
    return laps


def _csv_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "time", "ax", "ay", "az", "gx", "gy", "gz",
            "quat_w", "quat_x", "quat_y", "quat_z",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise RfidLogError(f"{path} is missing: {', '.join(sorted(missing))}")
        yield from reader


def read_real_imu_rows(
    path: str | Path,
    t_start_ms: Optional[float] = None,
    t_end_ms: Optional[float] = None,
    report: Optional[ReadReport] = None,
) -> np.ndarray:
    """Stream the SI-unit real-test IMU into the live-logger compact layout.

    Malformed physical rows are counted and skipped.  Valid rows become
    ``timestamp, accel[g], gyro[deg/s], quaternion[wxyz]``.
    """
    source = Path(path)
    report = report if report is not None else ReadReport()
    rows: list[tuple[float, ...]] = []
    names = ("time", "ax", "ay", "az", "gx", "gy", "gz",
             "quat_w", "quat_x", "quat_y", "quat_z")
    for row in _csv_rows(source):
        report.imu_rows += 1
        values = tuple(_number(row.get(name)) for name in names)
        if len(row) != 21 or not all(math.isfinite(v) for v in values):
            report.imu_skipped += 1
            continue
        t = values[0]
        if t_start_ms is not None and t < t_start_ms:
            continue
        if t_end_ms is not None and t > t_end_ms:
            # The file is chronological; do not scan another gigabyte.
            break
        rows.append((
            t,
            values[1] / G_TO_MS2, values[2] / G_TO_MS2, values[3] / G_TO_MS2,
            math.degrees(values[4]), math.degrees(values[5]), math.degrees(values[6]),
            values[7], values[8], values[9], values[10],
        ))
    if not rows:
        raise RfidLogError(f"{source} contains no usable IMU row in the selected window")
    data = np.asarray(rows, dtype=float)
    return data[np.argsort(data[:, IMU_TIME], kind="stable")]


def _anchor_route_speeds(
    anchors: list[RfidAnchor], network: RoadNetwork, max_gap_s: float = 600.0
) -> tuple[np.ndarray, list[Optional[Any]]]:
    """Centred checkpoint speeds from legal OSM route distance / elapsed time."""
    interval = np.full(max(0, len(anchors) - 1), np.nan)
    routes: list[Optional[Any]] = []
    for i, (left, right) in enumerate(zip(anchors[:-1], anchors[1:])):
        dt = (right.timestamp_ms - left.timestamp_ms) / 1000.0
        route = None
        if 0.0 < dt <= max_gap_s:
            a = network.frame.to_local(left.latitude, left.longitude)
            b = network.frame.to_local(right.latitude, right.longitude)
            route = network.route_between(a, b, max_snap_m=30.0)
            if route is not None:
                interval[i] = route.length_m / dt
        routes.append(route)
    speed = np.full(len(anchors), np.nan)
    for i in range(len(anchors)):
        neighbours = interval[max(0, i - 1):min(len(interval), i + 1)]
        finite = neighbours[np.isfinite(neighbours)]
        if finite.size:
            speed[i] = float(np.median(finite))
    return speed, routes


@dataclass
class RfidTripBuild:
    trip: Trip
    anchors: list[RfidAnchor]
    speeds_ms: np.ndarray
    routes: list[Optional[Any]]
    cutoff_anchor: int
    end_anchor: int
    report: dict[str, Any]


def build_rfid_trip(
    imu_csv: str | Path,
    anchor_csv: str | Path,
    network: RoadNetwork,
    cutoff_anchor: int,
    end_anchor: int,
    imu_rate_hz: float = 50.0,
    pre_roll_s: float = 30.0,
) -> RfidTripBuild:
    """Build one sparse-reference experiment without leaking test anchors."""
    all_anchors = read_rfid_anchors(anchor_csv)
    if not 2 <= cutoff_anchor < end_anchor < len(all_anchors):
        raise RfidLogError("cutoff/end anchor indices are out of range")
    anchors = all_anchors[:end_anchor + 1]
    speeds, routes = _anchor_route_speeds(anchors, network)
    if not np.isfinite(speeds[:cutoff_anchor + 1]).all():
        raise RfidLogError("a calibration RFID interval could not be routed on the graph")

    reader = ReadReport()
    start_ms = anchors[0].timestamp_ms - pre_roll_s * 1000.0
    end_ms = anchors[-1].timestamp_ms
    imu = read_real_imu_rows(imu_csv, start_ms, end_ms, reader)
    t0_ms = float(min(imu[0, IMU_TIME], anchors[0].timestamp_ms))

    gps = np.column_stack([
        np.array([a.timestamp_ms for a in anchors]),
        np.array([a.latitude for a in anchors]),
        np.array([a.longitude for a in anchors]),
        speeds,
        np.array([a.course_deg for a in anchors]),
    ])
    visible_gps = gps[:cutoff_anchor + 1]
    geometry = imu_geometry(imu[imu[:, IMU_TIME] <= anchors[cutoff_anchor].timestamp_ms])
    mount = estimate_mount(geometry, visible_gps)
    calibration = mount_calibration(mount, pre_roll_s)
    if calibration is None:
        raise RfidLogError(
            "RFID warm-up could not determine the fixed mount orientation: "
            + "; ".join(mount.notes)
        )

    motions = imu_to_motion_samples(imu, t0_ms, rate_hz=imu_rate_hz)
    accuracy = np.full(len(gps), 10.0)
    locations = gps_to_location_samples(
        visible_gps, t0_ms, accuracy[:cutoff_anchor + 1]
    )
    reference = gps_to_location_samples(
        gps[cutoff_anchor + 1:], t0_ms, accuracy[cutoff_anchor + 1:]
    )
    for sample, anchor in zip(locations, anchors[:cutoff_anchor + 1]):
        sample.source_information = {
            "recorder": "rfid_checkpoint", "anchor_id": anchor.anchor_id,
        }
    for sample, anchor in zip(reference, anchors[cutoff_anchor + 1:]):
        sample.source_information = {
            "recorder": "rfid_checkpoint", "anchor_id": anchor.anchor_id,
        }

    start = datetime.fromtimestamp(t0_ms / 1000.0, tz=timezone.utc)
    end = datetime.fromtimestamp(end_ms / 1000.0, tz=timezone.utc)
    timing = np.array([
        abs(a.residual_ms) for a in anchors if a.residual_ms is not None
    ])
    provenance = {
        "source": "real_tests_rfid_imu",
        "device_id": anchors[0].device_id,
        "anchors_total": len(anchors),
        "anchors_visible": len(locations),
        "anchors_withheld": len(reference),
        "cutoff_anchor_id": anchors[cutoff_anchor].anchor_id,
        "reader": reader.to_json(),
        "mount_estimate": mount.to_json(),
        "anchor_binding_residual_ms": {
            "median": round(float(np.median(timing)), 1) if timing.size else None,
            "max": round(float(np.max(timing)), 1) if timing.size else None,
            "note": "diagnostic only; physical anchor_ts_ms is used",
        },
        "reference_is": "withheld_sparse_rfid_checkpoints",
        "calibration_intervals": [
            {
                "start_s": round((anchors[i].timestamp_ms - t0_ms) / 1000.0, 6),
                "end_s": round((anchors[i + 1].timestamp_ms - t0_ms) / 1000.0, 6),
                "distance_m": round(float(routes[i].length_m), 6),
            }
            for i in range(cutoff_anchor)
            if routes[i] is not None
        ],
    }
    metadata = TripMetadata(
        trip_id=f"real-rfid-{start:%Y-%m-%d}-a{cutoff_anchor}-{end_anchor}",
        started_at=start,
        ended_at=end,
        device_model="vehicle IMU 100 Hz + sparse RFID checkpoints",
        calibration=calibration,
        location_sample_count=len(locations),
        motion_sample_count=len(motions),
        notes=(
            "RFID checkpoints through the cutoff calibrate the sensor. Later "
            "RFID checkpoints are withheld and used only for sparse scoring."
        ),
        extra={"rfid_import": provenance},
    )
    return RfidTripBuild(
        trip=Trip(metadata=metadata, locations=locations, motions=motions,
                  reference_locations=reference),
        anchors=anchors,
        speeds_ms=speeds,
        routes=routes,
        cutoff_anchor=cutoff_anchor,
        end_anchor=end_anchor,
        report=provenance,
    )
