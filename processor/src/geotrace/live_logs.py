"""Import the vehicle logger's day files into a trip directory.

The recorder here is not the iPhone. It is a box wired into the car that writes
two CSV files per day, both stamped with UTC epoch milliseconds:

    live_logs/gps_logs/<date>_GPS_logs.csv    timestamp_ms,lat,lon,speed,course
    live_logs/imu_logs/<date>_IMU_logs.csv    timestamp,temp,pressure,imu_ms,
                                              acc_*,gyr_*,mag_*,roll,pitch,yaw,quat_*

and, beside them, the raw NMEA the receiver emitted, which is the only place a
per-fix quality figure survives.

Three things differ from the phone export and each one is a decision, not a
format detail:

* **The file is a day, not a trip.** The logger keeps running between drives,
  so a day file holds several sessions separated by hours of nothing. They are
  split on gaps in the IMU clock and offered one at a time.

* **The accelerometer reports specific force, not CoreMotion's
  `userAcceleration`.** At rest it reads +1 g on Z; CoreMotion reads ~0 there
  and puts gravity in a separate channel. Gravity is removed here with the
  logger's own attitude quaternion, which measures to 0.003 g residual on a
  standing car — the tilt of the mount is real and the quaternion knows about
  it.

* **The gyro is in degrees per second** and the world frame its yaw integrates
  in is arbitrary: `yaw` is the plain integral of `gyr_z`, with no magnetometer
  correction anywhere (correlation with the column is 1.000). It is a relative
  heading that drifts, which is exactly what the trip format's
  `MountCalibration` exists to pin down.

The GPS in these logs runs for the whole drive. That is *not* how the trip is
written: only the opening `gps_warmup_s` seconds go into `samples.jsonl`, and
every later fix is put aside in `reference-samples.jsonl`. The reconstruction
then sees the case this repository is about — a fix to start from and then
nothing — while the fixes it was not allowed to see stay available to score it
against. That reference is real GPS, so it is a few metres wide itself; it is
a good yardstick, not a survey.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import numpy as np

from geotrace.config import G_TO_MS2, GPSQualityConfig
from geotrace.coordinates import haversine_m, wrap_angle
from geotrace.models import (
    LocationSample,
    MotionSample,
    MountCalibration,
    Trip,
    TripMetadata,
)
from geotrace.motion_model import quaternion_to_matrix

GPS_COLUMNS = ("timestamp_ms", "lat", "lon", "speed", "course")
"""The GPS columns, in the order every `gps` array below is laid out. `speed` is
metres per second and `course` is degrees clockwise from true north - both
checked against the positions themselves, not taken from the header."""

GPS_TIME, GPS_LAT, GPS_LON, GPS_SPEED, GPS_COURSE = range(5)
IMU_COLUMNS = (
    "timestamp", "acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z",
    "quat_w", "quat_x", "quat_y", "quat_z",
)
"""The IMU columns this module keeps, in the order every `imu` array below is
laid out. The day file has nine more (temperature, pressure, the logger's own
uptime counter, the magnetometer, the Euler angles); none is used, and carrying
them would double the memory a session needs. Build such an array with
`read_imu_rows`, which selects these by name - the day file's column order is
not assumed anywhere."""

IMU_TIME = 0
IMU_ACCEL = slice(1, 4)
IMU_GYRO = slice(4, 7)
IMU_QUATERNION = slice(7, 11)

DEFAULT_SESSION_GAP_S = 60.0
"""IMU silence longer than this ends a session. The logger's own restarts leave
holes of hours; a dropped USB frame leaves 4-5 s. Nothing in between has been
observed, so the exact value is not delicate."""

DEFAULT_IMU_RATE_HZ = 50.0
"""The logger samples at 100 Hz. The filters bin to `filter_dt_s` (0.1 s) and
the phone recorded at 50 Hz, so keeping every second frame costs nothing that
the binning would not have thrown away and halves the trip on disk."""

UERE_M = 4.0
"""User-equivalent range error multiplied by HDOP to get a horizontal accuracy.
A conventional single-frequency figure. It is a stand-in for the accuracy
CoreLocation reports directly, and it is recorded as such in the metadata."""

MAX_PLAUSIBLE_SPEED_MS = 60.0
"""~216 km/h between two fixes. Above it the receiver teleported. This gates
nothing; it is counted so that a session listing shows which drives the
receiver mangled, before one of them is chosen to reconstruct."""

FALLBACK_ACCURACY_M = 8.0
"""Assumed accuracy when no NMEA sentence can be matched to a fix. Deliberately
pessimistic: an invented accuracy must not make the filter trust a fix more
than a measured one."""


class LiveLogError(RuntimeError):
    """A problem with the day files the user can fix."""


# ------------------------------------------------------------------ reading


def _to_float(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


@dataclass
class ReadReport:
    """What the reader had to skip. Ends up in the trip metadata."""

    gps_rows: int = 0
    gps_skipped: int = 0
    imu_rows: int = 0
    imu_skipped: int = 0
    nmea_matched: int = 0
    nmea_unmatched: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "gps_rows": self.gps_rows,
            "gps_skipped": self.gps_skipped,
            "imu_rows": self.imu_rows,
            "imu_skipped": self.imu_skipped,
            "nmea_matched": self.nmea_matched,
            "nmea_unmatched": self.nmea_unmatched,
        }


def _iter_csv(path: Path, required: Sequence[str]) -> Iterator[dict[str, str]]:
    """Stream a logger CSV.

    The files are written while the car is running and the last line of a day
    is routinely half a row, so a short or unparseable line is skipped rather
    than being fatal. They are also up to a gigabyte, which is why this streams
    instead of loading.
    """
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise LiveLogError(f"{path} is empty")
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise LiveLogError(
                f"{path} is missing column(s) {', '.join(missing)}. "
                f"Found: {', '.join(reader.fieldnames)}"
            )
        for row in reader:
            yield row


def read_gps_rows(path: str | Path) -> tuple[np.ndarray, ReadReport]:
    """Read the GPS CSV into an (N, 5) array of t_ms, lat, lon, speed, course."""
    report = ReadReport()
    rows: list[tuple[float, float, float, float, float]] = []
    for row in _iter_csv(Path(path), GPS_COLUMNS):
        report.gps_rows += 1
        values = tuple(_to_float(row.get(c, "")) for c in GPS_COLUMNS)
        if not all(math.isfinite(v) for v in values[:3]):
            report.gps_skipped += 1
            continue
        if abs(values[1]) > 90.0 or abs(values[2]) > 180.0:
            report.gps_skipped += 1
            continue
        rows.append(values)  # type: ignore[arg-type]
    if not rows:
        raise LiveLogError(f"{path} contains no usable GPS row")
    data = np.array(rows, dtype=float)
    order = np.argsort(data[:, 0], kind="stable")
    return data[order], report


def read_imu_rows(
    path: str | Path,
    t_start_ms: Optional[float] = None,
    t_end_ms: Optional[float] = None,
    report: Optional[ReadReport] = None,
) -> np.ndarray:
    """Read the IMU CSV into an (N, 11) array, optionally clipped to a window.

    The window is applied while reading: a day of 100 Hz IMU is 900 MB on disk
    and there is no reason to hold a whole day in memory to reconstruct twenty
    minutes of it.
    """
    report = report if report is not None else ReadReport()
    rows: list[tuple[float, ...]] = []
    for row in _iter_csv(Path(path), IMU_COLUMNS):
        report.imu_rows += 1
        t = _to_float(row.get("timestamp", ""))
        if not math.isfinite(t):
            report.imu_skipped += 1
            continue
        if t_start_ms is not None and t < t_start_ms:
            continue
        if t_end_ms is not None and t > t_end_ms:
            continue
        values = tuple(_to_float(row.get(c, "")) for c in IMU_COLUMNS)
        if not all(math.isfinite(v) for v in values):
            report.imu_skipped += 1
            continue
        rows.append(values)
    if not rows:
        raise LiveLogError(f"{path} contains no usable IMU row in the requested window")
    data = np.array(rows, dtype=float)
    order = np.argsort(data[:, 0], kind="stable")
    return data[order]


def read_imu_timeline(path: str | Path) -> np.ndarray:
    """Just the IMU timestamps, for splitting a day into sessions cheaply."""
    times: list[float] = []
    for row in _iter_csv(Path(path), ("timestamp",)):
        t = _to_float(row.get("timestamp", ""))
        if math.isfinite(t):
            times.append(t)
    if not times:
        raise LiveLogError(f"{path} contains no usable timestamp")
    return np.sort(np.array(times, dtype=float))


_GGA = re.compile(r"^\$G[NPLA]GGA,(\d{2})(\d{2})(\d{2}(?:\.\d+)?),[^,]*,[^,]*,[^,]*,[^,]*,(\d),(\d+),([\d.]*)")


def read_nmea_quality(path: str | Path) -> np.ndarray:
    """Per-second (time-of-day, fix quality, satellites, HDOP) from GGA lines.

    The CSV keeps position, speed and course but drops every quality field, so
    a fix in it carries no accuracy at all. GGA still has the fix type, the
    satellite count and HDOP, and its time-of-day is the same UTC clock the CSV
    timestamps are in, which is enough to put them back together.
    """
    out: list[tuple[float, float, float, float]] = []
    with Path(path).open("r", encoding="ascii", errors="replace") as handle:
        for line in handle:
            match = _GGA.match(line.strip())
            if match is None:
                continue
            hh, mm, ss, quality, sats, hdop = match.groups()
            tod = int(hh) * 3600.0 + int(mm) * 60.0 + float(ss)
            hdop_value = _to_float(hdop)
            out.append((tod, float(quality), float(sats), hdop_value))
    if not out:
        return np.zeros((0, 4))
    data = np.array(out, dtype=float)
    return data[np.argsort(data[:, 0], kind="stable")]


def accuracy_from_nmea(
    t_ms: np.ndarray, quality: np.ndarray, report: Optional[ReadReport] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal accuracy per fix, from the nearest GGA within 0.5 s.

    Returns (accuracy_m, satellites). Fixes with no matching sentence get
    `FALLBACK_ACCURACY_M` and a satellite count of NaN.
    """
    n = len(t_ms)
    accuracy = np.full(n, FALLBACK_ACCURACY_M)
    sats = np.full(n, np.nan)
    if quality.size == 0:
        if report is not None:
            report.nmea_unmatched += n
        return accuracy, sats

    tod = np.mod(t_ms / 1000.0, 86400.0)
    idx = np.clip(np.searchsorted(quality[:, 0], tod), 1, len(quality) - 1)
    left = np.abs(tod - quality[idx - 1, 0])
    right = np.abs(quality[idx, 0] - tod)
    pick = np.where(left <= right, idx - 1, idx)
    delta = np.minimum(left, right)
    matched = (delta <= 0.5) & (quality[pick, 1] > 0) & np.isfinite(quality[pick, 3])
    accuracy[matched] = np.clip(quality[pick, 3][matched] * UERE_M, 1.0, 200.0)
    sats[matched] = quality[pick, 2][matched]
    if report is not None:
        report.nmea_matched += int(matched.sum())
        report.nmea_unmatched += int((~matched).sum())
    return accuracy, sats


# ----------------------------------------------------------------- sessions


@dataclass
class Session:
    """One continuous run of the logger inside a day file."""

    index: int
    start_ms: float
    end_ms: float
    imu_samples: int
    gps_fixes: int = 0
    moving_s: float = 0.0
    distance_m: float = 0.0
    gps_jumps: int = 0
    """Fix-to-fix steps implying more than `MAX_PLAUSIBLE_SPEED_MS`.

    `distance_m` sums the raw steps, so a teleport is counted as distance
    travelled - one session in the sample days reads 182 km across 21 minutes
    of driving. Rather than quietly clean that up, the count sits beside the
    distance: a session with jumps is one where the receiver, not the
    arithmetic, is the problem, and choosing it is a decision to make
    knowingly."""

    @property
    def duration_s(self) -> float:
        return (self.end_ms - self.start_ms) / 1000.0

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": _iso_ms(self.start_ms),
            "end": _iso_ms(self.end_ms),
            "duration_s": round(self.duration_s, 1),
            "imu_samples": self.imu_samples,
            "gps_fixes": self.gps_fixes,
            "moving_s": round(self.moving_s, 1),
            "gps_distance_km": round(self.distance_m / 1000.0, 2),
            "gps_jumps": self.gps_jumps,
        }


def _iso_ms(t_ms: float) -> str:
    return (
        datetime.fromtimestamp(t_ms / 1000.0, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def find_sessions(
    imu_times_ms: np.ndarray,
    gps: Optional[np.ndarray] = None,
    gap_s: float = DEFAULT_SESSION_GAP_S,
    min_duration_s: float = 60.0,
) -> list[Session]:
    """Split a day of IMU timestamps into logger sessions."""
    if imu_times_ms.size == 0:
        return []
    breaks = np.where(np.diff(imu_times_ms) > gap_s * 1000.0)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(imu_times_ms) - 1]))

    sessions: list[Session] = []
    for start, end in zip(starts, ends):
        t0, t1 = float(imu_times_ms[start]), float(imu_times_ms[end])
        if (t1 - t0) / 1000.0 < min_duration_s:
            continue
        session = Session(
            index=len(sessions),
            start_ms=t0,
            end_ms=t1,
            imu_samples=int(end - start + 1),
        )
        if gps is not None and gps.size:
            mask = (gps[:, GPS_TIME] >= t0) & (gps[:, GPS_TIME] <= t1)
            fixes = gps[mask]
            session.gps_fixes = int(len(fixes))
            if len(fixes) > 1:
                moving = fixes[:, GPS_SPEED] > 1.0
                session.moving_s = float(
                    np.sum(np.diff(fixes[:, GPS_TIME])[moving[:-1]]) / 1000.0
                )
                steps = np.array([
                    haversine_m(a[GPS_LAT], a[GPS_LON], b[GPS_LAT], b[GPS_LON])
                    for a, b in zip(fixes[:-1], fixes[1:])
                ])
                session.distance_m = float(np.sum(steps))
                gaps = np.diff(fixes[:, GPS_TIME]) / 1000.0
                session.gps_jumps = int(
                    np.sum(steps > MAX_PLAUSIBLE_SPEED_MS * np.maximum(gaps, 0.05))
                )
        sessions.append(session)
    return sessions


def motion_start_ms(
    gps: np.ndarray, session: Session, speed_ms: float = 1.5, hold_s: float = 3.0
) -> Optional[float]:
    """When the car in this session actually starts moving.

    A logger session usually opens with the car parked and the engine idling,
    sometimes for many minutes. Those minutes are not free: they are where the
    accelerometer and gyro bias are measured. But a quarter of an hour of them
    is only cost, so the import keeps a fixed pre-roll and drops the rest.
    """
    mask = (gps[:, GPS_TIME] >= session.start_ms) & (gps[:, GPS_TIME] <= session.end_ms)
    fixes = gps[mask]
    if len(fixes) < 2:
        return None
    moving = fixes[:, GPS_SPEED] >= speed_ms
    start: Optional[int] = None
    for i, flag in enumerate(moving):
        if flag and start is None:
            start = i
        elif not flag:
            start = None
            continue
        if start is not None and (fixes[i, GPS_TIME] - fixes[start, GPS_TIME]) / 1000.0 >= hold_s:
            return float(fixes[start, GPS_TIME])
    return None


# -------------------------------------------------------------- conversion


def _gravity_device(quaternion: np.ndarray) -> np.ndarray:
    """CoreMotion's `gravity`: the gravity vector in the device frame, in g.

    The logger's accelerometer reports specific force, so a device standing
    still reads +1 g along whichever axis points up. CoreMotion splits the same
    measurement into `gravity` (pointing down, magnitude 1) and
    `userAcceleration`, and the trip format is CoreMotion's. The relation is

        specific_force = userAcceleration - gravity_device

    so `gravity_device = -R(q)^T z_world`, and `userAcceleration` is the
    specific force with that subtracted back off.
    """
    rot = quaternion_to_matrix(quaternion)
    return -rot[2, :].copy()


def imu_to_motion_samples(
    imu: np.ndarray, t0_ms: float, rate_hz: float = DEFAULT_IMU_RATE_HZ
) -> list[MotionSample]:
    """Convert logger rows into `MotionSample`s on the trip's clock.

    Units in, units out: acceleration g -> g with gravity removed, gyro deg/s
    -> rad/s, quaternion straight through (it is already the (w, x, y, z)
    device -> world convention `quaternion_to_matrix` expects; its yaw matches
    the logger's own `yaw` column to 0.02 deg).
    """
    if imu.size == 0:
        return []
    stride = 1
    if rate_hz > 0 and len(imu) > 1:
        spacing_s = float(np.median(np.diff(imu[:, IMU_TIME]))) / 1000.0
        if spacing_s > 0:
            stride = max(1, int(round(1.0 / (rate_hz * spacing_s))))
    rows = imu[::stride]

    samples: list[MotionSample] = []
    for row in rows:
        t_ms = float(row[IMU_TIME])
        quaternion = tuple(float(v) for v in row[IMU_QUATERNION])
        gravity = _gravity_device(np.array(quaternion))
        acc = np.asarray(row[IMU_ACCEL], dtype=float)
        samples.append(
            MotionSample(
                monotonic_time=(t_ms - t0_ms) / 1000.0,
                user_acceleration_g=tuple(acc + gravity),  # type: ignore[arg-type]
                rotation_rate=tuple(np.radians(row[IMU_GYRO])),  # type: ignore[arg-type]
                gravity=tuple(gravity),  # type: ignore[arg-type]
                quaternion=quaternion,
                wall_time=datetime.fromtimestamp(t_ms / 1000.0, tz=timezone.utc),
            )
        )
    return samples


def gps_to_location_samples(
    gps: np.ndarray,
    t0_ms: float,
    accuracy_m: np.ndarray,
    satellites: Optional[np.ndarray] = None,
) -> list[LocationSample]:
    """Convert logger fixes into `LocationSample`s on the trip's clock.

    The course below the filters' own trust threshold is marked with a negative
    accuracy rather than dropped: a parked receiver still emits a course, and it
    is noise. The trip format already has a way to say "this field is present
    and not to be believed", and the gates downstream read it.
    """
    samples: list[LocationSample] = []
    for i, row in enumerate(gps):
        t_ms = float(row[GPS_TIME])
        lat, lon = float(row[GPS_LAT]), float(row[GPS_LON])
        speed, course = float(row[GPS_SPEED]), float(row[GPS_COURSE])
        source: dict[str, Any] = {"recorder": "vehicle_logger"}
        if satellites is not None and i < len(satellites) and math.isfinite(satellites[i]):
            source["satellites"] = int(satellites[i])
        samples.append(
            LocationSample(
                monotonic_time=(t_ms - t0_ms) / 1000.0,
                latitude=lat,
                longitude=lon,
                wall_time=datetime.fromtimestamp(t_ms / 1000.0, tz=timezone.utc),
                horizontal_accuracy=float(accuracy_m[i]),
                speed=speed if math.isfinite(speed) and speed >= 0.0 else None,
                # The receiver's course is meaningless standing still; the trip
                # format says so with a negative accuracy rather than by
                # dropping the field.
                course=course if math.isfinite(course) else None,
                course_accuracy=(
                    5.0 if speed >= GPSQualityConfig.min_speed_for_course_ms else -1.0
                ),
                speed_accuracy=1.0,
                source_information=source,
            )
        )
    return samples


# ------------------------------------------------------------- calibration


def _boxcar(values: np.ndarray, width: int) -> np.ndarray:
    """Moving average, used to take road vibration out of a 100 Hz channel."""
    if width <= 1 or values.size < width:
        return np.asarray(values, dtype=float)
    kernel = np.ones(width) / width
    return np.convolve(np.asarray(values, dtype=float), kernel, mode="same")


@dataclass
class ImuGeometry:
    """The per-sample quantities every calibration step below is built from."""

    t: np.ndarray
    """Seconds, on the IMU clock."""

    accel_xy: np.ndarray
    """Horizontal acceleration in the gyro's world frame, m/s^2."""

    yaw: np.ndarray
    """Device yaw in the gyro's world frame, radians. Arbitrary origin."""

    yaw_rate: np.ndarray
    """Vertical component of the rotation rate, rad/s, CCW positive."""

    quaternion: np.ndarray
    """The (w, x, y, z) attitude of every row, kept so that a calibration can
    name the exact frame the anchor heading was measured in."""


def imu_geometry(imu: np.ndarray) -> ImuGeometry:
    """Rotate every IMU row into the gyro's world frame."""
    n = len(imu)
    accel = np.empty((n, 2))
    yaw = np.empty(n)
    yaw_rate = np.empty(n)
    for i, row in enumerate(imu):
        quaternion = row[IMU_QUATERNION]
        rot = quaternion_to_matrix(quaternion)
        gravity = -rot[2, :]
        accel[i] = (rot @ ((row[IMU_ACCEL] + gravity) * G_TO_MS2))[:2]
        yaw_rate[i] = float(rot @ np.radians(row[IMU_GYRO]) @ np.array([0.0, 0.0, 1.0]))
        yaw[i] = math.atan2(rot[1, 0], rot[0, 0])
    return ImuGeometry(
        t=imu[:, IMU_TIME] / 1000.0,
        accel_xy=accel,
        yaw=yaw,
        yaw_rate=yaw_rate,
        quaternion=imu[:, IMU_QUATERNION].copy(),
    )


@dataclass
class ClockOffset:
    """How far the GPS file's clock runs behind the IMU file's."""

    seconds: float = 0.0
    correlation: float = 0.0
    source: str = "assumed_zero"
    note: Optional[str] = None

    def to_json(self) -> dict[str, Any]:
        return {
            "seconds": round(self.seconds, 3),
            "correlation": round(self.correlation, 3),
            "source": self.source,
            "note": self.note,
        }


def estimate_clock_offset(
    geometry: ImuGeometry,
    gps: np.ndarray,
    search_s: float = 15.0,
    coarse_step_s: float = 0.25,
    min_speed_ms: float = 4.0,
) -> ClockOffset:
    """Find the lag between the two files by matching turns to turns.

    **The two logs are not on the same clock.** The GPS file runs several
    seconds behind the IMU file — 4.5 to 5.0 s on every day checked, stable
    inside a session and sharply peaked (the correlation below falls from 0.77
    at the peak to 0.23 two seconds away). Left uncorrected it is the single
    largest error in the import: at 12 m/s it puts the seeding position 60 m
    down the road from where the car was, ties the initial heading to the wrong
    moment, and destroys the mount estimate outright — with the lag in place
    the acceleration coherence measured 0.23, and with it removed, 0.99.

    Turn rate is what makes it observable. The gyro's vertical rate and the
    derivative of the GPS course are the same physical quantity measured by
    two devices, so the lag that lines them up is the lag between the clocks.
    Speed would work too but is far too smooth to locate a peak.
    """
    result = ClockOffset()
    moving = gps[:, GPS_SPEED] > min_speed_ms
    if int(moving.sum()) < 100 or geometry.t.size < 600:
        result.note = "not enough driving to line the two clocks up"
        return result

    t_gps = gps[:, GPS_TIME] / 1000.0
    course = np.unwrap(np.radians(90.0 - gps[:, GPS_COURSE]))
    turn_rate = np.gradient(course[moving], t_gps[moving])
    width = _window_width(geometry.t, 0.5)
    yaw_rate = _boxcar(geometry.yaw_rate, width)

    # Only compare where a shift of the whole search range still lands inside
    # both records, so every candidate lag is scored on the same samples.
    inside = (geometry.t >= t_gps[moving][0] + search_s + 1.0) & (
        geometry.t <= t_gps[moving][-1] - search_s - 1.0
    )
    inside[:width] = False
    inside[-width:] = False
    if int(inside.sum()) < 300:
        result.note = (
            f"the overlap between the two files is shorter than the "
            f"{2 * search_s:g} s search window"
        )
        return result

    def score(lag: float) -> float:
        rate = _boxcar(np.interp(geometry.t, t_gps[moving] + lag, turn_rate), width)
        speed = np.interp(geometry.t, t_gps + lag, gps[:, GPS_SPEED])
        mask = inside & (speed > min_speed_ms)
        if int(mask.sum()) < 300:
            return float("-inf")
        if np.std(rate[mask]) < 1e-9 or np.std(yaw_rate[mask]) < 1e-9:
            return float("-inf")
        value = float(np.corrcoef(rate[mask], yaw_rate[mask])[0, 1])
        return value if math.isfinite(value) else float("-inf")

    grid = np.arange(-search_s, search_s + 1e-9, coarse_step_s)
    scores = [score(float(lag)) for lag in grid]
    best = int(np.argmax(scores))
    if scores[best] <= 0.0:
        result.note = "no lag lined the turns up; the clocks may be unrelated"
        return result
    fine = np.arange(
        grid[best] - coarse_step_s, grid[best] + coarse_step_s + 1e-9, coarse_step_s / 5.0
    )
    fine_scores = [score(float(lag)) for lag in fine]
    winner = int(np.argmax(fine_scores))

    result.seconds = float(fine[winner])
    result.correlation = float(fine_scores[winner])
    result.source = "gyro_vs_course_rate"
    if result.correlation < 0.35:
        result.note = (
            f"the turns line up only weakly (correlation {result.correlation:.2f}); "
            "the window may contain too little steering to place the lag "
            "confidently. Lengthen --gps-warmup or pass --clock-offset."
        )
    return result


def _window_width(t: np.ndarray, seconds: float) -> int:
    """Boxcar width in samples for a given number of seconds."""
    if t.size < 2:
        return 1
    spacing = float(np.median(np.diff(t)))
    if spacing <= 0.0:
        return 1
    return max(1, int(round(seconds / spacing)) | 1)


@dataclass
class MountEstimate:
    """The two angles that tie the logger's arbitrary yaw frame to the map."""

    world_yaw_offset_rad: float = 0.0
    """Rotation from the gyro's world frame onto local East/North."""

    initial_heading_rad: float = 0.0
    """Vehicle heading at the anchor fix, in the E/N convention."""

    mount_yaw_rad: float = 0.0
    """Angle from the logger's +x axis to the vehicle's forward axis. On the
    recordings so far this comes out within a degree of zero: the box is bolted
    in facing forwards. It is measured rather than assumed, because a box
    turned in its bracket would otherwise mirror every manoeuvre."""

    samples: int = 0
    coherence: float = 0.0
    """0..1. How consistently the measured and predicted acceleration vectors
    hold the same angle between them. Below ~0.7 the mount angle is a guess."""

    heading_spread_deg: float = 180.0
    quality: str = "unusable"
    reference_quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "world_yaw_offset_deg": round(math.degrees(self.world_yaw_offset_rad), 2),
            "initial_course_deg": round(
                float(np.mod(90.0 - math.degrees(self.initial_heading_rad), 360.0)), 2
            ),
            "initial_heading_psi_deg": round(
                math.degrees(wrap_angle(self.initial_heading_rad)), 2
            ),
            "mount_yaw_deg": round(math.degrees(self.mount_yaw_rad), 2),
            "samples": self.samples,
            "coherence": round(self.coherence, 3),
            "heading_spread_deg": round(self.heading_spread_deg, 2),
            "quality": self.quality,
            "notes": list(self.notes),
        }


def _circular_stats(angles: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Weighted circular mean and the spread around it, both in radians."""
    if angles.size == 0:
        return 0.0, math.pi
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0.0, math.pi
    c = float(np.sum(weights * np.cos(angles))) / total
    s = float(np.sum(weights * np.sin(angles))) / total
    mean = math.atan2(s, c)
    r = min(1.0, math.hypot(c, s))
    spread = math.sqrt(-2.0 * math.log(r)) if r > 1e-9 else math.pi
    return mean, spread


def estimate_mount(
    geometry: ImuGeometry,
    gps: np.ndarray,
    clock_offset_s: float = 0.0,
    min_speed_ms: float = 3.0,
    min_accel_ms2: float = 0.4,
) -> MountEstimate:
    """Tie the logger's yaw frame to the map, using the GPS warm-up window.

    Two angles, observable in two different ways:

    * **The mount angle** comes from acceleration, and the trick is what it is
      compared against. Differentiating the GPS *velocity vector* would work in
      principle and does not work in practice — at 10 Hz that derivative is
      mostly receiver noise. But a car's acceleration in its own frame is two
      quantities that are each measured cleanly: `dv/dt` from the GPS *speed*,
      which is smooth, and the centripetal term `v * yaw_rate`, whose yaw rate
      comes from the gyro itself. Predicting the acceleration vector that way
      and comparing its angle with the measured one gives the rotation from the
      device's own axes to the car's, with a coherence of 0.99 on real data.

    * **The initial heading** comes from the GPS course against the gyro's
      integrated yaw. That difference is constant except for gyro drift, so it
      is only meaningful near the start — which is where it is used.

    Their difference is the rotation from the gyro's world frame onto East and
    North, which is what the reconstruction needs and cannot recover on its own.
    """
    estimate = MountEstimate()
    if gps.size == 0 or geometry.t.size < 300:
        estimate.notes.append("no overlapping GPS and IMU in the warm-up window")
        return estimate

    t = geometry.t
    t_gps = gps[:, GPS_TIME] / 1000.0 + clock_offset_s
    width = _window_width(t, 1.0)

    speed = _boxcar(np.interp(t, t_gps, gps[:, GPS_SPEED]), width)
    yaw_rate = _boxcar(geometry.yaw_rate, width)
    longitudinal = np.gradient(speed, t)
    predicted = longitudinal + 1j * (speed * yaw_rate)

    # The measured vector, turned into the device's own yaw-referenced frame so
    # that what is left is the fixed angle of the mount.
    measured = (
        _boxcar(geometry.accel_xy[:, 0], width)
        + 1j * _boxcar(geometry.accel_xy[:, 1], width)
    ) * np.exp(-1j * geometry.yaw)

    edge = 3 * width
    core = np.zeros(t.size, dtype=bool)
    core[edge:-edge] = True
    usable = (
        core
        & (speed > min_speed_ms)
        & (np.abs(predicted) > min_accel_ms2)
        & (np.abs(measured) > min_accel_ms2)
    )
    if int(usable.sum()) < 200:
        estimate.notes.append(
            f"only {int(usable.sum())} instants in the warm-up window carried a "
            "measurable acceleration; the mount angle is not observable. Give "
            "the warm-up more driving with --gps-warmup."
        )
        return estimate

    product = measured[usable] * np.conj(predicted[usable])
    total = complex(np.sum(product))
    estimate.mount_yaw_rad = float(wrap_angle(math.atan2(total.imag, total.real)))
    estimate.samples = int(usable.sum())
    estimate.coherence = float(abs(total) / np.sum(np.abs(product)))

    # Heading: the GPS course against the gyro's own yaw, over moving fixes.
    moving = gps[:, GPS_SPEED] >= min_speed_ms
    if not np.any(moving):
        estimate.notes.append(
            f"the car never drove faster than {min_speed_ms:.0f} m/s during the "
            "warm-up window, so there is no trustworthy course to start from"
        )
        return estimate
    index = np.clip(np.searchsorted(t, t_gps[moving]), 0, t.size - 1)
    difference = np.radians(90.0 - gps[moving, GPS_COURSE]) - geometry.yaw[index]
    heading_offset, spread = _circular_stats(difference, np.ones(int(moving.sum())))
    estimate.heading_spread_deg = float(math.degrees(spread))

    anchor = int(index[0])
    estimate.reference_quaternion = tuple(
        float(v) for v in geometry.quaternion[anchor]
    )  # type: ignore[assignment]
    estimate.initial_heading_rad = float(wrap_angle(geometry.yaw[anchor] + heading_offset))
    estimate.world_yaw_offset_rad = float(
        wrap_angle(heading_offset - estimate.mount_yaw_rad)
    )

    if estimate.coherence >= 0.85 and estimate.samples >= 1000:
        estimate.quality = "good"
    elif estimate.coherence >= 0.7:
        estimate.quality = "usable"
    else:
        estimate.quality = "poor"
        estimate.notes.append(
            f"the measured and predicted acceleration hold a consistent angle "
            f"only {estimate.coherence:.2f} of the time; the mount rotation is a "
            "weak estimate and the reconstruction will inherit that error"
        )
    return estimate



def mount_calibration(
    estimate: MountEstimate, still_duration_s: float
) -> Optional[MountCalibration]:
    """Express a mount estimate in the trip format's calibration record.

    `forward_axis_device` is what the reconstruction actually consumes: the
    vehicle's forward direction written in device coordinates. Rotating it by
    the reference quaternion returns the car's bearing in the gyro world frame,
    and the difference between that and the true initial heading is precisely
    the rotation that puts the world frame back on East/North.
    """
    if estimate.quality == "unusable":
        return None
    bearing = wrap_angle(estimate.initial_heading_rad - estimate.world_yaw_offset_rad)
    forward_world = np.array([math.cos(bearing), math.sin(bearing), 0.0])
    rot = quaternion_to_matrix(np.array(estimate.reference_quaternion))
    forward_device = rot.T @ forward_world
    return MountCalibration(
        reference_quaternion=estimate.reference_quaternion,
        gravity_device=tuple(_gravity_device(np.array(estimate.reference_quaternion))),  # type: ignore[arg-type]
        forward_axis_device=tuple(float(v) for v in forward_device),  # type: ignore[arg-type]
        initial_heading_deg=float(np.mod(90.0 - math.degrees(estimate.initial_heading_rad), 360.0)),
        # Not "gps_course": that means one fix's course, and the pipeline reads
        # it as a hint it can improve on. This heading was measured across a
        # whole window of fixes against the gyro, so it is the better number
        # and `initial_heading` is told to take it as given.
        heading_source="gps_course_window",
        # The logger fuses its attitude by steering "down" toward the measured
        # specific force, so the quaternion leans under sustained acceleration
        # and the gravity subtraction eats it. Saying so here is what lets the
        # pipeline undo it; see `motion_model.leveling_correction`.
        attitude_source="accelerometer_levelled_ahrs",
        still_duration_s=float(still_duration_s),
        captured_at=datetime.now(timezone.utc),
    )


# ------------------------------------------------------------------ import


@dataclass
class ImportSpec:
    """Everything the import decides, in one place so it can be recorded."""

    gps_dir: Path
    imu_dir: Path
    day: str
    session_index: int = 0
    gps_warmup_s: float = 120.0
    keep_all_gps: bool = False
    imu_rate_hz: float = DEFAULT_IMU_RATE_HZ
    pre_roll_s: float = 20.0
    max_duration_s: Optional[float] = None
    session_gap_s: float = DEFAULT_SESSION_GAP_S
    trip_id: Optional[str] = None
    clock_align: str = "warmup"
    """"warmup" measures the GPS/IMU clock offset inside the window the
    reconstruction is allowed to see, which keeps the import honest. "session"
    uses the whole drive, including fixes that were withheld — a better
    estimate of a constant that belongs to the logger, but it does borrow from
    data the algorithm never gets, so it is recorded as such. "none" trusts the
    two files to already agree, which on this recorder they do not."""

    clock_align_s: Optional[float] = None
    """Length of the alignment window, when it should differ from the warm-up."""

    clock_offset_s: Optional[float] = None
    """A known offset, in seconds added to the GPS clock. Skips the search."""

    @property
    def gps_csv(self) -> Path:
        return self.gps_dir / f"{self.day}_GPS_logs.csv"

    @property
    def imu_csv(self) -> Path:
        return self.imu_dir / f"{self.day}_IMU_logs.csv"

    @property
    def nmea_txt(self) -> Path:
        return self.gps_dir / f"{self.day}_GPS_GNRMC_original_logs.txt"


def available_days(root: str | Path) -> list[str]:
    """Days that have both a GPS and an IMU file under `root`."""
    root = Path(root)
    days = set()
    for path in sorted((root / "gps_logs").glob("*_GPS_logs.csv")):
        day = path.name.split("_")[0]
        if (root / "imu_logs" / f"{day}_IMU_logs.csv").exists():
            days.add(day)
    return sorted(days)


def describe_sessions(spec: ImportSpec) -> list[Session]:
    """List the drives in a day without converting anything."""
    gps, _ = read_gps_rows(spec.gps_csv)
    return find_sessions(read_imu_timeline(spec.imu_csv), gps, gap_s=spec.session_gap_s)


def build_trip(spec: ImportSpec) -> tuple[Trip, dict[str, Any]]:
    """Convert one session of one day into a trip, and record what was done.

    The GPS split is the point of this function. Everything up to
    `gps_warmup_s` goes into the trip the reconstruction will read; everything
    after it goes into the reference file, which no filter ever sees. Nothing
    is deleted and nothing is invented - the fixes are only sorted into what
    the algorithm is allowed to know and what it will be marked against.
    """
    gps_all, report = read_gps_rows(spec.gps_csv)
    sessions = find_sessions(
        read_imu_timeline(spec.imu_csv), gps_all, gap_s=spec.session_gap_s
    )
    if not sessions:
        raise LiveLogError(f"{spec.imu_csv.name} contains no session longer than a minute")
    if not 0 <= spec.session_index < len(sessions):
        raise LiveLogError(
            f"--session {spec.session_index} is out of range: this day has "
            f"{len(sessions)} session(s), 0..{len(sessions) - 1}"
        )
    session = sessions[spec.session_index]
    if session.gps_fixes == 0:
        raise LiveLogError(
            f"session {spec.session_index} has no GPS fix at all, so there is "
            "nothing to seed the reconstruction with"
        )

    # Trim the parked minutes at the front, keeping a stationary pre-roll: the
    # accelerometer and gyro bias are measured there, and losing them costs
    # more than the samples are worth.
    start_ms = session.start_ms
    motion_ms = motion_start_ms(gps_all, session)
    trimmed_s = 0.0
    if motion_ms is not None:
        wanted = max(session.start_ms, motion_ms - spec.pre_roll_s * 1000.0)
        trimmed_s = (wanted - session.start_ms) / 1000.0
        start_ms = wanted
    end_ms = session.end_ms
    if spec.max_duration_s is not None:
        end_ms = min(end_ms, start_ms + spec.max_duration_s * 1000.0)

    # One pass over the IMU file. It is up to a gigabyte, and the clock
    # correction below only shifts the GPS side, so the padding here is enough
    # to keep every later window inside what was read.
    pad_ms = 30_000.0
    imu_all = read_imu_rows(spec.imu_csv, start_ms - pad_ms, end_ms, report)

    # Line the two clocks up before anything is measured from them together.
    align_end = (motion_ms if motion_ms is not None else start_ms) + (
        spec.clock_align_s if spec.clock_align_s is not None else spec.gps_warmup_s
    ) * 1000.0
    if spec.clock_align == "session":
        align_end = end_ms
    if spec.clock_align == "warmup" and not spec.keep_all_gps:
        warmup_limit = (motion_ms if motion_ms is not None else start_ms) + spec.gps_warmup_s * 1000.0
        if align_end > warmup_limit:
            raise LiveLogError("--clock-align-seconds must not exceed --gps-warmup when GPS is withheld")
    align_imu = imu_all[
        (imu_all[:, IMU_TIME] >= start_ms - pad_ms) & (imu_all[:, IMU_TIME] <= align_end + pad_ms)
    ]
    align_gps = gps_all[
        (gps_all[:, GPS_TIME] >= start_ms - pad_ms) & (gps_all[:, GPS_TIME] <= align_end)
    ]
    if spec.clock_offset_s is not None:
        offset = ClockOffset(
            seconds=spec.clock_offset_s, correlation=1.0, source="given_on_the_command_line"
        )
    elif spec.clock_align == "none":
        offset = ClockOffset(source="alignment_disabled")
    else:
        offset = estimate_clock_offset(imu_geometry(align_imu), align_gps)
        if spec.clock_align == "session":
            offset.source = "gyro_vs_course_rate_over_whole_session"
    gps_all = gps_all.copy()
    gps_all[:, GPS_TIME] += offset.seconds * 1000.0
    if motion_ms is not None:
        motion_ms += offset.seconds * 1000.0

    imu = imu_all[imu_all[:, IMU_TIME] >= start_ms]
    window = (gps_all[:, GPS_TIME] >= start_ms) & (gps_all[:, GPS_TIME] <= end_ms)
    gps = gps_all[window]
    if len(gps) < 2:
        raise LiveLogError("the selected window contains fewer than two GPS fixes")

    t0_ms = float(min(imu[0, IMU_TIME], gps[0, GPS_TIME]))
    quality = read_nmea_quality(spec.nmea_txt) if spec.nmea_txt.exists() else np.zeros((0, 4))
    # The NMEA sentences carry the receiver's own time-of-day, so they are
    # matched on the raw GPS clock, before the correction is applied.
    accuracy, satellites = accuracy_from_nmea(
        gps[:, GPS_TIME] - offset.seconds * 1000.0, quality, report
    )

    # The warm-up is counted from the moment the car moves, not from the first
    # sample. The stationary pre-roll before it is inside the window too, but it
    # is not spent: a parked car's course is noise, so counting those seconds
    # against the warm-up would quietly buy the mount estimate nothing while
    # looking like it had thirty seconds to work with.
    warmup_end_ms = (motion_ms if motion_ms is not None else t0_ms) + spec.gps_warmup_s * 1000.0
    warmup = gps[:, GPS_TIME] <= warmup_end_ms
    if not np.any(warmup):
        raise LiveLogError(
            f"--gps-warmup {spec.gps_warmup_s:g} leaves no fix to start from; "
            f"the first fix of this window arrives {(gps[0, GPS_TIME] - t0_ms) / 1000.0:.0f} s in"
        )
    withheld = np.zeros(len(gps), dtype=bool) if spec.keep_all_gps else ~warmup

    motions = imu_to_motion_samples(imu, t0_ms, spec.imu_rate_hz)
    locations = gps_to_location_samples(
        gps[~withheld], t0_ms, accuracy[~withheld], satellites[~withheld]
    )
    reference = gps_to_location_samples(
        gps[withheld], t0_ms, accuracy[withheld], satellites[withheld]
    )

    estimate = estimate_mount(
        imu_geometry(imu[imu[:, IMU_TIME] <= warmup_end_ms]), gps[warmup]
    )
    still_s = max(0.0, (motion_ms - t0_ms) / 1000.0) if motion_ms is not None else 0.0
    calibration = mount_calibration(estimate, still_s)

    provenance: dict[str, Any] = {
        "source": "vehicle_logger_day_files",
        "day": spec.day,
        "session": session.to_json(),
        "sessions_in_day": len(sessions),
        "window": {"start": _iso_ms(start_ms), "end": _iso_ms(end_ms)},
        "trimmed_before_motion_s": round(trimmed_s, 1),
        "clock_offset": offset.to_json(),
        "clock_offset_measured_over": spec.clock_align,
        "still_pre_roll_s": round(still_s, 1),
        "imu_rate_hz": spec.imu_rate_hz,
        "gps_warmup_s": None if spec.keep_all_gps else spec.gps_warmup_s,
        "gps_visible_until_s": (
            None if spec.keep_all_gps else round((warmup_end_ms - t0_ms) / 1000.0, 1)
        ),
        "gps_fixes_kept": int(len(locations)),
        "gps_fixes_withheld": int(len(reference)),
        "accuracy_source": "nmea_gga_hdop" if report.nmea_matched else "assumed_constant",
        "accuracy_model": f"horizontal_accuracy_m = HDOP * {UERE_M:g}",
        "mount_estimate": estimate.to_json(),
        "reader": report.to_json(),
        "reference_is": (
            None
            if spec.keep_all_gps
            else "withheld_real_gps"
        ),
    }

    metadata = TripMetadata(
        trip_id=spec.trip_id or f"live-{spec.day}-s{spec.session_index}",
        started_at=datetime.fromtimestamp(t0_ms / 1000.0, tz=timezone.utc),
        ended_at=datetime.fromtimestamp(end_ms / 1000.0, tz=timezone.utc),
        device_model="vehicle logger (GPS 10 Hz + IMU 100 Hz)",
        calibration=calibration,
        notes=(
            "Imported from day logs with the whole GPS track kept."
            if spec.keep_all_gps
            else (
                "Imported from day logs. The reconstruction is given GPS for "
                f"the first {spec.gps_warmup_s:g} s only; every later fix was "
                "withheld into reference-samples.jsonl."
            )
        ),
        extra={"live_import": provenance},
    )
    trip = Trip(
        metadata=metadata,
        locations=locations,
        motions=motions,
        reference_locations=reference,
    )
    return trip, provenance
