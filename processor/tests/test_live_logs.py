"""The vehicle logger import.

The fixtures here are synthetic but they are built to the *measured* properties
of the real day files, not to what the format description implies: the
accelerometer carries gravity, the gyro is in degrees per second, and the two
files do not share a clock. Each test names which of those it is pinning down.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from geotrace.config import G_TO_MS2
from geotrace.live_logs import (
    IMU_COLUMNS,
    ImportSpec,
    LiveLogError,
    accuracy_from_nmea,
    build_trip,
    estimate_clock_offset,
    estimate_mount,
    find_sessions,
    imu_geometry,
    imu_to_motion_samples,
    motion_start_ms,
    mount_calibration,
    read_gps_rows,
    read_imu_rows,
    read_nmea_quality,
)
from geotrace.coordinates import haversine_m
from geotrace.loader import load_trip, write_trip
from geotrace.motion_model import quaternion_to_matrix

EARTH_R = 6371000.0

DAY_FILE_COLUMNS = (
    "timestamp", "temp", "pressure", "imu_ms", "acc_x", "acc_y", "acc_z",
    "gyr_x", "gyr_y", "gyr_z", "mag_x", "mag_y", "mag_z", "roll", "pitch",
    "yaw", "quat_w", "quat_x", "quat_y", "quat_z",
)
"""Every column the logger writes. `synthetic_drive` produces this, because
that is what has to survive being written to a CSV and read back."""


def logger_rows(day_file_rows: np.ndarray) -> np.ndarray:
    """The reduced layout `read_imu_rows` produces, without going via a file."""
    index = [DAY_FILE_COLUMNS.index(name) for name in IMU_COLUMNS]
    return np.asarray(day_file_rows, dtype=float)[:, index]


def yaw_quaternion(yaw_rad: float, roll_rad: float = 0.0) -> tuple[float, float, float, float]:
    """(w, x, y, z) for a yaw, optionally tilted, device -> world."""
    cy, sy = math.cos(yaw_rad / 2), math.sin(yaw_rad / 2)
    cr, sr = math.cos(roll_rad / 2), math.sin(roll_rad / 2)
    # q = q_yaw * q_roll
    return (cy * cr, cy * sr, 0.0, sy * cr)


def synthetic_drive(
    duration_s: float = 240.0,
    imu_hz: float = 100.0,
    gps_hz: float = 10.0,
    clock_offset_s: float = 0.0,
    mount_yaw_deg: float = 0.0,
    world_yaw_deg: float = 40.0,
    lat0: float = 59.96,
    lon0: float = 30.27,
    t0_ms: float = 1_784_736_000_000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """A car that idles, accelerates, turns, and stops again.

    Returns (imu_rows, gps_rows) in exactly the layout the day files use.
    `world_yaw_deg` is the rotation between the gyro's arbitrary world frame
    and East/North; `clock_offset_s` is how far the GPS file's clock is put
    *behind* the IMU's, the defect the real recorder has.
    """
    t = np.arange(0.0, duration_s, 1.0 / imu_hz)
    # speed profile: still, accelerate, cruise, turn, brake to a stop
    speed = np.interp(
        t,
        [0, 20, 45, 90, 150, 200, duration_s],
        [0, 0, 12.0, 12.0, 9.0, 0.0, 0.0],
    )
    # Real traffic never holds a speed. The wobble matters: a smooth ramp is
    # forgiving of a clock error, and a test built on one would not notice the
    # defect this import exists to correct.
    speed = np.clip(speed + 2.0 * np.sin(2 * math.pi * t / 17.0) * (speed > 1.0), 0.0, None)
    # Two turns, the first inside any plausible warm-up: the clock offset and
    # the mount angle are both only observable while the car is steering, and a
    # fixture that drives straight for its first minute would be testing a case
    # the import cannot serve anyway.
    turn = np.where(
        ((t > 30) & (t < 52)) | ((t > 100) & (t < 122)), math.radians(9.0), 0.0
    )
    heading = np.cumsum(turn) / imu_hz  # heading in E/N, radians CCW from east
    heading += math.radians(30.0)
    accel = np.gradient(speed, t)

    east = np.cumsum(speed * np.cos(heading)) / imu_hz
    north = np.cumsum(speed * np.sin(heading)) / imu_hz

    world = math.radians(world_yaw_deg)
    mount = math.radians(mount_yaw_deg)
    # Device yaw inside the gyro's own world frame.
    device_yaw = heading - world - mount

    rows = []
    for i, ti in enumerate(t):
        q = yaw_quaternion(device_yaw[i])
        rot = quaternion_to_matrix(q)
        # True acceleration in the gyro world frame.
        a_world = np.array([
            accel[i] * math.cos(heading[i] - world) - speed[i] * turn[i] * 0.0,
            accel[i] * math.sin(heading[i] - world),
            0.0,
        ])
        a_world[0] += -speed[i] * turn[i] * math.sin(heading[i] - world)
        a_world[1] += speed[i] * turn[i] * math.cos(heading[i] - world)
        gravity_device = -rot[2, :]
        specific_force = (rot.T @ a_world) / G_TO_MS2 - gravity_device
        omega_world = np.array([0.0, 0.0, turn[i]])
        omega_device = np.degrees(rot.T @ omega_world)
        rows.append((
            t0_ms + ti * 1000.0, 20.0, 101000.0, ti * 1000.0,
            specific_force[0], specific_force[1], specific_force[2],
            omega_device[0], omega_device[1], omega_device[2],
            0.0, 0.0, 0.0, 0.0, 0.0, math.degrees(device_yaw[i]),
            q[0], q[1], q[2], q[3],
        ))
    imu = np.array(rows, dtype=float)

    g_t = np.arange(0.0, duration_s, 1.0 / gps_hz)
    g_east = np.interp(g_t, t, east)
    g_north = np.interp(g_t, t, north)
    g_speed = np.interp(g_t, t, speed)
    g_head = np.interp(g_t, t, heading)
    lat = lat0 + np.degrees(g_north / EARTH_R)
    lon = lon0 + np.degrees(g_east / (EARTH_R * math.cos(math.radians(lat0))))
    course = np.mod(90.0 - np.degrees(g_head), 360.0)
    gps = np.column_stack((
        t0_ms + (g_t - clock_offset_s) * 1000.0, lat, lon, g_speed, course
    ))
    return imu, gps


def write_day(root: Path, day: str, imu: np.ndarray, gps: np.ndarray) -> None:
    (root / "gps_logs").mkdir(parents=True, exist_ok=True)
    (root / "imu_logs").mkdir(parents=True, exist_ok=True)
    header_gps = "timestamp_ms,lat,lon,speed,course\n"
    lines = [
        f"{int(r[0])},{r[1]:.9f},{r[2]:.9f},{r[3]:.6f},{r[4]:.3f}" for r in gps
    ]
    (root / "gps_logs" / f"{day}_GPS_logs.csv").write_text(
        header_gps + "\n".join(lines) + "\n", encoding="utf-8"
    )
    header_imu = (
        "timestamp,temp,pressure,imu_ms,acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z,"
        "mag_x,mag_y,mag_z,roll,pitch,yaw,quat_w,quat_x,quat_y,quat_z\n"
    )
    # The timestamp is an integer millisecond in the real file, and it has to
    # stay one here: %g would round 1.78e12 to the nearest 10 s and silently
    # destroy the sample spacing everything downstream is derived from.
    rows = [
        ",".join([str(int(r[0]))] + [f"{v:.9g}" for v in r[1:]]) for r in imu
    ]
    (root / "imu_logs" / f"{day}_IMU_logs.csv").write_text(
        header_imu + "\n".join(rows) + "\n", encoding="utf-8"
    )


@pytest.fixture()
def day_root(tmp_path: Path) -> Path:
    imu, gps = synthetic_drive(clock_offset_s=4.85)
    write_day(tmp_path, "2026-07-22", imu, gps)
    return tmp_path


# ------------------------------------------------------------------ reading


def test_gps_reader_survives_a_truncated_final_line(tmp_path: Path) -> None:
    """The day file is being written while the car runs; the tail is often half a row."""
    path = tmp_path / "gps.csv"
    path.write_text(
        "timestamp_ms,lat,lon,speed,course\n"
        "1784736204853,59.9630,30.2660,0.18,54.0\n"
        "1784736204878,59.9630,30.2660,0.07,53.0\n"
        "17847362049",
        encoding="utf-8",
    )
    rows, report = read_gps_rows(path)
    assert len(rows) == 2
    assert report.gps_skipped == 1


def test_reader_rejects_a_file_without_the_columns(tmp_path: Path) -> None:
    path = tmp_path / "gps.csv"
    path.write_text("time,latitude\n1,2\n", encoding="utf-8")
    with pytest.raises(LiveLogError, match="missing column"):
        read_gps_rows(path)


def test_imu_reader_clips_to_the_window_while_reading(day_root: Path) -> None:
    """A day of IMU is up to a gigabyte; the window must not be applied after loading."""
    path = day_root / "imu_logs" / "2026-07-22_IMU_logs.csv"
    everything = read_imu_rows(path)
    clipped = read_imu_rows(path, everything[0, 0] + 10_000, everything[0, 0] + 20_000)
    assert len(clipped) < len(everything)
    assert clipped[0, 0] >= everything[0, 0] + 10_000
    assert clipped[-1, 0] <= everything[0, 0] + 20_000


def test_nmea_hdop_becomes_a_horizontal_accuracy(tmp_path: Path) -> None:
    path = tmp_path / "nmea.txt"
    path.write_text(
        "$GNGGA,160245.30,5957.78,N,03015.96,E,1,08,4.5,0.1,M,16.0,M,,*78\n"
        "$GNGGA,160246.30,,,,,0,00,9999.0,,,,,,*48\n"
        "$GNRMC,,V,,,,,,,,,,N,V*37\n",
        encoding="utf-8",
    )
    quality = read_nmea_quality(path)
    assert len(quality) == 2

    day_ms = 1_784_736_000_000.0 - (1_784_736_000_000.0 % 86_400_000)
    matched = day_ms + (16 * 3600 + 2 * 60 + 45.3) * 1000.0
    unmatched = day_ms + (16 * 3600 + 2 * 60 + 46.3) * 1000.0
    accuracy, sats = accuracy_from_nmea(np.array([matched, unmatched]), quality)
    assert accuracy[0] == pytest.approx(4.5 * 4.0)
    assert sats[0] == 8
    # A sentence reporting no fix at all is not a quality figure to be used.
    assert accuracy[1] == pytest.approx(8.0)
    assert math.isnan(sats[1])


# ----------------------------------------------------------------- sessions


def test_a_day_splits_into_sessions_on_the_imu_gap() -> None:
    """The logger runs between drives; a day file is several sessions, not one trip."""
    first = np.arange(0, 200_000, 10.0)
    second = np.arange(4_000_000, 4_200_000, 10.0)
    sessions = find_sessions(np.concatenate((first, second)))
    assert len(sessions) == 2
    assert sessions[0].duration_s == pytest.approx(200.0, abs=1.0)
    assert sessions[1].start_ms == pytest.approx(4_000_000)


def test_a_session_shorter_than_the_minimum_is_not_offered() -> None:
    times = np.concatenate((np.arange(0, 200_000, 10.0), np.arange(4_000_000, 4_010_000, 10.0)))
    assert len(find_sessions(times, min_duration_s=60.0)) == 1


def test_motion_start_ignores_a_single_noisy_fix(day_root: Path) -> None:
    """One fix reading 2 m/s while parked must not count as the drive starting."""
    gps, _ = read_gps_rows(day_root / "gps_logs" / "2026-07-22_GPS_logs.csv")
    gps[5, 3] = 9.0
    sessions = find_sessions(
        read_imu_rows(day_root / "imu_logs" / "2026-07-22_IMU_logs.csv")[:, 0], gps
    )
    start = motion_start_ms(gps, sessions[0])
    assert start is not None
    assert (start - gps[0, 0]) / 1000.0 > 15.0


# --------------------------------------------------------------- conversion


def test_gravity_is_removed_with_the_attitude_quaternion() -> None:
    """The logger reports specific force: +1 g on the up axis while parked.

    CoreMotion, which the trip format speaks, reports ~0 there and keeps
    gravity in its own channel. Getting this backwards would feed the filters
    a permanent 9.8 m/s^2 of acceleration.
    """
    roll = math.radians(12.0)
    q = yaw_quaternion(0.0, roll)
    rot = quaternion_to_matrix(q)
    specific_force = rot.T @ np.array([0.0, 0.0, 1.0])
    row = np.array([
        0.0, 20.0, 101000.0, 0.0,
        specific_force[0], specific_force[1], specific_force[2],
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, q[0], q[1], q[2], q[3],
    ])
    sample = imu_to_motion_samples(logger_rows(row[None, :]), 0.0, rate_hz=0.0)[0]
    assert np.allclose(sample.user_acceleration_g, np.zeros(3), atol=1e-9)
    assert np.allclose(
        quaternion_to_matrix(sample.quaternion) @ np.asarray(sample.gravity),
        [0.0, 0.0, -1.0],
        atol=1e-9,
    )


def test_the_gyro_is_converted_from_degrees_per_second() -> None:
    row = np.zeros(len(DAY_FILE_COLUMNS))
    row[DAY_FILE_COLUMNS.index("gyr_x")] = 30.0
    row[DAY_FILE_COLUMNS.index("gyr_z")] = -90.0
    row[DAY_FILE_COLUMNS.index("quat_w")] = 1.0
    sample = imu_to_motion_samples(logger_rows(row[None, :]), 0.0, rate_hz=0.0)[0]
    assert sample.rotation_rate == pytest.approx(
        (math.radians(30.0), 0.0, math.radians(-90.0))
    )


def test_decimation_targets_the_requested_rate(day_root: Path) -> None:
    imu = read_imu_rows(day_root / "imu_logs" / "2026-07-22_IMU_logs.csv")
    samples = imu_to_motion_samples(imu, imu[0, 0], rate_hz=50.0)
    assert len(samples) == pytest.approx(len(imu) / 2, rel=0.02)
    spacing = np.diff([s.monotonic_time for s in samples])
    assert float(np.median(spacing)) == pytest.approx(0.02, abs=1e-6)


# ------------------------------------------------------------- calibration


def test_the_clock_offset_between_the_two_files_is_recovered() -> None:
    """The real recorder stamps GPS ~4.85 s behind the IMU, and it matters.

    Uncorrected it puts the seeding position tens of metres down the road and
    makes the mount angle unmeasurable, so the import has to find it rather
    than trust the timestamps.
    """
    imu, gps = synthetic_drive(clock_offset_s=4.85)
    offset = estimate_clock_offset(imu_geometry(logger_rows(imu)), gps)
    assert offset.seconds == pytest.approx(4.85, abs=0.15)
    assert offset.correlation > 0.8
    assert offset.source == "gyro_vs_course_rate"


def test_two_files_already_on_one_clock_produce_no_shift() -> None:
    imu, gps = synthetic_drive(clock_offset_s=0.0)
    offset = estimate_clock_offset(imu_geometry(logger_rows(imu)), gps)
    assert offset.seconds == pytest.approx(0.0, abs=0.15)


def test_the_mount_rotation_is_recovered_from_the_warm_up() -> None:
    """Both angles: how the box sits in the car, and where the gyro frame points."""
    imu, gps = synthetic_drive(clock_offset_s=4.85, mount_yaw_deg=-15.0, world_yaw_deg=40.0)
    geometry = imu_geometry(logger_rows(imu))
    offset = estimate_clock_offset(geometry, gps)
    estimate = estimate_mount(geometry, gps, offset.seconds)
    assert math.degrees(estimate.mount_yaw_rad) == pytest.approx(-15.0, abs=4.0)
    assert math.degrees(estimate.world_yaw_offset_rad) == pytest.approx(40.0, abs=4.0)
    assert estimate.coherence > 0.9
    assert estimate.quality == "good"


def test_an_uncorrected_clock_puts_the_car_down_the_road(day_root: Path) -> None:
    """The clock correction is load-bearing, not a refinement.

    Left alone, every fix is filed against the wrong IMU instant, so the place
    the reconstruction is seeded from - and every reference point it is later
    scored against - sits `speed * offset` further along the route than the car
    ever was. At 12 m/s and 4.85 s that is nearly sixty metres before a single
    filter has run.
    """
    common = dict(
        gps_dir=day_root / "gps_logs",
        imu_dir=day_root / "imu_logs",
        day="2026-07-22",
        gps_warmup_s=80.0,
    )
    aligned, provenance = build_trip(ImportSpec(**common))
    raw, _ = build_trip(ImportSpec(**common, clock_align="none"))
    assert provenance["clock_offset"]["seconds"] == pytest.approx(4.85, abs=0.2)

    def place_at(trip, seconds: float) -> tuple[float, float]:
        fixes = trip.locations + trip.reference_locations
        sample = min(fixes, key=lambda s: abs(s.monotonic_time - seconds))
        return sample.latitude, sample.longitude

    at = 70.0
    displacement = haversine_m(*place_at(aligned, at), *place_at(raw, at))
    speed = next(
        s.speed
        for s in sorted(
            aligned.locations + aligned.reference_locations,
            key=lambda s: abs(s.monotonic_time - at),
        )
    )
    assert displacement == pytest.approx(speed * 4.85, rel=0.35)


def test_a_warm_up_with_no_driving_reports_itself_unusable() -> None:
    imu, gps = synthetic_drive(duration_s=18.0)
    estimate = estimate_mount(imu_geometry(logger_rows(imu)), gps)
    assert estimate.quality == "unusable"
    assert estimate.notes
    assert mount_calibration(estimate, still_duration_s=5.0) is None


def test_the_calibration_round_trips_through_the_reconstruction_s_own_formula() -> None:
    """`_estimate_reference_offset` must get the world rotation back out again.

    The calibration writes the vehicle's forward axis in device coordinates;
    the pipeline turns that back into the rotation from the gyro frame onto
    East/North. If the two disagree the whole route is rotated about its start.
    """
    from geotrace.motion_model import _estimate_reference_offset

    imu, gps = synthetic_drive(clock_offset_s=4.85, mount_yaw_deg=-15.0, world_yaw_deg=40.0)
    geometry = imu_geometry(logger_rows(imu))
    offset = estimate_clock_offset(geometry, gps)
    estimate = estimate_mount(geometry, gps, offset.seconds)
    calibration = mount_calibration(estimate, still_duration_s=20.0)
    assert calibration is not None

    recovered = _estimate_reference_offset(
        np.zeros((1, 3)), np.zeros(1), np.zeros(1),
        estimate.initial_heading_rad, calibration,
    )
    assert math.degrees(recovered) == pytest.approx(
        math.degrees(estimate.world_yaw_offset_rad), abs=0.5
    )


# ------------------------------------------------------------------ import


def test_import_withholds_every_fix_after_the_warm_up(day_root: Path, tmp_path: Path) -> None:
    """The point of the whole command: the filters see a start, and then nothing."""
    spec = ImportSpec(
        gps_dir=day_root / "gps_logs",
        imu_dir=day_root / "imu_logs",
        day="2026-07-22",
        gps_warmup_s=30.0,
    )
    trip, provenance = build_trip(spec)
    assert trip.locations
    assert trip.reference_locations
    assert max(s.monotonic_time for s in trip.locations) < min(
        s.monotonic_time for s in trip.reference_locations
    )
    assert provenance["reference_is"] == "withheld_real_gps"
    # Nothing was thrown away, only sorted.
    assert (
        provenance["gps_fixes_kept"] + provenance["gps_fixes_withheld"]
        == len(trip.locations) + len(trip.reference_locations)
    )


def test_the_warm_up_is_counted_from_when_the_car_moves(day_root: Path) -> None:
    """Twenty seconds of a parked car is not twenty seconds of usable GPS."""
    spec = ImportSpec(
        gps_dir=day_root / "gps_logs",
        imu_dir=day_root / "imu_logs",
        day="2026-07-22",
        gps_warmup_s=30.0,
        pre_roll_s=20.0,
    )
    trip, provenance = build_trip(spec)
    assert provenance["still_pre_roll_s"] > 5.0
    assert provenance["gps_visible_until_s"] > 30.0
    last_kept = max(s.monotonic_time for s in trip.locations)
    assert last_kept == pytest.approx(provenance["gps_visible_until_s"], abs=0.3)


def test_keep_all_gps_withholds_nothing(day_root: Path) -> None:
    spec = ImportSpec(
        gps_dir=day_root / "gps_logs",
        imu_dir=day_root / "imu_logs",
        day="2026-07-22",
        keep_all_gps=True,
    )
    trip, provenance = build_trip(spec)
    assert not trip.reference_locations
    assert provenance["reference_is"] is None
    assert provenance["gps_warmup_s"] is None


def test_an_out_of_range_session_is_refused(day_root: Path) -> None:
    spec = ImportSpec(
        gps_dir=day_root / "gps_logs",
        imu_dir=day_root / "imu_logs",
        day="2026-07-22",
        session_index=7,
    )
    with pytest.raises(LiveLogError, match="out of range"):
        build_trip(spec)


def test_an_imported_trip_loads_back_as_a_trip(day_root: Path, tmp_path: Path) -> None:
    """The import has to produce exactly what `load_trip` reads, not a near miss."""
    spec = ImportSpec(
        gps_dir=day_root / "gps_logs",
        imu_dir=day_root / "imu_logs",
        day="2026-07-22",
        gps_warmup_s=30.0,
    )
    trip, provenance = build_trip(spec)
    out = write_trip(trip, tmp_path / "trip", copy_photos=False)
    loaded, report = load_trip(out)
    assert report.malformed_lines == 0
    assert len(loaded.locations) == len(trip.locations)
    assert len(loaded.motions) == len(trip.motions)
    assert len(loaded.reference_locations) == len(trip.reference_locations)
    assert loaded.metadata.extra["live_import"]["clock_offset"]["seconds"] == pytest.approx(
        provenance["clock_offset"]["seconds"]
    )
    assert loaded.metadata.calibration is not None
    assert loaded.metadata.calibration.heading_source == "gps_course_window"


# ---------------------------------------------------------------- reporting


def test_the_report_does_not_call_withheld_gps_a_synthetic_corruption(
    day_root: Path,
) -> None:
    """The banner is a claim about what happened, and here it would be false.

    Every other trip with a reference track really was corrupted on purpose.
    This one was not: the fixes are real and were simply kept out of the
    filters' reach.
    """
    from geotrace.visualization import WITHHELD_GPS, reference_kind

    spec = ImportSpec(
        gps_dir=day_root / "gps_logs",
        imu_dir=day_root / "imu_logs",
        day="2026-07-22",
        gps_warmup_s=30.0,
    )
    withheld, _ = build_trip(spec)
    assert reference_kind(withheld) == WITHHELD_GPS

    control, _ = build_trip(
        ImportSpec(
            gps_dir=day_root / "gps_logs",
            imu_dir=day_root / "imu_logs",
            day="2026-07-22",
            keep_all_gps=True,
        )
    )
    assert reference_kind(control) is None

    # A trip that carries no live-import provenance keeps the old meaning.
    withheld.metadata.extra = {}
    assert reference_kind(withheld) == "synthetic"


def test_a_teleporting_receiver_is_counted_not_smoothed_away() -> None:
    """One sample day reads 182 km across 21 minutes of driving.

    The distance is the honest sum of what the receiver reported. Silently
    dropping the jumps would make a mangled session look like a good one in the
    listing, which is exactly when the user needs to see it.
    """
    times = np.arange(0, 200_000, 100.0)
    n = len(times)
    fixes = np.column_stack((
        times,
        59.96 + np.arange(n) * 1e-6,
        np.full(n, 30.27),
        np.full(n, 10.0),
        np.full(n, 90.0),
    ))
    clean = find_sessions(times, fixes)
    assert clean[0].gps_jumps == 0

    jumped = fixes.copy()
    jumped[500:, 1] += 0.05  # ~5.5 km sideways between two consecutive fixes
    flagged = find_sessions(times, jumped)
    assert flagged[0].gps_jumps == 1
    assert flagged[0].distance_m > clean[0].distance_m + 5000.0
