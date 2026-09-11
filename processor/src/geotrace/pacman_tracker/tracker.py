"""The Pacman tracker: one speed filter, many road hypotheses.

    IMU ──┬─ stop detector ──────→ v = 0, b_a, b_g
          ├─ longitudinal accel ─→ dv between anchors
          ├─ a_lat / yaw_rate ───→ precise speed anchor
          └─ spectral model ─────→ approximate v +/- sigma
                                          │
                                          ▼
                              GLOBAL SPEED FILTER   D, v, sigma_D
                                          │
                        ┌─────────────────┴─────────────────┐
                        ▼                                   ▼
                Pacman propagation                    turn events
                (s_i = D - route_offset_i)     (integrated yaw vs map angle)
                        └────────── road-graph scoring ─────┘
                                          │
                                   merge equivalents
                                          ▼
                                  route probability

GPS reaches this only through :func:`build_inputs`, which reads
``trip.locations`` - truncated at the outage by the recorder - and never
``trip.reference_locations``. ``tests/pacman/test_leakage.py`` mutates the
withheld stream and demands byte-identical output.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from geotrace.config import MotionConfig as LegacyMotionConfig
from geotrace.coordinates import LocalFrame, course_to_heading, wrap_angle
from geotrace.models import LocationSample, Trip
from geotrace.motion_model import build_imu_stream, quaternion_to_matrix
from geotrace.pacman_tracker.attitude import (
    G, AttitudeFilter, quat_from_gravity)
from geotrace.pacman_tracker.config import PacmanConfig
from geotrace.pacman_tracker.corridor import CorridorBuilder, CorridorSet
from geotrace.pacman_tracker.display_position import (
    DisplayPositionBranch, DisplayPositionSample, DisplayResidualModel)
from geotrace.pacman_tracker.curvature import CurvatureMatcher, MatchDiagnostics
from geotrace.pacman_tracker.intervals import DriftObservation, common_drift
from geotrace.pacman_tracker.manager import HypothesisManager
from geotrace.pacman_tracker.motion import ImuSample, detect_stationary
from geotrace.pacman_tracker.roadmap import RoadGeometry
from geotrace.pacman_tracker.spectral import SpectralSpeedModel, extract_features
from geotrace.pacman_tracker.speed import (
    BA, D, SPEED_DIM, GlobalSpeedTracker, SpeedSample)
from geotrace.pacman_tracker.retro_smooth import retrospective_distance_smooth
from geotrace.pacman_tracker.single_path import SinglePathManager
from geotrace.pacman_tracker.state import HypothesisSet, PacmanState, RouteNode, make_set
from geotrace.pacman_tracker.turns import HeadingIntegrator, detect_turns
from geotrace.road_graph import RoadNetwork


@dataclass
class TrackerInputs:
    """Everything the tracker is allowed to see."""

    samples: list[ImuSample]
    t_start: float
    init_edges: list[int]
    init_log_weights: list[float]
    init_s: list[float]
    speed0: float
    accel_bias0: float
    gyro_bias0: float
    origin_xy: tuple[float, float]
    heading0: float
    spectral: SpectralSpeedModel = field(default_factory=SpectralSpeedModel)
    oracle_speed: Optional[np.ndarray] = None
    """Benchmark-only. True speed per step, used by the oracle runs to separate
    "the route manager is wrong" from "the speed is wrong". Never set on the
    normal path; :func:`build_inputs` cannot produce it."""

    oracle_distance: Optional[np.ndarray] = None
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackerFrame:
    """One output tick."""

    t: float
    corridors: CorridorSet
    top: list[PacmanState]
    position: tuple[float, float]
    population: int
    stationary: bool
    speed: SpeedSample
    alive_edges: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def to_json(self) -> dict[str, Any]:
        return {
            "t": round(self.t, 2),
            "population": self.population,
            "stationary": self.stationary,
            "speed": self.speed.to_json(),
            "corridors": self.corridors.to_json(),
            "top": [s.to_json() for s in self.top[:5]],
        }


@dataclass
class TrackerResult:
    frames: list[TrackerFrame]
    final: HypothesisSet
    stats: dict[str, Any]
    inputs: TrackerInputs
    speed_trace: list[SpeedSample] = field(default_factory=list)
    position_trace: Optional[list[DisplayPositionSample]] = None
    """Zero-latency corrected *display* position, one per output tick, present
    only when ``display.position_branch_enabled``. It never influenced the
    committed route, ``speed_trace`` or ``frames`` - it is a marker overlay.
    See ``display_position.py`` and docs/DISPLAY_ODOMETRY_SPLIT.md."""
    retro_speed_trace: Optional[list[SpeedSample]] = None
    """Offline reconstruction of ``speed_trace`` with late bend-anchor
    corrections redistributed backward (see ``retro_smooth``). ``None`` unless
    ``single_path.retro_bend_smoothing_enabled`` and a bend anchor landed."""


class StepObserver:
    """Hook for diagnostics. Given a read-only view; returns nothing to the
    filter, so the tracker behaves identically with and without one."""

    def observe(self, t: float, hs: HypothesisSet, sample: ImuSample,
                match: MatchDiagnostics, manager: HypothesisManager,
                distance: float, sigma_s: float,
                speed: float) -> None:  # pragma: no cover
        return None


# --------------------------------------------------------------------- inputs


def _lateral_channel(trip: Trip) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray]:
    """Raw-rate lateral/longitudinal acceleration and yaw rate, plus raw axes.

    The mount is rigid and the recorder supplies the vehicle's forward axis in
    the device frame, so rotating it into the world and dropping its vertical
    component gives the direction of travel; the left-hand normal to that is
    the axis a turn accelerates the car along.
    """
    calibration = trip.metadata.calibration
    forward = None
    if calibration is not None and calibration.forward_axis_device is not None:
        candidate = np.asarray(calibration.forward_axis_device, dtype=float)
        if np.linalg.norm(candidate) > 1e-9:
            forward = candidate / np.linalg.norm(candidate)

    ordered = sorted(trip.motions, key=lambda s: s.monotonic_time)
    n = len(ordered)
    times = np.empty(n)
    a_lat = np.zeros(n)
    omega = np.zeros(n)
    accel = np.empty((n, 3))
    gyro = np.empty((n, 3))
    for i, sample in enumerate(ordered):
        times[i] = sample.monotonic_time
        accel[i] = sample.user_acceleration_ms2
        gyro[i] = sample.rotation_rate
        rot = quaternion_to_matrix(sample.quaternion)
        omega[i] = float((rot @ gyro[i])[2])
        if forward is None:
            continue
        a_world = rot @ accel[i]
        f_world = rot @ forward
        f_world[2] = 0.0
        norm = float(np.linalg.norm(f_world))
        if norm < 1e-6:
            continue
        f_world /= norm
        a_lat[i] = float(a_world @ np.array([-f_world[1], f_world[0], 0.0]))
    return times, a_lat, omega, accel, gyro


def _tally(reasons) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in reasons:
        key = r.split(";")[0].split(" below")[0].strip()
        out[key] = out.get(key, 0) + 1
    return out


def _raw_stationary(specific: np.ndarray, gyro: np.ndarray,
                    cfg: PacmanConfig) -> np.ndarray:
    """Rolling-window rest detector on the raw specific force and gyro."""
    width = max(3, int(round(cfg.attitude.zaru_window_s / 0.02)))
    n = len(specific)
    if n < width:
        return np.zeros(n, dtype=bool)
    kernel = np.ones(width) / width

    def spread(series: np.ndarray) -> np.ndarray:
        pad = (width // 2, width - 1 - width // 2)
        mean = np.convolve(np.pad(series, pad, mode="edge"), kernel, mode="valid")
        msq = np.convolve(np.pad(series**2, pad, mode="edge"), kernel, mode="valid")
        return np.sqrt(np.maximum(msq - mean**2, 0.0))

    a_std = spread(np.linalg.norm(specific, axis=1))
    w_std = spread(np.linalg.norm(gyro, axis=1))
    return (a_std < cfg.attitude.zaru_accel_std_ms2) & (
        w_std < cfg.attitude.zaru_gyro_std_rads)


def _attitude_channel(trip: Trip, cfg: PacmanConfig) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Longitudinal/lateral acceleration from our own attitude filter.

    The recorder's AHRS levels itself against measured specific force, so a
    sustained longitudinal acceleration is partly absorbed as pitch and the
    remaining tilt leaks gravity straight into the forward axis - at 0.3 deg
    that is 0.05 m/s^2, several m/s of velocity error per minute. Here the gyro
    propagates attitude and the accelerometer is used as a gravity reference
    only while the vehicle is not manoeuvring, which decouples the two.
    """
    calibration = trip.metadata.calibration
    forward = None
    if calibration is not None and calibration.forward_axis_device is not None:
        candidate = np.asarray(calibration.forward_axis_device, dtype=float)
        if np.linalg.norm(candidate) > 1e-9:
            forward = candidate / np.linalg.norm(candidate)

    ordered = sorted(trip.motions, key=lambda s: s.monotonic_time)
    n = len(ordered)
    times = np.empty(n)
    a_long = np.zeros(n)
    a_lat = np.zeros(n)
    omega = np.zeros(n)
    if n == 0 or forward is None:
        return times, a_long, a_lat, omega, {"enabled": False}

    specific = np.empty((n, 3))
    gyro = np.empty((n, 3))
    for i, sample in enumerate(ordered):
        times[i] = sample.monotonic_time
        specific[i] = (np.asarray(sample.user_acceleration_ms2, dtype=float)
                       + np.asarray(sample.gravity, dtype=float) * G)
        gyro[i] = sample.rotation_rate

    warm = min(n, max(1, int(round(2.0 / max(float(np.median(np.diff(times)))
                                             if n > 1 else 0.02, 1e-6)))))
    filt = AttitudeFilter(cfg.attitude,
                          q0=quat_from_gravity(specific[:warm].mean(axis=0)))
    gravity_world = np.array([0.0, 0.0, G])
    # Stationarity for ZARU, decided from the raw streams alone: this runs
    # before a_long exists, and must not depend on the attitude it is about to
    # correct. A parked car with the engine running still vibrates, so the test
    # is on the local spread rather than the absolute level.
    rest = _raw_stationary(specific, gyro, cfg)

    rolls = np.empty(n)
    pitches = np.empty(n)
    for i in range(n):
        dt = float(times[i] - times[i - 1]) if i else 0.02
        if not (0.0 < dt < 1.0):
            dt = 0.02
        if cfg.attitude.zaru_enabled and rest[i]:
            filt.zero_rotation(gyro[i], dt)
        rot = filt.update(specific[i], gyro[i], dt)
        rolls[i], pitches[i] = filt.roll_pitch()
        a_world = rot @ specific[i] - gravity_world
        omega[i] = float((rot @ (gyro[i] - filt.bias))[2])
        f_world = rot @ forward
        f_world[2] = 0.0
        norm = float(np.linalg.norm(f_world))
        if norm < 1e-6:
            continue
        f_world /= norm
        a_long[i] = float(a_world @ f_world)
        a_lat[i] = float(a_world @ np.array([-f_world[1], f_world[0], 0.0]))

    diagnostics = dict(filt.stats())
    diagnostics.update({
        "enabled": True,
        "roll_deg": {"mean": round(float(np.degrees(rolls).mean()), 3),
                     "std": round(float(np.degrees(rolls).std()), 3)},
        "pitch_deg": {"mean": round(float(np.degrees(pitches).mean()), 3),
                      "std": round(float(np.degrees(pitches).std()), 3)},
        # Tilt leaks gravity into the forward axis as g*sin(pitch); this is the
        # scale of the acceleration error the filter is there to remove.
        "gravity_leak_ms2": round(float(np.abs(G * np.sin(pitches)).mean()), 4),
        "stationary_fraction": round(float(rest.mean()), 4),
    })
    return times, a_long, a_lat, omega, diagnostics


def _smooth_to_steps(step_times: np.ndarray, raw_times: np.ndarray,
                     values: np.ndarray, window_s: float) -> np.ndarray:
    """Box-smooth a raw-rate channel and sample it onto the step timeline."""
    if raw_times.size == 0:
        return np.zeros(len(step_times))
    dt_raw = float(np.median(np.diff(raw_times))) if raw_times.size > 1 else 0.02
    width = max(1, int(round(window_s / max(dt_raw, 1e-6))))
    kernel = np.ones(width) / width
    pad = (width // 2, width - 1 - width // 2)
    smooth = np.convolve(np.pad(values, pad, mode="edge"), kernel, mode="valid")
    index = np.clip(np.searchsorted(raw_times, step_times), 0, len(raw_times) - 1)
    return smooth[index]


def build_imu_samples(trip: Trip, cfg: PacmanConfig, heading0: float,
                      spectral: Optional[SpectralSpeedModel] = None) -> list[ImuSample]:
    """Reuse the production IMU front end, then add what this tracker needs."""
    legacy = LegacyMotionConfig()
    legacy.filter_dt_s = cfg.motion.dt_s
    legacy.leveling_recovery_tau_s = cfg.leveling_recovery_tau_s
    legacy.accel_smooth_window_s = cfg.accel_smooth_window_s
    stream = build_imu_stream(
        trip.motions, legacy, calibration=trip.metadata.calibration,
        reference_heading_rad=heading0, t_start=trip.t0,
        t_end=trip.t0 + trip.duration_s,
    )
    if not stream.controls:
        return []
    times = np.array([c.t for c in stream.controls])
    a_long = np.array([0.0 if c.a_long is None else c.a_long for c in stream.controls])
    yaw = np.array([c.yaw_rate for c in stream.controls])
    stationary, a_std, w_std, run_s = detect_stationary(times, a_long, yaw, cfg.motion)

    raw_t, raw_lat, raw_omega, accel, gyro = _lateral_channel(trip)
    window = cfg.speed.lateral_smooth_s
    attitude_diagnostics: dict[str, Any] = {"enabled": False}
    if cfg.attitude.enabled:
        att_t, att_long, att_lat, att_omega, attitude_diagnostics = \
            _attitude_channel(trip, cfg)
        if att_t.size:
            raw_t, raw_lat, raw_omega = att_t, att_lat, att_omega
            a_long = _smooth_to_steps(times, att_t, att_long,
                                      cfg.accel_smooth_window_s)
            stationary, a_std, w_std, run_s = detect_stationary(
                times, a_long, yaw, cfg.motion)
    lat = _smooth_to_steps(times, raw_t, raw_lat, window)
    omega_smooth = _smooth_to_steps(times, raw_t, raw_omega, window)
    build_imu_samples.last_attitude_diagnostics = attitude_diagnostics

    spec_speed = np.full(len(times), np.nan)
    spec_sigma = np.full(len(times), np.nan)
    if spectral is not None and spectral.fitted:
        features = extract_features(raw_t, accel, gyro,
                                    cfg.spectral_window_s, cfg.spectral_hop_s)
        if len(features):
            predicted = spectral.predict_many(features.values)
            if spectral.prediction_smoothing_s > 0.0 and len(predicted) > 2:
                spacing = float(np.median(np.diff(features.times)))
                width = max(1, int(round(
                    spectral.prediction_smoothing_s / max(spacing, 1e-6))))
                if width > 1:
                    kernel = np.ones(width) / width
                    pad = (width // 2, width - 1 - width // 2)
                    predicted = np.convolve(
                        np.pad(predicted, pad, mode="edge"), kernel, mode="valid")
            spec_speed = np.interp(times, features.times, predicted)
            spec_sigma = np.full(len(times), spectral.deployment_sigma_ms)

    return [
        ImuSample(
            t=float(c.t), dt=float(c.dt), a_long=float(a_long[i]),
            yaw_rate=float(yaw[i]), a_lat=float(lat[i]),
            yaw_rate_smooth=float(omega_smooth[i]),
            stationary=bool(stationary[i]), stationary_run_s=float(run_s[i]),
            shock=bool(c.is_shock),
            gap=bool(c.gap_exceeded), accel_std=float(a_std[i]),
            gyro_std=float(w_std[i]), peak_accel=float(c.peak_accel_ms2),
            spectral_speed=float(spec_speed[i]), spectral_sigma=float(spec_sigma[i]),
        )
        for i, c in enumerate(stream.controls)
    ]


def build_inputs(trip: Trip, network: RoadNetwork, cfg: PacmanConfig,
                 gps_cutoff_t: Optional[float] = None) -> TrackerInputs:
    """Assemble tracker inputs from the *visible* GPS and the IMU."""
    visible = [s for s in trip.usable_locations
               if gps_cutoff_t is None or s.monotonic_time <= gps_cutoff_t]
    if not visible:
        raise ValueError("no usable GPS fix before the outage: nothing to initialise from")
    frame = network.frame
    last = visible[-1]
    heading0 = _initial_heading(visible, frame)

    spectral = _fit_spectral(trip, visible, cfg)
    samples = build_imu_samples(trip, cfg, heading0, spectral)
    if not samples:
        raise ValueError("motion samples produced an empty IMU stream")

    accel_bias0, gyro_bias0, bias_note = _initial_biases(samples, visible,
                                                         last.monotonic_time)
    origin = frame.to_local(last.latitude, last.longitude)
    course = course_to_heading(last.course) if last.has_valid_course else heading0
    edges, weights, s_values, snap_note = _initial_edges(network, origin, course, cfg)
    speed0 = float(last.speed) if last.has_valid_speed else 0.0

    return TrackerInputs(
        samples=samples, t_start=float(last.monotonic_time), init_edges=edges,
        init_log_weights=weights, init_s=s_values, speed0=speed0,
        accel_bias0=accel_bias0, gyro_bias0=gyro_bias0,
        origin_xy=(float(origin[0]), float(origin[1])), heading0=float(course),
        spectral=spectral,
        notes={
            "visible_fixes": len(visible),
            "gps_visible_until_s": float(last.monotonic_time),
            "imu_samples": len(samples),
            "bias_init": bias_note,
            "snap": snap_note,
            "spectral": spectral.to_json(),
        },
    )


def _fit_spectral(trip: Trip, visible: Sequence[LocationSample],
                  cfg: PacmanConfig) -> SpectralSpeedModel:
    """Calibrate the spectral speed model on the visible GPS window only."""
    speed_fixes = [f for f in visible if f.has_valid_speed]
    if len(speed_fixes) < 30 or not trip.motions:
        return SpectralSpeedModel(reason="no GPS speed in the visible window")
    cutoff = speed_fixes[-1].monotonic_time
    raw_t, _lat, _omega, accel, gyro = _lateral_channel(trip)
    inside = raw_t <= cutoff
    features = extract_features(raw_t[inside], accel[inside], gyro[inside],
                                cfg.spectral_window_s, cfg.spectral_hop_s)
    if not len(features):
        return SpectralSpeedModel(reason="visible window too short for one spectrum")
    rfid = trip.metadata.extra.get("rfid_import")
    intervals = rfid.get("calibration_intervals", []) if isinstance(rfid, dict) else []
    if intervals:
        return SpectralSpeedModel.fit_intervals(
            features.values,
            features.times,
            np.array([row["start_s"] for row in intervals], dtype=float),
            np.array([row["end_s"] for row in intervals], dtype=float),
            np.array([row["distance_m"] for row in intervals], dtype=float),
        )
    ft = np.array([f.monotonic_time for f in speed_fixes])
    fv = np.array([float(f.speed) for f in speed_fixes])
    return SpectralSpeedModel.fit(features.values, np.interp(features.times, ft, fv))


def _initial_heading(fixes: Sequence[LocationSample], frame: LocalFrame) -> float:
    moving = [f for f in fixes if f.has_valid_speed and (f.speed or 0.0) > 2.0]
    tail = (moving or list(fixes))[-40:]
    if len(tail) >= 2:
        a = np.asarray(frame.to_local(tail[0].latitude, tail[0].longitude))
        b = np.asarray(frame.to_local(tail[-1].latitude, tail[-1].longitude))
        d = b - a
        if float(np.hypot(*d)) > 5.0:
            return float(math.atan2(d[1], d[0]))
    for fix in reversed(fixes):
        if fix.has_valid_course:
            return course_to_heading(fix.course)
    return 0.0


def _initial_biases(samples: Sequence[ImuSample], fixes: Sequence[LocationSample],
                    t_end: float) -> tuple[float, float, dict[str, Any]]:
    """Bias seeds from stretches where the visible GPS proves the car was parked."""
    speeds = np.array([f.monotonic_time for f in fixes if f.has_valid_speed])
    stopped = np.array([f.monotonic_time for f in fixes
                        if f.has_valid_speed and (f.speed or 0.0) <= 0.4])
    trust_gps = speeds.size > 10
    a_vals: list[float] = []
    w_vals: list[float] = []
    for sample in samples:
        if sample.t > t_end:
            break
        parked = (stopped.size > 0 and float(np.min(np.abs(stopped - sample.t))) <= 0.6
                  ) if trust_gps else sample.stationary
        if parked:
            a_vals.append(sample.a_long)
            w_vals.append(sample.yaw_rate)
    accel_bias = float(np.median(a_vals)) if len(a_vals) >= 20 else 0.0
    gyro_bias = float(np.median(w_vals)) if len(w_vals) >= 20 else 0.0
    return accel_bias, gyro_bias, {
        "stationary_samples": len(a_vals),
        "source": "gps_confirmed_stop" if trust_gps else "imu_variance",
        "accel_bias": round(accel_bias, 4), "gyro_bias": round(gyro_bias, 6)}


def _initial_edges(network: RoadNetwork, origin, course: float,
                   cfg: PacmanConfig) -> tuple[list[int], list[float], list[float],
                                               dict[str, Any]]:
    """Candidate start edges, weighted by snap distance and heading agreement."""
    candidates = network.nearest_edges(origin, k=cfg.init_candidate_edges, radius=120.0)
    edges: list[int] = []
    logw: list[float] = []
    s_values: list[float] = []
    rejected = 0
    for index in candidates:
        s, offset = network.project(origin, index)
        if offset > cfg.init_max_offset_m:
            rejected += 1
            continue
        bearing = float(network.edges[index].bearing(s))
        dpsi = float(wrap_angle(bearing - course))
        edges.append(int(index))
        logw.append(-0.5 * (offset / 12.0) ** 2
                    - 0.5 * (dpsi / cfg.init_heading_sigma_rad) ** 2)
        s_values.append(float(s))
    if not edges:
        raise ValueError(
            f"no drivable edge within {cfg.init_max_offset_m:.0f} m of the last GPS fix; "
            "the graph probably does not cover the trip")
    return edges, logw, s_values, {"candidates": len(candidates),
                                   "rejected_by_offset": rejected}


# -------------------------------------------------------------------- tracker


class PacmanTracker:
    def __init__(self, network: RoadNetwork, cfg: Optional[PacmanConfig] = None,
                 geometry: Optional[RoadGeometry] = None) -> None:
        self.network = network
        self.cfg = cfg or PacmanConfig()
        self.geometry = geometry or RoadGeometry(network, self.cfg.geometry)
        self.matcher = CurvatureMatcher(self.geometry, self.cfg.match)
        self.corridors = CorridorBuilder(self.geometry, self.cfg.corridor)
        self.manager: Optional[HypothesisManager] = None
        self.speed: Optional[GlobalSpeedTracker] = None

    drift_log: list[DriftObservation]

    def seed(self, inputs: TrackerInputs) -> HypothesisSet:
        """A hypothesis starts at ``s`` on its edge, so its route offset is
        ``-s``: with ``D = 0`` at the outage, ``s = D - offset`` recovers it."""
        offsets = [-float(s) for s in inputs.init_s]
        routes = [RouteNode.root(e, inputs.t_start, o)
                  for e, o in zip(inputs.init_edges, offsets)]
        return make_set(inputs.init_edges, offsets, inputs.init_log_weights,
                        routes, inputs.t_start)

    def run(self, inputs: TrackerInputs,
            observer: Optional[StepObserver] = None) -> TrackerResult:
        started = time.perf_counter()
        cfg = self.cfg
        steps = [s for s in inputs.samples if s.t > inputs.t_start]
        integrator = HeadingIntegrator(
            np.array([s.t for s in steps]), np.array([s.yaw_rate for s in steps]),
            cfg.match.gyro_noise_rads)
        self._turn_events = detect_turns(
            np.array([s.t for s in steps]), np.array([s.yaw_rate for s in steps]),
            inputs.gyro_bias0)
        if cfg.tracker_mode in {"single_path", "single_path_rollback"}:
            cfg.single_path.rollback_enabled = cfg.tracker_mode == "single_path_rollback"
            self.manager = SinglePathManager(
                self.geometry, cfg.beam, cfg.single_path, integrator,
                self._turn_events)
        elif cfg.tracker_mode == "beam":
            self.manager = HypothesisManager(self.geometry, cfg.beam, integrator)
        else:
            raise ValueError(f"unknown tracker_mode: {cfg.tracker_mode}")
        self.speed = GlobalSpeedTracker(cfg.speed, v0=inputs.speed0,
                                        gyro_bias0=inputs.gyro_bias0,
                                        accel_bias0=inputs.accel_bias0)
        hs = self.seed(inputs)
        if cfg.tracker_mode != "beam" and len(hs) > 1:
            hs = hs.take(np.array([int(np.argmax(hs.logw))]))
            hs.logw[:] = 0.0
        self.drift_log: list[DriftObservation] = []
        self._event_candidates: dict[int, list[DriftObservation]] = {}
        self._processed_events: set[int] = set()
        self._spectral_hist: list[tuple[float, float]] = []
        self._interval_vspec_integral = 0.0
        self._anchor_drift = float("nan")
        self._anchor_sigma = float("nan")
        stop_active = False
        peak_population = len(hs)
        frames: list[TrackerFrame] = []
        speed_trace: list[SpeedSample] = []
        # Zero-latency corrected DISPLAY position (Phase 32). Runs only in
        # single-path mode and only when the flag is on; it is read-only w.r.t.
        # the tracker - no method below ever consults ``self._display``.
        self._display: Optional[DisplayPositionBranch] = None
        position_trace: Optional[list[DisplayPositionSample]] = None
        if (cfg.display.position_branch_enabled
                and isinstance(self.manager, SinglePathManager)
                and inputs.oracle_distance is None):
            model = DisplayResidualModel.load(
                cfg.display.model_path or None,
                leave_0726_out=cfg.display.leave_0726_out)
            self._display = DisplayPositionBranch(
                model, d0=self.speed.distance, frame=self.network.frame,
                edges=self.network.edges,
                correction_gain=cfg.display.correction_gain,
                max_correction_m=cfg.display.max_correction_m)
            position_trace = []
        speed_limits = self.network.speed_limits
        next_output = inputs.t_start
        last: Optional[ImuSample] = None

        for index, sample in enumerate(steps):
            last = sample
            stationary_applied = False
            if inputs.oracle_distance is not None:
                # Benchmark A: distance handed over from the withheld GPS.
                self.speed.x[0] = float(inputs.oracle_distance[index])
                if inputs.oracle_speed is not None:
                    self.speed.x[1] = float(inputs.oracle_speed[index])
                self.speed.P[0, 0] = cfg.speed.d_sigma_floor_m**2
            elif inputs.oracle_speed is not None:
                # Benchmark B: true speed, integrated here. Deliberately not
                # via `predict`, which advances the distance itself - doing
                # both counted every step twice and doubled D.
                v = float(inputs.oracle_speed[index])
                self.speed.x[0] += v * sample.dt
                self.speed.x[1] = v
                self.speed.P[1, 1] = cfg.speed.v_sigma_floor_ms**2
                self.speed.P[0, 0] += (0.02 * v * sample.dt) ** 2
            else:
                self.speed.predict(sample.a_long, sample.dt, sample.shock, sample.gap)
                if sample.stationary:
                    stationary_applied = self.speed.zero_velocity(
                        sample.a_long, sample.dt, sample.yaw_rate,
                        sample.stationary_run_s,
                        spectral_speed=sample.spectral_speed)
                if stationary_applied:
                    v_prior_display = self.speed.speed
                else:
                    self.speed.lateral_anchor(sample.a_lat, sample.yaw_rate_smooth,
                                              sample.dt, sample.shock,
                                              spectral_speed=sample.spectral_speed,
                                              t=sample.t)
                    v_prior_display = self.speed.speed   # IMU/lateral prior, pre-spectral
                    if math.isfinite(sample.spectral_speed):
                        self.speed.spectral_update(sample.spectral_speed,
                                                   sample.spectral_sigma, sample.dt)
                if self._display is not None:
                    self._display.step(
                        sample.t, sample.dt, v_route=self.speed.speed,
                        v_prior=v_prior_display, v_spectral=sample.spectral_speed,
                        d_route=self.speed.distance, stationary=stationary_applied)
            if (cfg.speed.spectral_scale_enabled
                    and math.isfinite(sample.spectral_speed)
                    and self.speed.anchor_open):
                # Running integral of the raw spectral speed since the interval
                # anchor opened - the Jacobian of an accepted map interval
                # w.r.t. k_s (Part 6 / docs).
                self._interval_vspec_integral += sample.spectral_speed * sample.dt
                limit = float(np.median(speed_limits[hs.edge])) if len(hs) else 16.7
                self.speed.envelope(limit * 1.35, sample.dt)

            if (cfg.single_path.bend_local_speed_anchor_enabled
                    and math.isfinite(sample.spectral_speed)):
                self._spectral_hist.append((sample.t, float(sample.spectral_speed)))
                if len(self._spectral_hist) > 1200:      # ~2 min at 10 Hz
                    self._spectral_hist = self._spectral_hist[-1200:]

            # A turn is reported only after its integration window matures.
            # Retaining the physical state now lets that delayed event attach
            # to D at the actual turn time rather than several seconds later.
            if (cfg.intervals.enabled or cfg.speed.fixed_lag_s > 0.0
                    or cfg.speed.lateral_delayed_correction_enabled):
                self.speed.remember(sample.t)

            distance = self.speed.distance
            hs = self.manager.advance(hs, distance, sample.t, self.speed.speed,
                                      self.speed.sigma_distance)
            if len(hs) == 0:
                break
            self.manager.resolve_turns(hs, sample.t, self.speed.gyro_bias,
                                       math.sqrt(self.speed.gyro_bias_var),
                                       self.speed.speed, distance)
            if (cfg.intervals.enabled and cfg.intervals.kind_turn
                    and inputs.oracle_distance is None
                    and isinstance(self.manager, SinglePathManager)):
                self._apply_single_path_intervals(sample.t, hs)
                distance = self.speed.distance
            if (isinstance(self.manager, SinglePathManager)
                    and self.manager.bend_anchors
                    and inputs.oracle_distance is None
                    and (cfg.single_path.bend_position_anchor_enabled
                         or cfg.single_path.bend_local_speed_anchor_enabled)):
                self._apply_bend_anchors(sample.t, hs, sample.spectral_speed)
                distance = self.speed.distance
            if (cfg.intervals.enabled and inputs.oracle_distance is None
                    and not isinstance(self.manager, SinglePathManager)):
                self._collect_drift(sample.t, hs)
                eligible_stop = (cfg.intervals.kind_stop and stationary_applied
                                 and sample.stationary_run_s
                                 >= cfg.motion.zupt_min_duration_s)
                if eligible_stop and not stop_active:
                    self._collect_stop(sample.t, hs)
                stop_active = eligible_stop
                distance = self.speed.distance
            ambiguous = self.manager.crossing_ambiguity(
                hs, distance, self.speed.sigma_distance)
            match = self.matcher.score(hs, sample, distance, self.speed.speed,
                                       self.speed.gyro_bias, sample.dt, ambiguous)
            hs = self.manager.merge(hs)
            hs = self.manager.prune(hs, distance, sample.t)
            peak_population = max(peak_population, len(hs))
            if observer is not None:
                observer.observe(sample.t, hs, sample, match, self.manager,
                                 distance, self.speed.sigma_distance,
                                 self.speed.speed)
            if sample.t >= next_output:
                frames.append(self._frame(hs, sample))
                speed_trace.append(self.speed.report(sample.t))
                if position_trace is not None:
                    position_trace.append(self._display_locate(hs, sample.t))
                next_output = sample.t + cfg.output_dt_s

        if last is not None and (not frames or frames[-1].t < last.t):
            frames.append(self._frame(hs, last))
            speed_trace.append(self.speed.report(last.t))
            if position_trace is not None:
                position_trace.append(self._display_locate(hs, last.t))

        if (cfg.intervals.enabled and inputs.oracle_distance is None
                and last is not None):
            self._collect_drift(last.t + cfg.intervals.event_settle_s + 1.0, hs)
        turns = detect_turns(np.array([s.t for s in steps]),
                             np.array([s.yaw_rate for s in steps]),
                             self.speed.gyro_bias)
        stats = {
            "tracker_mode": cfg.tracker_mode,
            "runtime_s": round(time.perf_counter() - started, 3),
            "steps": len(steps),
            "final_population": len(hs),
            "peak_population": peak_population,
            "branches": self.manager.stats.branches,
            "children": self.manager.stats.children,
            "merged": self.manager.stats.merged,
            "turns_scored": self.manager.stats.turns_scored,
            "turn_events_detected": len(turns),
            "dead_ends": self.manager.stats.dead_ends,
            "reversals": self.manager.stats.reversals,
            "offset_corrections": self.manager.stats.offset_corrections,
            "pruned_by_weight": self.manager.stats.pruned_weight,
            "pruned_by_beam": self.manager.stats.pruned_beam,
            "gyro_updates": self.matcher.updates,
            "speed": self.speed.stats(),
            "spectral": inputs.spectral.to_json(),
            "final_distance_m": round(self.speed.distance, 1),
            "final_sigma_distance_m": round(self.speed.sigma_distance, 1),
            "drift": {
                "observations": len(self.drift_log),
                "applied": sum(1 for d in self.drift_log if d.applied),
                "rejected": sum(1 for d in self.drift_log if d.reject_reason),
                "reject_reasons": _tally(d.reject_reason for d in self.drift_log
                                         if d.reject_reason),
                "total_correction_m": round(
                    sum(-d.drift_m for d in self.drift_log if d.applied), 1),
                "events": [d.to_json() for d in self.drift_log[:400]],
            },
            "accel_scale": {
                "enabled": cfg.speed.accel_scale_enabled,
                "value": round(self.speed.accel_scale, 5),
                "sigma": round(self.speed.sigma_accel_scale, 5),
            },
            "spectral_scale": {
                "enabled": cfg.speed.spectral_scale_enabled,
                "value": round(self.speed.spectral_scale_value, 5),
                "sigma": round(self.speed.sigma_spectral_scale, 5),
                "updates": self.speed.spectral_scale_updates,
                "from_lateral": self.speed.counts.get("spectral_scale_point", 0),
                "from_interval": self.speed.counts.get("spectral_scale_interval", 0),
                "trace": [[round(t, 1), round(k, 4), round(s, 4), kind]
                          for t, k, s, kind in self.speed.spectral_scale_trace],
            },
            "bend_scale": {
                "enabled": cfg.speed.bend_scale_enabled,
                "value": round(self.speed.bend_scale_value, 5),
                "sigma": round(self.speed.sigma_bend_scale, 5),
                "updates": self.speed.bend_scale_updates,
                "trace": [[round(t, 1), round(k, 4), round(s, 4)]
                          for t, k, s in self.speed.bend_scale_trace],
            },
            "spectral_censor": {
                "enabled": cfg.speed.spectral_censor_enabled,
                "saturation_known": self.speed.spectral_saturation_known,
                "plateau_ms": round(self.speed.spectral_plateau_ms, 2),
                "censored_updates": self.speed.spectral_censored_updates,
            },
            "attitude": getattr(build_imu_samples, "last_attitude_diagnostics",
                                {"enabled": False}),
        }
        if isinstance(self.manager, SinglePathManager):
            stats["single_path"] = self.manager.diagnostics()
            ivs = self.manager.turn_intervals
            applied = [iv for iv in ivs if iv.get("applied")]
            if ivs:
                stats["drift"] = {
                    "observations": len(ivs),
                    "applied": len(applied),
                    "rejected": sum(1 for iv in ivs if iv.get("reject_reason")),
                    "reject_reasons": _tally(iv["reject_reason"] for iv in ivs
                                             if iv.get("reject_reason")),
                    "total_correction_m": round(
                        sum(iv.get("D_after_m", 0.0) - iv.get("D_before_m", 0.0)
                            for iv in applied), 1),
                    "events": ivs,
                }
        retro_trace: Optional[list[SpeedSample]] = None
        if (isinstance(self.manager, SinglePathManager)
                and cfg.single_path.retro_bend_smoothing_enabled):
            retro_trace = retrospective_distance_smooth(
                speed_trace, self.manager.bend_anchors)
        if position_trace is not None:
            stats["display_position"] = {
                "enabled": True, "estimator": "iso-binary",
                "leave_0726_out": cfg.display.leave_0726_out,
                "ticks": len(position_trace),
                "gate_active_fraction": round(
                    float(np.mean([s.gate_active for s in position_trace]))
                    if position_trace else 0.0, 4),
                "final_delta_m": round(position_trace[-1].delta_m, 1) if position_trace else 0.0,
                "max_abs_delta_m": round(
                    max((abs(s.delta_m) for s in position_trace), default=0.0), 1),
                "max_excess_m": round(
                    max((s.excess_position_distance_m for s in position_trace), default=0.0), 1),
            }
        return TrackerResult(frames=frames, final=hs, stats=stats, inputs=inputs,
                             speed_trace=speed_trace, position_trace=position_trace,
                             retro_speed_trace=retro_trace)

    def _interval_reject_reason(self, iv: dict, length: float,
                                discrepancy: float) -> Optional[str]:
        """Quality / independence gate for a turn-to-turn interval.

        Replaces the old flat ``duration > 240 s`` / ``distance > 3000 m``
        cutoffs. A long interval is the *most* useful for accelerometer-scale
        observability and is not penalised for being long; it is refused only
        when the evidence at its endpoints or along its route is not clean.
        Hard bounds remain purely as numerical / runtime sanity, far above the
        old values.
        """
        cfg = self.cfg.intervals
        if not (cfg.min_length_m <= length <= cfg.interval_sanity_max_length_m):
            return f"length {length:.0f} m outside sanity bound"
        if float(iv["duration_s"]) > cfg.interval_sanity_max_duration_s:
            return f"duration {iv['duration_s']:.0f} s outside sanity bound"
        if abs(discrepancy) > cfg.max_drift_m:
            return (f"segment drift {discrepancy:.0f} m exceeds "
                    f"{cfg.max_drift_m:.0f} m (likely a wrong route, not a "
                    "wrong odometer)")
        if not iv.get("endpoint_a_ok", False):
            return (f"endpoint A weakly anchored "
                    f"(residual {iv['endpoint_residuals_m'][0]:.0f} m, angle z "
                    f"{iv['endpoint_a_angle_z']:.1f}, p "
                    f"{iv['endpoint_a_local_probability']:.2f})")
        if not iv.get("endpoint_b_ok", False):
            return (f"endpoint B weakly anchored "
                    f"(residual {iv['endpoint_residuals_m'][1]:.0f} m, angle z "
                    f"{iv['endpoint_b_angle_z']:.1f}, p "
                    f"{iv['endpoint_b_local_probability']:.2f})")
        if iv.get("interior_low_confidence", 0) > 0:
            return "a low-confidence junction lies inside the segment"
        if iv.get("upstream_unresolved_forks", 0) > 0:
            return ("an unresolved fork upstream of endpoint A leaves the "
                    "committed path (and its map length) uncertain")
        if iv.get("rolled_back_inside", False):
            return "a rollback rewound inside the segment"
        if not iv.get("unambiguous_committed_path", False):
            return "the committed map path A->B is not a single chain"
        return None

    def _apply_single_path_intervals(self, now: float, hs: HypothesisSet) -> None:
        """Apply committed turn-to-turn map-distance intervals from the single
        active path.

        Both endpoints are junctions the route reached because a preserved gyro
        turn event selected them (the distance term in the match is a soft prior
        that never selects a branch on its own), so the map length of the
        segment is not a function of the odometer it is about to constrain -
        that is the anti-circularity argument for a *committed single path*:
        independent selection, not multi-route corroboration.
        """
        mgr = self.manager
        assert isinstance(mgr, SinglePathManager) and self.speed is not None
        cfg = self.cfg.intervals
        while mgr._intervals_emitted < len(mgr.turn_intervals):
            iv = mgr.turn_intervals[mgr._intervals_emitted]
            mgr._intervals_emitted += 1
            length = float(iv["map_length_m"])
            discrepancy = float(iv["discrepancy_m"])
            reanchor = float(iv["t_b"])
            reason = self._interval_reject_reason(iv, length, discrepancy)
            if reason is not None:
                iv["reject_reason"] = reason
            elif not self.speed.anchor_open:
                iv["reject_reason"] = "no open interval anchor"
            else:
                # sigma_map grows with length; endpoint localization enters as
                # each turn centroid's distance-domain noise; timing sigma is
                # per-endpoint, not scaled by the whole interval.
                sigma = math.sqrt(
                    (cfg.sigma_map_frac * length) ** 2 + cfg.sigma_map_floor_m ** 2
                    + 2.0 * (self.cfg.single_path.event_offset_anchor_sigma_m) ** 2
                    + (cfg.sigma_timing_s * max(abs(self.speed.speed), 1.0)) ** 2)
                d_before = self.speed.distance
                v_before = self.speed.speed
                ba_before = float(self.speed.x[BA])
                ka_before = self.speed.accel_scale
                ka_sig_before = self.speed.sigma_accel_scale
                applied, ins, z = self.speed.apply_interval(
                    length, sigma, cfg.max_innovation_sigma, event_t=None)
                iv.update({
                    "applied": bool(applied), "sigma_m": round(sigma, 2),
                    "ins_length_m": round(float(ins), 2),
                    "innovation_sigma": round(float(z), 2),
                    "D_before_m": round(d_before, 1),
                    "D_after_m": round(self.speed.distance, 1),
                    "delta_D_m": round(self.speed.distance - d_before, 2),
                    "delta_v_ms": round(self.speed.speed - v_before, 4),
                    "delta_b_a_ms2": round(float(self.speed.x[BA]) - ba_before, 5),
                    "delta_k_a": round(self.speed.accel_scale - ka_before, 5),
                    "k_a_before": round(ka_before, 4),
                    "k_a_after": round(self.speed.accel_scale, 4),
                    "k_a_sigma_before": round(ka_sig_before, 4),
                    "k_a_sigma_after": round(self.speed.sigma_accel_scale, 4),
                })
                # Calibrate the spectral scale k_s from this interval - a
                # cruising interval strongly constrains k_s (Jacobian is the
                # integrated spectral speed) even where k_a is unobservable.
                # Done whether or not the D correction was applied, and it does
                # not consume the interval.
                if (self.cfg.speed.spectral_scale_enabled
                        and self._interval_vspec_integral > 30.0):
                    ks_b = self.speed.spectral_scale_value
                    ok_ks = self.speed.spectral_scale_interval(
                        length, sigma, self._interval_vspec_integral, t=reanchor)
                    iv.update({
                        "k_s_before": round(ks_b, 4),
                        "k_s_after": round(self.speed.spectral_scale_value, 4),
                        "k_s_updated": bool(ok_ks),
                        "integrated_v_spectral_m": round(self._interval_vspec_integral, 1),
                    })
                if applied:
                    delta_d = self.speed.distance - d_before
                    # The interval is a *better* estimate of the same odometer-
                    # vs-map disagreement the per-junction event matches have
                    # been tracking in offset_bias. Fold the correction into
                    # offset_bias rather than adding it (which would leave the
                    # old per-junction estimate stacked on top - the double
                    # correction) and clip so one interval cannot dominate.
                    take = float(np.clip(
                        cfg.interval_offset_bias_fraction * delta_d,
                        -cfg.interval_offset_bias_max_m,
                        cfg.interval_offset_bias_max_m))
                    hs.offset_bias += (delta_d - take)
                    hs.map_anchor_distance += delta_d
                    # odometer-frame bookkeeping moves by the full D delta
                    mgr.apply_common_distance_shift(delta_d, after_t=reanchor)
                    iv["route_offset_bias_shift_m"] = round(delta_d - take, 2)
                    iv["route_position_advance_m"] = round(take, 2)
                else:
                    iv["reject_reason"] = f"innovation {z:.2f} sigma exceeds gate"
            self.speed.open_interval(reanchor)
            self._interval_vspec_integral = 0.0
            mgr.interval_anchor_request = None
        req = mgr.interval_anchor_request
        if req is not None:
            mgr.interval_anchor_request = None
            self.speed.open_interval(float(req))
            self._interval_vspec_integral = 0.0

    def _apply_bend_anchors(self, now: float, hs: HypothesisSet,
                            spectral_now: float) -> None:
        """Apply accepted intra-edge bend anchors (Phase 17).

        Position: the bend's along-edge position is an absolute route-distance
        measurement. It is folded in exactly like a map interval - through the
        shared ``speed`` update (``k_a`` protected), ``offset_bias`` and the
        common-mode shift - so the active edge and route topology never move,
        only the along-route position.

        Speed: the bend's fitted local speed ``v_bar`` is a high-speed
        ``v_true`` sample; it calibrates ``k_high`` (``bend_scale``), a state
        separate from ``k_s`` and ``k_a``.
        """
        mgr = self.manager
        assert isinstance(mgr, SinglePathManager) and self.speed is not None
        cfg = self.cfg.single_path
        settle = self.cfg.single_path.event_settle_s + 2.0
        for ba in mgr.bend_anchors:
            if now < ba["t_peak"] + settle:
                continue
            if (cfg.bend_position_anchor_enabled and not ba["applied_position"]
                    and math.isfinite(ba["residual_s_m"])):
                ba["applied_position"] = True
                d_before = self.speed.distance
                H = np.zeros(SPEED_DIM)
                H[D] = 1.0
                self.speed._update(H, float(ba["residual_s_m"]),
                                   cfg.bend_position_anchor_sigma_m ** 2,
                                   allow_scale=False)
                delta_d = self.speed.distance - d_before
                take = float(np.clip(
                    self.cfg.intervals.interval_offset_bias_fraction * delta_d,
                    -self.cfg.intervals.interval_offset_bias_max_m,
                    self.cfg.intervals.interval_offset_bias_max_m))
                hs.offset_bias += (delta_d - take)
                hs.map_anchor_distance += delta_d
                mgr.apply_common_distance_shift(delta_d, after_t=ba["t_peak"])
                ba["delta_D_m"] = round(float(delta_d), 2)
                ba["t_applied"] = float(now)
            if (cfg.bend_local_speed_anchor_enabled and not ba["applied_speed"]):
                ba["applied_speed"] = True
                # robust median spectral speed over the SAME event window - the
                # instantaneous sample is far too noisy a denominator.
                win = [v for t, v in self._spectral_hist
                       if ba["t_start"] <= t <= ba["t_end"] and math.isfinite(v)]
                v_spec_event = (float(np.median(win)) if len(win) >= 3
                                else float(spectral_now))
                ba["v_spectral_event_ms"] = round(v_spec_event, 2)
                if math.isfinite(v_spec_event) and v_spec_event > 1e-6:
                    # A bend faster than the concurrent spectral reading proves
                    # the source under-reads there: it is saturated, not merely
                    # noisy. This is independent of whether k_high is estimated.
                    if (float(ba["v_bar_ms"]) >
                            v_spec_event + self.cfg.speed.spectral_saturation_margin_ms):
                        self.speed.spectral_saturation_known = True
                        self.speed.spectral_plateau_ms = max(
                            self.speed.spectral_plateau_ms, v_spec_event)
                    if self.cfg.speed.bend_scale_enabled:
                        sigma_v = max(0.5, 0.12 * abs(ba["v_bar_ms"]))
                        ok = self.speed.bend_speed_anchor(
                            float(ba["v_bar_ms"]), sigma_v, v_spec_event,
                            t=ba["t_peak"])
                        ba["k_high_after"] = round(self.speed.bend_scale_value, 4)
                        ba["speed_anchor_accepted"] = bool(ok)

    def _collect_drift(self, now: float, hs: HypothesisSet) -> None:
        """Collect route anchors and emit at most one constraint per real turn.

        Hypotheses mature their crossing windows at different times.  The
        inherited prototype applied every one of those as a new observation;
        on 07-26 that turned three physical turns into 145 candidates and 73
        Kalman updates.  Here a candidate must first coincide with a turn found
        directly in the gyro, then the strongest independently corroborated
        candidate for that physical event is selected after the event settles.
        The detector and the route support are both GPS-free.
        """
        assert self.speed is not None and self.manager is not None
        cfg = self.cfg.intervals
        if not cfg.kind_turn:
            for centre, innovations, weights, evidence_ids in self.manager.pending_drift:
                observation = common_drift(
                    cfg, centre, innovations, weights, evidence_ids, self.speed.speed)
                observation.reject_reason = "turn intervals disabled"
                self.drift_log.append(observation)
            self.manager.pending_drift.clear()
            return
        for centre, innovations, weights, evidence_ids in self.manager.pending_drift:
            match = None
            for j, event in enumerate(self._turn_events):
                if (event.t_start - cfg.event_match_margin_s <= centre
                        <= event.t_end + cfg.event_match_margin_s):
                    match = j
                    break
            observation = common_drift(
                cfg, centre, innovations, weights, evidence_ids, self.speed.speed)
            if match is None:
                observation.reject_reason = "no matching IMU turn event"
                self.drift_log.append(observation)
            elif match not in self._processed_events:
                self._event_candidates.setdefault(match, []).append(observation)
        self.manager.pending_drift.clear()

        for j, event in enumerate(self._turn_events):
            if j in self._processed_events or now < event.t_end + cfg.event_settle_s:
                continue
            candidates = self._event_candidates.pop(j, [])
            self._processed_events.add(j)
            if not candidates:
                self.drift_log.append(DriftObservation(
                    event.t_peak, 0.0, float("inf"), 0.0, 0.0, 0, 0,
                    reject_reason="turn had no independently supported route anchor"))
                continue
            valid = [c for c in candidates if not c.reject_reason]
            pool = valid or candidates
            chosen = max(pool, key=lambda c: (
                not bool(c.reject_reason), c.mass, c.effective_routes,
                -c.spread_m, c.n_hypotheses))
            self._apply_turn_interval(chosen, hs)
            self.drift_log.append(chosen)

    def _apply_turn_interval(self, observation: DriftObservation,
                             hs: HypothesisSet) -> None:
        """Apply one turn-to-turn integrated map-distance observation.

        At event i each route reports r_i = D_i - M_i.  Subtracting two events
        removes the route's arbitrary origin:

            M_B - M_A = (D_B - D_A) - (r_B - r_A)

        The right hand side is evaluated before the update.  This is an
        integral distance measurement, never an average-speed pseudo-update.

        Bookkeeping keeps the two uses disjoint. Shifting ``offset_bias`` by the
        same amount as ``D`` leaves every hypothesis's position on the map, and
        its own alignment residual, exactly where they were - so the evidence
        is spent once globally and once locally, never twice on the same
        quantity.
        """
        assert self.speed is not None
        cfg = self.cfg.intervals
        if observation.reject_reason or not math.isfinite(observation.sigma_m):
            return
        if not self.speed.anchor_open:
            self.speed.open_interval(observation.t)
            self._anchor_drift = observation.drift_m
            self._anchor_sigma = observation.sigma_m
            observation.anchor_started = True
            return

        observation.anchor_t = self.speed.anchor_t
        observation.anchor_drift_m = self._anchor_drift
        duration = observation.t - self.speed.anchor_t
        if duration <= 0.0:
            observation.reject_reason = "non-monotonic turn event"
            return
        if duration > cfg.max_duration_s:
            observation.reject_reason = (
                f"duration {duration:.0f} s exceeds {cfg.max_duration_s:.0f} s")
            self.speed.open_interval(observation.t)
            self._anchor_drift = observation.drift_m
            self._anchor_sigma = observation.sigma_m
            observation.anchor_started = True
            return

        ins = self.speed.travelled_since_anchor_at(observation.t)
        if not math.isfinite(ins):
            observation.reject_reason = "turn state fell outside retained history"
            return
        discrepancy = observation.drift_m - self._anchor_drift
        length = ins - discrepancy
        observation.ins_length_m = ins
        observation.map_length_m = length
        observation.discrepancy_m = discrepancy
        if abs(discrepancy) > cfg.max_drift_m:
            observation.reject_reason = (
                f"interval drift {discrepancy:.0f} m exceeds {cfg.max_drift_m:.0f} m")
            return
        if not (cfg.min_length_m <= length <= cfg.max_length_m):
            observation.reject_reason = f"length {length:.0f} m out of range"
            return

        sigma = math.sqrt(
            observation.sigma_m**2 + self._anchor_sigma**2
            + (cfg.sigma_map_frac * length)**2)
        observation.accel_bias_before = float(self.speed.x[BA])
        observation.accel_scale_before = self.speed.accel_scale
        observation.distance_before = self.speed.distance
        event_before = self.speed.distance_at(observation.t)
        applied, _, z = self.speed.apply_interval(
            length, sigma, cfg.max_innovation_sigma, event_t=observation.t)
        observation.applied = applied
        observation.innovation_sigma = z
        observation.states_modified = self.speed.last_interval_states_modified
        observation.lag_s = min(self.cfg.speed.fixed_lag_s, duration)
        observation.accel_bias_after = float(self.speed.x[BA])
        observation.accel_scale_after = self.speed.accel_scale
        observation.distance_after = self.speed.distance
        if not applied:
            observation.reject_reason = f"innovation {z:.2f} sigma exceeds gate"
            return

        # Preserve every route's map position while moving the common physical
        # distance. Differential route mismatch remains in offset_bias.
        hs.offset_bias += observation.distance_after - observation.distance_before
        event_after = self.speed.distance_at(observation.t)
        self.speed.open_interval(observation.t)
        self._anchor_drift = observation.drift_m + (event_after - event_before)
        self._anchor_sigma = observation.sigma_m

    def _collect_stop(self, t: float, hs: HypothesisSet) -> None:
        """Use a stop as an interval endpoint only after map localization.

        A stop supplies exact boundary velocity but, by itself, no position on
        the map. Treating a quiet patch as a map-distance anchor would be
        circular because its along-edge position was computed from D. A prior
        turn correction does not change that: it localises the turn, not the
        later stop. Until the map front end has an independent stop landmark
        (for example a mapped gate or charger observation), stop-to-stop map
        distance must therefore be rejected. ZUPT still supplies both endpoint
        velocity constraints to the physical filter.
        """
        assert self.speed is not None and self.manager is not None
        cfg = self.cfg.intervals
        obs = common_drift(cfg, t, hs.offset_bias.copy(), hs.weights(),
                           hs.edge.copy(), 0.0)
        obs.event_kind = "stop"
        obs.reject_reason = "stop is not independently map-localized"
        self.drift_log.append(obs)

    def _display_locate(self, hs: HypothesisSet, t: float) -> DisplayPositionSample:
        """Ask the read-only display branch to project ``D_position`` onto the
        route the tracker has *already* committed. Uses ``hs`` only to read the
        committed edge chain and its offset - never writes to it."""
        i = int(np.argmax(hs.logw)) if len(hs) > 1 else 0
        route_edges = hs.routes[i].edges() if hs.routes else [int(hs.edge[i])]
        return self._display.locate(
            t, route_edges=route_edges, edge_index=int(hs.edge[i]),
            route_offset=float(hs.route_offset[i]),
            offset_bias=float(hs.offset_bias[i]))

    def _frame(self, hs: HypothesisSet, sample: ImuSample) -> TrackerFrame:
        distance = self.speed.distance
        sigma = self.speed.sigma_distance
        map_sigma = self.manager.map_distance_sigma(hs, distance)
        total_sigma = np.hypot(sigma, map_sigma)
        top = hs.snapshot(distance, total_sigma, limit=8)
        best = top[0] if top else None
        if best is not None:
            edge = self.network.edges[best.edge]
            xy = edge.position(float(np.clip(best.s, 0.0, edge.length)))
        else:
            xy = (float("nan"), float("nan"))
        return TrackerFrame(
            t=float(sample.t),
            corridors=self.corridors.build(
                hs, float(sample.t), distance, sigma, map_sigma=map_sigma),
            top=top, position=(float(xy[0]), float(xy[1])), population=len(hs),
            stationary=bool(sample.stationary), speed=self.speed.report(sample.t),
            alive_edges=np.unique(hs.edge),
        )
