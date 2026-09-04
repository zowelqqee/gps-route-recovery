"""End-to-end reconstruction.

One pass over a merged IMU + GPS timeline drives, in lock-step:

  * the GPS trust state machine,
  * an EKF (which is both the `ekf_dead_reckoning` baseline and the source of
    the predicted position/covariance the Mahalanobis gate needs),
  * the `last_known_position` baseline,
  * and, for the main algorithm, the road particle filter.

Running all three together costs almost nothing beyond the particle filter and
means the metrics always contain a real comparison rather than a claim.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from geotrace.config import Config
from geotrace.coordinates import LocalFrame, course_to_heading, heading_to_course, wrap_angle
from geotrace.ekf import ExtendedKalmanFilter
from geotrace.gps_quality import GPSQualityMonitor, GPSState, measurement_sigma, state_intervals
from geotrace.metrics import (
    ErrorSeries,
    MetricsBundle,
    branch_accuracy,
    compute_error_series,
    coverage_and_area,
    gate_metrics,
    outage_end_error,
    trust_recovery_time,
)
from geotrace.models import LocationSample, Trip
from geotrace.parity_rng import make_rng
from geotrace.motion_model import build_imu_stream, estimate_initial_biases
from geotrace.particle_filter import RoadParticleFilter
from geotrace.parking_tracker import ParkingResult, ParkingTracker
from geotrace.polygons import UncertaintySet, branch_aware_estimate, build_uncertainty_set
from geotrace.road_graph import RoadNetwork
from geotrace.road_tracker import RoadTracker

ALGORITHMS = ("last_known_position", "ekf_dead_reckoning", "road_particle_filter")


class ReconstructionError(RuntimeError):
    pass


@dataclass
class Track:
    """A time series of estimated positions in the local frame."""

    name: str
    times: list[float] = field(default_factory=list)
    xy: list[tuple[float, float]] = field(default_factory=list)
    segment_starts: list[int] = field(default_factory=lambda: [0])
    _break_pending: bool = field(default=False, init=False, repr=False)

    def break_line(self) -> None:
        """Start a new visible segment at the next estimate.

        A GPS re-anchor is a correction, not a driven straight line between the
        stale inertial estimate and the recovered observation.  Keeping that
        discontinuity in the time series is useful; connecting it on a map is
        not.
        """
        self._break_pending = True

    def add(self, t: float, point: Sequence[float]) -> None:
        if self._break_pending and self.xy:
            self.segment_starts.append(len(self.xy))
        self._break_pending = False
        self.times.append(float(t))
        self.xy.append((float(point[0]), float(point[1])))

    @property
    def array(self) -> np.ndarray:
        return np.array(self.xy, dtype=float) if self.xy else np.zeros((0, 2))

    def to_geojson(self, frame: LocalFrame, properties: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        coords = frame.coords_to_geojson(self.array) if self.xy else []
        starts = self.segment_starts + [len(coords)]
        segments = [coords[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]
        segments = [segment for segment in segments if segment]
        props = {"name": self.name, "point_count": len(self.xy), "segment_count": len(segments)}
        if properties:
            props.update(properties)
        geometry = (
            {"type": "MultiLineString", "coordinates": segments}
            if len(segments) > 1
            else {"type": "LineString", "coordinates": coords}
        )
        return {
            "type": "Feature",
            "properties": props,
            "geometry": geometry,
        }


@dataclass
class ReconstructionResult:
    algorithm: str
    frame: LocalFrame
    tracks: dict[str, Track]
    uncertainty: list[UncertaintySet]
    gate_history: list[Any]
    diagnostics: dict[str, Any]
    gps_states: list[dict[str, Any]]
    network: Optional[RoadNetwork] = None
    particle_filter: Optional[RoadParticleFilter] = None
    outage_windows: list[dict[str, float]] = field(default_factory=list)
    parking_result: Optional[ParkingResult] = None

    @property
    def primary(self) -> Track:
        return self.tracks[self.algorithm]


@dataclass
class TrackingPipeline:
    """Production coordinator for independent road and parking trackers."""

    trip: Trip
    network: Optional[RoadNetwork]
    cfg: Config
    algorithm: str = "road_particle_filter"
    output_dt: float = 1.0
    progress: Optional[Any] = None

    def run(self) -> ReconstructionResult:
        return _run_tracking_pipeline(
            self.trip, self.network, self.cfg, self.algorithm, self.output_dt, self.progress
        )


def initial_heading(locations: Sequence[LocationSample], frame: LocalFrame, cfg: Config) -> float:
    """Heading at t0.

    Preference order: a trusted GPS course while actually moving, then the
    bearing between the first fix and the first fix ~20 m away. The magnetometer
    is never used - inside a car it is not reliable enough to seed a heading.
    """
    for sample in locations:
        if (
            sample.has_valid_course
            and sample.has_valid_speed
            and float(sample.speed or 0) >= cfg.gps.min_speed_for_course_ms
        ):
            return course_to_heading(float(sample.course or 0.0))
    if len(locations) >= 2:
        origin = frame.to_local(locations[0].latitude, locations[0].longitude)
        for sample in locations[1:]:
            point = frame.to_local(sample.latitude, sample.longitude)
            if math.dist(point, origin) >= 20.0:
                return math.atan2(point[1] - origin[1], point[0] - origin[0])
    return 0.0


def run_reconstruction(
    trip: Trip,
    network: Optional[RoadNetwork],
    cfg: Config,
    algorithm: str = "road_particle_filter",
    output_dt: float = 1.0,
    progress: Optional[Any] = None,
) -> ReconstructionResult:
    """Run the baseline dual-tracker coordinator."""
    return TrackingPipeline(trip, network, cfg, algorithm, output_dt, progress).run()


def _run_tracking_pipeline(
    trip: Trip,
    network: Optional[RoadNetwork],
    cfg: Config,
    algorithm: str = "road_particle_filter",
    output_dt: float = 1.0,
    progress: Optional[Any] = None,
) -> ReconstructionResult:
    """Run the whole filter chain over one trip."""
    if algorithm not in ALGORITHMS:
        raise ReconstructionError(
            f"unknown algorithm '{algorithm}'. Choose one of: {', '.join(ALGORITHMS)}"
        )
    if algorithm == "road_particle_filter" and network is None:
        raise ReconstructionError(
            "road_particle_filter needs a road graph. Pass --graph, or download "
            'one with: geotrace download-map --place "Saint Petersburg, Russia" '
            "--output cache/spb.graphml"
        )

    usable = trip.usable_locations
    if not usable:
        raise ReconstructionError(
            "the trip contains no usable GPS fix, so there is no origin to "
            "anchor the local frame to"
        )
    if not trip.motions and algorithm != "last_known_position":
        raise ReconstructionError(
            f"{algorithm} needs motion samples, but samples.jsonl contains none. "
            "Record the trip with CoreMotion enabled, or use "
            "--algorithm last_known_position."
        )

    started = time.perf_counter()
    frame = network.frame if network is not None else LocalFrame(usable[0].latitude, usable[0].longitude)
    heading0 = initial_heading(usable, frame, cfg)

    imu = build_imu_stream(
        trip.motions,
        cfg.motion,
        calibration=trip.metadata.calibration,
        reference_heading_rad=heading0,
        t_start=trip.t0,
        t_end=trip.t0 + trip.duration_s,
    )
    if not imu.controls and algorithm != "last_known_position":
        raise ReconstructionError("motion samples produced an empty control stream")

    # Biases come from the aligned stream, so the longitudinal projection uses
    # the same world frame the filters will.
    bias_a, bias_w = estimate_initial_biases(imu, heading0, cfg.motion)

    graph_fit = _graph_fit(usable, frame, network, cfg)

    origin_xy = frame.to_local(usable[0].latitude, usable[0].longitude)
    speed0 = float(usable[0].speed or 0.0) if usable[0].has_valid_speed else 0.0

    ekf = ExtendedKalmanFilter(
        cfg.motion,
        initial_state=[origin_xy[0], origin_xy[1], speed0, heading0, bias_a, bias_w],
    )
    monitor = GPSQualityMonitor(cfg.gps, max_accel_ms2=cfg.motion.max_accel_ms2)
    pf: Optional[RoadParticleFilter] = None
    if algorithm == "road_particle_filter":
        pf = RoadTracker(network, cfg, rng=make_rng(cfg.seed, cfg.rng_mode))

    tracks = {
        "last_known_position": Track("last_known_position"),
        "ekf_dead_reckoning": Track("ekf_dead_reckoning"),
    }
    if pf is not None:
        tracks["road_particle_filter"] = Track("road_particle_filter")

    uncertainty: list[UncertaintySet] = []
    gps_states: list[dict[str, Any]] = []
    last_known_xy = origin_xy
    last_trusted_t = trip.t0
    accepted_points: list[dict[str, Any]] = []
    rejected_points: list[dict[str, Any]] = []

    # Index the fixes by time so the loop can consume them in order.
    fixes = sorted(usable, key=lambda s: s.monotonic_time)
    # A terminal observation is only valid when the recorder explicitly ended
    # the trip.  Synthetic/open-ended traces may also slow down near their last
    # sample, but that is not evidence that the driver has parked.
    fix_cursor = 0
    previous_state = monitor.state
    reanchors: list[dict[str, Any]] = []
    map_assists: list[dict[str, Any]] = []
    next_output = trip.t0
    controls = imu.controls if imu.controls else _synthetic_controls(fixes, cfg)

    for control in controls:
        t = control.t
        ekf.predict(control.a_world, control.yaw_rate, control.dt)
        # A quiet IMU only means a stop if we also believe we are barely
        # moving; otherwise it just means steady cruising.
        #
        # GPS trust is deliberately evaluated before the road filter advances.
        if control.is_quiet and ekf.speed <= cfg.motion.zupt_max_speed_ms:
            ekf.zero_velocity_update()

        monitor.note_gap(t)
        if pf is not None and pf.initialized:
            # The particle cloud keeps moving through the road graph during an
            # outage, but is never reweighted by absent or rejected GPS.  It is
            # therefore a causal map prior, not retrospective map matching.
            pf.predict(control.a_world, control.yaw_rate, control.dt)
            if control.is_quiet and ekf.speed <= cfg.motion.zupt_max_speed_ms:
                pf.zero_velocity_update(cfg.motion.zupt_max_speed_ms)

        while fix_cursor < len(fixes) and fixes[fix_cursor].monotonic_time <= t:
            fix = fixes[fix_cursor]
            fix_cursor += 1
            xy = frame.to_local(fix.latitude, fix.longitude)
            road_distance = network.distance_to_road(xy) if network is not None else None
            result = monitor.update(
                fix,
                xy,
                predicted_xy=ekf.position,
                predicted_speed=ekf.speed,
                predicted_heading=ekf.heading,
                covariance=ekf.P,
                road_distance_m=road_distance,
            )
            sigma = measurement_sigma(fix, cfg.gps)
            course_rad = (
                course_to_heading(float(fix.course or 0.0))
                if (
                    fix.has_valid_course
                    and fix.has_valid_speed
                    and float(fix.speed or 0.0) >= cfg.gps.min_speed_for_course_ms
                )
                else None
            )
            # Trust has just come back after an outage: both filters must be
            # re-anchored on the recovered fix rather than blended with a stale
            # dead-reckoning solution.
            restored = (
                result.accepted
                and monitor.state is GPSState.TRUSTED
                and previous_state is not GPSState.TRUSTED
                and previous_state is not None
            )
            record = {
                "t": round(fix.monotonic_time - trip.t0, 2),
                "latitude": fix.latitude,
                "longitude": fix.longitude,
                "state": result.state.value,
                "reasons": result.reasons,
                "horizontal_accuracy": fix.horizontal_accuracy,
                # Gate margins, so a rejection can be compared against another
                # implementation rather than only counted.
                "mahalanobis": result.mahalanobis,
                "distance_m": result.distance_m,
                "max_distance_m": result.max_distance_m,
                "road_distance_m": result.road_distance_m,
                "sigma_m": result.sigma_m,
                "predicted_speed": round(float(ekf.speed), 4),
                "predicted_heading_deg": round(heading_to_course(float(ekf.heading)), 3),
            }
            if result.accepted:
                accepted_points.append(record)
                if restored and _had_real_outage(monitor):
                    ekf.reanchor(
                        xy,
                        sigma,
                        heading_rad=course_rad,
                        speed=float(fix.speed) if fix.has_valid_speed else None,
                    )
                    reanchors.append(
                        {"t": round(fix.monotonic_time - trip.t0, 2), "sigma_m": round(sigma, 2)}
                    )
                    # Do not draw a fictitious cross-city leg from the final
                    # inertial estimate to the GPS correction.
                    for track in tracks.values():
                        track.break_line()
                    if pf is not None and pf.initialized:
                        pf.reinitialize(
                            xy,
                            heading=course_rad if course_rad is not None else ekf.heading,
                            speed=ekf.speed,
                            sigma=sigma,
                        )
                # A RECOVERING point is evidence for the GPS state machine,
                # not yet a position measurement.  Applying it here caused a
                # kilometre-scale jump from stale inertial position to the
                # first recovery candidate before the required consecutive
                # confirmations had established TRUSTED GPS.
                if monitor.state is GPSState.TRUSTED:
                    ekf.update_position(xy, sigma)
                    if fix.has_valid_speed:
                        ekf.update_speed(
                            float(fix.speed or 0.0), max(1.0, float(fix.speed_accuracy or 1.0))
                        )
                    if (
                        fix.has_valid_course
                        and fix.has_valid_speed
                        and float(fix.speed or 0.0) >= cfg.gps.min_speed_for_course_ms
                    ):
                        ekf.update_heading(course_to_heading(float(fix.course or 0.0)), math.radians(25.0))
                    last_known_xy = xy
                    last_trusted_t = fix.monotonic_time
                    if pf is not None:
                        if not pf.initialized:
                            pf.initialize(
                                xy,
                                heading=ekf.heading,
                                speed=ekf.speed,
                                accel_bias=bias_a,
                                gyro_bias=bias_w,
                                position_sigma=sigma,
                            )
                        else:
                            pf.update_weights(
                                gps_xy=xy,
                                gps_sigma=sigma,
                                gps_course_rad=(
                                    course_to_heading(float(fix.course or 0.0))
                                    if fix.has_valid_course
                                    else None
                                ),
                                gps_speed=float(fix.speed) if fix.has_valid_speed else None,
                            )
                            if pf.has_diverged():
                                # Every particle is impossible under this fix:
                                # the filter followed the wrong branch.
                                pf.reinitialize(xy, ekf.heading, ekf.speed, sigma)
                            else:
                                pf.maybe_resample()
                                pf.inject_from_fix(xy, ekf.heading, ekf.speed, sigma)
            else:
                rejected_points.append(record)

            previous_state = monitor.state
            gps_states.append(
                {
                    "t": round(fix.monotonic_time - trip.t0, 2),
                    "state": result.state.value,
                    "accepted": result.accepted,
                    "reasons": result.reasons,
                }
            )

        if t + 1e-9 >= next_output:
            since_trusted = max(0.0, t - last_trusted_t)
            tracks["last_known_position"].add(t, last_known_xy)
            tracks["ekf_dead_reckoning"].add(t, ekf.position)

            if pf is not None and pf.initialized:
                if monitor.state is GPSState.TRUSTED:
                    pf.update_weights()
                    pf.maybe_resample()
                pf.snapshot(
                    t,
                    gps_state=monitor.state.value,
                    seconds_since_trusted=since_trusted,
                )
                unc = build_uncertainty_set(
                    network,  # type: ignore[arg-type]
                    pf.edge_idx,
                    pf.s,
                    pf.w,
                    cfg.polygon,
                    t=t,
                    gps_state=monitor.state.value,
                    seconds_since_trusted=since_trusted,
                )
                uncertainty.append(unc)
                displayed_xy = ekf.position
                if monitor.state is not GPSState.TRUSTED:
                    assist = _outage_map_assist(network, pf, unc, ekf.position, cfg)
                    if assist is not None:
                        displayed_xy, map_xy, probability, spread = assist
                        map_assists.append(
                            {
                                "t": round(t - trip.t0, 2),
                                "probability": round(probability, 3),
                                "spread_m": round(spread, 2),
                                "map_offset_m": round(float(math.dist(map_xy, ekf.position)), 2),
                            }
                        )
                tracks["road_particle_filter"].add(t, displayed_xy)
            elif algorithm == "road_particle_filter":
                # Bootstrap before the first trusted GPS correction has no
                # particle cloud yet, but it still has an EKF trajectory.
                tracks["road_particle_filter"].add(t, ekf.position)
                uncertainty.append(
                    _circular_uncertainty(
                        t, ekf.position, ekf, cfg, monitor.state.value,
                        since_trusted, "imu_dead_reckoning",
                    )
                )
            elif algorithm != "road_particle_filter":
                uncertainty.append(
                    _circular_uncertainty(
                        t,
                        ekf.position if algorithm == "ekf_dead_reckoning" else last_known_xy,
                        ekf,
                        cfg,
                        monitor.state.value,
                        since_trusted,
                        algorithm,
                    )
                )
            next_output += output_dt

    elapsed = time.perf_counter() - started
    outage_windows = _outage_windows(monitor.history, trip.t0, cfg)
    road_track = tracks.get("road_particle_filter") or tracks["ekf_dead_reckoning"]
    parking_end = trip.t0 + trip.duration_s
    parking_start = parking_end - cfg.parking_tracker.window_s
    if road_track.xy:
        road_index = int(np.argmin(np.abs(np.asarray(road_track.times) - parking_start)))
        road_prior = road_track.array[road_index]
        if len(road_track.xy) > 1:
            next_index = min(road_index + 1, len(road_track.xy) - 1)
            delta = road_track.array[next_index] - road_prior
            road_heading = math.atan2(delta[1], delta[0]) if np.linalg.norm(delta) > 0.1 else heading0
        else:
            road_heading = heading0
    else:
        road_prior, road_heading = origin_xy, heading0
    parking_result = ParkingTracker(cfg, frame).run(
        controls=controls,
        fixes=fixes,
        ended_at=parking_end,
        road_prior_xy=road_prior,
        road_heading=road_heading,
        road_speed=float(ekf.speed),
        calibration_present=trip.metadata.calibration is not None,
        road_distance=(network.distance_to_road if network is not None else None),
    )

    diagnostics: dict[str, Any] = {
        "algorithm": algorithm,
        "tracking_architecture": "dual-tracker-v1",
        "seed": cfg.seed,
        "frame_origin": {"latitude": frame.lat0, "longitude": frame.lon0},
        "initial_heading_deg": round(heading_to_course(heading0), 2),
        "initial_bias_estimate": {"accel_ms2": round(bias_a, 4), "gyro_rads": round(bias_w, 5)},
        "imu": {
            "control_steps": len(controls),
            "filter_dt_s": cfg.motion.filter_dt_s,
            "heading_reference_offset_deg": round(math.degrees(imu.heading_reference_offset), 2),
            "raw_motion_samples": len(trip.motions),
        },
        "gps": monitor.summary(),
        "gps_state_intervals": state_intervals(monitor.history),
        "outage_windows": outage_windows,
        "accepted_fixes": accepted_points,
        "rejected_fixes": rejected_points,
        "ekf": {"skipped_gaps": ekf.skipped_gaps, "final_state": ekf.state_json()},
        "reanchors": reanchors,
        "outage_map_assists": map_assists,
        "runtime_s": round(elapsed, 3),
        "road_graph": (
            None
            if network is None
            else {
                "edges": len(network),
                "nodes": len(network.node_xy),
                **graph_fit,
            }
        ),
        "road_result": {
            "endpoint": list(map(float, road_track.array[-1])) if road_track.xy else None,
            "confidence": (cfg.polygon.confidence if algorithm == "road_particle_filter" else None),
        },
        "parking_result": parking_result.to_json(frame),
        "final_vehicle_position": parking_result.to_json(frame)["position"],
        "final_vehicle_position_source": "parking_tracker",
    }
    if pf is not None:
        diagnostics["particle_filter"] = pf.result.to_json()
        diagnostics["particle_filter"]["n_particles"] = cfg.pf.n_particles
        diagnostics["particle_filter"]["initialized"] = pf.initialized
        if not pf.initialized:
            diagnostics["particle_filter"]["warning"] = (
                "GPS never reached TRUSTED, so the particle filter was never "
                "seeded. Nothing can be reconstructed from this trip."
            )

    return ReconstructionResult(
        algorithm=algorithm,
        frame=frame,
        tracks=tracks,
        uncertainty=uncertainty,
        gate_history=monitor.history,
        diagnostics=diagnostics,
        gps_states=gps_states,
        network=network,
        particle_filter=pf,
        outage_windows=outage_windows,
        parking_result=parking_result,
    )


def _graph_fit(
    locations: Sequence[LocationSample],
    frame: LocalFrame,
    network: Optional[RoadNetwork],
    cfg: Config,
) -> dict[str, Any]:
    """How well the road graph actually covers the recorded track.

    A graph centred on the right city but not containing the roads that were
    driven produces a plausible-looking but wrong reconstruction, because the
    particle filter will happily snap the trip onto whatever streets it does
    know. Measuring the track-to-road distance up front turns that silent
    failure into a stated one.
    """
    if network is None:
        return {}
    sample = locations[:: max(1, len(locations) // 60)]
    distances = [
        network.distance_to_road(frame.to_local(s.latitude, s.longitude)) for s in sample
    ]
    if not distances:
        return {}
    median = float(np.median(distances))
    fit = {
        "median_track_to_road_m": round(median, 2),
        "max_track_to_road_m": round(float(np.max(distances)), 2),
    }
    if median > cfg.gps.max_median_track_to_road_m:
        fit["warning"] = (
            f"The recorded track sits a median {median:.0f} m from the nearest "
            "road in this graph. The graph probably does not cover the roads "
            "that were actually driven, and the reconstruction will be snapped "
            "to the wrong streets. Download a graph covering this area."
        )
    return fit


def _had_real_outage(monitor: GPSQualityMonitor) -> bool:
    """True once the trip has lost GPS *after* having trusted it.

    Every trip begins in LOST, because nothing is trusted until several
    consistent fixes have arrived. That opening period is bootstrap, not an
    outage, and the first promotion to TRUSTED must not be treated as a
    recovery: there is no stale dead-reckoning solution to re-anchor away from.
    """
    trusted_yet = False
    for item in monitor.history:
        if item.state is GPSState.TRUSTED:
            trusted_yet = True
        elif trusted_yet and item.state is GPSState.LOST:
            return True
    return False


def _synthetic_controls(fixes: Sequence[LocationSample], cfg: Config) -> list[Any]:
    """A zero-IMU control stream so `last_known_position` works without motion."""
    from geotrace.motion_model import ImuControl

    if not fixes:
        return []
    t0, t1 = fixes[0].monotonic_time, fixes[-1].monotonic_time
    dt = cfg.motion.filter_dt_s
    steps = max(1, int((t1 - t0) / dt))
    return [ImuControl(t=t0 + dt * (i + 1), dt=dt, a_long=0.0, yaw_rate=0.0) for i in range(steps)]


def _outage_map_assist(
    network: RoadNetwork,
    pf: RoadParticleFilter,
    uncertainty: UncertaintySet,
    ekf_xy: Sequence[float],
    cfg: Config,
) -> Optional[tuple[np.ndarray, tuple[float, float], float, float]]:
    """Return a bounded, soft map correction only for a compact hypothesis.

    This deliberately has three independent checks. A connected OSM corridor is
    not enough: it can cover more than one plausible turn. The graph is allowed
    to nudge an IMU prediction, never replace it or bridge an arbitrary gap.
    """
    best = uncertainty.best
    if best is None or best.particle_indices.size == 0:
        return None
    probability = float(best.probability)
    if probability < cfg.pf.outage_map_assist_min_probability:
        return None

    member = best.particle_indices
    positions = network.positions_fast(pf.edge_idx[member], pf.s[member])
    weights = np.asarray(pf.w[member], dtype=float)
    if weights.sum() <= 0:
        return None
    centre = np.average(positions, axis=0, weights=weights)
    spread = math.sqrt(float(np.average(np.sum((positions - centre) ** 2, axis=1), weights=weights)))
    if spread > cfg.pf.outage_map_assist_max_spread_m:
        return None

    map_xy = branch_aware_estimate(network, uncertainty, pf.edge_idx, pf.s, pf.w)
    offset = math.dist(map_xy, ekf_xy)
    if offset > cfg.pf.outage_map_assist_max_offset_m:
        return None
    gain = cfg.pf.outage_map_assist_gain
    assisted = (1.0 - gain) * np.asarray(ekf_xy, dtype=float) + gain * np.asarray(map_xy)
    return assisted, map_xy, probability, spread


def _circular_uncertainty(
    t: float,
    xy: Sequence[float],
    ekf: ExtendedKalmanFilter,
    cfg: Config,
    state: str,
    since_trusted: float,
    algorithm: str,
) -> UncertaintySet:
    """IMU-only uncertainty: an explicitly non-map-constrained disc.

    During a GPS outage there is no observation that can justify choosing a
    graph branch.  The disc deliberately says less than a road corridor, but
    it is honest about what the inertial state alone can support.
    """
    import shapely
    from shapely.geometry import Point

    from geotrace.polygons import BranchComponent

    if algorithm == "ekf_dead_reckoning":
        radius = min(
            max(cfg.polygon.r_min_m, 2.0 * math.sqrt(max(1e-6, ekf.P[0, 0] + ekf.P[1, 1]))),
            2000.0,
        )
    else:
        radius = min(
            cfg.polygon.r_min_m
            + cfg.polygon.k_sigma
            * (
                cfg.polygon.cross_track_sigma_base_m
                + cfg.polygon.cross_track_sigma_per_s * since_trusted
            ),
            cfg.polygon.max_radius_m,
        )
    geometry = Point(float(xy[0]), float(xy[1])).buffer(radius)
    component = BranchComponent(
        component_id="branch-01",
        probability=cfg.polygon.confidence,
        geometry=geometry,
        particle_indices=np.zeros(0, dtype=np.int64),
        edge_indices=[],
        representative_xy=(float(xy[0]), float(xy[1])),
        area_m2=float(geometry.area),
        street_names=[],
    )
    return UncertaintySet(
        t=t,
        confidence=cfg.polygon.confidence,
        components=[component],
        gps_state=state,
        seconds_since_trusted=since_trusted,
        n_selected=0,
        n_particles=0,
    )


def _outage_windows(history: Sequence[Any], t0: float, cfg: Config) -> list[dict[str, float]]:
    """[start, end] of every period where GPS was not TRUSTED."""
    windows: list[dict[str, float]] = []
    open_window: Optional[dict[str, float]] = None
    for item in history:
        if item.monotonic_time is None:
            continue
        t_rel = item.monotonic_time - t0
        trusted = item.state is GPSState.TRUSTED
        if not trusted and open_window is None:
            open_window = {"start_s": round(t_rel, 2), "end_s": round(t_rel, 2),
                           "state": item.state.value}
        elif not trusted and open_window is not None:
            open_window["end_s"] = round(t_rel, 2)
            open_window["state"] = item.state.value
        elif trusted and open_window is not None:
            open_window["end_s"] = round(t_rel, 2)
            windows.append(open_window)
            open_window = None
    if open_window is not None:
        windows.append(open_window)
    # Drop the bootstrap window at the very start of the trip.
    return [w for w in windows if w["end_s"] - w["start_s"] > 1.0 and w["start_s"] > 0.5]


def build_metrics(
    trip: Trip,
    result: ReconstructionResult,
    cfg: Config,
    parking: Optional[dict[str, Any]] = None,
) -> MetricsBundle:
    """Compute every metric the specification asks for."""
    frame = result.frame
    has_reference = bool(trip.reference_locations)
    bundle = MetricsBundle(algorithm=result.algorithm, has_reference=has_reference)
    bundle.runtime = {
        "seconds": result.diagnostics.get("runtime_s"),
        "particles": cfg.pf.n_particles if result.algorithm == "road_particle_filter" else None,
        "seed": cfg.seed,
    }

    bundle.gps_gates = gate_metrics(
        result.gate_history,
        corrupted_times=_corrupted_times(trip),
    )
    bundle.trust_recovery = trust_recovery_time(result.gate_history)
    if parking:
        bundle.parking = parking
    elif result.parking_result is not None:
        bundle.parking = result.parking_result.to_json(frame)

    if not has_reference:
        bundle.notes.append(
            "No reference track: this trip was not synthetically corrupted, so "
            "there is no ground truth and position error is undefined. The "
            "reconstruction is a probabilistic estimate, not a measurement."
        )
        bundle.position_error = {"available": False}
        bundle.polygons = {
            "mean_area_m2": (
                float(np.mean([u.total_area_m2 for u in result.uncertainty]))
                if result.uncertainty
                else None
            ),
            "coverage_95": None,
            "note": "coverage requires a reference track",
        }
        bundle.branches = {
            "mean_branch_count": (
                float(np.mean([len(u.components) for u in result.uncertainty]))
                if result.uncertainty
                else None
            ),
            "top1_accuracy": None,
            "top3_recall": None,
        }
        return bundle

    reference = [s for s in trip.reference_locations if s.is_usable]
    ref_times = [s.monotonic_time for s in reference]
    ref_xy = frame.to_local_array(
        [s.latitude for s in reference], [s.longitude for s in reference]
    )

    if result.parking_result is not None and ref_xy.size:
        final_ref = ref_xy[-1]
        road_end = result.primary.array[-1] if result.primary.xy else None
        parking_xy = np.asarray(result.parking_result.position)
        bundle.parking.update({
            "parking_endpoint_error_m": float(np.linalg.norm(parking_xy - final_ref)),
            "road_endpoint_error_m": (
                float(np.linalg.norm(np.asarray(road_end) - final_ref)) if road_end is not None else None
            ),
        })
        if result.network is not None and road_end is not None:
            candidates = result.network.nearest_edges(final_ref, k=1)
            if candidates:
                edge = candidates[0]
                s_ref, _ = result.network.project(final_ref, edge)
                bearing = result.network.edges[edge].bearing(s_ref)
                delta = np.asarray(road_end) - final_ref
                bundle.parking["road_endpoint_along_error_m"] = float(
                    delta[0] * math.cos(bearing) + delta[1] * math.sin(bearing)
                )
                bundle.parking["road_endpoint_cross_error_m"] = float(
                    -delta[0] * math.sin(bearing) + delta[1] * math.cos(bearing)
                )

    outage_end = None
    if result.outage_windows:
        outage_end = trip.t0 + max(w["end_s"] for w in result.outage_windows)

    for name, track in result.tracks.items():
        series = compute_error_series(track.times, track.array, ref_times, ref_xy)
        stats = series.stats()
        stats["error_at_outage_end_m"] = outage_end_error(series, outage_end)
        stats["series"] = {
            "t": [round(t - trip.t0, 2) for t in series.times],
            "error_m": [round(e, 2) for e in series.errors],
        }
        if name == result.algorithm:
            bundle.position_error = stats
        else:
            bundle.baselines[name] = {k: v for k, v in stats.items() if k != "series"}

    bundle.polygons = coverage_and_area(result.uncertainty, ref_times, ref_xy)
    bundle.branches = branch_accuracy(result.uncertainty, ref_times, ref_xy)
    bundle.notes.append(
        "The reference track is ground truth only because the corruption was "
        "synthetic. A real GPS failure leaves no ground truth behind."
    )
    return bundle


def _corrupted_times(trip: Trip) -> Optional[set[float]]:
    """Timestamps the fault injector touched, from the `synthetic` flag."""
    if not trip.faults:
        return None
    return {round(s.monotonic_time, 3) for s in trip.locations if s.synthetic}
