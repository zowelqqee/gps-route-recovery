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
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Sequence

import numpy as np

from geotrace.config import Config
from geotrace.coordinates import LocalFrame, course_to_heading, heading_to_course, wrap_angle
from geotrace.display_route import DisplayRouteWalker
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
from geotrace.models import LocationSample, MountCalibration, Trip
from geotrace.parity_rng import make_rng
from geotrace.motion_model import ImuStream, build_imu_stream, estimate_initial_biases
from geotrace.particle_filter import RoadParticleFilter
from geotrace.parking_tracker import ParkingResult, ParkingTracker
from geotrace.polygons import UncertaintySet, branch_aware_estimate, build_uncertainty_set
from geotrace.road_graph import RoadNetwork
from geotrace.road_tracker import RoadTracker

ALGORITHMS = ("road_ekf", "last_known_position", "ekf_dead_reckoning", "road_particle_filter")


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
class RoadReconciliation:
    """Road geometry inferred after GPS confirms the far end of an outage.

    It is intentionally separate from :class:`Track`: it has no per-point
    timestamps and makes no claim that IMU measured its along-road progress.
    """

    start_t: float
    end_t: float
    coords: np.ndarray
    length_m: float
    start_offset_m: float
    end_offset_m: float
    edge_count: int


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
    road_reconciliations: list[RoadReconciliation] = field(default_factory=list)
    road_uncertainty: list[UncertaintySet] = field(default_factory=list)

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


def initial_heading(
    locations: Sequence[LocationSample],
    frame: LocalFrame,
    cfg: Config,
    calibration: Optional[MountCalibration] = None,
) -> float:
    """Heading at t0.

    Preference order: a heading the recorder measured across a window of fixes,
    then a trusted GPS course while actually moving, then the bearing between
    the first fix and the first fix ~20 m away. The magnetometer is never used -
    inside a car it is not reliable enough to seed a heading.

    Only `heading_source == "gps_course_window"` short-circuits the search. The
    plain "gps_course" a phone calibration writes is one fix's course, which is
    what the loop below would find anyway; a window is a genuinely better
    measurement, so a recorder that did the averaging is taken at its word
    rather than being second-guessed by a single sample of the same signal.
    """
    if (
        calibration is not None
        and calibration.heading_source == "gps_course_window"
        and calibration.initial_heading_deg is not None
    ):
        return course_to_heading(float(calibration.initial_heading_deg))
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
    algorithm: str = "road_ekf",
    output_dt: float = 1.0,
    progress: Optional[Any] = None,
) -> ReconstructionResult:
    """Run the road-coordinate EKF bank, or an explicitly selected baseline."""
    if algorithm == "road_ekf":
        from geotrace.road_ekf_pipeline import run_road_ekf
        return run_road_ekf(trip, network, cfg, output_dt, progress)
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
    start_fixes = usable[:1] if cfg.gps_start_only else usable
    heading0 = initial_heading(start_fixes, frame, cfg, trip.metadata.calibration)

    imu_calibration = trip.metadata.calibration
    if (trip.metadata.device_model == "synthetic" and imu_calibration is not None
            and imu_calibration.attitude_source != "rigid_mount_simulator_v2"):
        # Older saved simulations rotated acceleration but not their device
        # attitude. Preserve the world-vector path for those legacy recordings.
        imu_calibration = replace(imu_calibration, attitude_source="synthetic_fixed_world_legacy")
    imu = build_imu_stream(
        trip.motions,
        cfg.motion,
        calibration=imu_calibration,
        reference_heading_rad=heading0,
        t_start=trip.t0,
        t_end=trip.t0 + trip.duration_s,
    )
    if not imu.controls and algorithm != "last_known_position":
        raise ReconstructionError("motion samples produced an empty control stream")

    # Biases come from the aligned stream, so the longitudinal projection uses
    # the same world frame the filters will.
    # Quiet IMU alone also describes constant-speed driving. Initial bias
    # estimation requires observed stationary GPS, not the filter's own belief.
    stationary_controls = []
    speed_fixes = [s for s in start_fixes if s.has_valid_speed]
    if speed_fixes:
        speed_times = np.array([s.monotonic_time for s in speed_fixes])
        for c in imu.controls:
            if c.t > trip.t0 + 12.0:
                break
            j = int(np.searchsorted(speed_times, c.t, side="right")) - 1
            if j >= 0 and c.t - speed_times[j] <= 1.0 and float(speed_fixes[j].speed or 0.0) <= 0.5:
                stationary_controls.append(c)
    bias_a, bias_w = estimate_initial_biases(ImuStream(stationary_controls), heading0, cfg.motion)

    graph_fit = _graph_fit(start_fixes, frame, network, cfg)

    origin_xy = frame.to_local(usable[0].latitude, usable[0].longitude)
    speed0 = float(usable[0].speed or 0.0) if usable[0].has_valid_speed else 0.0

    ekf = ExtendedKalmanFilter(
        cfg.motion,
        initial_state=[origin_xy[0], origin_xy[1], speed0, heading0, bias_a, bias_w],
    )
    monitor = GPSQualityMonitor(
        cfg.gps, max_accel_ms2=cfg.motion.max_accel_ms2, max_speed_ms=cfg.motion.max_speed_ms
    )
    pf: Optional[RoadParticleFilter] = None
    if algorithm == "road_particle_filter":
        pf = RoadTracker(network, cfg, rng=make_rng(cfg.seed, cfg.rng_mode))
        if cfg.gps_start_only:
            pf.initialize(
                origin_xy,
                heading=heading0,
                speed=speed0,
                accel_bias=bias_a,
                gyro_bias=bias_w,
                position_sigma=cfg.gps.recovery_position_sigma_m,
            )

    tracks = {
        "last_known_position": Track("last_known_position"),
        "ekf_dead_reckoning": Track("ekf_dead_reckoning"),
    }
    if pf is not None:
        tracks["road_particle_filter"] = Track("road_particle_filter")
        tracks["road_posterior"] = Track("road_posterior")

    uncertainty: list[UncertaintySet] = []
    road_uncertainty: list[UncertaintySet] = []
    gps_states: list[dict[str, Any]] = []
    last_known_xy = origin_xy
    last_trusted_t = trip.t0
    last_known_sigma_m = cfg.gps.recovery_position_sigma_m
    accepted_points: list[dict[str, Any]] = []
    rejected_points: list[dict[str, Any]] = []

    # Index the fixes by time so the loop can consume them in order.
    fixes = [] if cfg.gps_start_only else sorted(usable, key=lambda s: s.monotonic_time)
    # A terminal observation is only valid when the recorder explicitly ended
    # the trip.  Synthetic/open-ended traces may also slow down near their last
    # sample, but that is not evidence that the driver has parked.
    fix_cursor = 0
    previous_state = monitor.state
    reanchors: list[dict[str, Any]] = []
    last_zupt_t = float("-inf")
    road_reconciliations: list[RoadReconciliation] = []
    map_assists: list[dict[str, Any]] = []
    shock_events: list[dict[str, Any]] = []
    active_shock: Optional[dict[str, Any]] = None
    walker: Optional[DisplayRouteWalker] = None
    display_route_stats: dict[str, Any] = {
        "walks": 0,
        "refused_off_graph": 0,
        "ticks": 0,
        "junctions_crossed": 0,
        "dead_ends": 0,
        "corrections_accepted": 0,
        "corrections_refused": 0,
        "off_best_branch_ticks": 0,
    }
    # The walker moves by however far the EKF itself moved, so the previous
    # tick's position and time are part of the loop state. Both are updated on
    # *every* output tick, trusted or not, so the first tick of an outage gets
    # one tick's worth of travel and not the whole preceding trusted stretch.
    previous_ekf_xy = np.asarray(origin_xy, dtype=float)
    previous_output_t = trip.t0
    output_speeds: list[float] = []
    # Recent observed GPS speed corroborates ZUPT; it is never a speed floor.
    last_trusted_speed_mps = speed0 if cfg.gps_start_only else 0.0
    next_output = trip.t0
    controls = imu.controls if imu.controls else _synthetic_controls(fixes, cfg)

    for control in controls:
        t = control.t
        # GPS trust is deliberately evaluated before the road filter advances.
        #
        # `has_gps_corroboration` gates two independent decisions below, both
        # for the same reason: a quiet, low-acceleration IMU signal is
        # identical for "stopped" and "cruising at a steady speed", so
        # without GPS to tell them apart the filter should neither commit to
        # zero (ZUPT) nor let a residual accelerometer bias slowly integrate
        # the speed towards zero on its own (the predict deadband). Once
        # trust has been lost after having been established, that
        # corroboration is gone until it returns. The opening calibration
        # stop is unaffected: `bootstrap_done` is false there too, but for
        # the opposite reason - nothing has been lost yet, so it is not a
        # real outage. See `GPSQualityMonitor.bootstrap_done`.
        monitor.note_gap(t)
        has_gps_corroboration = (
            not cfg.gps_start_only
            and (monitor.is_trusted or not monitor.bootstrap_done)
        )
        deadband = not has_gps_corroboration
        # A post-shock mount slip only matters while nothing else can catch
        # it. With GPS corroborating, a bad heading is corrected within one
        # fix cycle anyway, and most shocks are ordinary bumps hit while
        # tracking is otherwise fine (potholes, speed bumps) - discounting
        # their yaw rate there only throws away real steering signal and
        # makes GPS look like it disagrees with a now-lagging prediction.
        yaw_trust = control.yaw_trust if deadband else 1.0
        if control.gap_exceeded:
            # A resampled step still has a small dt inside a large source gap.
            # Do not mistake that small dt for an actual IMU observation.
            ekf.skipped_gaps += 1
            growth = cfg.motion.max_speed_ms**2 * cfg.motion.max_gap_s * control.dt
            ekf.P[0, 0] += growth
            ekf.P[1, 1] += growth
            ekf.P[3, 3] += (math.pi / 2)**2 * control.dt
        elif control.is_shock:
            # The phone may have moved independently of the car. Preserve the
            # vehicle's existing constant-velocity prediction, but do not turn
            # the impact into acceleration/yaw and make its uncertainty honest.
            ekf.predict((0.0, 0.0, 0.0), 0.0, control.dt, deadband=deadband, coast=True)
            ekf.inflate_for_mount_disturbance(
                cfg.motion.shock_position_noise_mpsqrt,
                cfg.motion.shock_heading_noise_radsqrt,
                control.dt,
            )
            report_shock = ekf.speed >= cfg.motion.shock_min_vehicle_speed_ms
            if report_shock and active_shock is None:
                lat, lon = frame.to_geo(*ekf.position)
                active_shock = {
                    "start_s": round(t - trip.t0, 2),
                    "latitude": lat,
                    "longitude": lon,
                    "peak_accel_ms2": control.peak_accel_ms2,
                    "peak_gyro_rads": control.peak_gyro_rads,
                }
            elif report_shock:
                active_shock["peak_accel_ms2"] = max(
                    active_shock["peak_accel_ms2"], control.peak_accel_ms2
                )
                active_shock["peak_gyro_rads"] = max(
                    active_shock["peak_gyro_rads"], control.peak_gyro_rads
                )
            elif active_shock is not None:
                active_shock["end_s"] = round(t - trip.t0, 2)
                shock_events.append(active_shock)
                active_shock = None
        else:
            if active_shock is not None:
                active_shock["end_s"] = round(t - trip.t0, 2)
                shock_events.append(active_shock)
                active_shock = None
            ekf.predict(
                control.a_world,
                control.yaw_rate,
                control.dt,
                deadband=deadband,
                yaw_trust=yaw_trust,
                a_vehicle=getattr(control, "a_long", None),
            )

        # Require observed low speed by default; IMU-only ZUPT is experimental.
        zupt_allowed = (
            ((has_gps_corroboration and last_trusted_speed_mps <= cfg.motion.zupt_max_speed_ms
              and t - last_trusted_t <= 1.0) or not cfg.motion.zupt_requires_gps)
            and (t - last_zupt_t) >= cfg.motion.zupt_min_interval_s
        )
        zupt_now = (
            not control.is_shock
            and not control.gap_exceeded
            and control.is_quiet
            and ekf.speed <= cfg.motion.zupt_max_speed_ms
            and zupt_allowed
        )
        if zupt_now:
            last_zupt_t = t
            ekf.zero_velocity_update()

        if pf is not None and pf.initialized and not control.gap_exceeded:
            # The particle cloud keeps moving through the road graph during an
            # outage, but is never reweighted by absent or rejected GPS.  It is
            # therefore a causal map prior, not retrospective map matching.
            pf.predict(
                (0.0, 0.0, 0.0) if control.is_shock else control.a_world,
                0.0 if control.is_shock else control.yaw_rate,
                control.dt,
                deadband=deadband,
                yaw_trust=yaw_trust,
                a_vehicle=None if control.is_shock else getattr(control, "a_long", None),
                coast=control.is_shock,
            )
            if zupt_now:
                pf.zero_velocity_update(cfg.motion.zupt_max_speed_ms)
            # Road/heading evidence remains available after GPS disappears.
            # Scale by elapsed time, not by GPS or report frequency.
            pf.update_weights(map_evidence_scale=control.dt)
            pf.maybe_resample()
        elif pf is not None and pf.initialized:
            pf.result.skipped_gaps += 1
            pf.s += pf.rng.normal(0.0, cfg.motion.max_speed_ms * math.sqrt(
                cfg.motion.max_gap_s * control.dt), len(pf.s))
            pf._resolve_edges()

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
                # A hard reanchor commits to this fix outright, bypassing the
                # Kalman gain entirely - sound only when the fix itself is
                # trustworthy. A recovering fix with a poor accuracy is the
                # receiver admitting it does not know where the car is; that
                # is folded in through the ordinary gain-weighted update below
                # instead, where a large sigma earns it only a small nudge.
                if restored and _had_real_outage(monitor) and sigma <= cfg.gps.max_reanchor_sigma_m:
                    # The two endpoints are now known: the last causal IMU
                    # position and this recovered GPS fix. Resolve the road
                    # geometry between them separately from the IMU trace. It
                    # deliberately has no timing claim, so a weak IMU
                    # odometer cannot erase a bridge from the *route* merely
                    # because it under-counted its travelled distance.
                    if walker is not None and network is not None:
                        route = network.route_between(
                            walker.position(), xy,
                            max_snap_m=cfg.pf.display_route_max_snap_m,
                        )
                        # A directed path that is wildly longer than the
                        # distance the live IMU/road walk represented is a
                        # graph loop caused by a stale endpoint, not a route
                        # the vehicle could have taken.  Do not turn it into
                        # a second visible journey.
                        plausible_length_m = max(100.0, 1.5 * walker.distance_m)
                        if route is not None and route.length_m <= plausible_length_m:
                            road_reconciliations.append(
                                RoadReconciliation(
                                    start_t=previous_output_t,
                                    end_t=fix.monotonic_time,
                                    coords=route.coords,
                                    length_m=route.length_m,
                                    start_offset_m=route.start_offset_m,
                                    end_offset_m=route.end_offset_m,
                                    edge_count=len(route.edge_indices),
                                )
                            )
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
                    last_known_sigma_m = sigma
                    if fix.has_valid_speed:
                        last_trusted_speed_mps = float(np.clip(
                            float(fix.speed or 0.0), 0.0, cfg.motion.max_speed_ms
                        ))
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
                                map_evidence_scale=0.0,
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
            output_speeds.append(ekf.speed)
            since_trusted = max(0.0, t - last_trusted_t)
            tracks["last_known_position"].add(t, last_known_xy)
            tracks["ekf_dead_reckoning"].add(t, ekf.position)

            if pf is not None and pf.initialized:
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
                road_uncertainty.append(unc)
                tracks["road_posterior"].add(t, branch_aware_estimate(
                    network, unc, pf.edge_idx, pf.s, pf.w,
                ))
                displayed_xy = ekf.position
                if monitor.state is GPSState.TRUSTED:
                    if walker is not None:
                        # Re-syncing onto the recovered position is the one jump
                        # the walker is otherwise forbidden to make, so it has to
                        # read as a break rather than a straight leg across
                        # whatever lies between. The hard reanchor above usually
                        # broke the line already; it does not when the recovering
                        # fix was too poor to earn one (max_reanchor_sigma_m).
                        # `break_line` only sets a pending flag, so asking twice
                        # is harmless.
                        if (
                            math.dist(walker.position(), ekf.position)
                            > cfg.pf.display_route_max_snap_m
                        ):
                            tracks["road_particle_filter"].break_line()
                        _merge_display_route_stats(display_route_stats, walker)
                        walker = None
                    road_distance_m = (
                        network.distance_to_road(last_known_xy) if network is not None else None
                    )
                    if road_distance_m is not None and road_distance_m > cfg.polygon.off_road_distance_m:
                        # Trusted GPS, but nowhere near any edge the graph
                        # knows about - a courtyard, a private drive, a gap
                        # in coverage. The particle cloud is still forced
                        # onto whichever real edge is nearest, which can be
                        # hundreds of metres away; reporting its corridor as
                        # 95%-confident would be a claim the GPS fix itself
                        # contradicts. Fall back to an honest disc around the
                        # real position instead.
                        uncertainty.append(
                            _circular_uncertainty(
                                t, last_known_xy, ekf, cfg, monitor.state.value,
                                since_trusted, "gps_off_road",
                                gps_sigma_m=last_known_sigma_m,
                            )
                        )
                    else:
                        # A trusted observation has just constrained the
                        # particle cloud, so its road corridors are an honest
                        # uncertainty visualisation.
                        uncertainty.append(unc)
                else:
                    # Position and heading are corrected independently here.
                    # A compact, confident branch is required below before
                    # position may be nudged at all - a real interchange with
                    # several plausible streets fails that outright. Heading
                    # is a weaker claim: those same streets can still all run
                    # the same way away from the fork, so this can correct
                    # heading even when no single branch is trusted enough to
                    # correct position (see Config.pf.outage_heading_assist_*).
                    # Keep the EKF baseline independent of the road posterior.
                    evidence = _map_hypothesis_evidence(network, pf, unc, ekf.position, cfg)
                    if evidence is not None:
                        map_xy, probability, spread = evidence
                        gain = cfg.pf.outage_map_assist_gain
                        displayed_xy = (1.0 - gain) * np.asarray(
                            ekf.position, dtype=float
                        ) + gain * np.asarray(map_xy, dtype=float)
                        # The same proof that permits a bounded map nudge also
                        # permits presenting a road-graph corridor.  Once it
                        # fails, rendering a merged graph component as a
                        # 95%-likely branch would be false confidence.
                        uncertainty.append(unc)
                        map_assists.append(
                            {
                                "t": round(t - trip.t0, 2),
                                "probability": round(probability, 3),
                                "spread_m": round(spread, 2),
                                "map_offset_m": round(float(math.dist(map_xy, ekf.position)), 2),
                            }
                        )
                    else:
                        uncertainty.append(
                            _circular_uncertainty(
                                t,
                                ekf.position,
                                ekf,
                                cfg,
                                monitor.state.value,
                                since_trusted,
                                "imu_dead_reckoning",
                            )
                        )

                    # The free inertial estimate is constrained by nothing and
                    # can leave the road entirely; walk a connected route
                    # instead. See `geotrace.display_route`.
                    if cfg.pf.display_route_constrained:
                        if walker is None:
                            walker = DisplayRouteWalker.start(network, pf, ekf, cfg)
                            if walker is None:
                                display_route_stats["refused_off_graph"] += 1
                            else:
                                display_route_stats["walks"] += 1
                        if walker is not None:
                            best = unc.best
                            walker_dt = max(1e-6, t - previous_output_t)
                            imu_distance_m = float(
                                np.linalg.norm(
                                    np.asarray(ekf.position, dtype=float)
                                    - previous_ekf_xy
                                )
                            )
                            displayed_xy = np.asarray(
                                walker.advance(
                                    pf,
                                    ekf,
                                    ds=imu_distance_m,
                                    dt=walker_dt,
                                    best_edges=(
                                        best.edge_indices if best is not None else ()
                                    ),
                                ),
                                dtype=float,
                            )
                            if walker.stalled:
                                # The route ran out under a moving car. Draw the
                                # unconstrained estimate rather than a position
                                # pinned to a dead end, break the line so the two
                                # are not joined, and let the next tick try to
                                # pick up a fresh route from wherever the car
                                # now is.
                                displayed_xy = ekf.position
                                tracks["road_particle_filter"].break_line()
                                display_route_stats["stalls"] = (
                                    display_route_stats.get("stalls", 0) + 1
                                )
                                _merge_display_route_stats(display_route_stats, walker)
                                walker = None
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
            previous_ekf_xy = np.asarray(ekf.position, dtype=float)
            previous_output_t = t
            next_output += output_dt

    elapsed = time.perf_counter() - started
    if walker is not None:
        _merge_display_route_stats(display_route_stats, walker)
    display_route_stats["active_at_end"] = walker is not None
    if active_shock is not None:
        active_shock["end_s"] = round(trip.duration_s, 2)
        shock_events.append(active_shock)
    outage_windows = _outage_windows(monitor.history, trip.t0, cfg, trip.duration_s)
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
        road_speed=output_speeds[road_index] if road_track.xy else speed0,
        calibration_present=trip.metadata.calibration is not None,
        road_distance=(network.distance_to_road if network is not None else None),
    )

    diagnostics: dict[str, Any] = {
        "algorithm": algorithm,
        "position_estimator": "continuous_display_route" if pf is not None else algorithm,
        "uncertainty_calibrated": False,
        "imu_processing": "offline centred windows; not zero-latency streaming",
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
        "imu_activity_windows": _imu_activity_windows(controls, trip.t0),
        "gps": monitor.summary(),
        "gps_state_intervals": state_intervals(monitor.history),
        "outage_windows": outage_windows,
        "accepted_fixes": accepted_points,
        "rejected_fixes": rejected_points,
        "ekf": {"skipped_gaps": ekf.skipped_gaps, "final_state": ekf.state_json()},
        "reanchors": reanchors,
        "road_reconciliations": [
            {
                "start_s": round(item.start_t - trip.t0, 2),
                "end_s": round(item.end_t - trip.t0, 2),
                "length_m": round(item.length_m, 1),
                "start_offset_m": round(item.start_offset_m, 1),
                "end_offset_m": round(item.end_offset_m, 1),
                "edge_count": item.edge_count,
            }
            for item in road_reconciliations
        ],
        "outage_map_assists": map_assists,
        "display_route": display_route_stats,
        "imu_shocks": [
            {
                **item,
                "peak_accel_ms2": round(float(item["peak_accel_ms2"]), 2),
                "peak_gyro_rads": round(float(item["peak_gyro_rads"]), 3),
            }
            for item in shock_events
        ],
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
            "confidence": None,
            "nominal_region_confidence": cfg.polygon.confidence,
            "calibrated": False,
        },
        "parking_result": parking_result.to_json(frame),
        "final_vehicle_position": (
            parking_result.to_json(frame)["position"]
            if parking_result.status in ("CONFIDENT", "PROBABLE")
            else dict(zip(("latitude", "longitude"), frame.to_geo(*road_track.array[-1])))
        ),
        "final_vehicle_position_source": (
            "parking_tracker" if parking_result.status in ("CONFIDENT", "PROBABLE") else "road_tracker"
        ),
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
        road_uncertainty=road_uncertainty,
        outage_windows=outage_windows,
        parking_result=parking_result,
        road_reconciliations=road_reconciliations,
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


def _imu_activity_windows(controls: Sequence[Any], t0: float, window_s: float = 10.0) -> list[dict[str, Any]]:
    """Compact activity diagnostics from the *normalised* IMU controls.

    Percentiles preserve short braking/turning events without letting one raw
    sensor spike dominate the graph.  These are diagnostics only: they never
    feed back into the state estimator.
    """
    buckets: dict[int, list[Any]] = {}
    for control in controls:
        bucket = int(max(0.0, control.t - t0) // window_s)
        buckets.setdefault(bucket, []).append(control)
    rows: list[dict[str, Any]] = []
    for bucket in sorted(buckets):
        chunk = buckets[bucket]
        acceleration = np.asarray([np.linalg.norm(item.a_world) for item in chunk], dtype=float)
        yaw = np.asarray([abs(item.yaw_rate) for item in chunk], dtype=float)
        rows.append(
            {
                "start_s": round(bucket * window_s, 1),
                "accel_p90_ms2": round(float(np.percentile(acceleration, 90)), 3),
                "yaw_p90_rads": round(float(np.percentile(yaw, 90)), 4),
                "shock_steps": sum(item.is_shock for item in chunk),
                "quiet_fraction": round(sum(item.is_quiet for item in chunk) / len(chunk), 3),
                "gap_steps": sum(item.gap_exceeded for item in chunk),
            }
        )
    return rows


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


def _outage_heading_assist(
    pf: RoadParticleFilter, ekf: ExtendedKalmanFilter, cfg: Config
) -> Optional[float]:
    """Bearing to nudge the EKF's heading towards, or None.

    Independent of `_outage_map_assist`: that one requires one compact,
    confident branch before it will touch position at all, which a real
    interchange with several plausible streets fails outright. Heading is a
    weaker claim - several genuinely different streets can still all run the
    same way away from the fork - so `RoadParticleFilter.heading_consensus`
    can answer it even when no single branch is trusted enough to correct
    position.

    That consensus is gated by two independent checks. Agreement within the
    particle cloud (`resultant`) is not by itself evidence the cloud is
    right: it is a causal, never retrospectively corrected prior, and after
    long enough without GPS it can be confidently wrong (confirmed against
    trip-b4faeae0-a941-4a87-9b18-de7aaa84f721: a 373 s outage after which one
    confidently-agreeing but wrong branch cost 134 GPS fixes once trust
    returned). Requiring the consensus to already be close to the EKF's own
    independent heading (`heading_gap`) keeps this to a fine correction of a
    small, plausible drift, never a wholesale redirection.

    A tempting third relief valve - trust a lower resultant once the bearing
    has held still for a few seconds - was tried and reverted: it cannot
    reliably tell a real, stable road apart from a fork the cloud is still
    actively deciding between. A slow-building divergence (confirmed against
    the fork integration test: bearing barely moves for the first few
    seconds of a decision, then accelerates) reads as "stable" under any
    window short enough to react in time, and broke
    test_the_polygons_cover_the_true_position. Known consequence: a real,
    merely moderately-spread stretch of road (resultant 0.7-0.85, e.g.
    trip-32c24d86-06af-45fd-a3a0-79354f2cbb70's Kantemirovsky bridge
    crossing) stays uncorrected by this function - see off_road_distance_m
    and yaw_trust for the other, independent mitigations that do apply
    there.
    """
    consensus = pf.heading_consensus()
    if consensus is None:
        return None
    bearing, resultant = consensus
    if resultant < cfg.pf.outage_heading_assist_min_resultant:
        return None
    heading_gap = abs(wrap_angle(bearing - ekf.heading))
    if heading_gap > cfg.pf.outage_heading_assist_max_gap_rad:
        return None
    return bearing


def _merge_display_route_stats(
    stats: dict[str, Any], walker: DisplayRouteWalker
) -> None:
    """Fold one finished walk into the run-level totals.

    `off_best_branch_ticks` is the one worth reading: a walker that took the
    wrong turn is smooth, plausible and confidently wrong for the rest of the
    outage, because re-syncing is exactly the jump it is forbidden to make.
    Counting it is what keeps that failure visible instead of invisible.
    """
    for key, value in walker.to_json().items():
        stats[key] = stats.get(key, 0) + value


def _map_hypothesis_evidence(
    network: RoadNetwork,
    pf: RoadParticleFilter,
    uncertainty: UncertaintySet,
    ekf_xy: Sequence[float],
    cfg: Config,
) -> Optional[tuple[tuple[float, float], float, float]]:
    """The map's own answer, and its supporting evidence, or None.

    Returns ``(map_xy, probability, spread)``. ``None`` means the belief is
    not one compact hypothesis and the map should not be presented as one.

    This deliberately has three independent checks. A connected OSM corridor is
    not enough: it can cover more than one plausible turn. The graph is allowed
    to nudge an IMU prediction, never replace it or bridge an arbitrary gap.

    The caller decides what to do with a passing hypothesis: blend it into the
    displayed position, and - the load-bearing second job - report the road
    corridor rather than an inertial disc as the uncertainty for this tick
    (see `test_noncompact_road_hypotheses_fall_back_to_an_imu_disc`).
    """
    best = uncertainty.best
    if best is None or best.particle_indices.size == 0:
        return None
    probability = float(best.probability)
    if probability < cfg.pf.outage_map_assist_min_probability:
        return None
    if uncertainty.total_area_m2 > cfg.pf.outage_map_assist_max_area_m2:
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
    return map_xy, probability, spread


def _circular_uncertainty(
    t: float,
    xy: Sequence[float],
    ekf: ExtendedKalmanFilter,
    cfg: Config,
    state: str,
    since_trusted: float,
    algorithm: str,
    gps_sigma_m: Optional[float] = None,
) -> UncertaintySet:
    """IMU-only uncertainty: an explicitly non-map-constrained disc.

    During a GPS outage there is no observation that can justify choosing a
    graph branch.  The disc deliberately says less than a road corridor, but
    it is honest about what the inertial state alone can support.

    ``algorithm == "gps_off_road"`` is the one caller with a *trusted* fix:
    the car is somewhere the road graph has no edge for, and ``gps_sigma_m``
    is that fix's own measurement sigma, not an outage-growth guess.
    """
    import shapely
    from shapely.geometry import Point

    from geotrace.polygons import BranchComponent

    # Circumscribe the 2-D Gaussian confidence ellipse. This is a model
    # interval, not an empirically calibrated promise; never truncate its size.
    quantile = -2.0 * math.log1p(-cfg.polygon.confidence)  # chi-square, 2 DOF
    largest_variance = max(0.0, float(np.linalg.eigvalsh(ekf.P[:2, :2])[-1]))
    radius = max(cfg.polygon.r_min_m, math.sqrt(quantile * largest_variance))
    if algorithm == "gps_off_road":
        radius = max(cfg.polygon.r_min_m, math.sqrt(quantile) * max(0.0, float(gps_sigma_m or 0.0)))
    elif algorithm != "ekf_dead_reckoning":
        radius = max(radius, cfg.polygon.r_min_m + cfg.polygon.k_sigma * (
            cfg.polygon.cross_track_sigma_base_m
            + cfg.polygon.cross_track_sigma_per_s * since_trusted
        ))
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


def _outage_windows(
    history: Sequence[Any], t0: float, cfg: Config, duration_s: float = 0.0
) -> list[dict[str, float]]:
    """[start, end] of every period where GPS was not TRUSTED.

    `duration_s` closes an outage that is still open when the recording stops.
    Trust is only re-evaluated when a fix arrives, so a GPS that goes away and
    never comes back leaves exactly one history entry - the dropout - and the
    window would otherwise be zero seconds long and be discarded as noise. An
    outage the trip ended inside of runs to the end of the trip.
    """
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
        open_window["end_s"] = round(max(open_window["end_s"], duration_s), 2)
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
    t0 = trip.t0
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
    ref_dt = np.diff(ref_times)
    ref_steps = np.linalg.norm(np.diff(ref_xy, axis=0), axis=1)
    jumps = ref_steps > cfg.motion.max_speed_ms * np.maximum(ref_dt, 0.0) + cfg.gps.physical_margin_m
    bundle.reference_quality = {
        "usable_fixes": len(reference), "implausible_steps": int(jumps.sum()),
        "gaps_over_2s": int(np.sum(ref_dt > 2.0)),
        "note": "GPS is a noisy reference, not surveyed truth; quality checks do not feed reconstruction",
    }
    if result.network is not None and len(ref_xy):
        stride = max(1, len(ref_xy) // 1000)
        road_distances = np.array([result.network.distance_to_road(p) for p in ref_xy[::stride]])
        bundle.reference_quality.update({
            "graph_check_samples": len(road_distances),
            "fraction_within_30m_of_graph": float(np.mean(road_distances <= 30.0)),
            "max_distance_from_graph_m": float(road_distances.max()),
        })

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
        outage_end = t0 + max(w["end_s"] for w in result.outage_windows)

    for name, track in result.tracks.items():
        series = compute_error_series(track.times, track.array, ref_times, ref_xy)
        stats = series.stats()
        stats["error_at_outage_end_m"] = outage_end_error(series, outage_end)
        stats["series"] = {
            "t": [round(t - t0, 2) for t in series.times],
            "error_m": [round(e, 2) for e in series.errors],
        }
        if name == result.algorithm:
            bundle.position_error = stats
        else:
            bundle.baselines[name] = {k: v for k, v in stats.items() if k != "series"}

    bundle.polygons = coverage_and_area(result.uncertainty, ref_times, ref_xy)
    bundle.polygons["calibrated"] = False
    bundle.polygons["road_posterior"] = coverage_and_area(result.road_uncertainty, ref_times, ref_xy)
    bundle.branches = branch_accuracy(result.road_uncertainty, ref_times, ref_xy)
    bundle.branches["definition"] = "reference containment in ranked connected road corridors, not street identification"
    if result.algorithm == "road_ekf":
        regions = result.uncertainty
        bundle.polygons["mean_represented_mass"] = float(np.mean([u.represented_mass for u in regions])) if regions else None
        bundle.polygons["lost_fraction"] = float(np.mean([u.status == "LOST" for u in regions])) if regions else None
        bundle.position_error["availability_fraction"] = bundle.position_error.get("count", 0) / max(1, bundle.polygons.get("samples", 0))
        bundle.position_error["tracking_availability_fraction"] = len(result.primary.times) / max(1, len(regions))
        bundle.notes.append("Road EKF corridors display a bounded subset of the conditional road posterior, not a 95% region. LOST timestamps count as misses in coverage and branch metrics; position error is defined only where an estimate exists.")
    else:
        bundle.notes.append("Nominal probability mass is not a calibrated coverage guarantee. Display-route and road-posterior errors are reported separately; EKF discs are excluded from branch metrics.")
    bundle.notes.append(_reference_provenance_note(trip))
    return bundle


def _reference_provenance_note(trip: Trip) -> str:
    """Say what the reference track actually is, because it differs.

    Two very different things end up in `reference-samples.jsonl` and the
    difference decides how far the error columns can be pushed:

    * a synthetically corrupted trip kept its own clean track aside, and that
      track *is* the answer the corruption was measured against;
    * a live import withheld real GPS from the reconstruction. Those fixes were
      never corrupted and never simulated, but they are still a receiver's
      output, with a receiver's own few metres of error.

    Neither is a survey. Printing one sentence for both would overstate the
    second and understate the first.
    """
    provenance = (trip.metadata.extra or {}).get("live_import")
    if isinstance(provenance, dict) and provenance.get("reference_is") == "withheld_real_gps":
        warmup = provenance.get("gps_warmup_s")
        return (
            "The reference track is real GPS that was withheld from the "
            f"reconstruction: every fix after the first {warmup:g} s was moved "
            "aside before the filters ran, so no algorithm saw it. It is a "
            "yardstick with a few metres of its own error, not survey ground "
            "truth, and errors below that floor are not resolvable."
        )
    return (
        "The reference track is ground truth only because the corruption was "
        "synthetic. A real GPS failure leaves no ground truth behind."
    )


def _corrupted_times(trip: Trip) -> Optional[set[float]]:
    """Timestamps the fault injector touched, from the `synthetic` flag."""
    if not trip.faults:
        return None
    return {round(s.monotonic_time, 3) for s in trip.locations if s.synthetic}
