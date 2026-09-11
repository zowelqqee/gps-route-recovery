"""Road EKF coordinator. Only visible locations are supplied to inference."""
from __future__ import annotations

from dataclasses import replace
import math
import time

import numpy as np
from shapely.ops import substring

from geotrace.coordinates import course_to_heading, heading_to_course
from geotrace.ekf import ExtendedKalmanFilter
from geotrace.gps_quality import GPSQualityMonitor, GPSState, measurement_sigma, state_intervals
from geotrace.motion_model import build_imu_stream
from geotrace.pipeline import Track, ReconstructionResult, ReconstructionError, initial_heading, _outage_windows
from geotrace.road_ekf import RoadEKF, S, V, PSI, BW, ROAD_ERROR


class RoadTrack(Track):
    """Timestamped filtered positions plus exact connecting road geometry.

    Competing mode changes break the line. We never draw a straight shortcut
    between disconnected roads, or invent a driven path to hide a correction.
    """
    def __init__(self, network):
        super().__init__("road_ekf")
        self.network = network
        self.road_segments: list[list[tuple[float, float]]] = []
        self.previous = None

    def add_mode(self, t, mode):
        point = self.network.edges[mode.edge].position(mode.x[S])
        geometry = []
        if self.previous is not None and not self._break_pending:
            old = self.previous
            if old.edge == mode.edge and mode.x[S] >= old.x[S] - 2.:
                geometry = list(substring(self.network.edges[mode.edge].line,
                                         old.x[S], mode.x[S]).coords)
            elif old.edge in mode.history[:-1]:
                index = len(mode.history) - 1 - mode.history[::-1].index(old.edge)
                path = mode.history[index:]
                for j, edge_index in enumerate(path):
                    edge = self.network.edges[edge_index]
                    low = old.x[S] if j == 0 else 0.
                    high = mode.x[S] if j == len(path)-1 else edge.length
                    geometry.extend(substring(edge.line, low, high).coords)
        if geometry:
            self.road_segments[-1].extend(geometry)
        else:
            self.break_line()
            self.road_segments.append([point])
        self.add(t, point)
        self.previous = mode.copy()

    def to_geojson(self, frame, properties=None):
        feature = super().to_geojson(frame, properties)
        segments = [frame.coords_to_geojson(s if len(s)>1 else s*2) for s in self.road_segments]
        feature['geometry'] = {'type':'MultiLineString', 'coordinates':segments}
        feature['properties']['geometry_source'] = 'directed_road_edges'
        return feature


def run_road_ekf(trip, network, cfg, output_dt=1.0, progress=None,
                 trace_hook=None):
    if network is None:
        raise ReconstructionError('road_ekf needs a road graph (--graph)')
    fixes = sorted(trip.usable_locations, key=lambda s: s.monotonic_time)
    if not fixes or not trip.motions:
        raise ReconstructionError('road_ekf needs an initial GPS fix and motion samples')
    if output_dt <= 0:
        raise ReconstructionError('output_dt must be positive')
    started = time.perf_counter()
    t0, duration = trip.t0, trip.duration_s
    if cfg.gps_start_only:
        fixes = fixes[:1]
    frame = network.frame
    first = fixes[0]
    # The importer's explicitly declared warm-up calibration is permitted.
    # Otherwise no future GPS course is inspected to seed the filter.
    calibration = trip.metadata.calibration
    heading = initial_heading([first], frame, cfg, calibration)
    heading_known = (first.has_valid_course and first.has_valid_speed and (first.speed or 0.) >= 2.) or (
        calibration is not None and calibration.heading_source == 'gps_course_window')
    if trip.metadata.device_model == 'synthetic' and calibration is not None and calibration.attitude_source != 'rigid_mount_simulator_v2':
        calibration = replace(calibration, attitude_source='synthetic_fixed_world_legacy')
    imu = build_imu_stream(trip.motions, replace(cfg.motion, filter_dt_s=cfg.road_ekf.step_s,
        shock_hold_s=0., shock_heading_recovery_s=0., shock_heading_gain=1.),
        calibration=calibration, reference_heading_rad=heading,
        t_start=first.monotonic_time, t_end=t0+duration)
    if not imu.controls:
        raise ReconstructionError('motion samples produced an empty control stream')
    origin = frame.to_local(first.latitude, first.longitude)
    speed = float(first.speed or 0.) if first.has_valid_speed else 0.
    stationary = []
    speed_fixes = [s for s in fixes if s.has_valid_speed]
    speed_times = np.array([s.monotonic_time for s in speed_fixes])
    for control in imu.controls:
        if control.t > first.monotonic_time+12.:
            break
        j = int(np.searchsorted(speed_times, control.t, side='right'))-1
        if (j >= 0 and control.t-speed_times[j] <= 1. and speed_fixes[j].speed < .3
                and control.is_quiet and not control.is_shock and control.a_long is not None):
            stationary.append(control)
    bias_a = float(np.median([c.a_long for c in stationary])) if len(stationary) >= 5 else 0.
    bias_w = float(np.median([c.yaw_rate for c in stationary])) if len(stationary) >= 5 else 0.
    bank = RoadEKF(network, cfg, trace_hook=trace_hook)
    bank.seed(origin, speed, heading, measurement_sigma(first, cfg.gps),
              bias_a=bias_a, bias_w=bias_w,
              heading_sigma=.3 if heading_known else math.pi)
    if bank.best is None and cfg.gps_start_only:
        raise ReconstructionError('initial GPS is more than 60 m from the road graph')
    baseline = ExtendedKalmanFilter(cfg.motion, [*origin, speed, heading, bias_a, bias_w])
    monitor = GPSQualityMonitor(cfg.gps, cfg.motion.max_accel_ms2, cfg.motion.max_speed_ms)
    primary = RoadTrack(network)
    baseline_track = Track('ekf_dead_reckoning')
    last_track = Track('last_known_position')
    tracks = {t.name:t for t in [primary, baseline_track, last_track]}
    regions, states, timeline = [], [], []
    last_xy = origin
    last_trusted = first.monotonic_time
    last_gps_update = first.monotonic_time - 1.
    index = 0
    next_output = first.monotonic_time
    accepted_points, rejected_points = [], []

    def gps_update(fix):
        nonlocal last_xy, last_trusted, last_gps_update
        xy = frame.to_local(fix.latitude, fix.longitude)
        # Gate against the closest live mode, rather than the mean of two roads.
        mode = min(bank.modes, key=lambda m: math.dist(bank.position(m), xy), default=None)
        covariance = None
        if mode is not None:
            theta = bank.tangent(mode)
            direction = np.array([math.cos(theta), math.sin(theta)])
            covariance = np.outer(direction, direction) * mode.P[S,S] + np.eye(2)*36.
        previous_state = monitor.state
        gate = monitor.update(fix, xy,
            predicted_xy=bank.position(mode),
            predicted_speed=float(mode.x[V]) if mode is not None else float(fix.speed or 0.),
            covariance=covariance, road_distance_m=network.distance_to_road(xy))
        (accepted_points if gate.accepted else rejected_points).append({
            't':fix.monotonic_time-t0, 'latitude':fix.latitude, 'longitude':fix.longitude,
            'state':gate.state.value, 'reasons':gate.reasons})
        if not gate.accepted or (not monitor.is_trusted and fix is not first):
            return
        last_xy, last_trusted = xy, fix.monotonic_time
        heading_fix = course_to_heading(fix.course) if fix.has_valid_course and fix.has_valid_speed and float(fix.speed or 0.) >= cfg.gps.min_speed_for_course_ms else None
        speed_fix = float(fix.speed) if fix.has_valid_speed else None
        sigma = measurement_sigma(fix, cfg.gps)
        if fix is first:
            # The first fix already defined the prior; do not count it twice.
            return
        if mode is None or (previous_state != GPSState.TRUSTED and monitor.is_trusted
                            and math.dist(bank.position(mode), xy) > 4*sigma + 30):
            bank.seed(xy, speed_fix or 0., heading_fix if heading_fix is not None else heading,
                      sigma, bias_w=bias_w, heading_sigma=.3 if heading_fix is not None else math.pi)
            primary.break_line()
        else:
            # 10 Hz fixes share receiver errors. Position information is limited
            # to approximately one independent fix per second.
            interval = max(.01, fix.monotonic_time-last_gps_update)
            bank.update_gps(xy, sigma, speed_fix, heading_fix,
                            noise_scale=math.sqrt(max(1., 1./interval)))
        last_gps_update = fix.monotonic_time
        bank.pruned_fraction_product = 1.0
        baseline.update_position(xy, sigma)
        if speed_fix is not None:
            baseline.update_speed(speed_fix, 1.)
        if heading_fix is not None:
            baseline.update_heading(heading_fix, .15)

    def output(t):
        if bank.best is not None:
            primary.add_mode(t, bank.best)
        else:
            primary.break_line()
        baseline_track.add(t, baseline.position)
        last_track.add(t, last_xy)
        region = bank.uncertainty(t, monitor.state.value, max(0., t-last_trusted))
        regions.append(region)
        best = bank.best
        timeline.append({'t':t-t0, 'status':region.status,
            'represented_mass':region.represented_mass, 'hypotheses':len(bank.modes),
            'best_weight':math.exp(best.log_weight) if best else 0.,
            'speed_ms':float(best.x[V]) if best else None,
            'edge_index':best.edge if best else None,
            's_m':float(best.x[S]) if best else None,
            'sigma_s_m':math.sqrt(max(0., best.P[S,S])) if best else None,
            'position_xy':bank.position(best),
            'heading_rad':float(best.x[PSI]) if best else None,
            'gyro_bias_rads':float(best.x[BW]) if best else None,
            'road_heading_error_rad':float(best.x[ROAD_ERROR]) if best else None,
            'accel_bias_ms2':float(best.x[2]) if best else None,
            'heading_nis':bank.last_heading_nis, 'lost_reason':bank.lost_reason,
            'retained_fraction_product':bank.pruned_fraction_product})
        states.append({'t':t-t0, 'state':monitor.state.value})
        if progress:
            progress((t-first.monotonic_time)/max(1., duration))

    # Split an IMU bin at fix timestamps: GPS at t must not correct state at t+.5.
    current = first.monotonic_time
    gps_update(first)
    index = 1
    output(current)
    next_output = current + output_dt
    for control in imu.controls:
        while current < control.t - 1e-8:
            target = min(control.t, fixes[index].monotonic_time if index < len(fixes) else math.inf)
            if target > current + 1e-8:
                piece = replace(control, t=target, dt=target-current)
                bank.predict(piece)
                baseline.predict(piece.a_world, piece.yaw_rate, piece.dt,
                    a_vehicle=piece.a_long, coast=piece.is_shock)
                current = target
            while index < len(fixes) and fixes[index].monotonic_time <= current + 1e-8:
                gps_update(fixes[index])
                index += 1
        monitor.note_gap(control.t)
        if control.t >= next_output - 1e-8:
            output(control.t)
            next_output = control.t + output_dt
    if regions[-1].t < imu.controls[-1].t - 1e-8:
        output(imu.controls[-1].t)
    endpoint = primary.xy[-1] if primary.xy and bank.best else None
    diagnostics = {
        'algorithm':'road_ekf', 'tracking_architecture':'road-coordinate-gaussian-sum-v1',
        'position_estimator':'road_coordinate_ekf_bank', 'uncertainty_calibrated':False,
        'imu_processing':'offline centred preprocessing; road state updated chronologically',
        'initial_heading_deg':heading_to_course(heading), 'gps':monitor.summary(),
        'initial_bias_estimate':{'accel_ms2':bias_a, 'gyro_rads':bias_w,
                                 'stationary_controls':len(stationary)},
        'gps_state_intervals':state_intervals(monitor.history),
        'accepted_fixes':accepted_points, 'rejected_fixes':rejected_points,
        'imu':{'control_steps':len(imu.controls), 'raw_motion_samples':len(trip.motions),
               'filter_dt_s':cfg.road_ekf.step_s},
        'road_ekf':{'timeline':timeline, 'max_modes_before_pruning':bank.max_modes_seen,
            'pruning_events':bank.pruning_events,
            'retained_fraction_product':bank.pruned_fraction_product,
            'retention_note':'product of conditional retention fractions since trusted GPS; not a posterior probability',
            'corridor_half_width_m':cfg.road_ekf.corridor_half_width_m,
            'max_corridor_length_m':cfg.road_ekf.cell_length_m,
            'max_display_hypotheses':cfg.road_ekf.max_display_hypotheses,
            'lost_reason':bank.lost_reason,
            'legacy_heuristics_used':False},
        'runtime_s':round(time.perf_counter()-started,3),
        'final_vehicle_position':dict(zip(('latitude','longitude'), frame.to_geo(*endpoint))) if endpoint else None,
        'final_vehicle_position_source':'road_ekf' if endpoint else 'unavailable',
    }
    outages = _outage_windows(monitor.history, t0, cfg, duration)
    diagnostics['outage_windows'] = outages
    return ReconstructionResult('road_ekf', frame, tracks, regions, monitor.history,
        diagnostics, states, network=network, outage_windows=outages, road_uncertainty=regions)
