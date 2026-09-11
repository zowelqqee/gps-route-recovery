"""Gaussian-sum inertial filter in directed road coordinates.

Each mode is (edge, successor, recent path, [s, v, b_a, psi, b_w]).
A road tangent is a state constraint, not a second gyro measurement. Its
Jacobian includes curvature: turns can correct distance and speed through
cross covariance. At junctions the prior is split before evaluating this
constraint, so the same gyro evidence is never used twice to choose a turn.

Gaussian truncation partitions probability at road boundaries and into short
longitudinal cells. Beam pruning is approximate and recorded. Fixed-width
corridors display a bounded subset of this conditional posterior; their mass
is reported explicitly, and is NOT advertised as calibrated 95% coverage.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Callable, Optional

import numpy as np
from scipy.special import ndtr, logsumexp
from shapely.ops import substring

from geotrace.config import Config
from geotrace.coordinates import wrap_angle
from geotrace.motion_model import bounded_displacement, ImuControl, longitudinal_acceleration
from geotrace.polygons import BranchComponent, UncertaintySet
from geotrace.road_graph import RoadNetwork

S, V, BA, PSI, BW, ROAD_ERROR = range(6)


@dataclass
class Mode:
    edge: int
    x: np.ndarray
    P: np.ndarray
    log_weight: float
    history: tuple[int, ...]
    successor: Optional[int] = None
    last_map_heading: Optional[float] = None
    trace_id: int = 0
    parent_trace_id: Optional[int] = None
    debug: Optional[dict[str, Any]] = None

    def copy(self) -> "Mode":
        return replace(self, x=self.x.copy(), P=self.P.copy(),
                       debug=dict(self.debug or {}))


def truncate(mode: Mode, low: float, high: float) -> Optional[Mode]:
    """Moment match a Gaussian conditioned on low <= s < high.

    Keep the correlations with velocity and biases, not just marginal s.
    The returned weight includes the probability of the interval.
    """
    variance = max(float(mode.P[S, S]), 1e-12)
    sigma = math.sqrt(variance)
    a, b = (low - mode.x[S]) / sigma, (high - mode.x[S]) / sigma
    # Survival probabilities avoid cancellation in the positive tail.
    mass = float(ndtr(-a) - ndtr(-b) if a > 0 else ndtr(b) - ndtr(a))
    if mass < 1e-10:
        return None
    pa = math.exp(-0.5 * a * a) / math.sqrt(2 * math.pi)
    pb = math.exp(-0.5 * b * b) / math.sqrt(2 * math.pi)
    shift = (pa - pb) / mass
    apa = a * pa if math.isfinite(a) else 0.0
    bpb = b * pb if math.isfinite(b) else 0.0
    new_var = variance * max(1e-8, 1 + (apa - bpb) / mass - shift * shift)
    result = mode.copy()
    column = mode.P[:, S].copy()
    result.x += column / sigma * shift
    result.x[PSI] = wrap_angle(result.x[PSI])
    result.P += np.outer(column, column) * ((new_var - variance) / variance**2)
    result.P = (result.P + result.P.T) * 0.5
    result.log_weight += math.log(mass)
    return result


def scalar_update(mode: Mode, residual: float, H: np.ndarray, variance: float) -> float:
    """Joseph-form measurement update; return the pre-update log likelihood."""
    innovation = max(float(H @ mode.P @ H + variance), 1e-12)
    K = mode.P @ H / innovation
    mode.x += K * residual
    mode.x[PSI] = wrap_angle(mode.x[PSI])
    A = np.eye(6) - np.outer(K, H)
    mode.P = A @ mode.P @ A.T + np.outer(K, K) * variance
    mode.P = 0.5 * (mode.P + mode.P.T)
    return -0.5 * (residual**2 / innovation + math.log(2 * math.pi * innovation))


class RoadEKF:
    def __init__(self, network: RoadNetwork, cfg: Config,
                 trace_hook: Optional[Callable[[dict[str, Any]], None]] = None):
        self.network, self.cfg, self.rc = network, cfg, cfg.road_ekf
        self.modes: list[Mode] = []
        self.pruned_fraction_product = 1.0
        self.pruning_events = 0
        self.max_modes_seen = 0
        self.inconsistent_steps = 0
        self.last_heading_nis = 0.0
        self.lost_reason: Optional[str] = None
        self.yaw_since_constraint = 0.0
        self.accel_bias_centre = 0.0
        self.trace_hook = trace_hook
        self._trace_t = 0.0
        self._next_trace_id = 1
        self._trace_events: list[dict[str, Any]] = []

    def _assign_child(self, child: Mode, parent: Mode) -> Mode:
        child.parent_trace_id = parent.trace_id
        child.trace_id = self._next_trace_id
        self._next_trace_id += 1
        return child

    def _record_pruning(self, mode: Mode, reason: str, **details: Any) -> None:
        self._trace_events.append({
            "trace_id": mode.trace_id,
            "parent_trace_id": mode.parent_trace_id,
            "edge": mode.edge,
            "reason": reason,
            "log_weight": float(mode.log_weight),
            **details,
        })

    def _emit_trace(self, stage: str) -> None:
        if self.trace_hook is None:
            self._trace_events.clear()
            return
        normalizer = (float(logsumexp([m.log_weight for m in self.modes]))
                      if self.modes else -math.inf)
        modes = []
        for rank, mode in enumerate(sorted(self.modes, key=lambda m: m.log_weight,
                                           reverse=True), 1):
            edge = self.network.edges[mode.edge]
            modes.append({
                "trace_id": mode.trace_id,
                "parent_trace_id": mode.parent_trace_id,
                "rank": rank,
                "edge": mode.edge,
                "edge_id": [str(part) for part in edge.edge_id],
                "road_name": edge.name,
                "s_m": float(mode.x[S]),
                "edge_length_m": float(edge.length),
                "speed_ms": float(mode.x[V]),
                "accel_bias_ms2": float(mode.x[BA]),
                "heading_rad": float(mode.x[PSI]),
                "gyro_bias_rads": float(mode.x[BW]),
                "road_heading_error_rad": float(mode.x[ROAD_ERROR]),
                "sigma_s_m": math.sqrt(max(float(mode.P[S, S]), 0.0)),
                "log_weight": float(mode.log_weight),
                "weight": (math.exp(mode.log_weight-normalizer)
                           if math.isfinite(normalizer) else 0.0),
                "history": list(mode.history),
                "successor": mode.successor,
                **(mode.debug or {}),
            })
        self.trace_hook({"t": self._trace_t, "stage": stage,
                         "modes": modes, "events": list(self._trace_events)})
        self._trace_events.clear()

    def seed(self, xy, speed: float, heading: float, sigma: float,
             bias_a: Optional[float] = None, bias_w: float = 0.0,
             heading_sigma: float = 0.3) -> None:
        self.modes = []
        self._next_trace_id = 1
        if bias_a is not None:
            self.accel_bias_centre = bias_a
        bias_a = self.accel_bias_centre
        self.yaw_since_constraint = 0.0
        self.inconsistent_steps = 0
        for edge_index in self.network.nearest_edges(xy, k=12, radius=60.0):
            s, distance = self.network.project(xy, edge_index)
            if distance > 60:
                continue
            edge = self.network.edges[edge_index]
            delta = wrap_angle(heading - edge.bearing(s))
            P = np.diag([sigma**2, 2.0**2, self.rc.accel_error_sigma_ms2**2, heading_sigma**2,
                         self.rc.initial_gyro_bias_sigma**2, self.rc.heading_sigma_rad**2])
            x = np.array([s, max(0., speed), bias_a, heading, bias_w, 0.])
            logw = -0.5 * (distance**2 / sigma**2 + delta**2 / (heading_sigma**2 + .2**2))
            self.modes.append(Mode(edge_index, x, P, logw, (edge_index,),
                                   last_map_heading=edge.bearing(s),
                                   trace_id=self._next_trace_id, debug={}))
            self._next_trace_id += 1
        self.lost_reason = None if self.modes else "initial_position_outside_graph"
        self._reduce("seed")

    @property
    def best(self) -> Optional[Mode]:
        return max(self.modes, key=lambda m: m.log_weight, default=None)

    def position(self, mode: Optional[Mode] = None):
        mode = mode or self.best
        return None if mode is None else self.network.edges[mode.edge].position(mode.x[S])

    def _path_position(self, m: Mode, s: float) -> np.ndarray:
        edge = self.network.edges[m.edge]
        if s < 0 and len(m.history) > 1:
            previous = self.network.edges[m.history[-2]]
            return np.array(previous.position(previous.length + s))
        if s > edge.length and m.successor is not None:
            return np.array(self.network.edges[m.successor].position(s - edge.length))
        return np.array(edge.position(s))

    def tangent(self, m: Mode, s: Optional[float] = None) -> float:
        s = float(m.x[S]) if s is None else s
        span = self.rc.tangent_span_m * 0.5
        d = self._path_position(m, s + span) - self._path_position(m, s - span)
        if np.linalg.norm(d) < 1e-5:
            return self.network.edges[m.edge].bearing(s)
        return math.atan2(d[1], d[0])

    def _plan(self, modes: list[Mode]) -> list[Mode]:
        planned = []
        for m in modes:
            edge = self.network.edges[m.edge]
            reach = m.x[S] + 4 * math.sqrt(max(m.P[S, S], 0)) + self.rc.tangent_span_m
            if m.successor is None and reach >= edge.length:
                successors = self.network.allowed_successors(m.edge, m.history)
                if successors:
                    for index in successors:
                        child = self._assign_child(m.copy(), m)
                        child.successor = index
                        penalty = -math.log(len(successors))
                        child.log_weight += penalty
                        child.debug["transition_penalty_log"] = penalty
                        child.debug["transition_kind"] = "planned_successor"
                        planned.append(child)
                    continue
            planned.append(m)
        return planned

    def predict(self, control: ImuControl) -> None:
        self._trace_t = float(control.t)
        self._trace_events.clear()
        self._emit_trace("before_predict")
        if control.gap_exceeded:
            self.modes = []
            self.lost_reason = "imu_gap"
            return
        dt = control.dt
        bias_decay = math.exp(-dt/self.rc.accel_error_tau_s)
        bias_average_gain = self.rc.accel_error_tau_s/dt*(1-bias_decay)
        for m in self.modes:
            old_x, old_P = m.x.copy(), m.P.copy()
            a = (control.a_long if control.a_long is not None else
                 longitudinal_acceleration(control.a_world, m.x[PSI]))
            coast = control.is_shock
            average_bias = self.accel_bias_centre + (m.x[BA]-self.accel_bias_centre)*bias_average_gain
            accel = 0.0 if coast else float(np.clip(a - average_bias,
                -self.cfg.motion.max_accel_ms2, self.cfg.motion.max_accel_ms2))
            m.debug = {
                "raw_accel_ms2": float(a),
                "accel_residual_ms2": float(a-average_bias),
                "yaw_rate_rads": float(control.yaw_rate),
                "gyro_increment_rad": float((control.yaw_rate-m.x[BW])*dt),
                "transition_penalty_log": 0.0,
                "transition_kind": None,
                "pruning_reason": None,
            }
            accel_gain = float(not coast and abs(a - average_bias) < self.cfg.motion.max_accel_ms2)
            d, dv, da = bounded_displacement(float(m.x[V]), accel, dt, self.cfg.motion.max_speed_ms)
            F = np.eye(6)
            F[S, V] = dv
            F[S, BA] = -da * accel_gain * bias_average_gain
            speed = m.x[V] + accel * dt
            interior = 0 < speed < self.cfg.motion.max_speed_ms
            F[V, V] = float(interior)
            F[V, BA] = -dt * accel_gain * bias_average_gain if interior else 0.
            F[BA, BA] = bias_decay
            # A pothole's accelerometer spike is not evidence that the gyro
            # stopped measuring a real turn. Gate the channels independently.
            gyro_corrupt = control.peak_gyro_rads >= self.cfg.motion.shock_gyro_rads
            trust = 0.0 if gyro_corrupt else 1.0
            F[PSI, BW] = -dt * trust
            if control.a_long is None and not coast:
                derivative = (-control.a_world[0] * math.sin(m.x[PSI])
                              + control.a_world[1] * math.cos(m.x[PSI]))
                F[S, PSI] = da * derivative * accel_gain
                F[V, PSI] = dt * derivative * accel_gain if interior else 0.
            spatial_decay = math.exp(-d / self.rc.road_heading_correlation_m)
            F[ROAD_ERROR, ROAD_ERROR] = spatial_decay
            F[ROAD_ERROR, V] = -old_x[ROAD_ERROR]*spatial_decay*dv / self.rc.road_heading_correlation_m
            F[ROAD_ERROR, BA] = old_x[ROAD_ERROR]*spatial_decay*da*accel_gain*bias_average_gain / self.rc.road_heading_correlation_m
            F[ROAD_ERROR, PSI] = -old_x[ROAD_ERROR]*spatial_decay*F[S,PSI] / self.rc.road_heading_correlation_m
            curvature = wrap_angle(self.tangent(m, m.x[S]+1.)-self.tangent(m, m.x[S]-1.))/2.
            road_error_variance = (self.rc.heading_sigma_rad**2
                + (curvature*self.rc.corridor_half_width_m*.5)**2)
            m.x[ROAD_ERROR] *= spatial_decay
            m.x[BA] = self.accel_bias_centre + (old_x[BA]-self.accel_bias_centre)*bias_decay
            m.x[S] += d
            m.x[V] = np.clip(speed, 0., self.cfg.motion.max_speed_ms)
            m.x[PSI] = wrap_angle(m.x[PSI] + (control.yaw_rate - m.x[BW]) * dt * trust)
            Q = np.zeros((6, 6))
            q = self.cfg.motion.accel_noise**2 * (16 if coast else 1)
            Q[S, S], Q[S, V], Q[V, S], Q[V, V] = q*dt**3/3, q*dt**2/2, q*dt**2/2, q*dt
            bias_variance = self.rc.accel_error_sigma_ms2**2*(1-bias_decay**2)
            bias_noise_direction = np.array([-dt*dt/6., -dt/2., 1., 0., 0., 0.])
            Q += bias_variance*np.outer(bias_noise_direction,bias_noise_direction)
            Q[PSI, PSI] = (self.cfg.motion.shock_heading_noise_radsqrt**2
                          if gyro_corrupt else self.rc.gyro_noise**2) * dt
            Q[BW, BW] = self.rc.gyro_bias_walk**2 * dt
            Q[ROAD_ERROR, ROAD_ERROR] = road_error_variance * (1-spatial_decay**2)
            m.P = F @ m.P @ F.T + Q
            speed_sigma = math.sqrt(max(old_P[V,V] + dt*dt*old_P[BA,BA]
                                       - 2*dt*old_P[V,BA], 0.))
            if speed < 3*speed_sigma or speed > self.cfg.motion.max_speed_ms - 3*speed_sigma:
                # At a speed boundary, the derivative of clip at the mean is
                # not the uncertainty of the distribution. Integrate positive
                # cubature points instead: stopped and moving support survives.
                root = np.linalg.cholesky((old_P+old_P.T)*.5 + np.eye(6)*1e-10)
                points = np.vstack([old_x + math.sqrt(6)*root.T,
                                    old_x - math.sqrt(6)*root.T])
                decay_squared_sum = 0.
                for x in points:
                    raw = (control.a_long if control.a_long is not None else
                           longitudinal_acceleration(control.a_world, x[PSI]))
                    bi = self.accel_bias_centre+(x[BA]-self.accel_bias_centre)*bias_average_gain
                    ai = 0. if coast else float(np.clip(raw-bi,
                        -self.cfg.motion.max_accel_ms2, self.cfg.motion.max_accel_ms2))
                    vi = float(np.clip(x[V], 0., self.cfg.motion.max_speed_ms))
                    di = bounded_displacement(vi, ai, dt, self.cfg.motion.max_speed_ms)[0]
                    x[S] += di
                    x[ROAD_ERROR] *= math.exp(-di/self.rc.road_heading_correlation_m)
                    decay_squared_sum += math.exp(-2*di/self.rc.road_heading_correlation_m)
                    x[BA] = self.accel_bias_centre+(x[BA]-self.accel_bias_centre)*bias_decay
                    x[V] = np.clip(vi + ai*dt, 0., self.cfg.motion.max_speed_ms)
                    x[PSI] += (control.yaw_rate-x[BW])*dt*trust
                m.x = points.mean(axis=0)
                deviations = points-m.x
                Q[ROAD_ERROR,ROAD_ERROR] = road_error_variance*(1-decay_squared_sum/len(points))
                m.P = deviations.T @ deviations / len(points) + Q
                m.x[PSI] = wrap_angle(m.x[PSI])
        self.modes = self._plan(self.modes)
        self._emit_trace("after_plan")
        self._partition()
        self._emit_trace("after_partition")
        if control.peak_gyro_rads < self.cfg.motion.shock_gyro_rads:
            self.yaw_since_constraint += control.yaw_rate*dt
        new_gyro_direction = abs(self.yaw_since_constraint) >= self.rc.heading_sigma_rad
        if new_gyro_direction:
            self.yaw_since_constraint = 0.0
        best_nis = math.inf
        evaluated = False
        for m in self.modes:
            # The residual's slope in s makes a bend a distance observation.
            theta = self.tangent(m)
            previous_map_heading = m.last_map_heading
            m.debug["map_heading_rad"] = float(theta)
            m.debug["map_heading_change_rad"] = (float(wrap_angle(theta-previous_map_heading))
                                                   if previous_map_heading is not None else None)
            if (not new_gyro_direction and m.last_map_heading is not None
                    and abs(wrap_angle(theta-m.last_map_heading)) < self.rc.heading_sigma_rad):
                m.debug["gyro_map_residual_rad"] = float(
                    wrap_angle(theta + m.x[ROAD_ERROR] - m.x[PSI]))
                m.debug["heading_constraint_applied"] = False
                continue
            evaluated = True
            m.last_map_heading = theta
            curvature = wrap_angle(self.tangent(m, m.x[S] + 1.) - self.tangent(m, m.x[S] - 1.)) / 2.
            H = np.array([-curvature, 0., 0., 1., 0., -1.])
            residual = wrap_angle(theta + m.x[ROAD_ERROR] - m.x[PSI])
            variance = .01**2
            nis = residual**2 / max(float(H @ m.P @ H + variance), 1e-12)
            best_nis = min(best_nis, nis)
            m.debug["gyro_map_residual_rad"] = float(residual)
            m.debug["heading_nis"] = float(nis)
            m.debug["heading_constraint_applied"] = True
            # New geometric constraints have compatibility in [0,1]. A
            # Gaussian density (>1 in radians) must not reward a mode merely
            # because it crossed a map vertex while another mode did not.
            m.log_weight += scalar_update(m, residual, H, variance) + .5*math.log(2*math.pi*variance)
            m.x[V] = np.clip(m.x[V], 0., self.cfg.motion.max_speed_ms)
        self.last_heading_nis = best_nis if math.isfinite(best_nis) else 0.
        if new_gyro_direction and evaluated:
            self.inconsistent_steps = self.inconsistent_steps + 1 if best_nis > 16 else 0
        if self.inconsistent_steps >= 3:
            self.modes = []
            self.lost_reason = "no_road_hypothesis_explains_gyro"
        self._emit_trace("before_reduce")
        self._reduce("predict")
        self._emit_trace("after_reduce")

    def _partition(self) -> None:
        """Partition s at junctions, then in finite cells, preserving mass.

        A negative tail follows known ancestry (uncertainty about progress, not
        reverse driving). At the initial edge it is conditioned on s >= 0.
        At a dead end the outgoing mass is dropped and recorded as model loss.
        """
        queue = [(m, 0) for m in self.modes]
        floor = max((m.log_weight for m in self.modes), default=0.) - 30.
        result = []
        while queue:
            m, depth = queue.pop()
            if m.log_weight < floor:
                self._record_pruning(m, "partition_relative_log_weight_floor",
                                     floor_log_weight=float(floor), depth=depth)
                continue
            edge = self.network.edges[m.edge]
            sigma = math.sqrt(max(m.P[S, S], 1e-12))
            low = max(0., m.x[S] - 7 * sigma)
            high = min(edge.length, m.x[S] + 7 * sigma)
            if high > low:
                first = int(low // self.rc.cell_length_m)
                last = int(high // self.rc.cell_length_m)
                for cell in range(first, last + 1):
                    piece = truncate(m, cell*self.rc.cell_length_m,
                                     min((cell+1)*self.rc.cell_length_m, edge.length))
                    if piece is not None:
                        self._assign_child(piece, m)
                        piece.debug["partition_interval_m"] = [
                            cell*self.rc.cell_length_m,
                            min((cell+1)*self.rc.cell_length_m, edge.length),
                        ]
                        result.append(piece)
            if depth >= 12:
                self._record_pruning(m, "partition_depth_limit", depth=depth)
                continue
            forward = truncate(m, edge.length, math.inf)
            if forward is not None:
                successors = ([m.successor] if m.successor is not None else
                              self.network.allowed_successors(m.edge, m.history))
                if successors:
                    for index in successors:
                        child = self._assign_child(forward.copy(), m)
                        child.x[S] -= edge.length
                        child.edge, child.successor = index, None
                        child.history = (m.history + (index,))[-max(8, self.network.restriction_history_limit):]
                        penalty = -math.log(len(successors))
                        child.log_weight += penalty
                        child.debug["transition_penalty_log"] = (
                            float(child.debug.get("transition_penalty_log", 0.0)) + penalty)
                        child.debug["transition_kind"] = "crossed_junction"
                        queue.append((child, depth + 1))
                else:
                    self._record_pruning(forward, "dead_end_no_successor", depth=depth)
            backward = truncate(m, -math.inf, 0.)
            if backward is not None and len(m.history) > 1:
                backward.edge = m.history[-2]
                backward.successor = m.edge
                backward.history = m.history[:-1]
                backward.x[S] += self.network.edges[backward.edge].length
                self._assign_child(backward, m)
                queue.append((backward, depth + 1))
        before = float(logsumexp([m.log_weight for m in self.modes])) if self.modes else 0.
        after = float(logsumexp([m.log_weight for m in result])) if result else -math.inf
        retained = min(1., math.exp(after - before))
        if retained < .999:
            self.pruned_fraction_product *= retained
            self.pruning_events += 1
        self.modes = result
        if not result and self.lost_reason is None:
            self.lost_reason = "road_support_exhausted"

    def _reduce(self, context: str = "reduce") -> None:
        if not self.modes:
            return
        groups: dict[tuple, list[Mode]] = {}
        for m in self.modes:
            key = (m.edge, int(max(0, m.x[S]) // self.rc.cell_length_m),
                   m.successor, m.history)
            groups.setdefault(key, []).append(m)
        merged = []
        for modes in groups.values():
            total = float(logsumexp([m.log_weight for m in modes]))
            top = max(modes, key=lambda m: m.log_weight).copy()
            for mode in modes:
                if mode.trace_id != top.trace_id:
                    self._record_pruning(mode, "merged_equivalent_mode",
                                         retained_trace_id=top.trace_id,
                                         reduce_context=context)
            if len(modes) > 1:
                weights = np.exp(np.array([m.log_weight for m in modes]) - total)
                states = np.array([m.x for m in modes])
                states[:, PSI] = top.x[PSI] + np.array([wrap_angle(x - top.x[PSI]) for x in states[:, PSI]])
                mean = weights @ states
                top.P = sum(w * (m.P + np.outer(x-mean, x-mean))
                            for w, m, x in zip(weights, modes, states))
                top.x = mean
                top.x[PSI] = wrap_angle(top.x[PSI])
            top.log_weight = total
            merged.append(top)
        self.max_modes_seen = max(self.max_modes_seen, len(merged))
        normalizer = float(logsumexp([m.log_weight for m in merged]))
        merged.sort(key=lambda m: m.log_weight, reverse=True)
        kept = [m for m in merged[:self.rc.max_hypotheses]
                if m.log_weight >= merged[0].log_weight - 30.]
        kept_ids = {m.trace_id for m in kept}
        for rank, mode in enumerate(merged, 1):
            if mode.trace_id in kept_ids:
                continue
            reason = ("beam_width_limit" if rank > self.rc.max_hypotheses
                      else "reduce_relative_log_weight_floor")
            self._record_pruning(mode, reason, rank=rank,
                                 max_hypotheses=self.rc.max_hypotheses,
                                 best_log_weight=float(merged[0].log_weight),
                                 reduce_context=context)
        kept_z = float(logsumexp([m.log_weight for m in kept]))
        if len(kept) < len(merged):
            self.pruning_events += 1
            self.pruned_fraction_product *= math.exp(kept_z - normalizer)
        for m in kept:
            m.log_weight -= kept_z
        self.modes = kept

    def update_gps(self, xy, sigma: float, speed: Optional[float],
                   heading: Optional[float], noise_scale: float = 1.0) -> None:
        sigma *= noise_scale
        for m in self.modes:
            edge = self.network.edges[m.edge]
            p = np.asarray(edge.position(m.x[S]))
            theta = edge.bearing(m.x[S])
            along = np.array([math.cos(theta), math.sin(theta)])
            across = np.array([-along[1], along[0]])
            residual = np.asarray(xy) - p
            H = np.array([1., 0., 0., 0., 0., 0.])
            m.log_weight += scalar_update(m, float(residual @ along), H, sigma**2)
            m.log_weight += -0.5 * (float(residual @ across)**2 / (sigma**2 + 3.**2))
            if speed is not None:
                H = np.array([0., 1., 0., 0., 0., 0.])
                m.log_weight += scalar_update(m, speed - m.x[V], H, noise_scale**2)
            if heading is not None:
                H = np.array([0., 0., 0., 1., 0., 0.])
                m.log_weight += scalar_update(m, wrap_angle(heading - m.x[PSI]), H, (.15*noise_scale)**2)
            m.x[V] = np.clip(m.x[V], 0., self.cfg.motion.max_speed_ms)
        self._reduce("gps_update")

    def uncertainty(self, t: float, gps_state: str, elapsed: float) -> UncertaintySet:
        components = []
        # Display each cell separately, even if buffers touch at intersections.
        # A long Gaussian is represented by only the probability IN this window.
        for i, m in enumerate(sorted(self.modes, key=lambda m: m.log_weight, reverse=True)):
            edge = self.network.edges[m.edge]
            half = min(self.rc.cell_length_m / 2, 2. * math.sqrt(max(m.P[S, S], 0.)))
            low, high = max(0., m.x[S]-half), min(edge.length, m.x[S]+half)
            if high <= low:
                continue
            selected = truncate(m, low, high)
            if selected is None:
                continue
            mass = math.exp(selected.log_weight)
            geometry = substring(edge.line, low, high).buffer(self.rc.corridor_half_width_m)
            components.append(BranchComponent(f"road-ekf-{i}", mass, geometry,
                np.array([i]), [m.edge], edge.position(m.x[S]), geometry.area,
                [edge.name] if edge.name else []))
        components.sort(key=lambda c: c.probability, reverse=True)
        components = components[:self.rc.max_display_hypotheses]
        mass = sum(c.probability for c in components)
        status = "LOST" if not self.modes else "AMBIGUOUS"
        if components and components[0].probability >= .8 and self.pruned_fraction_product >= .9:
            status = "CONCENTRATED"
        return UncertaintySet(t, self.cfg.polygon.confidence, components,
            gps_state, elapsed, len(components), len(self.modes), mass, status)
