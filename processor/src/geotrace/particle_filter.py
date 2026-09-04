"""Road-constrained particle filter.

Every particle is a hypothesis about *where on the road network* the car is:

    edge_id, distance_along_edge, speed, heading, accel_bias, gyro_bias, weight

Because a particle lives on an edge rather than in free space, the belief after
a junction is a set of distinct branches, not a blob covering the buildings in
between. That is the whole point: when GPS dies just before a fork, the honest
answer is "branch A with p=0.71, branch B with p=0.29", never their average.

Propagation follows the same motion model as the EKF, plus per-particle process
noise:

    s_{i,t+1}   = s + v dt + 0.5 a_hat dt^2 + eps_s
    v_{i,t+1}   = max(0, v + a_hat dt + eps_v)
    psi_{i,t+1} = psi + w_hat dt + eps_psi

When a particle runs past the end of its edge it is handed to a junction model
that samples an outgoing edge with

    P(e' | p_i) ~ exp(-wrap(theta_e' - psi_i)^2 / (2 sigma_turn^2)) * P_route(e')

Weights combine a GPS term, a heading term, a speed-limit term and a map term:

    w~_i = w_i * L_GPS * L_psi * L_v * L_map,   w_i = w~_i / sum_j w~_j

with systematic resampling whenever N_eff = 1 / sum(w_i^2) drops below N/2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from geotrace.config import Config, MotionConfig, ParticleFilterConfig
from geotrace.coordinates import wrap_angle
from geotrace.road_graph import RoadNetwork

# Route prior: without turn restrictions or traffic data, the only honest prior
# is that a driver is somewhat more likely to stay on a larger road.
ROUTE_PRIOR = {
    "motorway": 1.6, "motorway_link": 1.2, "trunk": 1.5, "trunk_link": 1.2,
    "primary": 1.4, "primary_link": 1.1, "secondary": 1.25, "secondary_link": 1.05,
    "tertiary": 1.1, "unclassified": 1.0, "residential": 1.0,
    "living_street": 0.8, "service": 0.6, "road": 1.0,
}


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Low-variance systematic resampling.

    One uniform draw, then N equally spaced pointers into the cumulative weight.
    Returns the array of chosen indices, which always has length ``len(weights)``.
    """
    n = len(weights)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0  # guard against floating point falling short of 1
    return np.searchsorted(cumulative, positions).clip(0, n - 1).astype(np.int64)


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Normalise to sum 1. A fully collapsed set falls back to uniform, which is
    the correct statement of "the filter knows nothing" rather than a crash."""
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    total = w.sum()
    if total <= 0 or not math.isfinite(total):
        return np.full(len(w), 1.0 / max(1, len(w)))
    return w / total


def effective_sample_size(weights: np.ndarray) -> float:
    """N_eff = 1 / sum(w_i^2)."""
    w = np.asarray(weights, dtype=float)
    denom = float(np.sum(w * w))
    return float("inf") if denom <= 0 else 1.0 / denom


@dataclass
class ParticleSnapshot:
    """The belief at one output timestamp, kept compactly."""

    t: float
    edge_idx: np.ndarray
    s: np.ndarray
    weights: np.ndarray
    heading: np.ndarray
    speed: np.ndarray
    gps_state: str = "UNKNOWN"
    gps_accepted: Optional[bool] = None
    seconds_since_trusted: float = 0.0
    n_eff: float = 0.0


@dataclass
class ParticleFilterResult:
    snapshots: list[ParticleSnapshot] = field(default_factory=list)
    resample_count: int = 0
    junction_splits: int = 0
    dead_ends: int = 0
    skipped_gaps: int = 0
    reinitializations: int = 0
    injected: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "snapshots": len(self.snapshots),
            "resample_count": self.resample_count,
            "junction_transitions": self.junction_splits,
            "dead_end_particles": self.dead_ends,
            "skipped_gaps": self.skipped_gaps,
            "reinitializations": self.reinitializations,
            "injected_particles": self.injected,
        }


class RoadParticleFilter:
    """The `road_particle_filter` algorithm."""

    def __init__(
        self,
        network: RoadNetwork,
        cfg: Config,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.net = network
        self.cfg = cfg
        self.pf: ParticleFilterConfig = cfg.pf
        self.motion: MotionConfig = cfg.motion
        self.rng = rng if rng is not None else np.random.default_rng(cfg.seed)

        n = self.pf.n_particles
        self.edge_idx = np.zeros(n, dtype=np.int64)
        self.s = np.zeros(n, dtype=float)
        self.v = np.zeros(n, dtype=float)
        self.psi = np.zeros(n, dtype=float)
        self.b_a = np.zeros(n, dtype=float)
        self.b_w = np.zeros(n, dtype=float)
        self.w = np.full(n, 1.0 / n, dtype=float)
        self.map_likelihood = np.ones(n, dtype=float)
        """L_map: carries the dead-end penalty from prediction into the update."""

        self.last_gps_likelihood = 1.0
        """Best particle's GPS likelihood at the last update; drives divergence
        detection."""


        self.result = ParticleFilterResult()
        self.successors = self.net.successor_table(allow_uturn=False)
        self.successors_uturn = self.net.successor_table(allow_uturn=True)
        self._route_prior = np.array(
            [ROUTE_PRIOR.get(str(e.highway), 1.0) for e in self.net.edges], dtype=float
        )
        self._lengths = self.net.lengths
        self._speed_limits = self.net.speed_limits
        self._initialized = False

    # ---------------------------------------------------------------- setup

    def initialize(
        self,
        xy: Sequence[float],
        heading: float,
        speed: float = 0.0,
        accel_bias: float = 0.0,
        gyro_bias: float = 0.0,
        position_sigma: Optional[float] = None,
    ) -> None:
        """Seed the cloud around the last trusted fix.

        Particles are spread over the several nearest drivable edges, not just
        the closest one: at 15 m accuracy the nearest edge is often the wrong
        side of a dual carriageway.
        """
        n = self.pf.n_particles
        radius = position_sigma if position_sigma is not None else self.pf.init_radius_m
        candidates = self.net.nearest_edges(
            xy, k=self.pf.init_candidate_edges, radius=max(radius * 3, 120.0)
        )
        if not candidates:
            raise ValueError(
                f"no drivable road within reach of {xy}; the cached graph does "
                "not cover this trip"
            )

        # Prefer edges that both lie close to the fix and point the right way.
        scores = []
        for index in candidates:
            s_on, offset = self.net.project(xy, index)
            bearing = self.net.edges[index].bearing(s_on)
            align = math.exp(
                -(float(wrap_angle(bearing - heading)) ** 2)
                / (2 * self.pf.sigma_heading_rad**2)
            )
            proximity = math.exp(-(offset**2) / (2 * max(radius, 5.0) ** 2))
            scores.append(max(align * proximity, 1e-6))
        scores_arr = np.array(scores) / np.sum(scores)
        counts = self.rng.multinomial(n, scores_arr)

        cursor = 0
        for index, count in zip(candidates, counts):
            if count == 0:
                continue
            sl = slice(cursor, cursor + count)
            s_on, _offset = self.net.project(xy, index)
            self.edge_idx[sl] = index
            self.s[sl] = np.clip(
                s_on + self.rng.normal(0.0, max(radius * 0.5, 3.0), count),
                0.0,
                self._lengths[index],
            )
            bearing = self.net.edges[index].bearing(s_on)
            self.psi[sl] = wrap_angle(
                bearing + self.rng.normal(0.0, self.pf.sigma_heading_rad * 0.5, count)
            )
            cursor += count
        self.v[:] = np.clip(
            speed + self.rng.normal(0.0, max(0.5, speed * 0.15), n), 0.0, self.motion.max_speed_ms
        )
        # A wide spread on purpose: the accelerometer bias is what decides how
        # far the cloud travels during an outage, and it is only observable
        # through the GPS speed likelihood while GPS is still there. Seeding it
        # too tightly means there is no particle carrying the right value to
        # select.
        self.b_a[:] = accel_bias + self.rng.normal(0.0, 0.12, n)
        self.b_w[:] = gyro_bias + self.rng.normal(0.0, 0.008, n)
        self.w[:] = 1.0 / n
        self.map_likelihood[:] = 1.0
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    # ------------------------------------------------------------- predict

    def predict(self, a_world: Sequence[float], yaw_rate: float, dt: float) -> None:
        """One motion step for every particle."""
        if dt <= 0 or not self._initialized:
            return
        if dt > self.motion.max_gap_s:
            # Do not integrate across a hole in the IMU stream. Instead diffuse
            # the cloud along the roads so the belief widens honestly.
            self.result.skipped_gaps += 1
            self.s += self.rng.normal(0.0, self.motion.max_speed_ms * dt * 0.5, len(self.s))
            self._resolve_edges()
            return

        rng = self.rng
        n = len(self.s)
        a_world = np.asarray(a_world, dtype=float)

        # Longitudinal projection, per particle: a_par = aE cos(psi) + aN sin(psi)
        a_long = a_world[0] * np.cos(self.psi) + a_world[1] * np.sin(self.psi)
        a_hat = np.clip(a_long - self.b_a, -self.motion.max_accel_ms2, self.motion.max_accel_ms2)
        w_hat = yaw_rate - self.b_w

        scale = math.sqrt(dt / max(self.motion.filter_dt_s, 1e-6))
        eps_psi = rng.normal(0.0, self.pf.sigma_psi_rad * scale, n)
        eps_s = rng.normal(0.0, self.pf.sigma_s * scale, n)
        eps_v = rng.normal(0.0, self.pf.sigma_v * scale, n)

        self.psi = wrap_angle(self.psi + w_hat * dt + eps_psi)
        ds = self.v * dt + 0.5 * a_hat * dt * dt + eps_s
        self.v = np.clip(self.v + a_hat * dt + eps_v, 0.0, self.motion.max_speed_ms)
        self.s = self.s + np.maximum(ds, 0.0)

        # Bias random walk.
        self.b_a += rng.normal(0.0, self.motion.accel_bias_rw * math.sqrt(dt), n)
        self.b_w += rng.normal(0.0, self.motion.gyro_bias_rw * math.sqrt(dt), n)

        self._resolve_edges()

    def _resolve_edges(self) -> None:
        """Move particles that ran off the end of their edge through junctions.

        A particle may cross several short edges in one step, so this iterates
        until nothing overflows (with a hard cap so a pathological graph cannot
        hang the run).
        """
        self.s = np.maximum(self.s, 0.0)
        for _ in range(12):
            over = np.nonzero(self.s > self._lengths[self.edge_idx])[0]
            if over.size == 0:
                return
            self._cross_junctions(over)
        # Anything still overflowing is clamped to the end of its edge.
        self.s = np.minimum(self.s, self._lengths[self.edge_idx])

    def _cross_junctions(self, indices: np.ndarray) -> None:
        rng = self.rng
        sigma_turn_sq = 2.0 * self.pf.sigma_turn_rad**2
        gain = self.pf.heading_snap_gain
        for i in indices:
            current = int(self.edge_idx[i])
            remainder = float(self.s[i] - self._lengths[current])
            allow_uturn = self.pf.allow_uturn and float(self.v[i]) <= self.pf.uturn_max_speed_ms
            options = self.successors_uturn[current] if allow_uturn else self.successors[current]
            if options.size == 0:
                # Dead end: the car cannot be here. Park the particle at the end
                # of the edge and mark it so the weight update kills it.
                self.s[i] = self._lengths[current]
                self.v[i] = 0.0
                self.map_likelihood[i] = self.pf.dead_end_weight
                self.result.dead_ends += 1
                continue

            psi_i = float(self.psi[i])
            if options.size == 1:
                choice = int(options[0])
            else:
                bearings = self.net.bearings_fast(options, np.zeros(options.size))
                turn = np.asarray(wrap_angle(bearings - psi_i), dtype=float)
                logits = -(turn**2) / sigma_turn_sq
                probs = np.exp(logits - logits.max()) * self._route_prior[options]
                total = probs.sum()
                if total <= 0 or not math.isfinite(total):
                    choice = int(options[rng.integers(options.size)])
                else:
                    choice = int(options[rng.choice(options.size, p=probs / total)])
                self.result.junction_splits += 1

            self.edge_idx[i] = choice
            self.s[i] = min(remainder, float(self._lengths[choice]))
            # Nudge the heading onto the new road; the gyro still drives the rest.
            new_bearing = float(self.net.edges[choice].start_bearing)
            self.psi[i] = wrap_angle(psi_i + gain * float(wrap_angle(new_bearing - psi_i)))

    # -------------------------------------------------------------- update

    def positions(self) -> np.ndarray:
        return self.net.positions_fast(self.edge_idx, self.s)

    def road_bearings(self) -> np.ndarray:
        return self.net.bearings_fast(self.edge_idx, self.s)

    def update_weights(
        self,
        gps_xy: Optional[Sequence[float]] = None,
        gps_sigma: Optional[float] = None,
        gps_course_rad: Optional[float] = None,
        gps_speed: Optional[float] = None,
    ) -> None:
        """w~_i = w_i * L_GPS * L_psi * L_v * L_map, then normalise."""
        likelihood = self.map_likelihood.copy()

        # L_psi - particle heading against the bearing of the road it sits on.
        heading_error = np.asarray(wrap_angle(self.psi - self.road_bearings()), dtype=float)
        likelihood *= np.exp(
            -(heading_error**2) / (2 * self.pf.sigma_heading_rad**2)
        )

        # L_v - a particle must not be doing 90 km/h in a courtyard.
        limits = self._speed_limits[self.edge_idx]
        excess = np.maximum(0.0, self.v - limits * 1.4)
        likelihood *= np.exp(-(excess**2) / (2 * self.pf.sigma_speed_ms**2))

        self.last_gps_likelihood = 1.0
        if gps_xy is not None:
            sigma = math.sqrt(
                (gps_sigma if gps_sigma is not None else self.pf.sigma_gps_m) ** 2
                + self.pf.sigma_gps_m**2
            )
            delta = self.positions() - np.asarray(gps_xy, dtype=float)
            d2 = np.einsum("ij,ij->i", delta, delta)
            gps_term = np.exp(-d2 / (2 * sigma**2))
            self.last_gps_likelihood = float(gps_term.max())
            likelihood *= gps_term

            if gps_course_rad is not None and (gps_speed or 0.0) >= self.cfg.gps.min_speed_for_course_ms:
                course_error = np.asarray(wrap_angle(self.psi - gps_course_rad), dtype=float)
                likelihood *= np.exp(-(course_error**2) / (2 * (self.pf.sigma_heading_rad * 1.5) ** 2))

            if gps_speed is not None and gps_speed >= 0:
                likelihood *= np.exp(
                    -((self.v - gps_speed) ** 2) / (2 * self.pf.sigma_gps_speed_ms**2)
                )

        self.w = normalize_weights(self.w * likelihood)
        self.map_likelihood[:] = 1.0

    def zero_velocity_update(self, max_speed_ms: float = 1.5) -> None:
        """The vehicle is provably standing still.

        Particles that believe they are still rolling are down-weighted and then
        stopped. This is what pins down the accelerometer bias without GPS: at a
        red light the true speed is known exactly, so any speed the particle has
        accumulated is integrated bias, and resampling keeps the particles whose
        bias explained it.
        """
        if not self._initialized:
            return
        # Only particles that already agree the car is nearly stopped are
        # affected; one that is convinced it is doing 40 km/h is a different
        # hypothesis, not a bias error, and is left for the weights to judge.
        self.w = normalize_weights(
            self.w * np.exp(-(self.v**2) / (2 * self.pf.zupt_speed_sigma_ms**2))
        )
        self.v[:] = np.where(self.v <= max_speed_ms, 0.0, self.v)

    def apply_likelihood(self, likelihood: np.ndarray) -> None:
        """External evidence (a photo, a parking zone) folded into the weights."""
        self.w = normalize_weights(self.w * np.asarray(likelihood, dtype=float))

    def has_diverged(self) -> bool:
        """True when no particle can explain the current trusted fix."""
        return self.last_gps_likelihood < self.pf.divergence_likelihood

    def inject_from_fix(
        self, xy: Sequence[float], heading: float, speed: float, sigma: float
    ) -> None:
        """Replace the worst particles with fresh ones drawn around a trusted fix.

        Standard rejuvenation. It is what lets the filter climb back out of a
        branch it wrongly committed to during an outage, and it costs nothing
        while the filter is already tracking correctly, because fresh particles
        near the fix are exactly where the good ones already are.
        """
        n = len(self.w)
        count = max(1, int(self.pf.inject_fraction * n))
        # Stable, so that equal weights - which is exactly the state right
        # after a resample - break ties by index in both languages.
        victims = np.argsort(self.w, kind="stable")[:count]
        candidates = self.net.nearest_edges(xy, k=self.pf.init_candidate_edges,
                                            radius=max(sigma * 4, 120.0))
        if not candidates:
            return
        mean_bias_a = float(np.average(self.b_a, weights=self.w))
        mean_bias_w = float(np.average(self.b_w, weights=self.w))
        picks = self.rng.choice(len(candidates), size=count)
        for slot, pick in zip(victims, picks):
            index = candidates[int(pick)]
            s_on, _ = self.net.project(xy, index)
            self.edge_idx[slot] = index
            self.s[slot] = float(np.clip(
                s_on + self.rng.normal(0.0, max(sigma, 5.0)), 0.0, self._lengths[index]
            ))
            self.psi[slot] = wrap_angle(
                self.net.edges[index].bearing(s_on)
                + self.rng.normal(0.0, self.pf.sigma_heading_rad * 0.5)
            )
            self.v[slot] = max(0.0, speed + self.rng.normal(0.0, 1.0))
            self.b_a[slot] = mean_bias_a + self.rng.normal(0.0, 0.05)
            self.b_w[slot] = mean_bias_w + self.rng.normal(0.0, 0.005)
            self.w[slot] = float(np.median(self.w))
        self.w = normalize_weights(self.w)
        self.result.injected += count

    def reinitialize(
        self, xy: Sequence[float], heading: float, speed: float, sigma: float
    ) -> None:
        """Full re-seed after divergence, keeping the learned sensor biases."""
        mean_bias_a = float(np.average(self.b_a, weights=self.w))
        mean_bias_w = float(np.average(self.b_w, weights=self.w))
        self.initialize(
            xy, heading=heading, speed=speed,
            accel_bias=mean_bias_a, gyro_bias=mean_bias_w, position_sigma=sigma,
        )
        self.result.reinitializations += 1

    def maybe_resample(self) -> bool:
        """Resample when N_eff drops below the configured fraction of N."""
        n = len(self.w)
        if effective_sample_size(self.w) >= self.pf.resample_threshold * n:
            return False
        picks = systematic_resample(self.w, self.rng)
        self.edge_idx = self.edge_idx[picks]
        self.s = self.s[picks].copy()
        self.v = self.v[picks].copy()
        self.psi = self.psi[picks].copy()
        self.b_a = self.b_a[picks].copy()
        self.b_w = self.b_w[picks].copy()
        self.map_likelihood = self.map_likelihood[picks].copy()
        self.w = np.full(n, 1.0 / n)
        self.result.resample_count += 1
        return True

    # ---------------------------------------------------------- diagnostics

    def snapshot(
        self,
        t: float,
        gps_state: str = "UNKNOWN",
        gps_accepted: Optional[bool] = None,
        seconds_since_trusted: float = 0.0,
    ) -> ParticleSnapshot:
        snap = ParticleSnapshot(
            t=float(t),
            edge_idx=self.edge_idx.copy().astype(np.int32),
            s=self.s.copy().astype(np.float32),
            weights=self.w.copy().astype(np.float32),
            heading=self.psi.copy().astype(np.float32),
            speed=self.v.copy().astype(np.float32),
            gps_state=gps_state,
            gps_accepted=gps_accepted,
            seconds_since_trusted=float(seconds_since_trusted),
            n_eff=effective_sample_size(self.w),
        )
        self.result.snapshots.append(snap)
        return snap

    @property
    def n_eff(self) -> float:
        return effective_sample_size(self.w)

    def weighted_mean_position(self) -> np.ndarray:
        """Only meaningful when the belief is unimodal - use the branch-aware
        estimate in :mod:`geotrace.polygons` when it may not be."""
        return np.average(self.positions(), axis=0, weights=self.w)
