"""All tunable constants live here.

Nothing in the algorithms hard-codes a threshold; every gate, sigma and
probability limit is read from this configuration so that it can be tuned per
city, per vehicle or per experiment without touching the math.

A configuration can be loaded from / dumped to JSON so a run is reproducible.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

G_TO_MS2 = 9.80665
"""Standard gravity. CoreMotion reports userAcceleration in g."""

EARTH_RADIUS_M = 6371000.0
"""Spherical earth radius used by the simple local tangent-plane model."""


@dataclass
class MotionConfig:
    """Strapdown / dead-reckoning limits."""

    max_gap_s: float = 1.0
    """Largest timestamp gap that may still be integrated. Beyond this the
    propagation is refused: integrating across a long hole in the IMU stream
    produces silently wrong positions."""

    max_accel_ms2: float = 6.0
    """Physically plausible longitudinal acceleration of a passenger car."""

    max_speed_ms: float = 45.0
    """~162 km/h. Anything above is treated as a sensor/holder failure."""

    zupt_accel_ms2: float = 0.25
    """Longitudinal |a| below this, together with low rotation, means the IMU is
    quiet. Quiet is NOT the same as stopped: a car cruising at a constant speed
    on a straight road has almost no acceleration and almost no yaw rate, and
    looks identical to a parked one. A zero-velocity update is therefore only
    applied when the filter's own speed estimate is also low."""

    accel_deadband_ms2: float = 1.0
    """Bias-compensated |a_hat| below this is treated as exactly zero.

    Speed is only observable from acceleration through braking/accelerating
    events - a car holding a constant speed produces zero true longitudinal
    acceleration. Without a deadband, any leftover bias-compensated residual
    is integrated on every step regardless of magnitude, and over a long
    unaided outage even a small one steadily drags the speed estimate down
    (the speed floor at zero then holds it there) even though the car never
    slowed down. Below this bar the model instead coasts at its last known
    speed; a real deliberate manoeuvre (braking, accelerating hard) is well
    above it and is integrated exactly as before.

    This is set well above plain sensor bias on purpose. A real recorded
    outage (trip-b4faeae0) showed a confirmed-steady highway cruise produce a
    sustained apparent deceleration for minutes right after GPS was lost,
    still present (just slower) at 0.35 m/s^2 - too large to be residual
    accelerometer bias alone. The likely compounding cause is heading
    uncertainty: once GPS stops correcting course, drifting yaw leaks part of
    a real lateral/curve acceleration onto the longitudinal axis, and that
    projection error is not bounded by how good the accelerometer itself is.
    1.0 clears the worst case observed so far with real margin, while staying
    below what a deliberate driving manoeuvre registers - but it is an
    empirical margin over one confirmed case, not a derived bound, and may
    need to move again."""

    zupt_max_speed_ms: float = 1.5
    """Filter speed below which a quiet IMU may be treated as a real stop."""

    zupt_gyro_rads: float = 0.03
    zupt_window_s: float = 1.0
    """How long the stillness condition must hold before a ZUPT is applied."""

    shock_accel_ms2: float = 9.0
    """Horizontal-or-vertical user acceleration above this is not a plausible
    passenger-car control input and may be a dropped or dislodged phone."""

    shock_gyro_rads: float = 4.0
    """Angular-rate magnitude that flags a likely mount disturbance rather
    than a normal vehicle yaw manoeuvre. Normal hard vehicle turns are well
    below this; the threshold is deliberately conservative."""

    shock_hold_s: float = 1.0
    """How long an impact disables direct IMU control integration."""

    shock_min_vehicle_speed_ms: float = 2.0
    """A shock is reported as a route-relevant mount disturbance only while
    the estimated vehicle is moving. Handling the phone after parking must not
    flood the route report with irrelevant markers."""

    shock_position_noise_mpsqrt: float = 10.0
    shock_heading_noise_radsqrt: float = 0.35
    """Extra uncertainty injected while a mount-disturbance hold is active.
    The state keeps constant velocity and heading, but is explicitly marked as
    less reliable until GPS or a stable IMU sequence can constrain it again."""

    shock_heading_recovery_s: float = 5.0
    """A shock hard enough to flag (see ``shock_accel_ms2``/``shock_gyro_rads``)
    can leave the phone sitting at a new angle in its mount rather than
    bouncing back - the gyro then keeps reporting a real rotation, just the
    phone's relative to the car, not the car's relative to the road. That is
    indistinguishable from a real, sustained turn using the gyro alone, so
    for this long after the hold ends the yaw rate is discounted (see
    ``shock_heading_gain``) rather than trusted outright, until GPS - which
    resets heading directly - is available again.

    Only 5 s, deliberately conservative: on a real trip with frequent bumps
    (trip-b4faeae0-a941-4a87-9b18-de7aaa84f721, 28 shocks in 35 minutes) a
    longer window keeps discounting real turns that happen to follow a bump
    during an already-untrusted stretch, which cascades into far more GPS
    fixes being rejected than it saves - confirmed by direct comparison
    against real-trip GPS accept/reject counts, not just the one bridge
    case this exists for."""

    shock_heading_gain: float = 0.5
    """Fraction of the measured yaw rate kept during shock_heading_recovery_s.
    Not zero: a genuinely sharp turn right after a bump must still register,
    only damped rather than taken at face value."""

    accel_bias_rw: float = 0.008
    """Random-walk sigma of the accelerometer bias, m/s^2 per sqrt(s).

    Kept small on purpose. The bias is only observable while GPS speed is
    available; if it is allowed to wander fast, whatever was learned before an
    outage is forgotten within seconds of the outage starting and the
    dead-reckoned distance runs away."""

    gyro_bias_rw: float = 0.002
    """Random-walk sigma of the gyro bias, rad/s per sqrt(s)."""

    accel_noise: float = 0.35
    gyro_noise: float = 0.02

    filter_dt_s: float = 0.1
    """The IMU stream (50 Hz on iPhone) is resampled to this step before it is
    fed to the filters. 10 Hz is plenty for vehicle dynamics and keeps the
    particle filter affordable."""


@dataclass
class GPSQualityConfig:
    """GPS trust state machine."""

    max_horizontal_accuracy_m: float = 50.0
    """Fallback sigma when CoreLocation omits an accuracy value.

    A large reported accuracy is not, by itself, evidence of a false position:
    it is represented as a larger measurement covariance in the EKF gate and
    update.  In particular it must not make a continuous real GPS trace turn
    into an outage.
    """

    physical_margin_m: float = 25.0
    """`m` in d_max = v*dt + 0.5*a_max*dt^2 + m. Absorbs GPS noise on both the
    previous and the current fix."""

    mahalanobis_threshold: float = 9.21
    """chi^2 with 2 dof at p = 0.99.

    Applied to every usable fix. During an IMU-only interval its innovation
    covariance is deliberately expanded to represent dead-reckoning drift, so
    a plausible returning fix can recover the filter while a teleport still
    fails the same statistical test."""

    min_speed_for_course_ms: float = 3.0
    """Below this speed CoreLocation course is noise and is ignored."""

    max_course_error_deg: float = 60.0
    max_speed_mismatch_ms: float = 8.0

    recovery_course_error_deg: float = 55.0
    """LOST / RECOVERING: how far the course a fix reports may differ from the
    bearing implied by the previous accepted fix."""

    recovery_speed_mismatch_ms: float = 7.0
    """LOST / RECOVERING: how far a reported speed may differ from the speed
    implied by the previous accepted fix."""

    recovery_min_step_m: float = 8.0
    """Below this fix-to-fix distance the implied bearing is meaningless."""

    recovery_position_sigma_m: float = 30.0
    """Additional one-sigma dead-reckoning uncertainty at the start of a GPS
    recovery interval, in metres."""

    recovery_position_sigma_growth_mps: float = 3.0
    """Additional one-sigma uncertainty growth while GPS is untrusted.

    This is a conservative representation of uncalibrated IMU drift used only
    in the innovation gate. The EKF is re-anchored after consecutive coherent
    fixes restore trust.
    """

    recovery_position_sigma_cap_m: float = 250.0
    """Ceiling on `recovery_position_sigma_m + growth_mps * since`.

    Without a ceiling the innovation gate's tolerance grows without bound the
    longer GPS stays untrusted, so after a long enough outage it will admit a
    fix however far away it lands - a frozen or cell-tower-derived position is
    not made more plausible by the clock running. The car cannot actually be
    reached by dead reckoning past a few hundred metres of honest drift, so
    the gate should not pretend otherwise.
    """

    max_reanchor_sigma_m: float = 100.0
    """A recovering fix may only trigger the *hard* re-anchor (EKF state
    overwrite plus particle-filter reinitialization) when its own measurement
    sigma is at least this good.

    `reanchor` intentionally skips the Kalman gain and commits to the fix
    outright (see `ExtendedKalmanFilter.reanchor`), which is only sound when
    the fix itself is trustworthy. A fix whose own reported accuracy is
    hundreds or thousands of metres wide is not "GPS is back" - it is the
    receiver admitting it does not know where the car is - and forcing it
    through the hard reset discards a perfectly good dead-reckoning track for
    a worse one. Below this bar the fix is still folded in, just through the
    ordinary gain-weighted EKF/particle updates, where a large sigma
    correctly earns it only a small nudge.
    """

    max_median_track_to_road_m: float = 25.0
    """Whole-track check, not a per-fix one: if the *median* distance from the
    recorded track to the nearest road exceeds this, the graph does not cover
    the roads that were driven and the run is warned about.

    A well-matched graph gives a median of a couple of metres; a graph for the
    wrong area gives tens. This is report-only metadata and never participates
    in a per-fix GPS decision."""

    suspect_to_lost_count: int = 3
    """Consecutive rejected fixes that push SUSPECT -> LOST."""

    recover_count: int = 4
    """Consecutive accepted, mutually consistent fixes required to return to
    TRUSTED. This is what filters out the false points that a receiver emits
    right after re-acquiring a signal."""

    lost_gap_s: float = 5.0
    """No fix at all for this long => LOST (dropout)."""

    min_accuracy_sigma_m: float = 5.0
    """Floor for the measurement sigma. Receivers routinely under-report."""

    allow_simulated_fixes: bool = False
    """Whether to accept fixes CoreLocation flagged as simulated by software.

    False in normal use: a spoofed location is not evidence about where a car
    was. Set it (via --allow-simulated) to run the pipeline on a trip recorded
    in the iOS Simulator, where every fix carries the flag."""

    accuracy_sigma_scale: float = 1.0


@dataclass
class ParticleFilterConfig:
    n_particles: int = 5000
    resample_threshold: float = 0.5
    """Resample when N_eff < threshold * N."""

    init_radius_m: float = 40.0
    init_candidate_edges: int = 6
    """How many nearby drivable edges seed the initial particle cloud."""

    sigma_s: float = 0.15
    """Process noise on distance along the edge, metres per filter step.

    These three are per-step and therefore accumulate as a random walk over an
    outage: sigma * sqrt(steps). At 10 Hz a 45 s outage is 450 steps, so a
    seemingly innocent 0.3 m/s speed noise would diffuse the speed by 6 m/s and
    put the along-track estimate hundreds of metres out. They are set to match
    the actual sensor noise, not to "add some spread"."""

    sigma_v: float = 0.05
    """m/s per step. Roughly accel_noise * filter_dt."""

    sigma_psi_rad: float = 0.008
    """rad per step. Roughly gyro_noise * filter_dt."""

    sigma_gps_m: float = 12.0
    """Base GPS likelihood sigma; combined with the reported accuracy."""

    sigma_heading_rad: float = 0.60
    """Heading likelihood: particle heading vs. edge bearing."""

    sigma_turn_rad: float = 0.70
    """Turn model at a junction: exp(-wrap(theta_out - psi)^2 / 2 sigma^2)."""

    sigma_speed_ms: float = 4.0
    """Speed likelihood against the edge speed limit - deliberately loose, it
    only has to catch a particle doing 90 km/h through a courtyard."""

    sigma_gps_speed_ms: float = 1.5
    """Speed likelihood against the GPS Doppler speed. This one is tight: it is
    the only measurement that makes the per-particle accelerometer bias
    observable, and therefore the only thing that keeps along-track drift under
    control once GPS disappears."""

    zupt_speed_sigma_ms: float = 0.3
    """Zero-velocity update: how much residual speed a particle may claim while
    the vehicle is provably standing still."""

    allow_uturn: bool = False
    uturn_max_speed_ms: float = 1.5
    """A U-turn onto the reverse of the current edge is only ever considered
    below this speed (i.e. the car actually stopped)."""

    dead_end_weight: float = 1e-9
    """Weight given to a particle that ran into a dead end."""

    inject_fraction: float = 0.02
    """While GPS is TRUSTED, this fraction of the lowest-weight particles is
    replaced by fresh ones drawn around the fix. Without it a filter that has
    committed to the wrong branch during an outage can never recover: every
    particle is far from the returning fixes, all likelihoods underflow to zero
    and the normalisation falls back to uniform on a cloud that is entirely
    wrong."""

    divergence_likelihood: float = 1e-6
    """If the best particle's GPS likelihood falls below this while GPS is
    TRUSTED, the filter has diverged and is re-seeded from the fix."""

    reinit_route_continuity_hops: int = 8
    """When re-seeding after divergence, an edge reachable within this many
    graph hops of where the cloud already was scores normally; every other
    edge is scored down by `reinit_disconnected_penalty`.

    Divergence's own candidate search is purely geometric (nearest edge to
    the fix, weighted by heading), which has no way to prefer "the street the
    car was already driving on" over "some other street that happens to be
    just as close" - exactly the failure mode near parallel embankments either
    side of a narrow river, where re-seeding can lock onto the wrong bank and
    never recover, because every subsequent divergence check re-runs the same
    geometry-only search and finds the same wrong edge again."""

    reinit_disconnected_penalty: float = 0.15
    """Score multiplier for a re-seed candidate that is not within
    `reinit_route_continuity_hops` of the pre-divergence cloud. A penalty, not
    a hard filter: a real route change (a U-turn, backtracking, a long enough
    outage that the car could plausibly be anywhere) must still be reachable,
    just less preferred than continuing on the connected road."""

    heading_snap_gain: float = 0.35
    """After a junction the particle heading is pulled towards the new edge
    bearing by this gain; the gyro still drives the rest."""

    outage_map_assist_min_probability: float = 0.85
    """Minimum posterior mass of one road hypothesis before the map may softly
    assist the displayed IMU route during a GPS outage."""

    outage_map_assist_max_spread_m: float = 45.0
    """The dominant graph hypothesis must remain spatially compact.  A single
    connected OSM component can still contain several junction choices, so its
    probability alone is not sufficient evidence."""

    outage_map_assist_max_area_m2: float = 35_000.0
    """Largest 95% road-corridor footprint that may be presented as one map
    hypothesis during an outage.  Above this, connected road buffers can join
    several turns into one visually misleading component."""

    outage_map_assist_max_offset_m: float = 120.0
    """Never let a map hypothesis pull the EKF estimate across the city, even
    when the graph itself happens to be topologically unambiguous."""

    outage_map_assist_gain: float = 0.45
    """Fraction of the validated map correction blended into the IMU estimate.
    The road graph is a soft prior, not an independent position measurement."""

    outage_heading_assist_min_resultant: float = 0.9
    """Position and heading are corrected independently during an outage.
    `_outage_map_assist` above requires one compact, confident branch before
    it will touch position at all - a real river-confluence interchange with
    several plausible streets fails that outright. But *direction* is a much
    weaker claim than *which branch*: several genuinely different streets can
    still all run the same way away from the fork itself. `heading_consensus`
    (see RoadParticleFilter) measures exactly that agreement as a mean
    resultant length in [0, 1]; only this concentrated does the road's own
    bearing correct the EKF's heading (see outage_heading_assist_gain),
    regardless of whether position could be corrected at all.

    0.9, not lower: confirmed empirically against
    test_the_polygons_cover_the_true_position (the fork-junction scenario) -
    a mid-range resultant (~0.8) shows up exactly while the cloud is still
    genuinely deciding between two branches, and nudging heading then biases
    that decision. Above 0.9 the cliff disappears; 0.9-0.95 score equally
    well, so 0.9 is kept for more real-trip coverage."""

    outage_heading_assist_max_gap_rad: float = 0.6
    """~35 degrees. The consensus bearing must already be this close to the
    EKF's own independent heading before it may nudge it at all - agreement
    within the particle cloud is not evidence the cloud itself is right, only
    that it agrees with itself; after a long enough outage it is a causal,
    never retrospectively corrected prior, and can be confidently on the
    wrong street entirely. This keeps the correction to a small, plausible
    drift, never a wholesale redirection - confirmed necessary against
    trip-b4faeae0-a941-4a87-9b18-de7aaa84f721 (a 373 s outage where, without
    this gap check, a confidently-agreeing but wrong branch cost 134 GPS
    fixes once trust returned)."""

    outage_heading_assist_gain: float = 0.35
    """Fraction of the gap to the road's own bearing closed per output tick
    when outage_heading_assist_min_resultant is met. This only rotates the
    EKF's heading mean directly (`EKF.nudge_heading`) - deliberately not an
    ordinary Kalman update: `heading_consensus` carries no position
    information of its own, so pulling position along via P's cross-terms
    (as a real course measurement legitimately would) is not justified and
    measurably hurts a genuinely undecided branch split (see
    test_the_polygons_cover_the_true_position)."""



@dataclass
class PolygonConfig:
    confidence: float = 0.95
    """gamma: probability mass the polygon set must contain."""

    r_min_m: float = 8.0
    """Half a carriageway. A particle sits on the centreline, the car does not."""

    k_sigma: float = 2.0
    cross_track_sigma_base_m: float = 4.0
    cross_track_sigma_per_s: float = 0.15
    """Cross-track uncertainty grows while GPS is unavailable."""

    max_radius_m: float = 60.0
    simplify_tolerance_m: float = 1.0
    min_component_probability: float = 0.01
    """Components below this are dropped from the output."""

    off_road_distance_m: float = 60.0
    """A trusted GPS fix farther than this from every known graph edge means
    the car is somewhere the road graph has no edge for at all - a courtyard,
    a private drive, a car park, a coverage gap - not a mismatch between two
    nearby streets. The particle filter still snaps onto whichever real edge
    happens to be nearest, which can be hundreds of metres away and is then
    not a 95%-confident corridor but a guess the GPS itself contradicts. Past
    this distance the output falls back to an honest GPS-accuracy disc around
    the fix instead of that corridor."""


@dataclass
class ParkingConfig:
    selected_probability: float = 0.98
    margin_to_second: float = 0.20


@dataclass
class ParkingTrackerConfig:
    """Free-space terminal manoeuvre estimator; intentionally separate from PF."""

    window_s: float = 60.0
    terminal_cluster_window_s: float = 30.0
    low_speed_ms: float = 3.0
    min_cluster_fixes: int = 3
    max_accuracy_m: float = 80.0
    physical_margin_m: float = 25.0
    process_position_sigma_m: float = 3.0
    uncertain_radius_m: float = 55.0


@dataclass
class PhotoConfig:
    min_ocr_confidence: float = 0.30
    address_boost_radius_m: float = 80.0
    address_boost_alpha: float = 2.0
    """Exponent applied to the image likelihood, `alpha` in the weight update."""

    ocr_boost_beta: float = 1.0
    vpr_temperature: float = 0.07


@dataclass
class Config:
    motion: MotionConfig = field(default_factory=MotionConfig)
    gps: GPSQualityConfig = field(default_factory=GPSQualityConfig)
    pf: ParticleFilterConfig = field(default_factory=ParticleFilterConfig)
    polygon: PolygonConfig = field(default_factory=PolygonConfig)
    parking: ParkingConfig = field(default_factory=ParkingConfig)
    parking_tracker: ParkingTrackerConfig = field(default_factory=ParkingTrackerConfig)
    photo: PhotoConfig = field(default_factory=PhotoConfig)
    seed: int = 42

    rng_mode: str = "numpy"
    """Which random generator the particle filter draws from.

    "numpy" is the default and is what the baseline has always used. "parity"
    switches to `geotrace.parity_rng.ParityGenerator`, whose bit stream and
    derived distributions are reimplemented exactly in Swift, so a run can be
    compared against the on-device implementation draw for draw instead of only
    through the geometry it happens to produce."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def dump(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            value = data[f.name]
            if is_dataclass(f.type) or isinstance(value, dict):
                sub = {
                    "motion": MotionConfig,
                    "gps": GPSQualityConfig,
                    "pf": ParticleFilterConfig,
                    "polygon": PolygonConfig,
                    "parking": ParkingConfig,
                    "parking_tracker": ParkingTrackerConfig,
                    "photo": PhotoConfig,
                }.get(f.name)
                kwargs[f.name] = sub(**value) if sub else value
            else:
                kwargs[f.name] = value
        return cls(**kwargs)

    @classmethod
    def load(cls, path: Path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
