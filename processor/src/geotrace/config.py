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

    zupt_max_speed_ms: float = 1.5
    """Filter speed below which a quiet IMU may be treated as a real stop."""

    zupt_gyro_rads: float = 0.03
    zupt_window_s: float = 1.0
    """How long the stillness condition must hold before a ZUPT is applied."""

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
    """Reported accuracy worse than this is never treated as TRUSTED."""

    physical_margin_m: float = 25.0
    """`m` in d_max = v*dt + 0.5*a_max*dt^2 + m. Absorbs GPS noise on both the
    previous and the current fix."""

    mahalanobis_threshold: float = 9.21
    """chi^2 with 2 dof at p = 0.99.

    Applied only while GPS is TRUSTED or SUSPECT, i.e. while the filter is
    actually tracking and its predicted position means something. In LOST and
    RECOVERING the prediction has drifted and gating against it would reject
    exactly the fixes needed to recover, so trust is rebuilt from the mutual
    consistency of several consecutive fixes instead."""

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

    max_distance_to_road_m: float = 60.0
    """Diagnostic graph-coverage threshold. A fix beyond it is reported as
    outside the road prior, but is never rejected solely for that reason:
    incomplete or stale map data must not manufacture a GPS outage."""

    max_median_track_to_road_m: float = 25.0
    """Whole-track check, not a per-fix one: if the *median* distance from the
    recorded track to the nearest road exceeds this, the graph does not cover
    the roads that were driven and the run is warned about.

    A well-matched graph gives a median of a couple of metres; a graph for the
    wrong area gives tens. Reusing max_distance_to_road_m here would be too
    lenient, because that is an outlier threshold for a single fix."""

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

    heading_snap_gain: float = 0.35
    """After a junction the particle heading is pulled towards the new edge
    bearing by this gain; the gyro still drives the rest."""



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
