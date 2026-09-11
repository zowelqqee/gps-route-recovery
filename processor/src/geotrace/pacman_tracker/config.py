"""Tuning surface for the Pacman road tracker.

Everything the algorithm can be tuned by lives here so a benchmark run can
serialise the exact configuration it used. Defaults are physically motivated or
measured against the withheld reference, not fitted to it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from geotrace.pacman_tracker.speed import SpeedConfig
from .attitude import AttitudeConfig
from .intervals import IntervalConfig


@dataclass
class GeometryConfig:
    """How OSM polylines are turned into a heading/curvature profile."""

    sample_ds_m: float = 2.0
    heading_smooth_m: float = 10.0
    """Gaussian smoothing length for heading before differentiating. A car
    takes a street corner over 10-25 m, so a corner OSM stores as one sharp
    vertex is spread over a comparable distance. The integral of curvature over
    the corner - the total turn - is unaffected."""

    junction_smooth_m: float = 9.0
    junction_memory: int = 2
    curvature_sigma_floor: float = 0.002
    curvature_sigma_scale: float = 0.6


@dataclass
class MotionConfigP:
    """The IMU step and the stop detector. Speed estimation lives in
    :class:`~geotrace.pacman_tracker.speed.SpeedConfig`."""

    dt_s: float = 0.1

    zupt_window_s: float = 2.0
    zupt_accel_std_ms2: float = 0.05
    zupt_gyro_std_rads: float = 0.006
    zupt_min_duration_s: float = 1.0


@dataclass
class MatchConfig:
    """Gyro-vs-map scoring. Scoring only - it never touches the speed."""

    gyro_noise_rads: float = 0.02
    model_sigma_rads: float = 0.04
    """Everything the pointwise curvature model does not represent.

    Large, because on this map the model has almost nothing to say: |kappa| on
    the true road has a p90 of 0.00057 rad/m, so at 10 m/s it predicts a yaw
    rate of 0.006 rad/s against a gyro noise of 0.02. A tight sigma here does
    not add discrimination, it adds noise - and on 2026-07-22 it drove the true
    hypothesis to a normalised RMS of 1.47 and 10 pruning strikes even when the
    distance was perfect. The discrimination lives in the junction turns."""
    map_error_correlation_s: float = 2.5
    """A 10 Hz gyro stream looks like ten independent measurements per second.
    It is not: the map's curvature error over a bend, the driver's line through
    it and the mount's misalignment are one error sampled ten times. Counting
    them as independent multiplies the information by tau/dt and the filter
    converges hard onto whatever it believed first."""

    student_dof: float = 4.0
    score_window_s: float = 25.0
    """Forgetting horizon. Evidence older than this stops mattering, so a
    hypothesis penalised by one bad stretch can come back."""

    rms_window_s: float = 20.0

    junction_loglik_floor: float = -1.0
    """Per-step floor in a crossing zone, a contamination likelihood for the
    junction motion missing from edge-centreline curvature. It remains weak
    evidence rather than making an ambiguous hypothesis artificially perfect."""


@dataclass
class BeamConfig:
    """Hypothesis population management."""

    max_hypotheses: int = 6000
    min_hypotheses: int = 40
    prune_log_margin: float = 25.0
    """Keep everything within this many nats of the best. Wide, deliberately:
    the leading edge of the belief is exactly the part that scores slightly
    worse and exactly the part that must survive."""

    prune_patience: int = 12
    """Steps a hypothesis must stay below the margin before removal. One bad
    sample must never be fatal."""

    merge_s_tol_m: float = 8.0
    max_children: int = 6

    crossing_tolerance_sigma: float = 1.0
    crossing_tolerance_min_m: float = 12.0
    crossing_tolerance_max_m: float = 60.0
    """Half-width of the stretch over which a junction crossing is ambiguous.
    Successors are spawned on entering it; the parent survives until clear."""

    map_distance_relative_sigma: float = 0.01
    map_distance_sigma_floor_m: float = 4.0
    map_distance_sigma_max_m: float = 35.0
    """Route-length uncertainty between geometric anchors.

    OSM centreline length and physical tyre-path distance differ by a local
    scale and by junction placement. One percent is the prior standard
    deviation of that scale; the cap prevents old uncertainty from making
    arbitrary neighbouring edges plausible forever.
    """

    offset_min_turn_deg: float = 20.0
    """Only a real turn re-times a crossing; a gentle bend does not localise it."""

    offset_max_correction_m: float = 30.0
    offset_correction_gain: float = 0.5
    """Damped, because the turn's centre and the map's node are not the same
    point either. Corrections accumulate over successive junctions."""

    offset_anchor_sigma_m: float = 8.0
    """Distance-domain noise of locating an OSM node from a turn centroid."""

    turn_model_sigma_deg: float = 9.0
    """How far a driver's line through a junction departs from the map's angle
    between two edges. They cut corners, they swing wide, and the node is not
    where the turn's centre actually is."""


@dataclass
class CorridorConfig:
    k_sigma: float = 2.0
    lateral_buffer_m: float = 6.0
    max_corridors: int = 4
    min_branch_mass: float = 0.05
    confident_mass: float = 0.60
    confident_sigma_s_m: float = 60.0
    low_confidence_sigma_s_m: float = 250.0
    confident_effective_streets: float = 1.6
    """Largest effective number of occupied streets, exp(H) over per-edge mass,
    that still counts as CONFIDENT. A belief spread over twenty streets is not
    confident whatever its favourite one looks like."""


@dataclass
class SinglePathConfig:
    low_confidence_margin_log: float = 1.5
    min_confident_probability: float = 0.75
    rollback_enabled: bool = False
    rollback_max_age_s: float = 180.0
    contradiction_match_margin_s: float = 5.0
    event_settle_s: float = 1.0

    # Bounded DFS backtracking. On an unexplained strong turn (or a dead end)
    # the active route is rewound to a recent decision that has an unused
    # dormant sibling and one branch is re-committed. This is depth-first with
    # a single active branch, never a parallel beam.
    rollback_max_depth: int = 3
    """How many committed decisions back the rewind may reach."""
    rollback_max_alternatives_per_junction: int = 2
    """How many siblings of one junction may be tried before giving up on it."""
    rollback_replay_window_s: float = 120.0
    """Longest span of buffered IMU the rewind is allowed to re-consume."""
    rollback_require_low_confidence: bool = True
    """Default true: only rewind decisions the local score was already unsure
    about. Setting it false lets a decisive unexplained contradiction rewind a
    confident decision too - which is the failure mode the review data shows,
    but the experiment (docs/SINGLE_PATH_FORENSICS.md) is a clean negative:
    without a full buffered-IMU replay it swaps one committed error for
    another and route survival drops."""
    rollback_sibling_direction_match: bool = True
    """Prefer the rewind target whose dormant sibling turns the way the
    unexplained gyro event did."""

    # Event-aligned junction matching. A detected gyro turn is an immutable
    # physical observation; it is matched to a reachable map junction despite
    # odometer error, and its frozen signed angle scores the outgoing edges -
    # instead of re-integrating the gyro around the (lagged) crossing time.
    event_min_angle_deg: float = 25.0
    """Only turns at least this strong steer a junction decision."""
    event_match_k_sigma: float = 2.5
    """sigma_D / sigma_map multiplier in the event-to-junction match tolerance."""
    event_match_drift_allowance_m: float = 40.0
    """Bounded extra tolerance for accumulated odometer drift, on top of sigma."""
    event_match_min_tol_m: float = 20.0
    event_match_max_tol_m: float = 160.0
    """Hard clamp on the match tolerance - it cannot widen without limit."""
    event_max_junction_distance_m: float = 220.0
    """A match may not associate an event with a junction further than this."""
    event_max_age_s: float = 60.0
    """An unmatched strong turn expires after this and becomes a contradiction."""
    event_distance_prior_weight: float = 0.4
    """Weight on the soft distance term. < 1 so distance never selects a branch
    on its own - the turn angle does that, distance only breaks ties."""
    event_offset_gain: float = 0.85
    """Cap on the fraction of the event-junction residual folded into this
    route's offset_bias. The actual gain is Kalman-like in the position
    uncertainty, so a run whose D is already good barely moves."""
    event_offset_anchor_sigma_m: float = 15.0
    """Localization noise of an OSM node from a turn centroid, in the distance
    domain - the measurement noise for the offset_bias correction."""
    event_offset_max_correction_m: float = 160.0
    """Clip on that per-event offset_bias correction."""
    straight_commit_extra_wait_s: float = 2.0
    """Grace period after the odometer reaches an edge end before the junction
    is committed as a no-turn crossing, in case a turn event is still settling."""
    shallow_road_class_downgrade_penalty_log: float = 2.5
    """Penalty per OSM road-class downgrade when the local gyro window is too
    shallow to identify a branch. Real turn events remain angle-driven."""
    compound_connector_max_m: float = 40.0
    """Collapse a very short connector plus its following node into one
    physical manoeuvre for shallow/no-turn decisions. OSM often represents one
    swept car manoeuvre as two opposite node angles; the gyro observes their
    net angle, not either artificial angle in isolation."""
    compound_plan_max_residual_deg: float = 15.0
    """Lock the second edge only for a tightly fitting compound manoeuvre.
    Looser look-ahead may rank the first connector, but the next junction must
    then use its own evidence instead of inheriting a brittle forced choice."""
    terminal_backtrack_enabled: bool = True
    """At a mapped dead end, continue over the reverse twin instead of pinning
    the marker forever. With unsigned IMU speed this is the only observable
    model of reversing out: distance keeps increasing while map position moves
    back along the same road. It is used only when no forward successor exists."""
    blocked_successor_node_pairs: tuple[tuple[int, int], ...] = (
        (12180325731, 12180325730),
        (12180325770, 12180325748),
        # Novgorodskaya ul. / Starorusskaya ul., St. Petersburg: two short
        # (6.8 m, 8.8 m) one-/bi-directional Starorusskaya connectors, one
        # node apart, both geometrically and angularly indistinguishable (by
        # speed, road class or turn kinematics) from a genuine turn out of the
        # one-way Novgorodskaya corridor - the same physical gyro event
        # matches whichever of the two the tracker still reaches. Every real
        # drive through here (2026-09-08 real_tests, RFID-confirmed) instead
        # continues straight down the corridor (921->911->916->949).
    )
    """Manual exception list: OSM (u, v) node pairs - stable across graph
    reloads, unlike internal edge indices - that single_path must never offer
    as a successor, regardless of sensor evidence. Not a general heuristic:
    every entry here is a confirmed-wrong junction from ground truth the
    tracker cannot see in production (e.g. sparse RFID checkpoints), left as a
    manually curated list because no sensor-only signal (speed, road class,
    turn kinematics) distinguished it from a genuine turn - see
    docs/SINGLE_PATH_FORENSICS.md and the 2026-09-08 real_tests review."""

    # Soft turn tier: gentle turns below the strong-event threshold.
    soft_turn_enabled: bool = True
    soft_turn_min_rate_rads: float = 0.035
    soft_turn_min_angle_deg: float = 14.0
    soft_turn_map_turn_min_deg: float = 16.0
    soft_turn_map_turn_max_deg: float = 62.0
    """A soft event may only steer a junction whose best-matching successor
    turns by an angle in this band - a real but moderate corner. Outside it the
    soft event is more likely a lane change or a curved street."""
    soft_turn_sigma_inflation_deg: float = 7.0
    soft_turn_min_probability: float = 0.55
    """The soft event must still pick one successor over the others by at least
    this local probability, or the junction is committed by geometry instead."""
    soft_turn_max_position_residual_m: float = 60.0
    """A gentle event is not authoritative enough to bridge an arbitrarily
    large odometer mismatch. Strong turns retain the wider event matcher, but
    a soft turn must remain local to the junction it is allowed to steer."""
    soft_turn_pair_max_gap_s: float = 1.5
    soft_turn_pair_max_net_angle_deg: float = 8.0
    soft_turn_pair_min_balance_ratio: float = 0.5
    """Adjacent opposite soft events with a small net heading change are one
    S-manoeuvre (lane change or intra-edge weave), not two junction turns."""

    # Delayed local commitment for a low-confidence fork.
    provisional_fork_enabled: bool = True
    provisional_max_alternatives: int = 2
    provisional_min_direction_split_deg: float = 18.0
    """Minimum angular separation between the best two branches for retaining
    the runner-up. Keeps tiny OSM wiggles out while preserving shallow,
    genuinely ambiguous left-vs-right forks."""
    provisional_min_alternative_probability: float = 0.10
    """Smallest runner-up mass retained at an ambiguous fork. Ten percent
    keeps a plausible opposite branch available without retaining remote
    alternatives that the local turn evidence has effectively ruled out."""
    provisional_max_events_to_resolve: int = 2
    """How many subsequent strong turn events a provisional fork may wait for
    before it is force-committed to the greedy choice."""
    provisional_sequence_margin: float = 2.0
    """Log-likelihood margin by which one branch's forced turn sequence must
    beat the other before a provisional fork is switched or confirmed."""

    # Intra-edge road-curvature ("bend") events. A long curving street polyline
    # integrates the same yaw as a junction turn without any node; the junction
    # matcher, forced to bind it somewhere, plants it at the wrong node and
    # shortens the committed map length (07-26 ev3: -740 m). See
    # docs/SPECTRAL_CALIBRATION_FORENSICS.md Phase 17 and `bend_anchor.py`.
    bend_classification_enabled: bool = False
    """Classify a settled gyro event as an intra-edge bend when it is spread and
    low-rate, the committed active edge's own polyline contains a unique
    matching curvature feature (shape match, NO distance prior), and no nearby
    junction offers a compatible outgoing turn. A bend event is then excluded
    from junction matching, straight-commit revision, rollback and interval
    anchoring - it never becomes a fake junction or a fake map interval."""
    bend_event_max_peak_rate_rads: float = 0.16
    bend_event_min_duration_s: float = 6.0
    bend_min_uniqueness_margin: float = 0.7
    bend_max_shape_rms_deg: float = 4.0
    bend_min_rate_corr: float = 0.55
    bend_junction_gap_reject_deg: float = 18.0
    """If a reachable junction offers an outgoing turn within this of the
    measured angle, the event is left to the junction matcher, not called a
    bend."""

    bend_position_anchor_enabled: bool = False
    """Apply an accepted bend's along-edge position as an absolute route-distance
    measurement (through the shared speed / common-mode machinery - the active
    edge and route topology are unchanged). Off by default: needs the
    real-trip ablations in Phase 17 to justify a production default."""
    bend_position_anchor_sigma_m: float = 45.0
    bend_local_speed_anchor_enabled: bool = False
    """Feed an accepted bend's fitted local speed ``v_bar`` (from the
    time->arc-length scale of the shape match) to the spectral scale ``k_s`` as
    a high-speed absolute-speed anchor, blended in by ``v_spectral`` regime.
    Requires ``speed.spectral_scale_enabled``."""
    bend_local_speed_regime_lo_ms: float = 9.0
    bend_local_speed_regime_hi_ms: float = 14.0
    """``v_spectral`` below `lo` keeps the low-speed scale (~1); above `hi` the
    bend-derived high-speed scale applies in full; linear in between. This is
    the only online-available regime indicator - Phase 17 B5/B6 found the
    spectral features themselves do not separate the saturated regime."""

    retro_bend_smoothing_enabled: bool = False
    """OFFLINE post-processing only. When a bend position anchor lands ~450 s
    into the outage it corrects the live distance in one forward jump, but the
    trajectory *before* the bend keeps the full accumulated undershoot - on the
    07-26 review trip that is the ~700 m worst-case along-route error, almost
    all of it pre-bend (Phase 26 TASK 1). This flag makes ``run`` also emit a
    ``retro_speed_trace``: the folded correction from each applied bend anchor
    spread backward over the pre-anchor trajectory in proportion to time spent
    moving, so the reconstructed history joins the already-corrected future with
    no step. It uses only the bend's own map residual - no withheld GPS, no
    future information beyond the anchor it is attached to, no route-topology
    change - and never touches the causal ``speed_trace``. Phase 26 TASK 4:
    07-26 whole-outage max\\|D_err\\| 707 -> 394 m, p90 613 -> 224 m, endpoint
    and 07-22 unchanged. Off by default; it is a reconstruction aid, not a live
    estimate."""


@dataclass
class DisplayConfig:
    """The zero-latency corrected *display* position branch (Phase 32).

    A second longitudinal state ``D_position`` for the on-map vehicle marker
    only. It never feeds back into the tracker: the committed route, junction
    timing, branch weights and ``speed_trace`` are byte-identical whether or not
    this runs. See ``display_position.py`` and docs/DISPLAY_ODOMETRY_SPLIT.md.
    """

    position_branch_enabled: bool = False
    """When False, ``D_position`` ≡ ``D_route`` and the panel behaves exactly as
    before. When True, the frozen Phase 32 iso-binary saturation correction is
    applied to the displayed marker only."""

    correction_gain: float = 1.5
    """Scale the positive saturation residual before integration."""

    max_correction_m: float = 300.0
    """Hard cross-trip safety bound on ``D_position - D_route``. Saturation
    evidence alone does not prove that the route odometer is behind."""

    leave_0726_out: bool = False
    """Use the leave-07-26-out isotonic curve instead of the full-pool one.
    Only for reproducing the Phase 32 diagnostic number on 07-26; the shipped
    default model is the full in-regime pool."""

    model_path: str | None = None
    """Override the frozen-model artifact path (tests / experiments)."""


@dataclass
class PacmanConfig:
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    motion: MotionConfigP = field(default_factory=MotionConfigP)
    speed: SpeedConfig = field(default_factory=SpeedConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    beam: BeamConfig = field(default_factory=BeamConfig)
    corridor: CorridorConfig = field(default_factory=CorridorConfig)
    attitude: AttitudeConfig = field(default_factory=AttitudeConfig)
    intervals: IntervalConfig = field(default_factory=IntervalConfig)
    single_path: SinglePathConfig = field(default_factory=SinglePathConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)

    tracker_mode: str = "beam"
    """``beam``, ``single_path``, or ``single_path_rollback``."""

    seed: int = 42
    leveling_recovery_tau_s: float = 20.0
    """Passed to `geotrace.motion_model.build_imu_stream`. The review recorder
    levels its attitude against measured specific force, which eats roughly half
    of sustained longitudinal acceleration; this undoes it."""

    accel_smooth_window_s: float = 0.0
    init_candidate_edges: int = 8
    init_max_offset_m: float = 45.0
    init_heading_sigma_rad: float = 0.5
    output_dt_s: float = 1.0

    spectral_window_s: float = 2.56
    spectral_hop_s: float = 0.5

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
