"""One speed and one distance for the whole system.

There is one car. It cannot be doing 7 m/s because it is on Nevsky and 13 m/s
because it is on the embankment, and the previous design let exactly that
happen: every hypothesis carried its own ``v``, every hypothesis pulled that
``v`` towards whatever its own road's curvature implied, and the map ended up
deciding how fast the car was going. Speed lives here instead, estimated once
from the IMU, and the road hypotheses consume it.

State::

    x = [ D, v, b_a, D_anchor, k_a ]

    D    distance travelled since the outage began, metres
    v    vehicle speed, m/s (signed; reversing is real, if rare)
    b_a  longitudinal accelerometer bias, m/s^2
    D_anchor  frozen distance at the start of an accepted map interval
    k_a  longitudinal accelerometer scale (disabled unless explicitly ablated)

The gyro bias is also global - it belongs to the sensor, not to a road - and is
estimated separately at stops, where the true yaw rate is known to be zero.

Five sources can feed it, in descending order of authority:

===================  ==========================================================
stop                 ``v = 0``. The strongest anchor there is, and it also
                     re-measures ``b_a``. Proposed by IMU variance and vetoed
                     when the independent spectral source still sees motion.
``a_lat / omega``    Circular motion. Accurate to a few tenths of a m/s in a
                     real turn, needs no map and no integration. About 8 % of
                     samples on the review recordings.
spectral model       Speed regressed on IMU band powers. Mediocre but
                     continuous - it is what holds the estimate together
                     between turns.
longitudinal accel   Propagation *between* anchors only. On this recorder its
                     zero point moves 0.65 m/s^2 between parked and driving, so
                     it is never allowed to set an absolute speed.
map interval         Optional integrated ``D_B - D_A = L_map`` observation,
                     gated by independent route agreement. Off by default.
===================  ==========================================================

What is deliberately absent: pointwise map-derived speed. ``v = omega / kappa``
is not used here.
On the review recordings usable curvature exists on 1.5 % of steps and is
wrong by a median of -5.8 m/s where it does, and letting it set the speed is
what closed the loop between "which road do I think I am on" and "how fast do I
think I am going".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

D, V, BA, DA, KA = 0, 1, 2, 3, 4
SPEED_DIM = 5
"""``DA`` is the distance frozen when an interval constraint opened. It has
no dynamics and no process noise, so ``D - DA`` accumulates exactly the
uncertainty the filter has genuinely added since the anchor - which is what
makes an integrated-distance measurement a linear update rather than an
approximation.

``KA`` is the accelerometer scale in ``a_true = k_a * (a_measured - b_a)``. It
is a separate state from ``b_a`` because the two are not interchangeable: an
additive bias puts a velocity error proportional to *elapsed time* into the
estimate, a scale error one proportional to *distance travelled*. On these
recordings the measured longitudinal acceleration reproduces only 0.79 (07-22)
and 0.66 (07-26) of the true 10 s velocity change, and the distance shortfall
tracks that, so no value of ``b_a`` can absorb it. ``k_a`` is unobservable from
the IMU alone and is moved only by the interval constraints."""


@dataclass
class SpeedConfig:
    accel_noise_ms2: float = 0.45
    accel_bias_rw: float = 0.12
    """m/s^2 per sqrt(s). Large because the quantity is not a bias: it is the
    attitude filter's tilt error, and it moves 0.65 m/s^2 between parked and
    driving on this recorder."""

    initial_v_sigma_ms: float = 1.0
    initial_accel_bias_sigma: float = 0.35
    initial_d_sigma_m: float = 5.0

    v_sigma_floor_ms: float = 0.25
    accel_bias_sigma_floor: float = 0.05
    d_sigma_floor_m: float = 5.0

    max_speed_ms: float = 33.0
    reverse_speed_limit_ms: float = 1.5
    envelope_sigma_ms: float = 2.5
    envelope_correlation_s: float = 1.0
    """The rails describe a standing condition, not a stream of observations.
    Applied at their face sigma ten times a second they become the strongest
    thing in the filter: a speed limit the car is legitimately exceeding drags
    the estimate down through it, past zero, and into the reversing rail on the
    other side. Inflating by tau/dt makes them a nudge on a one-second
    timescale, which is what a plausibility bound should be."""

    # ---- stop anchor ----
    zupt_speed_sigma_ms: float = 0.3
    zupt_bias_sigma_ms2: float = 0.6

    zupt_speed_correlation_s: float = 4.0
    """A stop is one observation however long it lasts, and what is uncertain
    about it is the *detector*, whose mistakes last as long as the quiet patch
    that caused them. Applying ``v = 0`` at 10 Hz with its raw sigma lets a
    one-second false positive - a smooth stretch of cruising, which is what a
    variance detector cannot tell from idling - overwhelm everything else and
    pin the speed to zero. Inflating by tau/dt leaves a real 30 s stop
    overwhelming (300 steps of it) and a 1 s false one merely a nudge."""

    zupt_gyro_correlation_s: float = 1.0
    """The gyro half needs almost no inflation: the true yaw rate of a parked
    car really is zero, so the only error is the gyro's own white noise, and
    that does average down. This is the one measurement in the whole filter
    that is exactly what it claims to be."""

    zupt_max_decel_ms2: float = 2.5
    """How much speed a *quiet* stretch can plausibly have shed, m/s^2.

    Not the car's braking limit - the braking limit is irrelevant here, because
    a stretch the detector called quiet is by definition one in which the
    accelerometer reported nothing. Any real braking happened *before* the run
    began and has already been integrated. What is left is what the
    accelerometer could be hiding inside its own zero-point error, which on
    this recorder is around 0.65 m/s^2; 2.5 is a generous multiple of it.

    ``v = 0`` at sigma 0.3 m/s is the strongest claim in this filter, and a
    variance detector cannot tell a smooth stretch of cruising from idling - so
    something has to decide whether a detected stop is credible before it is
    believed. Dynamics decide it: a car cannot be stopped now if it was doing
    12 m/s one second ago, because that is 12 m/s^2 of braking and the
    accelerometer did not report it. A stop is accepted in full once the quiet
    stretch has lasted long enough for the current speed estimate to have been
    braked away (``|v| <= a_max * run``), and discounted smoothly before that.
    A real stop is preceded by deceleration and passes immediately; a false one
    has to wait, by which time the detector has usually let go."""

    zupt_bias_correlation_s: float = 10.0
    """The bias half is worse still: what is measured while parked is the
    *parked* offset, and on this recorder that is 0.65 m/s^2 away from the
    offset while driving."""

    zupt_motion_guard_enabled: bool = True
    zupt_motion_guard_spectral_min_ms: float = 5.0
    """Reject an IMU-variance stop claim while the independent spectral
    source still reports clear road speed. A variance detector alone cannot
    distinguish a parked car from smooth constant-speed cruising."""

    # ---- lateral anchor ----
    lateral_enabled: bool = True
    lateral_accel_noise_ms2: float = 0.30
    lateral_gyro_noise_rads: float = 0.02
    lateral_smooth_s: float = 1.0
    accel_scale_enabled: bool = False
    """Estimate ``k_a`` in ``a_true = k_a * (a_measured - b_a)``.

    Off by default: it is unobservable without an absolute distance reference,
    and a free scale with nothing to pin it is a licence to drift. It only
    becomes identifiable once the interval constraints are supplying map
    distance, so the two belong in the same ablation step."""

    accel_scale0: float = 1.0
    initial_accel_scale_sigma: float = 0.25
    accel_scale_rw: float = 0.002
    """Random walk, per root-second. Deliberately tiny: the mount's scale error
    is a property of the hardware, not something that changes minute to
    minute. It is here so a converged estimate can still be revised, not so it
    can chase noise."""

    accel_scale_min: float = 0.6
    accel_scale_max: float = 2.2
    """Physical bounds. Outside this the explanation is a broken axis or a
    wrong forward vector, not a calibration constant."""

    accel_scale_sigma_floor: float = 0.02

    fixed_lag_s: float = 0.0
    """Bounded online smoothing horizon. Zero keeps the production baseline
    purely forward. Interval anchoring still retains twelve seconds of filter
    history so a turn reported after its integration window can be attached to
    the state at the turn rather than to the later state that noticed it."""

    lateral_accel_bias_ms2: float = 0.07
    """Lateral accelerometer offset that does NOT average down, m/s^2.

    The variance below divides the accelerometer noise by sqrt(n) for the
    smoothing window, which is right for white noise and wrong for an offset.
    There is an offset: measured on the review recordings, a_lat while the car
    is provably parked has a median of +0.005 (2026-07-22) and -0.036
    (2026-07-26) m/s^2, and the same quantity measured while driving straight
    disagrees with those - so it is neither zero nor constant, and it cannot be
    calibrated away from one regime and used in another.

    It matters because the measurement divides by omega. Fitting
    a_lat = k*(v*omega) + c over the accepted anchors gives c = -0.069 m/s^2 on
    2026-07-26, which at the median anchor |omega| of 0.028 rad/s is -2.5 m/s of
    speed error - and indeed the measured bias in that bin is -2.7 m/s. Carrying
    the offset in sigma rather than sqrt-averaging it away makes the 1/omega
    amplification de-weight exactly those anchors.
    """

    lateral_model_sigma_ms: float = 1.0
    lateral_min_omega_rads: float = 0.02
    """Not a quality gate - the variance below is the quality gate. This only
    keeps the arithmetic finite."""

    lateral_max_sigma_ms: float = 12.0
    """Beyond this the measurement is not worth applying at all."""

    lateral_correlation_s: float = 1.0

    lateral_consensus_guard_enabled: bool = True
    lateral_consensus_deadband_ms: float = 4.0
    lateral_consensus_scale_ms: float = 2.0
    lateral_consensus_max_factor: float = 25.0
    """Reduce a downward lateral anchor's authority when both the pre-update
    state and the independent spectral source say the car is faster. The
    variance multiplier grows smoothly; the anchor is never hard-rejected."""

    lateral_delayed_correction_enabled: bool = False
    """Phase 35 diagnostic, OFF by default. ``a_lat``/``yaw_rate_smooth`` are a
    1 s CENTERED box-smooth of the raw channel (``_smooth_to_steps`` in
    tracker.py: symmetric padding, +/- window/2 raw samples around each step),
    so by construction the anchor's own timestamp already equals the current
    step's time to within one raw sample (~1/50 s) - there is no built-in
    multi-tick delay to correct. This flag exists to let that be checked
    empirically rather than assumed: when on, the lateral measurement is
    applied at ``t - lateral_anchor_delay_s`` via the SAME bounded
    fixed-lag-smoother machinery ``apply_interval`` already uses for delayed
    turn-to-turn map constraints (``remember`` / ``_history_at`` /
    ``_smooth_delayed_history``, exact joint-covariance propagation, not a
    ``D += dv*dt`` patch), then the correction is propagated forward through
    every retained intervening state - never a raw ``D`` rewrite. With
    ``lateral_anchor_delay_s == 0`` this reduces to today's undelayed update
    exactly (the history lookup lands on the current tick itself, so no replay
    happens) - a built-in no-op correctness check for the plumbing itself."""

    lateral_anchor_delay_s: float = 0.5
    """Assumed physical age, in seconds, of a lateral-anchor measurement when
    ``lateral_delayed_correction_enabled``. Only meaningful test candidate:
    half of ``lateral_smooth_s`` (the smoothing window's own half-width) - the
    most a genuinely mistimed centered-window anchor could be off by. 0.0
    exercises the same code path as a pure no-op check."""

    # ---- spectral model ----
    spectral_enabled: bool = True
    spectral_correlation_s: float = 2.5
    spectral_sigma_scale: float = 1.3

    spectral_saturation_guard_enabled: bool = True
    spectral_guard_min_raw_ms: float = 11.0
    spectral_guard_max_raw_ms: float = 16.0
    spectral_guard_deadband_ms: float = 1.5
    spectral_guard_min_accel_ms2: float = -0.25
    spectral_guard_factor: float = 8.0
    """Causal fallback before a bend can prove saturation. A reading in the
    measured plateau band is down-weighted only when it pulls speed downward
    and the longitudinal channel does not report braking. Upward updates and
    real deceleration retain their normal authority."""

    # ---- spectral absolute-speed calibration k_s ----
    spectral_scale_enabled: bool = False
    """Estimate a scalar ``k_s`` in ``v_true ~ k_s * v_spectral``, calibrated
    online during the outage from lateral (``v = a_lat/omega``) anchors and
    accepted turn-to-turn map intervals only. Off by default: on the two review
    trips the anchors that exist do not sample the high-speed cruising regime
    where the spectral model saturates, so ``k_s`` is not well constrained
    where it matters (docs/SPECTRAL_CALIBRATION_FORENSICS.md). It is a separate
    state from ``k_a`` - accelerometer scale - and never absorbs the other's
    error."""
    spectral_scale0: float = 1.0
    initial_spectral_scale_sigma: float = 0.15
    spectral_scale_rw: float = 0.004
    """Per root-second. The spectral bias is a property of the vehicle/mount,
    not something that changes minute to minute; this is here so a converged
    ``k_s`` can still be revised by a later anchor, not so it chases noise."""
    spectral_scale_min: float = 0.7
    spectral_scale_max: float = 1.7
    spectral_scale_sigma_floor: float = 0.03
    spectral_scale_anchor_min_speed_ms: float = 5.0
    """A calibrating anchor must predict a spectral speed at least this high -
    below it the model is near its floor and the ratio ``v_anchor/v_spectral``
    is dominated by the floor, not by the scale."""
    spectral_scale_anchor_model_frac: float = 0.15
    """Fractional model uncertainty of a lateral anchor as a scale reference
    (tyre slip, body roll take a heading-dependent share of a_lat)."""
    spectral_scale_max_innovation_sigma: float = 3.5

    # ---- high-speed spectral scale from road-curvature ("bend") anchors ----
    bend_scale_enabled: bool = False
    """A SECOND spectral scale ``k_high`` for the saturated high-speed regime,
    calibrated only by road-curvature bend anchors (`bend_anchor.py`): a bend's
    fitted local speed ``v_bar`` divided by the concurrent spectral speed is a
    direct high-speed ``v_true / v_spectral`` measurement. ``spectral_update``
    then blends ``k_low`` (the existing ``spectral_scale``, ~1 from mid-speed
    lateral anchors) and ``k_high`` by a ``v_spectral`` regime weight. Off by
    default: Phase 17 review-trip ablations justify the mechanism but not yet a
    production default. Separate state - a bend anchor never moves ``k_low``,
    ``k_a`` or ``k_s``."""
    bend_scale0: float = 1.0
    initial_bend_scale_sigma: float = 0.30
    """Wide: nothing constrains the high-speed regime until a bend anchor does,
    and it must stay wide on a trip that has no bend (07-22)."""
    bend_scale_rw: float = 0.002
    bend_scale_min: float = 0.7
    bend_scale_max: float = 1.7
    bend_scale_sigma_floor: float = 0.04
    bend_scale_regime_lo_ms: float = 8.0
    bend_scale_regime_hi_ms: float = 12.0
    """``v_spectral`` below `lo` uses ``k_low`` only; above `hi` uses ``k_high``
    in full; linear between. The spectral output is the only online speed signal
    that separates the regimes - Phase 17 B5/B6 found the raw features do not -
    and it saturates near 13 m/s on the trip that needs the correction, so the
    high regime has to trigger a little below that."""
    bend_scale_max_innovation_sigma: float = 3.5
    bend_scale_anchor_model_frac: float = 0.06
    """A bend's ``v_bar`` comes from time->arc-length alignment of the gyro
    against the map polyline; the residual model error is small (~0.8 deg RMS on
    ev3) compared with a lateral anchor's body-roll share."""

    # ---- down-censor the spectral source once its plateau is known ----
    spectral_censor_enabled: bool = False
    """Once a bend speed anchor proves the spectral source under-reads at high
    speed (``v_bar`` clearly exceeds the concurrent reading), record that this
    source *has* a plateau (``spectral_saturation_known``, permanent - it is a
    property of the trip's model, not of the car's current speed). From then
    on, a reading near the plateau that would drag ``v`` *down* is heavily
    down-weighted: the model reads ~13 m/s whether the car is at 14 or 22, so a
    downward pull there is uninformative. Readings above the plateau are the
    source's right-skewed noise, whose upper tail *does* track true speed
    (Phase 0: ``v_spectral`` 13-14 -> ``v_true`` p90 ~20), so they update
    normally and legitimately lift ``v``. Phase 23-25: this takes 07-26
    ``D/D_true`` 0.927 -> ~0.98 and the speed bias -1.3 -> +0.3; without
    ``k_high`` it is ~0.96 with bias -0.05. Inert on a trip with no bend anchor
    (07-22 stays 0.950 exactly, zero censored updates). Off by default; the
    trigger needs only ``single_path.bend_local_speed_anchor_enabled``, not
    ``bend_scale_enabled``. (Phase 25 tested the strict ``z = min(v, plateau)``
    form that also forbids the upward lift - it caps ``v`` at the plateau and
    loses ~6 points of ``D/D_true``; the noise asymmetry is worth keeping.)"""
    spectral_censor_factor: float = 15.0
    """Variance inflation for a downward spectral pull near the plateau while
    saturation is known. 07-26 ``D/D_true``: 8 -> 0.971, 15 -> ~0.98, 50 ->
    0.988 (the endpoint keeps improving but the mid-trip swing grows)."""
    spectral_censor_band_ms: float = 1.5
    """A reading more than this above the plateau is a real higher speed and
    updates in full (band 1 -> 0.988, band 6 -> 0.972 on 07-26)."""
    spectral_censor_deadband_ms: float = 0.5
    """Only down-weight a downward pull larger than this."""
    spectral_saturation_margin_ms: float = 1.0
    """A bend confirms the plateau only when ``v_bar`` exceeds the concurrent
    spectral reading by at least this."""
    """A bend anchor confirms saturation only when ``v_bend`` exceeds the
    concurrent spectral reading by at least this."""

    # ---- gyro bias ----
    gyro_bias_rw: float = 0.0004
    initial_gyro_bias_sigma: float = 0.004
    gyro_bias_sigma_floor: float = 0.0005


@dataclass
class SpeedSample:
    """What the speed tracker reports at one instant."""

    t: float
    distance_m: float
    sigma_distance_m: float
    speed_ms: float
    sigma_speed_ms: float
    accel_bias_ms2: float
    gyro_bias_rads: float
    accel_scale: float = 1.0
    sigma_accel_scale: float = 0.0
    spectral_scale: float = 1.0
    sigma_spectral_scale: float = 0.0
    bend_scale: float = 1.0
    sigma_bend_scale: float = 0.0
    source: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "t": round(self.t, 2),
            "distance_m": round(self.distance_m, 2),
            "sigma_distance_m": round(self.sigma_distance_m, 2),
            "speed_ms": round(self.speed_ms, 3),
            "sigma_speed_ms": round(self.sigma_speed_ms, 3),
            "accel_bias_ms2": round(self.accel_bias_ms2, 4),
            "gyro_bias_rads": round(self.gyro_bias_rads, 6),
            "accel_scale": round(self.accel_scale, 5),
            "sigma_accel_scale": round(self.sigma_accel_scale, 5),
            "spectral_scale": round(self.spectral_scale, 5),
            "sigma_spectral_scale": round(self.sigma_spectral_scale, 5),
            "bend_scale": round(self.bend_scale, 5),
            "sigma_bend_scale": round(self.sigma_bend_scale, 5),
            "source": self.source,
        }


@dataclass
class _HistoryState:
    """A recent filtered state and its covariance with the live state.

    The cross covariance is enough to apply a delayed scalar measurement at a
    turn time to the current filter exactly.  Only a bounded history is kept.
    The pairwise cross-covariances make storage quadratic in the retained
    sample count: the normal 10 Hz, 30 s case is about 9 MB of numeric arrays
    (plus small Python-object overhead), and cannot grow with trip duration.
    """

    t: float
    x: np.ndarray
    P: np.ndarray
    cross: np.ndarray
    past_crosses: list[np.ndarray] = field(default_factory=list)
    samples: list[SpeedSample] = field(default_factory=list)


class GlobalSpeedTracker:
    """The single speed/distance filter for a trip."""

    def __init__(self, cfg: Optional[SpeedConfig] = None, v0: float = 0.0,
                 gyro_bias0: float = 0.0, accel_bias0: float = 0.0) -> None:
        self.cfg = cfg or SpeedConfig()
        self.x = np.array([0.0, float(v0), float(accel_bias0), 0.0,
                           float(self.cfg.accel_scale0)])
        self.P = np.diag([
            self.cfg.initial_d_sigma_m**2,
            self.cfg.initial_v_sigma_ms**2,
            self.cfg.initial_accel_bias_sigma**2,
            self.cfg.initial_d_sigma_m**2,
            (self.cfg.initial_accel_scale_sigma**2
             if self.cfg.accel_scale_enabled else 0.0),
        ])
        self.anchor_open = False
        self.anchor_t = float("nan")
        self.gyro_bias = float(gyro_bias0)
        self.gyro_bias_var = self.cfg.initial_gyro_bias_sigma**2
        # Spectral absolute-speed calibration - a scalar side state, not in x,
        # so adding it changes no EKF dimension. Its variance grows with a slow
        # random walk and is reduced only by the two absolute-speed anchors.
        self.spectral_scale = float(self.cfg.spectral_scale0)
        self.spectral_scale_var = (self.cfg.initial_spectral_scale_sigma**2
                                   if self.cfg.spectral_scale_enabled else 0.0)
        self.spectral_scale_updates = 0
        self.spectral_scale_trace: list[tuple[float, float, float, str]] = []
        # k_high: a second spectral scale for the saturated high-speed regime,
        # moved only by road-curvature bend anchors. Its variance never shrinks
        # without one - a trip with no bend keeps the wide prior.
        self.bend_scale = float(self.cfg.bend_scale0)
        self.bend_scale_var = (self.cfg.initial_bend_scale_sigma**2
                               if self.cfg.bend_scale_enabled else 0.0)
        self.bend_scale_updates = 0
        self.bend_scale_trace: list[tuple[float, float, float]] = []
        # Set true once a bend anchor proves the spectral source under-reads at
        # high speed; from then on a downward spectral pull inside the plateau
        # band is censored (see spectral_update / spectral_censor_enabled).
        self.spectral_saturation_known = False
        self.spectral_plateau_ms = 0.0
        self.spectral_censored_updates = 0
        self.last_longitudinal_accel = 0.0
        self.counts = {"predict": 0, "zupt": 0, "lateral": 0, "spectral": 0,
                       "envelope": 0, "lateral_rejected": 0,
                       "zupt_motion_rejected": 0,
                       "lateral_consensus_downweighted": 0,
                       "spectral_guarded": 0,
                       "interval_applied": 0, "interval_rejected": 0,
                       "spectral_scale_point": 0, "spectral_scale_interval": 0,
                       "bend_speed_anchor": 0, "lateral_delayed": 0}
        self.applied: list[str] = []
        self.history: list[_HistoryState] = []
        self.current_t = float("nan")
        self.last_interval_states_modified = 0

    # ------------------------------------------------------------- accessors

    @property
    def distance(self) -> float:
        return float(self.x[D])

    @property
    def speed(self) -> float:
        return float(self.x[V])

    @property
    def sigma_distance(self) -> float:
        return float(math.sqrt(max(self.P[D, D], 0.0)))

    @property
    def sigma_speed(self) -> float:
        return float(math.sqrt(max(self.P[V, V], 0.0)))

    @property
    def accel_scale(self) -> float:
        return float(self.x[KA]) if self.cfg.accel_scale_enabled else 1.0

    @property
    def sigma_accel_scale(self) -> float:
        return (float(math.sqrt(max(self.P[KA, KA], 0.0)))
                if self.cfg.accel_scale_enabled else 0.0)

    @property
    def spectral_scale_value(self) -> float:
        return (float(self.spectral_scale)
                if self.cfg.spectral_scale_enabled else 1.0)

    @property
    def sigma_spectral_scale(self) -> float:
        return (float(math.sqrt(max(self.spectral_scale_var, 0.0)))
                if self.cfg.spectral_scale_enabled else 0.0)

    @property
    def bend_scale_value(self) -> float:
        return float(self.bend_scale) if self.cfg.bend_scale_enabled else 1.0

    @property
    def sigma_bend_scale(self) -> float:
        return (float(math.sqrt(max(self.bend_scale_var, 0.0)))
                if self.cfg.bend_scale_enabled else 0.0)

    def _regime_weight(self, raw_spectral: float) -> float:
        """0 below the low-speed cutoff, 1 above the high-speed cutoff - how
        much of the blended spectral scale comes from ``k_high``."""
        lo, hi = self.cfg.bend_scale_regime_lo_ms, self.cfg.bend_scale_regime_hi_ms
        if hi <= lo:
            return 1.0 if raw_spectral >= hi else 0.0
        return float(np.clip((float(raw_spectral) - lo) / (hi - lo), 0.0, 1.0))

    def report(self, t: float) -> SpeedSample:
        sample = SpeedSample(
            t=float(t), distance_m=self.distance, sigma_distance_m=self.sigma_distance,
            speed_ms=self.speed, sigma_speed_ms=self.sigma_speed,
            accel_bias_ms2=float(self.x[BA]), gyro_bias_rads=self.gyro_bias,
            accel_scale=self.accel_scale,
            sigma_accel_scale=self.sigma_accel_scale,
            spectral_scale=self.spectral_scale_value,
            sigma_spectral_scale=self.sigma_spectral_scale,
            bend_scale=self.bend_scale_value,
            sigma_bend_scale=self.sigma_bend_scale,
            source="+".join(self.applied),
        )
        item = self._history_at(float(t))
        if item is not None and abs(item.t - float(t)) < 1e-6:
            item.samples.append(sample)
        return sample

    # ------------------------------------------------------------ prediction

    def predict(self, a_long: float, dt: float, shock: bool = False,
                gap: bool = False) -> float:
        """Advance one IMU step. Returns the distance travelled this step."""
        cfg = self.cfg
        self.applied = []
        raw = 0.0 if (gap or shock) else float(a_long) - float(self.x[BA])
        scale = float(self.x[KA]) if cfg.accel_scale_enabled else 1.0
        a = scale * raw
        self.last_longitudinal_accel = float(a)
        ds = self.x[V] * dt + 0.5 * a * dt * dt
        self.x[D] += ds
        self.x[V] += a * dt

        F = np.eye(SPEED_DIM)
        F[D, V] = dt
        F[D, BA] = -0.5 * scale * dt * dt
        F[V, BA] = -scale * dt
        # d(a)/d(k_a) = raw, so the scale enters exactly where the bias does but
        # weighted by how hard the car was accelerating - which is why only a
        # manoeuvring vehicle can ever make it observable.
        F[D, KA] = 0.5 * raw * dt * dt
        F[V, KA] = raw * dt
        # DA is frozen: identity row, no process noise.
        self.P = F @ self.P @ F.T
        for item in self.history:
            item.cross = item.cross @ F.T

        sigma_a = cfg.accel_noise_ms2 * (4.0 if shock else 1.0)
        g = np.array([0.5 * scale * dt * dt, scale * dt, 0.0, 0.0, 0.0])
        Q = (sigma_a**2) * np.outer(g, g)
        Q[BA, BA] += (cfg.accel_bias_rw**2) * dt
        if cfg.accel_scale_enabled:
            Q[KA, KA] += (cfg.accel_scale_rw**2) * dt
        self.P += Q
        self.gyro_bias_var += (cfg.gyro_bias_rw**2) * dt
        if cfg.spectral_scale_enabled:
            self.spectral_scale_var += (cfg.spectral_scale_rw**2) * dt
        if cfg.bend_scale_enabled:
            self.bend_scale_var += (cfg.bend_scale_rw**2) * dt
        if cfg.accel_scale_enabled:
            self.x[KA] = float(np.clip(self.x[KA], cfg.accel_scale_min,
                                       cfg.accel_scale_max))
        self.counts["predict"] += 1
        return float(ds)

    # --------------------------------------------------------------- anchors

    def zero_velocity(self, a_long: float, dt: float, yaw_rate: float,
                      run_s: float = 1e9,
                      spectral_speed: float = float("nan")) -> bool:
        """The car is provably parked: v = 0, and both biases are observable."""
        cfg = self.cfg
        if (cfg.zupt_motion_guard_enabled and math.isfinite(spectral_speed)
                and float(spectral_speed) >= cfg.zupt_motion_guard_spectral_min_ms):
            self.counts["zupt_motion_rejected"] += 1
            return False
        v_inflation = max(1.0, cfg.zupt_speed_correlation_s / max(dt, 1e-6))
        # Speed only. "The car is not moving" is a statement about the speed;
        # it is not evidence about the accelerometer's zero point, and letting
        # it rewrite b_a through the cross-covariance starts a feedback loop -
        # the correction is attributed to bias, the raised bias decelerates the
        # next prediction, which invites a further correction. The bias is
        # measured separately just below, where the claim is explicit.
        r_v = (cfg.zupt_speed_sigma_ms**2) * v_inflation
        reachable = cfg.zupt_max_decel_ms2 * max(float(run_s), 0.0)
        credibility = 1.0 if abs(self.x[V]) <= reachable else (
            reachable / max(abs(float(self.x[V])), 1e-6))
        self._update_speed_only(-float(self.x[V]) * credibility,
                                r_v / max(credibility**2, 1e-6))
        inflation = max(1.0, cfg.zupt_bias_correlation_s / max(dt, 1e-6))
        self._update(np.array([0.0, 0.0, 1.0, 0.0, 0.0]), float(a_long) - self.x[BA],
                     (cfg.zupt_bias_sigma_ms2**2) * inflation)
        # The true yaw rate while parked is zero, so whatever the gyro reports
        # is bias. Same correlation argument, same inflation.
        gyro_inflation = max(1.0, cfg.zupt_gyro_correlation_s / max(dt, 1e-6))
        r = (cfg.lateral_gyro_noise_rads**2) * gyro_inflation
        s = self.gyro_bias_var + r
        k = self.gyro_bias_var / s
        self.gyro_bias += k * (float(yaw_rate) - self.gyro_bias)
        self.gyro_bias_var = max((1.0 - k) * self.gyro_bias_var,
                                 cfg.gyro_bias_sigma_floor**2)
        self.counts["zupt"] += 1
        self.applied.append("stop")
        self._floors()
        return True

    def lateral_anchor(self, a_lat: float, yaw_rate_smooth: float, dt: float,
                       shock: bool = False,
                       spectral_speed: float = float("nan"),
                       t: Optional[float] = None) -> Optional[tuple[float, float]]:
        """``v = a_lat / omega`` - the speed of circular motion.

        No threshold decides whether this is usable; its own variance does.
        Propagating the ratio,

            sigma_v^2  =  sigma_a^2 / omega^2  +  a_lat^2 * sigma_omega^2 / omega^4

        the uncertainty rises steeply as ``omega`` approaches zero, so a barely
        perceptible drift in the steering contributes a measurement so vague it
        changes nothing, and a real turn contributes a sharp one. That is the
        behaviour a threshold was crudely approximating.
        """
        cfg = self.cfg
        if not cfg.lateral_enabled or shock:
            return None
        omega = float(yaw_rate_smooth) - self.gyro_bias
        if abs(omega) < cfg.lateral_min_omega_rads:
            return None
        a_lat = float(a_lat)
        measured = a_lat / omega
        # In circular motion the lateral acceleration points into the turn, so
        # a_lat and omega share a sign. A confident disagreement is not a turn.
        if measured < 0.0 and abs(measured) > 1.0:
            self.counts["lateral_rejected"] += 1
            return None
        if not math.isfinite(measured) or abs(measured) > cfg.max_speed_ms * 1.5:
            self.counts["lateral_rejected"] += 1
            return None

        n_eff = max(1.0, cfg.lateral_smooth_s / max(dt, 1e-6))
        # White noise averages down over the smoothing window; the offset does
        # not, so it enters at full size and dominates as omega gets small.
        sa = math.hypot(cfg.lateral_accel_noise_ms2 / math.sqrt(n_eff),
                        cfg.lateral_accel_bias_ms2)
        sw = cfg.lateral_gyro_noise_rads / math.sqrt(n_eff)
        var = (sa / omega) ** 2 + (a_lat**2) * (sw**2) / (omega**4)
        var += cfg.lateral_model_sigma_ms**2
        sigma = math.sqrt(var)
        if sigma > cfg.lateral_max_sigma_ms:
            return None
        inflation = max(1.0, cfg.lateral_correlation_s / max(dt, 1e-6))
        R = var * inflation
        if (cfg.lateral_consensus_guard_enabled
                and math.isfinite(spectral_speed)):
            supported = min(float(self.x[V]), float(spectral_speed))
            excess = supported - measured - cfg.lateral_consensus_deadband_ms
            if excess > 0.0:
                scale = max(cfg.lateral_consensus_scale_ms, 1e-6)
                factor = min((1.0 + excess / scale) ** 2,
                             cfg.lateral_consensus_max_factor)
                R *= factor
                self.counts["lateral_consensus_downweighted"] += 1
        H = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        delayed = None
        if (cfg.lateral_delayed_correction_enabled and t is not None
                and cfg.lateral_anchor_delay_s > 0.0):
            delayed = self._history_at(float(t) - float(cfg.lateral_anchor_delay_s))
            if delayed is not None and delayed.t >= float(t) - 1e-6:
                delayed = None          # anchor time == now: nothing to rewind
        if delayed is None:
            innovation = measured - self.x[V]
            self._update(H, innovation, R)
        else:
            # Delayed scalar measurement: exact joint-covariance Kalman gain
            # onto the CURRENT state from the retained cross-covariance with
            # the state at t_anchor (identical machinery to apply_interval's
            # delayed branch), then propagate the same correction through
            # every retained state between t_anchor and now. No raw D rewrite.
            innovation = measured - delayed.x[V]
            S = float(H @ (delayed.P @ H) + R)
            if math.isfinite(S) and S > 0.0:
                cross = delayed.cross.T @ H
                gain = cross / S
                self.x = self.x + gain * innovation
                self.P = self.P - np.outer(cross, cross) / S
                self.P = 0.5 * (self.P + self.P.T)
                self._smooth_delayed_history(delayed, H, innovation, S)
                self.counts["lateral_delayed"] = self.counts.get("lateral_delayed", 0) + 1
        self.counts["lateral"] += 1
        self.applied.append("lateral")
        self._floors()
        # A clean, well-conditioned circular-motion anchor is also an absolute
        # speed reference for the spectral scale - but only when omega is large
        # enough that the ratio is sharp, so a lane-drift anchor cannot nudge
        # k_s. The scale calibrator applies its own min-speed and innovation
        # gates.
        if (cfg.spectral_scale_enabled and abs(omega) > 3.0 * cfg.lateral_min_omega_rads
                and sigma < cfg.lateral_max_sigma_ms * 0.35):
            self.spectral_scale_point(measured, sigma, float(spectral_speed),
                                      t=self.current_t)
        return measured, sigma

    def spectral_update(self, speed: float, sigma: float, dt: float) -> None:
        """Speed regressed on IMU band powers. The between-turns filler.

        When ``spectral_scale_enabled``, the incoming ``speed`` is corrected by
        the online scale ``k_s`` and the measurement variance is inflated by the
        scale's own uncertainty, ``(speed * sigma_k_s)^2`` - so an unconstrained
        ``k_s`` widens ``sigma_v`` and ``sigma_D`` rather than pretending to a
        correction it has not earned. This update does not touch ``k_s``:
        letting the spectral source calibrate its own scale is the circular
        loop the anchors exist to break.
        """
        cfg = self.cfg
        if not cfg.spectral_enabled or not math.isfinite(speed):
            return
        raw = float(speed)
        # Blend the low-speed scale k_s (spectral_scale, from lateral anchors)
        # and the high-speed scale k_high (bend_scale, from curvature anchors)
        # by a v_spectral regime weight. With bend_scale disabled w is unused
        # and k == k_s; with both disabled k == 1.
        w = self._regime_weight(raw) if cfg.bend_scale_enabled else 0.0
        k = (1.0 - w) * self.spectral_scale_value + w * self.bend_scale_value
        corrected = k * raw
        extra = 0.0
        if cfg.spectral_scale_enabled:
            extra += ((1.0 - w) * raw * self.sigma_spectral_scale) ** 2
        if cfg.bend_scale_enabled:
            extra += (w * raw * self.sigma_bend_scale) ** 2
        inflation = max(1.0, cfg.spectral_correlation_s / max(dt, 1e-6))
        r = ((sigma * cfg.spectral_sigma_scale) ** 2 + extra) * inflation
        # `spectral_saturation_known` (permanent, set by the first bend that beat
        # the concurrent reading): this source *has* a plateau. Once known, a
        # reading close to the plateau that would drag v DOWN is (near)
        # uninformative - the model reads ~the same there for a wide range of
        # true speeds - so its pull is heavily down-weighted. Readings clearly
        # above the plateau are the source's right-skewed noise, whose upper
        # tail does track true speed (Phase 0: v_spectral 13-14 -> v_true p90
        # ~20), so they are left alone and legitimately lift v. Phase 25 tried
        # the strict `z = min(v, plateau)` form that also forbids the upward
        # lift; it caps v at the plateau and loses ~6 points of D/D_true - the
        # noise asymmetry is real and worth keeping.
        if (cfg.spectral_censor_enabled and self.spectral_saturation_known
                and raw <= self.spectral_plateau_ms + cfg.spectral_censor_band_ms
                and corrected < self.x[V] - cfg.spectral_censor_deadband_ms):
            r *= cfg.spectral_censor_factor
            self.spectral_censored_updates += 1
        elif (cfg.spectral_saturation_guard_enabled
              and cfg.spectral_guard_min_raw_ms <= raw <= cfg.spectral_guard_max_raw_ms
              and corrected < self.x[V] - cfg.spectral_guard_deadband_ms
              and self.last_longitudinal_accel >= cfg.spectral_guard_min_accel_ms2):
            r *= cfg.spectral_guard_factor
            self.counts["spectral_guarded"] += 1
        self._update(np.array([0.0, 1.0, 0.0, 0.0, 0.0]), corrected - self.x[V], r)
        self.counts["spectral"] += 1
        self.applied.append("spectral")
        self._floors()

    def _spectral_scale_kalman(self, z: float, r: float, t: float,
                               kind: str) -> bool:
        """Scalar Kalman step on ``k_s`` from an implied-scale observation
        ``z ≈ k_s`` with variance ``r``. Only the two absolute-speed anchors
        reach here; ordinary spectral observations never do."""
        cfg = self.cfg
        if not cfg.spectral_scale_enabled or not math.isfinite(z) or r <= 0.0:
            return False
        s = self.spectral_scale_var + r
        innovation = z - self.spectral_scale
        if abs(innovation) / math.sqrt(max(s, 1e-12)) > cfg.spectral_scale_max_innovation_sigma:
            return False
        k = self.spectral_scale_var / s
        self.spectral_scale = float(np.clip(
            self.spectral_scale + k * innovation,
            cfg.spectral_scale_min, cfg.spectral_scale_max))
        self.spectral_scale_var = max((1.0 - k) * self.spectral_scale_var,
                                      cfg.spectral_scale_sigma_floor ** 2)
        self.spectral_scale_updates += 1
        self.counts[f"spectral_scale_{kind}"] += 1
        self.spectral_scale_trace.append(
            (float(t), float(self.spectral_scale),
             float(math.sqrt(self.spectral_scale_var)), kind))
        return True

    def spectral_scale_point(self, v_anchor: float, sigma_anchor: float,
                             v_spectral_now: float, t: float = float("nan")) -> bool:
        """Calibrate ``k_s`` from a point absolute-speed anchor (``a_lat/omega``).

        The anchor says the true speed here is ``v_anchor``; the spectral model
        said ``v_spectral_now``; so ``k_s ≈ v_anchor / v_spectral_now``. Only
        used above ``spectral_scale_anchor_min_speed_ms`` where the ratio is
        meaningful.
        """
        cfg = self.cfg
        if (not cfg.spectral_scale_enabled or not math.isfinite(v_spectral_now)
                or v_spectral_now < cfg.spectral_scale_anchor_min_speed_ms):
            return False
        z = float(v_anchor) / float(v_spectral_now)
        r = ((float(sigma_anchor) / v_spectral_now) ** 2
             + (cfg.spectral_scale_anchor_model_frac * z) ** 2)
        return self._spectral_scale_kalman(z, r, t, "point")

    def spectral_scale_interval(self, length_m: float, sigma_m: float,
                                integrated_v_spectral: float,
                                t: float = float("nan")) -> bool:
        """Calibrate ``k_s`` from an accepted turn-to-turn map interval.

        ``L_map ≈ k_s · ∫ v_spectral dt`` over the segment, so
        ``k_s ≈ L_map / ∫v_spectral dt`` and the Jacobian ``∂L_pred/∂k_s`` is
        ``∫v_spectral dt`` - large on a long cruising interval even when the
        longitudinal accelerometer (and therefore ``k_a``) is nearly
        unobservable there.
        """
        cfg = self.cfg
        iv = float(integrated_v_spectral)
        if (not cfg.spectral_scale_enabled or not math.isfinite(iv)
                or iv < 30.0):
            return False
        z = float(length_m) / iv
        r = (float(sigma_m) / iv) ** 2
        return self._spectral_scale_kalman(z, r, t, "interval")

    def bend_speed_anchor(self, v_bend: float, sigma_v_bend: float,
                          v_spectral_now: float, t: float = float("nan")) -> bool:
        """Calibrate ``k_high`` from a road-curvature bend anchor.

        A bend's fitted local speed ``v_bend`` (time->arc-length scale of the
        gyro-vs-polyline shape match) is an absolute speed at a high-speed point
        the spectral model has saturated at ``v_spectral_now``. So
        ``k_high ≈ v_bend / v_spectral_now``. Scalar Kalman on ``bend_scale``,
        innovation-gated; only bend anchors reach it, and it never moves
        ``k_s``, ``k_a`` or the EKF state.
        """
        cfg = self.cfg
        if (not cfg.bend_scale_enabled or not math.isfinite(v_bend)
                or not math.isfinite(v_spectral_now) or v_spectral_now < 1e-6):
            return False
        z = float(v_bend) / float(v_spectral_now)
        r = ((float(sigma_v_bend) / v_spectral_now) ** 2
             + (cfg.bend_scale_anchor_model_frac * z) ** 2)
        if r <= 0.0:
            return False
        s = self.bend_scale_var + r
        innovation = z - self.bend_scale
        if abs(innovation) / math.sqrt(max(s, 1e-12)) > cfg.bend_scale_max_innovation_sigma:
            return False
        k = self.bend_scale_var / s
        self.bend_scale = float(np.clip(self.bend_scale + k * innovation,
                                        cfg.bend_scale_min, cfg.bend_scale_max))
        self.bend_scale_var = max((1.0 - k) * self.bend_scale_var,
                                  cfg.bend_scale_sigma_floor ** 2)
        self.bend_scale_updates += 1
        self.counts["bend_speed_anchor"] += 1
        self.bend_scale_trace.append(
            (float(t), float(self.bend_scale),
             float(math.sqrt(self.bend_scale_var))))
        # A bend that is faster than the concurrent spectral reading proves the
        # source under-reads in that band - it is saturated, not merely noisy.
        if float(v_bend) > float(v_spectral_now) + cfg.spectral_saturation_margin_ms:
            self.spectral_saturation_known = True
            self.spectral_plateau_ms = max(self.spectral_plateau_ms,
                                           float(v_spectral_now))
        return True

    def envelope(self, speed_limit_ms: float, dt: float = 0.1) -> None:
        """Physical rails. A car does not reverse down a street at road speed,
        and the sign of v decides which way the map predicts the road turns."""
        cfg = self.cfg
        upper = min(speed_limit_ms, cfg.max_speed_ms)
        lower = -abs(cfg.reverse_speed_limit_ms)
        target = None
        if self.x[V] > upper:
            target = upper
        elif self.x[V] < lower:
            target = lower
        if target is None:
            return
        inflation = max(1.0, cfg.envelope_correlation_s / max(dt, 1e-6))
        # Speed only, for the same reason: a rail says where the speed cannot
        # be. It says nothing whatever about the accelerometer.
        self._update_speed_only(target - self.x[V],
                                (cfg.envelope_sigma_ms**2) * inflation)
        self.counts["envelope"] += 1
        self._floors()

    # ------------------------------------------------------------- internals

    def open_interval(self, t: float) -> None:
        """Freeze the current distance as the start of an interval.

        ``DA := D`` makes the two perfectly correlated at this instant, so the
        covariance is copied across rather than invented. From here the
        variance of ``D - DA`` grows with exactly the process noise the filter
        actually adds, which is what the constraint later measures against.
        """
        anchor = self._history_at(float(t))
        if anchor is None:
            self.x[DA] = self.x[D]
            self.P[DA, :] = self.P[D, :]
            self.P[:, DA] = self.P[:, D]
            self.P[DA, DA] = self.P[D, D]
        else:
            # DA becomes the *past* D at the physical event time.  Its
            # covariance with today's state comes from the retained joint
            # covariance, not from pretending the delayed event happened now.
            cross = anchor.cross[D, :].copy()
            self.x[DA] = anchor.x[D]
            self.P[DA, :] = cross
            self.P[:, DA] = cross
            self.P[DA, DA] = anchor.P[D, D]
        self.anchor_open = True
        self.anchor_t = float(t)
        # States recorded before DA existed carry an unrelated fourth
        # component. They cannot participate in the next interval.
        self.history.clear()
        if math.isfinite(self.current_t):
            self.remember(self.current_t)

    def travelled_since_anchor(self) -> float:
        return float(self.x[D] - self.x[DA])

    def distance_at(self, t: float) -> float:
        item = self._history_at(float(t))
        return float(item.x[D]) if item is not None else float("nan")

    def travelled_since_anchor_at(self, t: float) -> float:
        item = self._history_at(float(t))
        return (float(item.x[D] - item.x[DA])
                if item is not None else float("nan"))

    def apply_interval(self, length_m: float, sigma_m: float,
                       max_innovation_sigma: float = 4.0,
                       event_t: Optional[float] = None) -> tuple[bool, float, float]:
        """Apply ``D - D_anchor = L_map`` as a measurement.

        This is the only absolute distance information available without GPS,
        and unlike every other source it constrains a long fast straight, where
        the accelerometer has drifted, the lateral anchors are silent and the
        spectral model has saturated.

        Returns ``(applied, ins_length, innovation_sigma)``. A wildly
        disagreeing constraint is refused rather than clipped: at that size the
        likely explanation is a wrong route, not a wrong distance, and a bad
        route must not be allowed to rewrite the speed.
        """
        if not self.anchor_open or not math.isfinite(sigma_m):
            return False, float("nan"), float("nan")
        H = np.zeros(SPEED_DIM)
        H[D] = 1.0
        H[DA] = -1.0
        delayed = self._history_at(float(event_t)) if event_t is not None else None
        state = delayed.x if delayed is not None else self.x
        covariance = delayed.P if delayed is not None else self.P
        ins = float(state[D] - state[DA])
        innovation = float(length_m) - ins
        S = float(H @ (covariance @ H) + sigma_m**2)
        z = abs(innovation) / math.sqrt(max(S, 1e-12))
        if z > max_innovation_sigma:
            self.counts["interval_rejected"] += 1
            return False, ins, z
        if delayed is None:
            self._update(H, innovation, sigma_m**2, allow_scale=True)
            self.last_interval_states_modified = 1
        else:
            cross = delayed.cross.T @ H
            gain = cross / S
            self.x = self.x + gain * innovation
            self.P = self.P - np.outer(cross, cross) / S
            self.P = 0.5 * (self.P + self.P.T)

            self.last_interval_states_modified = self._smooth_delayed_history(
                delayed, H, innovation, S)
        if self.cfg.accel_scale_enabled:
            self.x[KA] = float(np.clip(
                self.x[KA], self.cfg.accel_scale_min, self.cfg.accel_scale_max))
        self.counts["interval_applied"] += 1
        self.applied.append("interval")
        self._floors()
        return True, ins, z

    def remember(self, t: float) -> None:
        """Retain a bounded state/cross-covariance history for delayed events."""
        self.current_t = float(t)
        keep_s = max(12.0, float(self.cfg.fixed_lag_s) + 2.0)
        cutoff = self.current_t - keep_s
        drop = next((i for i, h in enumerate(self.history) if h.t >= cutoff),
                    len(self.history))
        if drop:
            self.history = self.history[drop:]
            for item in self.history:
                item.past_crosses = item.past_crosses[drop:]
        if self.history and abs(self.history[-1].t - self.current_t) < 1e-6:
            old_samples = self.history[-1].samples
            self.history.pop()
        else:
            old_samples = []
        past = [h.cross.copy() for h in self.history]
        self.history.append(_HistoryState(
            self.current_t, self.x.copy(), self.P.copy(), self.P.copy(),
            past_crosses=past, samples=old_samples))

    def _history_at(self, t: float) -> Optional[_HistoryState]:
        if not self.history or not math.isfinite(t):
            return None
        index = int(np.argmin([abs(h.t - t) for h in self.history]))
        item = self.history[index]
        return item if abs(item.t - t) <= 1.0 else None

    def _smooth_delayed_history(self, target: _HistoryState, H: np.ndarray,
                                innovation: float, S: float) -> int:
        """Condition recent stored states on a measurement at ``target``.

        Cross-covariances are captured when every state is appended, so this
        is the scalar fixed-lag Kalman update, not a distance correction spread
        by hand.  States older than the configured lag are immutable outputs.
        With lag zero only the delayed endpoint is revised internally.
        """
        try:
            j = self.history.index(target)
        except ValueError:
            return 1
        lag = max(float(self.cfg.fixed_lag_s), 0.0)
        earliest = target.t - lag
        modified = 0
        for i, item in enumerate(self.history):
            if item.t < earliest:
                continue
            if i < j:
                cross = target.past_crosses[i]
            elif i == j:
                cross = target.P
            else:
                cross = item.past_crosses[j].T
            c = cross @ H
            item.x = item.x + (c / S) * innovation
            item.P = item.P - np.outer(c, c) / S
            item.P = 0.5 * (item.P + item.P.T)
            if lag > 0.0:
                self._sync_samples(item)
            modified += 1
        return max(modified, 1)

    @staticmethod
    def _sync_samples(item: _HistoryState) -> None:
        for sample in item.samples:
            sample.distance_m = float(item.x[D])
            sample.speed_ms = float(item.x[V])
            sample.accel_bias_ms2 = float(item.x[BA])
            sample.accel_scale = float(item.x[KA])
            sample.sigma_distance_m = float(math.sqrt(max(item.P[D, D], 0.0)))
            sample.sigma_speed_ms = float(math.sqrt(max(item.P[V, V], 0.0)))
            sample.sigma_accel_scale = float(math.sqrt(max(item.P[KA, KA], 0.0)))

    def _update(self, H: np.ndarray, innovation: float, R: float,
                allow_scale: bool = False) -> None:
        PH = self.P @ H
        S = float(H @ PH + R)
        if not math.isfinite(S) or S <= 0.0:
            return
        K = PH / S
        history_projection = [item.cross @ H for item in self.history]
        if self.cfg.accel_scale_enabled and not allow_scale:
            # Refuse both a mean update and an information update for k_a.
            # Merely restoring the mean after an ordinary Kalman update still
            # shrinks P[KA, KA], falsely reporting that a non-map source has
            # calibrated scale. A zero constrained gain keeps its marginal
            # uncertainty unchanged while retaining a PSD Joseph update.
            K[KA] = 0.0
        self.x = self.x + K * float(innovation)
        # Lateral/spectral speed and stop-bias observations are not absolute
        # distance references. They may shape the covariance (and therefore
        # how a later interval identifies scale), but they must not calibrate
        # k_a themselves. Otherwise enabling the state lets the saturated
        # spectral source silently drive it, which happened in the interrupted
        # prototype even with zero accepted map intervals.
        IKH = np.eye(SPEED_DIM) - np.outer(K, H)
        self.P = IKH @ self.P @ IKH.T + R * np.outer(K, K)
        self.P = 0.5 * (self.P + self.P.T)
        for item, projection in zip(self.history, history_projection):
            # Keep the retained joint distribution conditioned on every
            # intervening measurement. Only expose those revised samples when
            # fixed-lag output is explicitly enabled.
            L = projection / S
            if self.cfg.accel_scale_enabled and not allow_scale:
                L[KA] = 0.0
            item.x = item.x + L * float(innovation)
            item.P = (item.P - np.outer(L, projection)
                      - np.outer(projection, L) + S * np.outer(L, L))
            item.P = 0.5 * (item.P + item.P.T)
            item.cross = (item.cross - np.outer(L, PH)
                          - np.outer(projection, K) + S * np.outer(L, K))
            if self.cfg.fixed_lag_s > 0.0:
                self._sync_samples(item)

    def _update_speed_only(self, innovation: float, R: float) -> None:
        """Kalman update on ``v`` alone, leaving the other states' means where
        they are.

        Used for the two observations that are constraints on the speed rather
        than information about the sensors: a detected stop, and the physical
        rails. The covariance is updated consistently - ``P[V, :]`` and
        ``P[:, V]`` shrink by the same gain - so the filter stays calibrated,
        but ``b_a`` is not asked to explain a constraint it had no part in.
        """
        pv = float(self.P[V, V])
        S = pv + R
        if not math.isfinite(S) or S <= 0.0:
            return
        k = pv / S
        self.x[V] += k * float(innovation)
        self.P[V, :] *= (1.0 - k)
        self.P[:, V] *= (1.0 - k)
        self.P[V, V] = max((1.0 - k) * pv, 0.0)
        self.P = 0.5 * (self.P + self.P.T)
        for item in self.history:
            item.cross[:, V] *= (1.0 - k)

    def _floors(self) -> None:
        cfg = self.cfg
        floors = np.array([cfg.d_sigma_floor_m**2, cfg.v_sigma_floor_ms**2,
                           cfg.accel_bias_sigma_floor**2, 0.0,
                           (cfg.accel_scale_sigma_floor**2
                            if cfg.accel_scale_enabled else 0.0)])
        short = np.maximum(floors - np.diag(self.P), 0.0)
        self.P = self.P + np.diag(short)
        sigma = np.sqrt(np.maximum(np.diag(self.P), 0.0))
        bound = 0.999 * np.outer(sigma, sigma)
        self.P = np.clip(self.P, -bound, bound)
        np.fill_diagonal(self.P, sigma**2)
        self.gyro_bias_var = max(self.gyro_bias_var, cfg.gyro_bias_sigma_floor**2)

    def stats(self) -> dict[str, Any]:
        return dict(self.counts)
