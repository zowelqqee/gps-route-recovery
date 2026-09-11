"""Curvature matching - scoring only, and never the speed.

    omega_gyro  ~=  b_g  +  v * kappa(s)

The equation is unchanged; what changed is which way it is read. It is now used
only to ask *how well does this road explain the gyro*, and the answer moves the
hypothesis's weight. It no longer moves the speed.

That restriction is not tidiness, it closes a loop. When each hypothesis owned a
speed and the curvature update could change it, every hypothesis pulled its own
``v`` towards whatever its own road implied - and then that ``v`` was used to
decide how far along the road it had travelled. The map was setting the speed
and the speed was choosing the map. Measured on the review recordings the map is
in no position to: usable curvature exists on 1.5 % of moving steps, and
``v = omega / kappa`` is wrong there by a median of -5.8 m/s, because a smoothed
OSM corner is tighter than the line a driver takes through it.

Speed comes from :mod:`~geotrace.pacman_tracker.speed`. This module reads it.

**On a straight road the score is identical for every hypothesis**, because
``kappa = 0`` makes the prediction ``b_g`` regardless of where along the road a
hypothesis sits. The gyro confirms the road is straight and says nothing about
position along it - which is a fact about the geometry, not a special case in
the code, and ``tests/pacman/test_straight_road.py`` holds it to it.

Scoring uses a Student-t log-likelihood of the normalised residual rather than a
residual standard deviation: ``[10, 10, 10, 10]`` degrees has zero standard
deviation and is a bad match. The heavy tails mean a pothole costs a hypothesis
some weight and never all of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from geotrace.pacman_tracker.config import MatchConfig
from geotrace.pacman_tracker.motion import ImuSample
from geotrace.pacman_tracker.roadmap import RoadGeometry
from geotrace.pacman_tracker.state import HypothesisSet


@dataclass
class MatchDiagnostics:
    """Per-step, per-hypothesis numbers kept for the death report.

    Indexed by hypothesis *id*, not by position: these are produced before
    merging and pruning reshuffle the population.
    """

    ids: np.ndarray
    kappa: np.ndarray
    omega_map: np.ndarray
    omega_gyro: float
    residual: np.ndarray
    z: np.ndarray
    sigma: np.ndarray
    log_likelihood: np.ndarray

    def index_of(self, hypothesis_id: int) -> Optional[int]:
        if self.ids.size == 0:
            return None
        hit = np.nonzero(self.ids == int(hypothesis_id))[0]
        return int(hit[0]) if hit.size else None


EMPTY = MatchDiagnostics(np.zeros(0, dtype=np.int64), *(np.zeros(0),), *(np.zeros(0),),
                         0.0, np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))


class CurvatureMatcher:
    """Scores each road hypothesis against the measured yaw rate."""

    def __init__(self, geometry: RoadGeometry, cfg: MatchConfig) -> None:
        self.geometry = geometry
        self.cfg = cfg
        self.updates = 0

    def score(self, hs: HypothesisSet, sample: ImuSample, distance: float,
              speed: float, gyro_bias: float, dt: float,
              ambiguous: Optional[np.ndarray] = None) -> MatchDiagnostics:
        cfg = self.cfg
        n = len(hs)
        if n == 0:
            return EMPTY

        s = np.clip(hs.s(distance), 0.0, self.geometry.lengths[hs.edge])
        kappa, _grad, sigma_kappa = self.geometry.curvature(hs.edge, s)
        omega_map = float(speed) * kappa
        predicted = float(gyro_bias) + omega_map
        residual = float(sample.yaw_rate) - predicted

        # Sensor noise really is white at 10 Hz; map error, mount misalignment
        # and the driver's line through a bend are one error sampled ten times,
        # so their variance carries the effective-sample-size correction.
        inflation = max(1.0, cfg.map_error_correlation_s / max(dt, 1e-6))
        correlated = cfg.model_sigma_rads**2 + (abs(float(speed)) * sigma_kappa) ** 2
        var = cfg.gyro_noise_rads**2 + inflation * correlated
        if sample.shock:
            var = var + (10.0 * cfg.gyro_noise_rads) ** 2
        sigma = np.sqrt(np.maximum(var, 1e-12))
        z = residual / sigma

        dof = max(cfg.student_dof, 1.0)
        loglik = -0.5 * (dof + 1.0) * np.log1p((z * z) / dof)
        if ambiguous is not None:
            ambiguous = np.asarray(ambiguous, dtype=bool)
            # Edge polylines do not contain the instantaneous junction turn;
            # that evidence is scored by the integrated turn model. Applying
            # endpoint curvature while either adjacent edge is plausible
            # double-counts a wrong model before the turn event has matured.
            loglik[ambiguous] = np.maximum(
                loglik[ambiguous], float(cfg.junction_loglik_floor))

        lam = math.exp(-dt / max(cfg.score_window_s, 1e-6))
        hs.logw *= lam
        hs.logw += loglik
        hs.logw -= hs.logw.max()

        lam_rms = math.exp(-dt / max(cfg.rms_window_s, 1e-6))
        hs.ewm_z2 *= lam_rms
        hs.ewm_w *= lam_rms
        valid = np.ones(n, dtype=bool) if ambiguous is None else ~ambiguous
        hs.ewm_z2[valid] += z[valid] * z[valid]
        hs.ewm_w[valid] += 1.0

        self.updates += 1
        return MatchDiagnostics(
            ids=hs.ids.copy(), kappa=kappa, omega_map=np.full(n, omega_map),
            omega_gyro=float(sample.yaw_rate), residual=np.full(n, 0.0) + residual,
            z=z, sigma=sigma, log_likelihood=loglik,
        )
