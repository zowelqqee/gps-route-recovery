"""A biased sensor with an honest sigma is useful; with a small one it is not.

Both mechanisms here were added after measuring, against withheld GPS, that the
speed filter was confidently wrong: on 2026-07-26 its distance error reached
1969 m while sigma_D said 153 m, and its 2-sigma coverage was 5 % where 95 % was
claimed.
"""

import math

import numpy as np
import pytest

from geotrace.pacman_tracker.config import PacmanConfig
from geotrace.pacman_tracker.speed import GlobalSpeedTracker, SpeedConfig
from geotrace.pacman_tracker.spectral import SpectralSpeedModel


# ------------------------------------------------------- spectral deployment


def _model(holdout_rmse, in_sample_rmse, fitted=True):
    return SpectralSpeedModel(
        coefficients=np.zeros(3), intercept=0.0, sigma_ms=holdout_rmse,
        fitted=fitted, in_sample_sigma_ms=in_sample_rmse,
        generalisation_gap=holdout_rmse / max(in_sample_rmse, 1e-6), reason="ok")


def test_deployment_sigma_charges_for_the_generalisation_gap():
    """A model that already doubled its error on held-out calibration data will
    degrade further on driving it has never seen."""
    tight = _model(holdout_rmse=2.0, in_sample_rmse=1.9)     # gap ~1.05
    loose = _model(holdout_rmse=2.0, in_sample_rmse=0.7)     # gap ~2.9
    assert loose.deployment_sigma_ms > tight.deployment_sigma_ms
    assert tight.deployment_sigma_ms >= tight.sigma_ms


def test_deployment_sigma_is_never_smaller_than_the_measured_error():
    """The holdout RMSE is a floor, not a ceiling: deployment is strictly
    further from the training distribution than the holdout was."""
    for holdout, in_sample in ((2.16, 0.80), (2.85, 1.36), (4.0, 4.0), (1.6, 3.0)):
        m = _model(holdout, in_sample)
        assert m.deployment_sigma_ms >= m.sigma_ms - 1e-9


def test_deployment_sigma_is_bounded_in_both_directions():
    """It is an extrapolation, not a measurement, so it may not run away - and
    a well-behaved model must not be punished into uselessness."""
    wild = _model(holdout_rmse=12.0, in_sample_rmse=0.001)
    assert wild.deployment_sigma_ms <= 20.0
    perfect = _model(holdout_rmse=1.5, in_sample_rmse=1.5)
    assert 1.5 <= perfect.deployment_sigma_ms <= 1.5 * 3.0 + 1e-9


def test_an_unfitted_model_reports_its_fallback():
    m = SpectralSpeedModel()
    assert not m.fitted
    assert m.deployment_sigma_ms == m.fallback_sigma_ms


def test_the_gap_is_computed_per_session_not_shared():
    """Vibration-to-speed behaviour differs between vehicles, mountings and
    road surfaces, so the charge for it has to be measured per session."""
    a = _model(holdout_rmse=2.0, in_sample_rmse=1.9)
    b = _model(holdout_rmse=2.0, in_sample_rmse=0.5)
    assert a.deployment_sigma_ms != b.deployment_sigma_ms


def test_a_biased_spectral_reading_cannot_make_the_filter_confident():
    """The failure this exists to prevent is *confidence*, not movement.

    With no competing evidence a filter should converge on a persistent
    measurement - that is correct. What must not happen is the thing measured
    on 2026-07-26: converging on a reading that is 8 m/s wrong while reporting
    a sigma that denies it. The sigma has to stay the size of the sensor.
    """
    cfg = SpeedConfig()
    t = GlobalSpeedTracker(cfg, v0=18.0)
    for _ in range(600):                       # a minute of "you are doing 12"
        t.predict(0.0, 0.1)
        t.spectral_update(12.0, 6.0, 0.1)
    assert t.sigma_speed > 2.0, "a 6 m/s sensor must not produce sub-m/s certainty"
    assert abs(t.speed - 12.0) < 6.0


def test_a_tighter_sensor_earns_more_confidence_than_a_loose_one():
    """The sigma must actually gate how much the filter commits."""
    ends = []
    for sigma in (1.0, 6.0):
        t = GlobalSpeedTracker(SpeedConfig(), v0=18.0)
        for _ in range(600):
            t.predict(0.0, 0.1)
            t.spectral_update(12.0, sigma, 0.1)
        ends.append(t.sigma_speed)
    assert ends[0] < ends[1]


# ---------------------------------------------------------- lateral offset


def _lateral_sigma(cfg, a_lat, omega, dt=0.1):
    t = GlobalSpeedTracker(cfg, v0=a_lat / omega)
    got = t.lateral_anchor(a_lat, omega, dt)
    return None if got is None else got[1]


def test_the_lateral_offset_does_not_average_down():
    """Smoothing reduces white noise by sqrt(n). It does nothing to an offset,
    and the measurement divides by omega, so the offset is what dominates when
    the turn is gentle."""
    cfg = SpeedConfig()
    bare = SpeedConfig(lateral_accel_bias_ms2=0.0)
    omega, v = 0.03, 14.0
    with_offset = _lateral_sigma(cfg, v * omega, omega)
    without = _lateral_sigma(bare, v * omega, omega)
    assert with_offset > without
    assert with_offset > 2.0, "a 0.07 m/s^2 offset at omega=0.03 is >2 m/s of speed"


def test_a_strong_turn_is_still_a_sharp_anchor():
    """The offset must not blunt the anchors that are actually accurate: on the
    review recordings |omega| > 0.1 anchors were unbiased to within 0.3 m/s."""
    cfg = SpeedConfig()
    sigma = _lateral_sigma(cfg, 8.0 * 0.35, 0.35)
    assert sigma is not None and sigma < 1.5


def test_lateral_sigma_grows_as_the_turn_flattens():
    cfg = SpeedConfig()
    sigmas = [_lateral_sigma(cfg, 12.0 * w, w) for w in (0.03, 0.08, 0.2, 0.5)]
    assert all(s is not None for s in sigmas)
    assert sigmas == sorted(sigmas, reverse=True)


def test_a_gentle_turn_anchor_moves_the_state_less_than_a_sharp_one():
    """The point of the sigma: on 2026-07-26 the |omega| 0.02-0.05 anchors
    carried a -2.7 m/s bias and the |omega| > 0.2 anchors carried +0.3."""
    cfg = SpeedConfig()
    moves = []
    for omega in (0.03, 0.35):
        t = GlobalSpeedTracker(cfg, v0=14.0)
        t.predict(0.0, 0.1)
        before = t.speed
        t.lateral_anchor(8.0 * omega, omega, 0.1)   # both claim v = 8
        moves.append(abs(t.speed - before))
    assert moves[1] > moves[0]


def test_the_deployment_sigma_is_a_function_of_the_fit_alone():
    """It must depend on nothing but numbers the fit itself produced, so there
    is no path for withheld GPS to influence it."""
    a = _model(2.5, 1.0)
    b = SpectralSpeedModel(
        coefficients=np.arange(3, dtype=float), intercept=7.0, sigma_ms=2.5,
        fitted=True, in_sample_sigma_ms=1.0, generalisation_gap=2.5,
        calibration_correlation=0.1, calibration_samples=99, reason="ok")
    assert a.deployment_sigma_ms == b.deployment_sigma_ms
