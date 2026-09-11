"""The GPS-free stop detector.

The one thing the vibration signal is genuinely good at. What it is *not* good
at - measuring speed - is the subject of
:mod:`geotrace.pacman_tracker.spectral` and of the diagnosis in
``docs/PACMAN_TRACKER.md``.
"""

import numpy as np

from geotrace.pacman_tracker.config import MotionConfigP
from geotrace.pacman_tracker.motion import detect_stationary


def test_it_separates_idling_from_driving():
    rng = np.random.default_rng(1)
    t = np.arange(0, 60, 0.1)
    a = np.where(t < 30, rng.normal(0, 0.008, len(t)), rng.normal(0, 0.25, len(t)))
    w = np.where(t < 30, rng.normal(0, 0.0008, len(t)), rng.normal(0, 0.02, len(t)))
    stationary, _, _, _ = detect_stationary(t, a, w, MotionConfigP())
    assert stationary[(t > 5) & (t < 28)].mean() > 0.9
    assert stationary[t > 35].mean() < 0.05


def test_one_quiet_sample_in_a_cruise_is_not_a_stop():
    rng = np.random.default_rng(2)
    t = np.arange(0, 30, 0.1)
    a = rng.normal(0, 0.3, len(t))
    w = rng.normal(0, 0.02, len(t))
    a[150] = 0.0
    w[150] = 0.0
    stationary, _, _, _ = detect_stationary(t, a, w, MotionConfigP())
    assert not stationary.any()


def test_it_reports_how_long_each_stop_has_lasted():
    """The run length is what makes a stop physically credible: a car cannot be
    stopped now if it was doing 12 m/s during the second the detector itself
    called quiet."""
    rng = np.random.default_rng(3)
    t = np.arange(0, 40, 0.1)
    a = np.where(t < 20, rng.normal(0, 0.3, len(t)), rng.normal(0, 0.008, len(t)))
    w = np.where(t < 20, rng.normal(0, 0.02, len(t)), rng.normal(0, 0.0008, len(t)))
    stationary, _, _, run = detect_stationary(t, a, w, MotionConfigP())
    assert run[t < 20].max() < 2.0
    assert run[-1] > 15.0
    assert np.all(np.diff(run[stationary & (t > 25)]) > 0)


def test_a_missing_signal_does_not_crash_it():
    stationary, a_std, w_std, run = detect_stationary(
        np.zeros(0), np.zeros(0), np.zeros(0), MotionConfigP())
    assert len(stationary) == len(a_std) == len(w_std) == len(run) == 0
