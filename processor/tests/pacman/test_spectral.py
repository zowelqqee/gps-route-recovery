"""The spectral speed model - band powers, not total power.

The variance odometer this replaces has no conditional speed information at all
(correlation 0.13 and -0.01 on the two review recordings once the car is
moving). Band-resolved power does, because tyre and road excitation move up in
frequency with speed while the total energy need not change.
"""

import numpy as np
import pytest

from geotrace.pacman_tracker.spectral import BANDS, SpectralSpeedModel, extract_features


def _synthetic(duration=240.0, fs=50.0, seed=0):
    """A signal whose *frequency* tracks speed while its amplitude does not.

    This is the whole point of the module: a total-power feature is blind to
    this by construction, and a band-resolved one is not.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration, 1.0 / fs)
    speed = 6.0 + 5.0 * np.sin(2 * np.pi * t / 90.0) + 4.0
    phase = np.cumsum(speed / 0.32 / (2 * np.pi)) / fs * 2 * np.pi
    tone = 0.4 * np.sin(phase)
    accel = np.column_stack([tone + rng.normal(0, 0.4, len(t)) for _ in range(3)])
    gyro = np.column_stack([0.05 * tone + rng.normal(0, 0.02, len(t)) for _ in range(3)])
    return t, accel, gyro, speed


def test_features_have_one_row_per_window_and_the_expected_width():
    t, accel, gyro, _ = _synthetic(duration=60.0)
    features = extract_features(t, accel, gyro)
    assert features.values.shape[1] == 6 * (len(BANDS) + 1)
    assert len(features) == features.values.shape[0] > 50
    assert np.all(np.isfinite(features.values))


def test_a_short_recording_yields_no_windows_rather_than_nonsense():
    t = np.arange(0.0, 1.0, 0.02)
    features = extract_features(t, np.zeros((len(t), 3)), np.zeros((len(t), 3)))
    assert len(features) == 0


def test_it_learns_speed_from_a_frequency_that_tracks_it():
    t, accel, gyro, speed = _synthetic()
    features = extract_features(t, accel, gyro)
    target = np.interp(features.times, t, speed)
    half = len(target) // 2
    model = SpectralSpeedModel.fit(features.values[:half], target[:half])
    assert model.fitted, model.reason
    predicted = model.predict_many(features.values[half:])
    assert float(np.corrcoef(predicted, target[half:])[0, 1]) > 0.5


def test_the_sigma_is_measured_on_held_out_calibration():
    """In-sample residuals flatter a least-squares fit, and this sigma is the
    only thing between a mediocre predictor and a confidently wrong filter."""
    t, accel, gyro, speed = _synthetic()
    features = extract_features(t, accel, gyro)
    target = np.interp(features.times, t, speed)
    model = SpectralSpeedModel.fit(features.values, target)
    residual = model.predict_many(features.values) - target
    in_sample = float(np.sqrt(np.mean(residual**2)))
    assert model.sigma_ms >= in_sample


def test_a_refused_fit_falls_back_rather_than_inventing_a_speed():
    model = SpectralSpeedModel.fit(np.zeros((10, 42)), np.zeros(10))
    assert not model.fitted
    speed, sigma = model.predict(np.zeros(42))
    assert speed == model.fallback_speed_ms
    assert sigma == model.fallback_sigma_ms


def test_a_constant_signal_is_refused():
    """No variation, no information - and no pretending otherwise."""
    t = np.arange(0.0, 240.0, 0.02)
    accel = np.ones((len(t), 3))
    gyro = np.ones((len(t), 3))
    features = extract_features(t, accel, gyro)
    model = SpectralSpeedModel.fit(features.values, np.full(len(features), 7.0))
    assert not model.fitted or model.sigma_ms >= 1.5


def test_predictions_are_clipped_to_plausible_speeds():
    t, accel, gyro, speed = _synthetic()
    features = extract_features(t, accel, gyro)
    model = SpectralSpeedModel.fit(features.values, np.interp(features.times, t, speed))
    wild = model.predict(np.full(features.values.shape[1], 1e6))[0]
    assert 0.0 <= wild <= 33.0


def test_interval_fit_uses_distance_constraints_without_pointwise_labels():
    rng = np.random.default_rng(17)
    intervals = 40
    per_interval = 8
    aggregate = rng.normal(size=(intervals, 3))
    values = np.repeat(aggregate, per_interval, axis=0)
    values += rng.normal(scale=0.05, size=values.shape)
    times = np.arange(len(values), dtype=float) * 0.5
    starts = np.arange(intervals, dtype=float) * per_interval * 0.5 - 0.25
    ends = starts + per_interval * 0.5
    speed = 4.0 + 1.2 * aggregate[:, 0] - 0.7 * aggregate[:, 1]
    distances = speed * (ends - starts)

    model = SpectralSpeedModel.fit_intervals(
        values, times, starts, ends, distances, min_intervals=30
    )

    assert model.fitted
    assert model.calibration_basis == "rfid_interval_distance"
    assert model.calibration_intervals == intervals
    predicted = np.array([
        model.predict_many(values[i * per_interval:(i + 1) * per_interval]).mean()
        for i in range(intervals)
    ])
    assert np.mean(np.abs(predicted - speed)) < 0.2


def test_interval_fit_refuses_too_few_rfid_intervals():
    model = SpectralSpeedModel.fit_intervals(
        np.ones((20, 2)),
        np.arange(20.0),
        np.array([0.0, 10.0]),
        np.array([9.0, 19.0]),
        np.array([30.0, 30.0]),
    )
    assert not model.fitted
    assert model.calibration_basis == "rfid_interval_distance"
