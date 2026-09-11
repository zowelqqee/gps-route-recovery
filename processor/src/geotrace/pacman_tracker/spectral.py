"""Speed from the shape of the IMU spectrum, not its size.

The variance-based odometer this replaces (deleted along with it) carried no
speed information at all once the car was moving: conditional on ``v > 2 m/s`` its correlation with true
speed is 0.13 on 2026-07-22 and -0.01 on 2026-07-26, and it emits a near
constant 5.4 / 9.5 m/s for every speed above walking pace. Its apparent skill was entirely
the ability to tell stopped from moving, which
:func:`~geotrace.pacman_tracker.motion.detect_stationary` already does, so that
is all that survives of it.

Band-resolved power is a different matter. Tyre and road excitation, the engine,
and wheel-rotation harmonics all move *up in frequency* with speed while the
total energy need not change, so where the total is blind the distribution over
bands is not. Measured on the same recordings, calibrated on the visible GPS
window only and evaluated over the whole outage, conditional on the car moving:

    total power (the old features)      corr  0.13  /  -0.01
    band powers, 6 bands x 6 axes       corr  0.74  /   0.45

with the standard deviation of the prediction rising from 0.27 to 0.67 of the
truth's - that is, it starts tracking the variation instead of emitting the
mean.

This is still a mediocre speedometer and it is used as one: a middling
pseudo-measurement that holds the estimate together between turns, well below
a detected stop or an ``a_lat / omega`` anchor in authority. Its job is to keep
the between-anchor bias under about 1-2 m/s, which by
``delta_s ~ delta_v * T`` is what keeps a 30-60 s gap between anchors from
costing more than the ~60 m that killed the true hypothesis before.

Ridge regression on log band powers, fitted with numpy - deliberately no new
dependency and no model that cannot be inspected. A gradient-boosted or small
convolutional model is the obvious next step if this proves out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

BANDS = ((0.5, 2.0), (2.0, 5.0), (5.0, 8.0), (8.0, 12.0), (12.0, 18.0), (18.0, 25.0))
"""Hz. Chosen to straddle body/suspension motion at the bottom, wheel-rotation
harmonics in the middle (a 0.32 m wheel turns at 7.5 Hz at 15 m/s) and tyre
roar at the top, within the 25 Hz the 50 Hz recorder can see."""


@dataclass
class SpectralFeatures:
    times: np.ndarray
    values: np.ndarray

    def __len__(self) -> int:
        return int(self.times.shape[0])


def extract_features(
    times: np.ndarray,
    accel: np.ndarray,
    gyro: np.ndarray,
    window_s: float = 2.56,
    hop_s: float = 0.5,
) -> SpectralFeatures:
    """Log band powers plus log total power, per axis, per window."""
    times = np.asarray(times, dtype=float)
    if times.size < 8:
        return SpectralFeatures(np.zeros(0), np.zeros((0, 6 * (len(BANDS) + 1))))
    fs = 1.0 / float(np.median(np.diff(times)))
    win = max(8, int(round(window_s * fs)))
    hop = max(1, int(round(hop_s * fs)))
    if len(times) <= win:
        return SpectralFeatures(np.zeros(0), np.zeros((0, 6 * (len(BANDS) + 1))))

    channels = np.column_stack([np.asarray(accel, dtype=float),
                                np.asarray(gyro, dtype=float)])
    taper = np.hanning(win)
    freqs = np.fft.rfftfreq(win, 1.0 / fs)
    masks = [(freqs >= lo) & (freqs < hi) for lo, hi in BANDS]

    starts = np.arange(0, len(times) - win, hop)
    rows = np.empty((len(starts), channels.shape[1] * (len(BANDS) + 1)))
    for i, s0 in enumerate(starts):
        seg = channels[s0 : s0 + win]
        seg = seg - seg.mean(axis=0)
        power = np.abs(np.fft.rfft(seg * taper[:, None], axis=0)) ** 2
        col = 0
        for ch in range(channels.shape[1]):
            for mask in masks:
                rows[i, col] = math.log(float(power[mask, ch].sum()) + 1e-12)
                col += 1
            rows[i, col] = math.log(float(power[:, ch].sum()) + 1e-12)
            col += 1
    return SpectralFeatures(times[starts + win // 2], rows)


@dataclass
class SpectralSpeedModel:
    """Ridge regression from band powers to speed."""

    coefficients: np.ndarray = field(default_factory=lambda: np.zeros(0))
    intercept: float = 0.0
    sigma_ms: float = 8.0
    fitted: bool = False
    calibration_correlation: float = 0.0
    calibration_samples: int = 0
    calibration_intervals: int = 0
    calibration_basis: str = "pointwise_gps_speed"
    prediction_smoothing_s: float = 0.0
    in_sample_sigma_ms: float = 0.0
    """Residual spread on the data the fit was computed from. Only useful as
    the denominator of the generalisation gap below."""

    generalisation_gap: float = 1.0
    """How much worse the model got from in-sample to held-out calibration.

    This is the one honest signal available at fit time about how the model
    will degrade on data it has not seen. A model whose error already doubles
    over a 43-second holdout drawn from its own 145-second window will degrade
    further over seventeen minutes of unseen driving, and `deployment_sigma_ms`
    charges for that."""

    reason: str = "not fitted"
    fallback_speed_ms: float = 8.0
    fallback_sigma_ms: float = 8.0

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        speeds: np.ndarray,
        ridge: float = 1.0,
        min_samples: int = 60,
        min_moving: int = 30,
        holdout: float = 0.3,
    ) -> "SpectralSpeedModel":
        """Fit on the visible window, and *score on a held-out tail of it*.

        The residual spread on the data a least-squares fit was computed from
        is an optimistic estimate of its error, and this measurement's sigma is
        the only thing standing between a mediocre predictor and a confidently
        wrong filter. Holding out the last 30 % of the calibration window costs
        a little accuracy and buys an honest sigma.
        """
        features = np.asarray(features, dtype=float)
        speeds = np.asarray(speeds, dtype=float)
        good = np.all(np.isfinite(features), axis=1) & np.isfinite(speeds)
        features, speeds = features[good], speeds[good]
        if len(speeds) < min_samples or int(np.sum(speeds > 2.0)) < min_moving:
            return cls(reason=f"only {len(speeds)} calibration windows, "
                              f"{int(np.sum(speeds > 2.0))} of them moving")

        split = max(min_samples // 2, int(len(speeds) * (1.0 - holdout)))
        split = min(split, len(speeds) - 5)
        train_x, train_y = features[:split], speeds[:split]
        test_x, test_y = features[split:], speeds[split:]

        def solve(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
            centre = x.mean(axis=0)
            xc = x - centre
            gram = xc.T @ xc + ridge * np.eye(xc.shape[1])
            coef = np.linalg.solve(gram, xc.T @ (y - y.mean()))
            return coef, float(y.mean() - centre @ coef)

        coef, intercept = solve(train_x, train_y)
        held = test_x @ coef + intercept
        if held.size < 5 or float(np.std(held)) < 1e-9:
            return cls(reason="prediction is constant on held-out calibration")
        residual = float(np.sqrt(np.mean((held - test_y) ** 2)))
        correlation = float(np.corrcoef(held, test_y)[0, 1])
        if not math.isfinite(correlation):
            correlation = 0.0

        in_sample = float(np.sqrt(np.mean(
            (train_x @ coef + intercept - train_y) ** 2)))
        gap = residual / max(in_sample, 1e-6)
        if not math.isfinite(gap):
            gap = 1.0

        # Refit on everything now that the error has been measured honestly.
        coef, intercept = solve(features, speeds)
        return cls(
            coefficients=coef,
            intercept=intercept,
            sigma_ms=float(np.clip(residual, 1.5, 12.0)),
            fitted=True,
            calibration_correlation=correlation,
            calibration_samples=len(speeds),
            in_sample_sigma_ms=in_sample,
            generalisation_gap=float(gap),
            reason="ok",
        )

    @classmethod
    def fit_intervals(
        cls,
        features: np.ndarray,
        feature_times: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray,
        distances_m: np.ndarray,
        ridge: float = 3.0,
        min_intervals: int = 30,
        prediction_smoothing_s: float = 15.0,
    ) -> "SpectralSpeedModel":
        """Fit sparse RFID supervision without inventing pointwise speeds.

        An RFID pair says only ``integral(v dt) = road_distance``.  A linear
        spectral model has a useful property here: its mean prediction over an
        interval is the prediction at the interval's mean feature vector.  We
        therefore regress those aggregate vectors against ``distance / time``
        and never interpolate an average interval speed into thousands of fake
        instantaneous labels.

        Columns are standardised for the solve and converted back to raw-space
        coefficients afterward.  This matters across recorder generations:
        their band-power variances differ greatly even after the log transform.
        """
        features = np.asarray(features, dtype=float)
        feature_times = np.asarray(feature_times, dtype=float)
        starts = np.asarray(starts, dtype=float)
        ends = np.asarray(ends, dtype=float)
        distances_m = np.asarray(distances_m, dtype=float)
        rows: list[np.ndarray] = []
        speeds: list[float] = []
        durations: list[float] = []
        windows = 0
        for start, end, distance in zip(starts, ends, distances_m):
            duration = float(end - start)
            if duration <= 0.0 or not math.isfinite(distance) or distance < 0.0:
                continue
            mask = (feature_times > start) & (feature_times <= end)
            mask &= np.all(np.isfinite(features), axis=1)
            count = int(mask.sum())
            if count < 3:
                continue
            rows.append(features[mask].mean(axis=0))
            speeds.append(float(distance / duration))
            durations.append(duration)
            windows += count
        if len(rows) < min_intervals:
            return cls(
                reason=f"only {len(rows)} usable RFID calibration intervals",
                calibration_basis="rfid_interval_distance",
                calibration_intervals=len(rows),
            )

        x = np.asarray(rows, dtype=float)
        y = np.asarray(speeds, dtype=float)
        duration = np.asarray(durations, dtype=float)
        split = max(min_intervals // 2, int(len(y) * 0.7))
        split = min(split, len(y) - 5)

        def solve(xx: np.ndarray, yy: np.ndarray, dd: np.ndarray) -> tuple[np.ndarray, float]:
            centre = xx.mean(axis=0)
            scale = np.maximum(xx.std(axis=0), 0.2)
            normal = (xx - centre) / scale
            weight = np.clip(dd / max(float(np.median(dd)), 1e-6), 0.25, 4.0)
            design = np.column_stack([np.ones(len(normal)), normal])
            penalty = np.diag(np.r_[0.0, np.full(normal.shape[1], ridge)])
            beta = np.linalg.solve(
                design.T @ (weight[:, None] * design) + penalty,
                design.T @ (weight * yy),
            )
            raw_coef = beta[1:] / scale
            raw_intercept = float(beta[0] - centre @ raw_coef)
            return raw_coef, raw_intercept

        coef, intercept = solve(x[:split], y[:split], duration[:split])
        held = x[split:] @ coef + intercept
        residual = float(np.sqrt(np.mean((held - y[split:]) ** 2)))
        correlation = (
            float(np.corrcoef(held, y[split:])[0, 1])
            if len(held) >= 2 and np.std(held) > 1e-9 else 0.0
        )
        if not math.isfinite(correlation):
            correlation = 0.0
        train_pred = x[:split] @ coef + intercept
        in_sample = float(np.sqrt(np.mean((train_pred - y[:split]) ** 2)))
        gap = residual / max(in_sample, 1e-6)
        coef, intercept = solve(x, y, duration)
        return cls(
            coefficients=coef,
            intercept=intercept,
            sigma_ms=float(np.clip(residual, 1.5, 12.0)),
            fitted=True,
            calibration_correlation=correlation,
            calibration_samples=windows,
            calibration_intervals=len(rows),
            calibration_basis="rfid_interval_distance",
            prediction_smoothing_s=float(prediction_smoothing_s),
            in_sample_sigma_ms=in_sample,
            generalisation_gap=float(gap) if math.isfinite(gap) else 1.0,
            reason="ok",
            fallback_speed_ms=float(np.median(y)),
        )

    @property
    def deployment_sigma_ms(self) -> float:
        """The sigma this model should be believed at *in deployment*.

        ``sigma_ms`` is the held-out calibration RMSE, and it is honest about
        exactly one thing: the error on data drawn from the same 145-second
        window as the fit. It cannot see the error on driving unlike anything
        in that window, and on these recordings that error is far larger - the
        model saturates, predicting 12.6 m/s where the truth is 21.2, an 8.6
        m/s bias that its own reported 2.85 m/s sigma denies.

        Nothing in the features, the prediction, or the fit reveals *when* that
        happens (feature leverage correlates -0.38 and +0.08 with the actual
        error; the saturated predictions sit at the same quantile of the
        calibration output distribution as the unbiased ones). So the sigma
        cannot be made conditional. What it can be is honest on average, and
        the model's own generalisation gap is the available evidence: it
        already degraded by that factor once, from in-sample to holdout, and
        deployment is a further step away.

        The factor is bounded because it is an extrapolation, not a
        measurement, and it is computed per session so a well-calibrated
        vehicle is not punished for a badly-calibrated one.
        """
        if not self.fitted:
            return self.fallback_sigma_ms
        factor = float(np.clip(self.generalisation_gap, 1.5, 3.0))
        return float(np.clip(self.sigma_ms * factor, 1.5, 20.0))

    def predict(self, row: np.ndarray) -> tuple[float, float]:
        if not self.fitted:
            return self.fallback_speed_ms, self.fallback_sigma_ms
        row = np.asarray(row, dtype=float)
        if row.shape[0] != self.coefficients.shape[0] or not np.all(np.isfinite(row)):
            return self.fallback_speed_ms, self.fallback_sigma_ms
        value = float(row @ self.coefficients + self.intercept)
        return float(np.clip(value, 0.0, 33.0)), self.sigma_ms

    def predict_many(self, rows: np.ndarray) -> np.ndarray:
        if not self.fitted or rows.size == 0:
            return np.full(len(rows), self.fallback_speed_ms)
        return np.clip(np.asarray(rows, dtype=float) @ self.coefficients + self.intercept,
                       0.0, 33.0)

    def to_json(self) -> dict[str, Any]:
        return {
            "fitted": self.fitted,
            "reason": self.reason,
            "n_features": int(self.coefficients.shape[0]),
            "sigma_ms": round(self.sigma_ms, 2),
            "in_sample_sigma_ms": round(self.in_sample_sigma_ms, 2),
            "generalisation_gap": round(self.generalisation_gap, 2),
            "deployment_sigma_ms": round(self.deployment_sigma_ms, 2),
            "holdout_correlation": round(self.calibration_correlation, 3),
            "calibration_windows": self.calibration_samples,
            "calibration_intervals": self.calibration_intervals,
            "calibration_basis": self.calibration_basis,
            "prediction_smoothing_s": round(self.prediction_smoothing_s, 1),
        }
