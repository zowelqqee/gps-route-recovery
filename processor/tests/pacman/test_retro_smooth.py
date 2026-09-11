"""Offline retrospective distance reconstruction (Phase 26 TASK 4).

Gated by ``single_path.retro_bend_smoothing_enabled`` (OFF by default); it only
reshapes the emitted trace after the run and never touches the live filter.
"""
import inspect
import math

import pytest

from geotrace.pacman_tracker.config import PacmanConfig, SinglePathConfig
from geotrace.pacman_tracker.retro_smooth import retrospective_distance_smooth
from geotrace.pacman_tracker.speed import SpeedSample
from geotrace.pacman_tracker.tracker import TrackerResult


def _sample(t, d, v):
    return SpeedSample(t=t, distance_m=d, sigma_distance_m=5.0, speed_ms=v,
                       sigma_speed_ms=1.0, accel_bias_ms2=0.0, gyro_bias_rads=0.0)


def _trace(vs, dt=1.0):
    """Constant-speed-segment trace: vs is a list of per-second speeds."""
    out, d, t = [], 0.0, 0.0
    for v in vs:
        out.append(_sample(t, d, v))
        d += v * dt
        t += dt
    out.append(_sample(t, d, vs[-1]))
    return out


def _bend(t_applied, delta, t_peak=None):
    return {"applied_position": True, "t_applied": float(t_applied),
            "delta_D_m": float(delta),
            "t_peak": float(t_applied if t_peak is None else t_peak)}


def test_no_applied_bend_anchor_returns_none():
    tr = _trace([10.0] * 20)
    assert retrospective_distance_smooth(tr, []) is None
    assert retrospective_distance_smooth(tr, [{"applied_position": False}]) is None


def test_tiny_correction_is_ignored():
    tr = _trace([10.0] * 20)
    assert retrospective_distance_smooth(tr, [_bend(10.0, 0.4)]) is None


def test_backward_spread_preserves_the_endpoint_and_pulls_history_forward():
    # 20 s of cruising, then a bend at t=15 folds +120 m forward.
    tr = _trace([10.0] * 20)
    causal_end = tr[-1].distance_m
    out = retrospective_distance_smooth(tr, [_bend(15.0, 120.0)])
    assert out is not None and len(out) == len(tr)
    # endpoint distance unchanged - only the shape of the history moved
    assert out[-1].distance_m == pytest.approx(causal_end)
    # every pre-apply sample is lifted toward truth, and never past +delta
    for a, b in zip(tr, out):
        if a.t < 15.0:
            assert 0.0 <= b.distance_m - a.distance_m <= 120.0 + 1e-6
    # correction is monotone non-decreasing up to the apply time
    corr = [b.distance_m - a.distance_m for a, b in zip(tr, out) if a.t < 15.0]
    assert all(y >= x - 1e-9 for x, y in zip(corr, corr[1:]))
    assert corr[-1] == pytest.approx(120.0, abs=1.0)


def test_reconstruction_is_continuous_across_the_apply_time():
    tr = _trace([10.0] * 20)
    out = retrospective_distance_smooth(tr, [_bend(15.0, 120.0)])
    # after the apply time the causal jump already carries the correction,
    # so the reconstructed trace must not add it a second time
    after = [b.distance_m - a.distance_m for a, b in zip(tr, out) if a.t >= 15.0]
    assert all(abs(x) < 1e-6 for x in after)


def test_only_moving_samples_absorb_the_correction():
    # stationary for the first 10 s, then moving
    tr = _trace([0.0] * 10 + [12.0] * 10)
    out = retrospective_distance_smooth(tr, [_bend(20.0, 90.0)])
    for a, b in zip(tr, out):
        if a.t < 10.0:
            assert b.distance_m == pytest.approx(a.distance_m)      # untouched
            assert b.speed_ms == pytest.approx(a.speed_ms)
    moved = [b.speed_ms - a.speed_ms for a, b in zip(tr, out) if 10.0 <= a.t < 20.0]
    assert all(x > 0.0 for x in moved)


def test_speed_corrections_integrate_to_the_folded_distance():
    tr = _trace([9.0] * 12 + [15.0] * 8)
    delta = 100.0
    out = retrospective_distance_smooth(tr, [_bend(20.0, delta)])
    dv_dt = sum((b.speed_ms - a.speed_ms) * 1.0
                for a, b in zip(tr, out) if a.t < 20.0)
    assert dv_dt == pytest.approx(delta, abs=1.0)


def test_covariance_fields_pass_through_unchanged():
    tr = _trace([10.0] * 20)
    out = retrospective_distance_smooth(tr, [_bend(15.0, 120.0)])
    for a, b in zip(tr, out):
        assert b.sigma_distance_m == a.sigma_distance_m
        assert b.sigma_speed_ms == a.sigma_speed_ms


def test_gated_off_by_default():
    assert SinglePathConfig().retro_bend_smoothing_enabled is False
    assert PacmanConfig().single_path.retro_bend_smoothing_enabled is False
    assert TrackerResult(frames=[], final=None, stats={},
                         inputs=None).retro_speed_trace is None


def test_no_hidden_gps_tokens_in_the_retro_path():
    src = inspect.getsource(retrospective_distance_smooth)
    src += inspect.getsource(
        inspect.getmodule(retrospective_distance_smooth))
    for banned in ("reference_location", "withheld", "oracle", "truth",
                   "map_match", "position_at", "edge_at"):
        assert banned not in src
