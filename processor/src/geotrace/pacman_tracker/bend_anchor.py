"""Road curvature as a GPS-free intra-edge position anchor.

A gyro heading-change event is usually a junction turn - a sharp, compact swing
selecting one successor edge at a node. But a long curving street polyline
produces the same integrated yaw without any node: on 07-26 event ev3 (+42 deg
over 11 s, peak rate 0.10 rad/s) is the car following the bend of a single
1039 m edge, and the junction matcher, forced to bind it to *some* node, planted
it ~740 m too early and shortened the committed map interval by exactly that.

When the active edge is already committed (topology fixed by earlier, sharper
turns), the shape of that edge's own curvature profile can say *where along the
edge* the bend event happened - an absolute along-edge position, from map
geometry alone, needing no node and no GPS.

This module is standalone: nothing in the tracker imports it yet. It is the
prototype for `docs/SPECTRAL_CALIBRATION_FORENSICS.md` Phase 17.

Method
------
The gyro gives ``psi_gyro(t)`` (cumulative yaw over the event window). The edge
gives ``psi_map(s)`` (cumulative heading along its polyline). Speed is unknown,
so time and arc length are related by an unknown scale; we search a bounded
affine map ``s(t) = s0 + v_bar * (t - t0)`` and compare the two heading-change
curves by shape (RMS angle residual + rate correlation), not just total angle -
a long edge can contain several bends of similar total angle. A match is
returned only if it is unique: the best ``s0`` must beat every other ``s0``
more than ``min_margin`` away from it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["BendMatch", "match_curvature_event", "classify_event"]


@dataclass(frozen=True)
class BendMatch:
    """Where a curvature event sits on one edge, from geometry + gyro only."""

    s_start: float
    s_peak: float
    s_end: float
    v_bar: float
    map_delta_psi: float
    gyro_delta_psi: float
    angle_resid_rad: float
    """RMS of ``psi_map(s(t)) - psi_gyro(t)`` over the window."""
    shape_corr: float
    """Correlation of the two yaw-*rate* profiles; 1.0 is a perfect shape."""
    margin: float
    """best score minus the best score at any ``s0`` more than ``separation_m``
    away. A lone gentle bend on a simple edge sits near 0.8; 07-26 ev3 on the
    curvature-rich edge 4326 scores ~1.5."""
    n_local_maxima: int

    @property
    def total_angle_resid_rad(self) -> float:
        return abs(self.map_delta_psi - self.gyro_delta_psi)


def _cum_yaw(t: np.ndarray, rate: np.ndarray) -> np.ndarray:
    out = np.concatenate([[0.0], np.cumsum(rate[:-1] * np.diff(t))])
    return out - out[0]


def match_curvature_event(
    edge_s: np.ndarray,
    edge_heading: np.ndarray,
    edge_length: float,
    gyro_t: np.ndarray,
    gyro_rate: np.ndarray,
    t_peak: float,
    *,
    vbar_bounds: tuple[float, float] = (6.0, 28.0),
    vbar_step: float = 0.5,
    s0_step: float = 4.0,
    separation_m: float = 60.0,
    min_margin: float = 0.7,
    max_angle_resid_rad: float = math.radians(6.0),
    min_shape_corr: float = 0.55,
    max_total_resid_rad: float = math.radians(12.0),
) -> Optional[BendMatch]:
    """Best along-edge position for a heading-change event, or ``None``.

    ``edge_s`` / ``edge_heading`` are a monotone arc-length grid and the
    smoothed unwrapped heading on it (``RoadGeometry.profile``). ``gyro_t`` /
    ``gyro_rate`` are the bias-removed yaw rate over the event window (a few
    seconds of margin each side). No distance / D prior is taken - the search
    is over the whole edge.
    """
    gyro_t = np.asarray(gyro_t, dtype=float)
    gyro_rate = np.asarray(gyro_rate, dtype=float)
    if gyro_t.size < 5 or edge_length <= s0_step:
        return None
    t_rel = gyro_t - gyro_t[0]
    psi_g = _cum_yaw(gyro_t, gyro_rate)
    rate_g = np.gradient(psi_g, t_rel)
    gyro_total = float(np.sum(gyro_rate[:-1] * np.diff(gyro_t)))

    def psi_map(s: np.ndarray) -> np.ndarray:
        return np.interp(np.clip(s, 0.0, edge_length), edge_s, edge_heading)

    vbars = np.arange(vbar_bounds[0], vbar_bounds[1] + 1e-9, vbar_step)
    s0s = np.arange(0.0, edge_length - s0_step, s0_step)
    per_s0 = np.full(s0s.size, -1e18)
    best = (-1e18, 0.0, 0.0)  # score, s0, vbar

    for i, s0 in enumerate(s0s):
        for vb in vbars:
            traj = s0 + vb * t_rel
            if traj[-1] > edge_length or traj[0] < 0.0:
                continue
            pm = psi_map(traj)
            pm = pm - pm[0]
            resid = float(np.sqrt(np.mean((pm - psi_g) ** 2)))
            total = abs(float(pm[-1] - psi_g[-1]))
            rate_m = np.gradient(pm, t_rel)
            if np.std(rate_m) < 1e-6 or np.std(rate_g) < 1e-6:
                corr = 0.0
            else:
                corr = float(np.corrcoef(rate_m, rate_g)[0, 1])
            score = -(resid / 0.06) - 0.5 * (total / 0.10) + 1.5 * corr
            if score > per_s0[i]:
                per_s0[i] = score
            if score > best[0]:
                best = (score, float(s0), float(vb))

    score, s0b, vbb = best
    if not np.isfinite(score):
        return None

    order = np.argsort(per_s0)[::-1]
    margin = math.inf
    for k in order[1:]:
        if abs(s0s[k] - s0s[order[0]]) > separation_m:
            margin = float(per_s0[order[0]] - per_s0[k])
            break
    n_lm = int(sum(
        1 for j in range(1, s0s.size - 1)
        if per_s0[j] >= per_s0[j - 1] and per_s0[j] >= per_s0[j + 1]
        and per_s0[j] > per_s0.max() - 3.0
    ))

    traj = s0b + vbb * t_rel
    pm = psi_map(traj)
    pm = pm - pm[0]
    resid = float(np.sqrt(np.mean((pm - psi_g) ** 2)))
    rate_m = np.gradient(pm, t_rel)
    corr = (0.0 if (np.std(rate_m) < 1e-6 or np.std(rate_g) < 1e-6)
            else float(np.corrcoef(rate_m, rate_g)[0, 1]))
    map_dpsi = float(pm[-1])

    if (resid > max_angle_resid_rad or corr < min_shape_corr
            or margin < min_margin or abs(map_dpsi - gyro_total) > max_total_resid_rad):
        return None

    return BendMatch(
        s_start=s0b,
        s_peak=float(s0b + vbb * (t_peak - gyro_t[0])),
        s_end=float(traj[-1]),
        v_bar=vbb,
        map_delta_psi=map_dpsi,
        gyro_delta_psi=gyro_total,
        angle_resid_rad=resid,
        shape_corr=corr,
        margin=margin,
        n_local_maxima=n_lm,
    )


def classify_event(
    delta_psi_rad: float,
    peak_rate_rads: float,
    duration_s: float,
    bend: Optional[BendMatch],
    junction_angle_gap_deg: float,
    *,
    junction_gap_ok_deg: float = 18.0,
    junction_peak_min_rads: float = 0.18,
    junction_compact_min: float = 0.75,
    bend_peak_max_rads: float = 0.16,
    bend_duration_min_s: float = 6.0,
) -> str:
    """``"junction"`` / ``"bend"`` / ``"ambiguous"`` / ``"reject"``.

    Rules-first. ``junction_angle_gap_deg`` is ``||map turn| - |gyro turn||`` for
    the best compatible successor at the nearby node. ``bend`` is the result of
    :func:`match_curvature_event` on the committed active edge (or ``None``).
    Thresholds are shape/rate scales, not tuned to any one event.
    """
    compact = abs(delta_psi_rad) / max(peak_rate_rads * duration_s, 1e-6)
    junction_ok = junction_angle_gap_deg < junction_gap_ok_deg
    bend_ok = bend is not None
    peaky = peak_rate_rads > junction_peak_min_rads and compact > junction_compact_min
    spread = peak_rate_rads < bend_peak_max_rads and duration_s > bend_duration_min_s

    if peaky and junction_ok and not (spread and bend_ok):
        return "junction"
    if spread and bend_ok and not peaky:
        return "bend"
    if bend_ok and junction_ok:
        return "ambiguous"
    if bend_ok:
        return "bend"
    if junction_ok:
        return "junction"
    return "reject"
