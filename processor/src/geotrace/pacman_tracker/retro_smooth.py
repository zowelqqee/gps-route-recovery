"""Offline retrospective reconstruction of the distance trajectory.

A bend position anchor is often found late - on the 07-26 review trip about
450 s into the outage - and it corrects the live distance in a single forward
jump. Everything the filter emitted *before* the bend still carries the full
accumulated undershoot from the saturated spectral speed source, which is where
the worst along-route position error lives (Phase 26 TASK 1: ~700 m, almost all
of it pre-bend).

This module reshapes that history *after the fact*. It is not part of the live
filter and is never called from the estimation path: it takes the already
emitted causal trace plus the accepted bend anchors and returns a *separate*
reconstructed trace. The causal trace is not modified.

What it uses: only each bend anchor's own folded correction (`delta_D_m`), the
time it was applied, and the emitted speed samples. No hidden GPS. No future
information beyond the anchor being applied. It cannot and does not change route
topology - it only moves the along-route position, exactly like the causal
anchor already did, but backward in time instead of forward.

Method: the causal distance steps up by `delta_D_m` at the apply time. Spread
that same step backward over the trajectory since the previous anchor (or the
outage start), in proportion to the time the car spent moving in each interval -
the saturated-spectral undershoot accrues roughly in proportion to distance
cruised, so moving-time is the right shape (Phase 26 TASK 4 found uniform
moving-time beats sigma- or residual-weighted redistribution, which spike the
implied speed correction). The reconstructed history then meets the
already-corrected future with no discontinuity.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Optional, Sequence

from .speed import SpeedSample

__all__ = ["retrospective_distance_smooth"]


def _applied_bend_corrections(
    bend_anchors: Sequence[dict[str, Any]],
) -> list[tuple[float, float]]:
    """`(t_applied, delta_D_m)` for every bend whose position anchor landed,
    in apply-time order. A tiny correction is ignored - it is not worth a
    reconstruction pass and only adds noise."""
    out: list[tuple[float, float]] = []
    for ba in bend_anchors:
        if not ba.get("applied_position"):
            continue
        t_ap = ba.get("t_applied", float("nan"))
        delta = float(ba.get("delta_D_m", 0.0))
        if not math.isfinite(t_ap) or abs(delta) < 1.0:
            continue
        out.append((float(t_ap), delta))
    out.sort(key=lambda p: p[0])
    return out


def retrospective_distance_smooth(
    speed_trace: Sequence[SpeedSample],
    bend_anchors: Sequence[dict[str, Any]],
    *,
    moving_speed_ms: float = 2.0,
) -> Optional[list[SpeedSample]]:
    """Return a reconstructed copy of ``speed_trace`` with each applied bend
    anchor's forward jump redistributed backward over the preceding trajectory.

    Returns ``None`` when there is nothing to do (no applied bend anchor, or an
    unusable trace), so the caller can simply skip attaching a retro trace.
    ``distance_m`` and ``speed_ms`` are rewritten; the covariance fields are
    copied through unchanged (they describe the causal estimate and a
    reconstruction does not earn tighter ones).
    """
    corrections = _applied_bend_corrections(bend_anchors)
    if not corrections or len(speed_trace) < 3:
        return None

    samples = list(speed_trace)
    t = [float(s.t) for s in samples]
    dist = [float(s.distance_m) for s in samples]
    spd = [float(s.speed_ms) for s in samples]
    n = len(samples)

    # per-sample time step (leading step mirrors the first gap)
    dt = [0.0] * n
    for i in range(1, n):
        dt[i] = max(0.0, min(2.0, t[i] - t[i - 1]))
    dt[0] = dt[1] if n > 1 else 0.0

    d_add = [0.0] * n
    v_add = [0.0] * n
    boundary = t[0]
    for t_ap, delta in corrections:
        idx = [i for i in range(n)
               if boundary < t[i] < t_ap and spd[i] > moving_speed_ms]
        wsum = sum(dt[i] for i in idx)
        if wsum <= 0.0:
            boundary = t_ap
            continue
        # this correction's own distance ramp: rises 0 -> delta across its
        # moving samples, then collapses to 0 once the causal jump takes over
        run = 0.0
        for i in range(n):
            if boundary < t[i] < t_ap and spd[i] > moving_speed_ms:
                share = delta * dt[i] / wsum
                v_add[i] += share / dt[i] if dt[i] > 0 else 0.0
                run += share
            d_add[i] += run if t[i] < t_ap else 0.0
        boundary = t_ap

    out: list[SpeedSample] = []
    for i, s in enumerate(samples):
        out.append(replace(s,
                           distance_m=dist[i] + d_add[i],
                           speed_ms=spd[i] + v_add[i]))
    return out
