"""Zero-latency corrected *display* position for the Pacman marker.

This is the Phase 32 result made into a runtime component. It maintains a second
longitudinal state ``D_position`` that is allowed to use a more aggressive
spectral-saturation speed correction than the conservative route odometer
``D_route``. ``D_position`` is used **only** to place the displayed vehicle
marker along the route the baseline tracker has *already* committed.

Hard architectural rule (docs/DISPLAY_ODOMETRY_SPLIT.md): there is **no feedback
path** from anything in this module back into the tracker. ``D_position`` cannot
influence edge choice, branch weights, junction timing, commits, rollback,
pruning, turn/bend recognition, PF weighting or route-hypothesis selection. The
committed route and ``speed_trace`` are byte-identical whether or not this runs.

The estimator ("iso-binary") and its coefficients are frozen in
``data/display_residual_iso.json`` and are never retrained here:

    residual_hat = clip( isotonic( mean_3s(v_prior - v_spectral) ), -2, +12 )
    gate ON  iff  frac_30s(v_spectral > 15) > 0.02  or  max_20s(v_spectral) > 15
    v_position = v_route + 1.5 * gate * max(0, (v_spectral + residual_hat) - v_route)
    D_position(t) = D_route(t) + clip(integral(v_position - v_route) dt, 0, 300 m)

``D_position`` may only move along the committed edge chain. If it reaches the
committed frontier before the route commits a continuation, the marker is held
at the frontier and the surplus is reported as ``excess_position_distance``;
it is released automatically once the route commits the next edge (the frontier
moves and the accumulated integral lands on the new geometry).
"""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

_ARTIFACT = Path(__file__).with_name("data") / "display_residual_iso.json"


@dataclass
class DisplayResidualModel:
    """Frozen isotonic residual model + binary saturation gate."""

    iso_x: np.ndarray
    iso_y: np.ndarray
    resid_clip: tuple[float, float]
    resid_feature_window_s: float
    gate_frac_level_ms: float
    gate_frac_window_s: float
    gate_frac_threshold: float
    gate_max_window_s: float
    gate_max_threshold_ms: float
    estimator: str = "iso-binary"
    source: str = ""

    @classmethod
    def load(cls, path: Optional[Path] = None, *, leave_0726_out: bool = False
             ) -> "DisplayResidualModel":
        art = json.loads(Path(path or _ARTIFACT).read_text(encoding="utf-8"))
        gx = "isotonic_grid_x_leave_0726_out" if leave_0726_out else "isotonic_grid_x"
        gy = "isotonic_grid_y_leave_0726_out" if leave_0726_out else "isotonic_grid_y"
        g = art["gate"]
        return cls(
            iso_x=np.asarray(art[gx], dtype=float),
            iso_y=np.asarray(art[gy], dtype=float),
            resid_clip=tuple(art["residual_clip_ms"]),
            resid_feature_window_s=float(art["residual_feature_window_s"]),
            gate_frac_level_ms=float(g["frac_above_level_ms"]),
            gate_frac_window_s=float(g["frac_window_s"]),
            gate_frac_threshold=float(g["frac_threshold"]),
            gate_max_window_s=float(g["max_window_s"]),
            gate_max_threshold_ms=float(g["max_threshold_ms"]),
            estimator=art.get("estimator", "iso-binary"),
            source=str(path or _ARTIFACT),
        )

    def residual_hat(self, vprior_minus_vspec: float) -> float:
        if not math.isfinite(vprior_minus_vspec):
            return 0.0
        y = float(np.interp(vprior_minus_vspec, self.iso_x, self.iso_y))
        return float(np.clip(y, self.resid_clip[0], self.resid_clip[1]))

    def gate(self, frac_above: float, max_recent: float) -> bool:
        return bool(frac_above > self.gate_frac_threshold
                    or max_recent > self.gate_max_threshold_ms)


@dataclass
class DisplayPositionSample:
    """What the display branch reports at one output tick."""

    t: float
    distance_m: float                 # D_position
    route_distance_m: float            # D_route (for reference)
    delta_m: float                     # D_position - D_route (bounded only by the gate)
    excess_position_distance_m: float  # surplus held at the committed frontier
    gate_active: bool
    v_route_ms: float
    v_position_ms: float
    edge: int
    s_m: float                         # distance along ``edge``
    lat: float
    lon: float
    at_frontier: bool
    source: str = "display:iso-binary"

    def to_json(self) -> dict[str, Any]:
        return {
            "t": round(self.t, 3),
            "distance_m": round(self.distance_m, 2),
            "route_distance_m": round(self.route_distance_m, 2),
            "delta_m": round(self.delta_m, 2),
            "excess_position_distance_m": round(self.excess_position_distance_m, 2),
            "gate_active": bool(self.gate_active),
            "v_route_ms": round(self.v_route_ms, 3),
            "v_position_ms": round(self.v_position_ms, 3),
            "edge": int(self.edge),
            "s_m": round(self.s_m, 2),
            "lat": round(self.lat, 7),
            "lon": round(self.lon, 7),
            "at_frontier": bool(self.at_frontier),
            "source": self.source,
        }


class DisplayPositionBranch:
    """Runs alongside the tracker; never feeds anything back into it."""

    def __init__(self, model: DisplayResidualModel, d0: float, frame: Any,
                 edges: Any, correction_gain: float = 1.0,
                 max_correction_m: float = float("inf")) -> None:
        self._m = model
        self._frame = frame
        self._edges = edges
        self._correction_gain = max(0.0, float(correction_gain))
        self._max_correction = max(0.0, float(max_correction_m))
        self._delta = 0.0
        self._d_route_seen = float(d0)
        self._v_route = 0.0
        self._v_position = 0.0
        self._gate = False
        self._excess = 0.0
        # Trailing feature samples on a fixed 0.5 s cadence - the grid the Phase
        # 32 estimator was defined on. Sampling the (noisy) v_prior/v_spectral
        # signals at the raw IMU rate instead changes the rolling-window means
        # enough to shift the result; decimating here keeps the runtime faithful
        # to the frozen diagnostic.
        self._grid_dt = 0.5
        self._next_sample_t = float("nan")
        self._buf: deque[tuple[float, float, float]] = deque()  # (t, v_prior-v_spec, v_spec)
        self._window = max(model.resid_feature_window_s, model.gate_frac_window_s,
                           model.gate_max_window_s) + 1.0

    # ------------------------------------------------------------- per IMU step
    def step(self, t: float, dt: float, v_route: float, v_prior: float,
             v_spectral: float, d_route: float, stationary: bool) -> None:
        """Advance the display velocity/distance by one shared IMU step.

        ``v_prior`` is the route EKF speed *before* the spectral update (the
        IMU/lateral prior); ``v_route`` is the fused route speed after it.
        """
        self._d_route_seen = float(d_route)
        self._v_route = float(v_route)
        has_spec = math.isfinite(v_spectral)
        if math.isnan(self._next_sample_t):
            self._next_sample_t = float(t)
        if has_spec and t >= self._next_sample_t:
            self._buf.append((t, float(v_prior) - float(v_spectral), float(v_spectral)))
            while t - self._next_sample_t >= 0.0:
                self._next_sample_t += self._grid_dt
            while self._buf and t - self._buf[0][0] > self._window:
                self._buf.popleft()

        dv = 0.0
        self._gate = False
        # NB: no stationary check - the frozen Phase 32 estimator applies the
        # correction purely on its observable gate (the ~20 s ``v_spectral``
        # memory keeps it active briefly into a stop). Adding a stationary veto
        # here changes the proven estimator (07-26 median 57 -> ~95 m).
        if has_spec and self._buf:
            ts = np.fromiter((s[0] for s in self._buf), dtype=float)
            diff = np.fromiter((s[1] for s in self._buf), dtype=float)
            spec = np.fromiter((s[2] for s in self._buf), dtype=float)
            win = ts >= t - self._m.resid_feature_window_s
            f_resid = diff[win]
            f_spec = spec[win]
            f_frac = spec[ts >= t - self._m.gate_frac_window_s]
            f_max = spec[ts >= t - self._m.gate_max_window_s]
            # Phase 32 features are window MEANS on a 0.5 s grid: the residual
            # feature is mean(v_prior - v_spectral) and v_ml is built from
            # mean(v_spectral) over the same trailing window, not the noisy
            # instantaneous value.
            feat = float(np.mean(f_resid)) if f_resid.size else float(diff[-1])
            vspec_mean = float(np.mean(f_spec)) if f_spec.size else float(v_spectral)
            frac_above = (float(np.mean(f_frac > self._m.gate_frac_level_ms))
                          if f_frac.size else 0.0)
            max_recent = float(np.max(f_max)) if f_max.size else 0.0
            self._gate = self._m.gate(frac_above, max_recent)
            if self._gate:
                v_ml = float(np.clip(vspec_mean + self._m.residual_hat(feat), 0.0, 33.0))
                dv = max(0.0, v_ml - float(v_route))
        requested_dv = self._correction_gain * dv
        previous_delta = self._delta
        self._delta = min(self._max_correction,
                          previous_delta + requested_dv * float(dt))
        applied_dv = ((self._delta - previous_delta) / float(dt)
                      if dt > 0.0 else 0.0)
        self._v_position = float(v_route) + applied_dv

    # --------------------------------------------------------- per output tick
    def locate(self, t: float, route_edges: list[int], edge_index: int,
               route_offset: float, offset_bias: float) -> DisplayPositionSample:
        """Project ``D_position`` onto the *already committed* edge chain."""
        d_route = self._d_route_seen
        d_position = d_route + self._delta
        # arc coordinate from the route start, in the odometer's own frame
        l_route = d_route - offset_bias
        l_disp = l_route + self._delta

        # The committed chain can already extend a few edges past the edge the
        # route odometer currently sits on - the single-path manager commits the
        # continuation as each junction is decided. The display marker may travel
        # to the END of that whole committed chain, but not one edge further.
        chain = list(route_edges) if route_edges else [edge_index]
        try:
            k = len(chain) - 1 - chain[::-1].index(edge_index)
        except ValueError:
            chain = chain + [edge_index]
            k = len(chain) - 1
        n = len(chain)
        offs = [0.0] * n
        offs[k] = float(route_offset)                     # anchor on the current edge
        for j in range(k + 1, n):
            offs[j] = offs[j - 1] + float(self._edges[chain[j - 1]].length)
        for j in range(k - 1, -1, -1):
            offs[j] = offs[j + 1] - float(self._edges[chain[j]].length)
        frontier = offs[-1] + float(self._edges[chain[-1]].length)

        excess = 0.0
        at_frontier = False
        if l_disp >= frontier:
            excess = l_disp - frontier
            at_frontier = True
            place_edge, place_s = chain[-1], float(self._edges[chain[-1]].length)
        elif l_disp <= offs[0]:
            place_edge, place_s = chain[0], 0.0
        else:
            place_edge, place_s = chain[-1], l_disp - offs[-1]
            for j in range(n):
                e = chain[j]
                elen = float(self._edges[e].length)
                if offs[j] <= l_disp < offs[j] + elen:
                    place_edge, place_s = e, l_disp - offs[j]
                    break
        self._excess = excess
        x, y = self._edges[place_edge].position(float(np.clip(place_s, 0.0,
                                                self._edges[place_edge].length)))
        lat, lon = self._frame.to_geo(x, y)
        return DisplayPositionSample(
            t=float(t), distance_m=d_position, route_distance_m=d_route,
            delta_m=self._delta, excess_position_distance_m=excess,
            gate_active=self._gate, v_route_ms=self._v_route,
            v_position_ms=self._v_position, edge=int(place_edge), s_m=float(place_s),
            lat=float(lat), lon=float(lon), at_frontier=at_frontier)
