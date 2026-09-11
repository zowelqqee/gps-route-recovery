"""Arc-length heading / curvature profile of a road network.

The tracker never asks "where is this point on the map"; it asks "what yaw rate
would the map produce if I were here, going this fast". That question needs one
thing the plain :class:`~geotrace.road_graph.RoadNetwork` does not expose:
curvature as a function of distance along an edge, sampled densely enough to be
differentiated and cheap enough to evaluate for a whole beam at once.

Two facts drive the construction.

*Corners are not vertices.* OSM stores a street corner as a single polyline
vertex, i.e. a heading step. A car takes that corner over 10-25 m. Smoothing the
heading over ``heading_smooth_m`` before differentiating turns the step into a
finite curvature hump whose integral is still exactly the turn angle, which is
the quantity the window matcher compares against the integrated gyro.

*The biggest turn of the trip belongs to no edge at all.* Turning from one
street into another is a heading step at a graph **node**; neither edge's own
polyline contains it. It is therefore not baked into this index. The tracker
injects it per hypothesis when the hypothesis crosses the node
(:func:`junction_turn`), because only the hypothesis knows which way it went.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from geotrace.coordinates import wrap_angle
from geotrace.pacman_tracker.config import GeometryConfig
from geotrace.road_graph import RoadNetwork

CURVATURE_EPS = 1e-9
"""rad/m below which a curvature is numerical noise, not geometry."""


def _gaussian_kernel(sigma_samples: float) -> np.ndarray:
    radius = max(1, int(math.ceil(3.0 * sigma_samples)))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / max(sigma_samples, 1e-6)) ** 2)
    return k / k.sum()


def _smooth_reflect(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve with edge-reflecting padding.

    Reflection rather than zero padding: heading is an angle, and padding it
    with zeros would invent a turn at both ends of every edge.
    """
    radius = (len(kernel) - 1) // 2
    if len(values) == 0:
        return values
    if radius == 0:
        return values.copy()
    pad = min(radius, len(values) - 1) if len(values) > 1 else 0
    if pad <= 0:
        return np.full_like(values, values.mean())
    left = 2.0 * values[0] - values[pad:0:-1]
    right = 2.0 * values[-1] - values[-2 : -pad - 2 : -1]
    if len(left) < radius:
        left = np.concatenate([np.full(radius - len(left), left[0] if len(left) else values[0]), left])
    if len(right) < radius:
        right = np.concatenate([right, np.full(radius - len(right), right[-1] if len(right) else values[-1])])
    padded = np.concatenate([left, values, right])
    return np.convolve(padded, kernel, mode="valid")


@dataclass
class EdgeProfile:
    """Sampled geometry of one edge. Kept for tests and diagnostics."""

    index: int
    length: float
    s: np.ndarray
    heading: np.ndarray
    kappa: np.ndarray
    kappa_grad: np.ndarray
    sigma_kappa: np.ndarray


class RoadGeometry:
    """Flat, vectorised (edge, s) -> heading / curvature index.

    Every edge is resampled onto a uniform ``sample_ds_m`` grid and all grids
    are concatenated into single arrays, so a lookup for N hypotheses spread
    over N different edges is one ``take`` and not a Python loop.
    """

    def __init__(self, network: RoadNetwork, cfg: Optional[GeometryConfig] = None) -> None:
        self.network = network
        self.cfg = cfg or GeometryConfig()
        self._build()

    # ------------------------------------------------------------------ build

    def _build(self) -> None:
        cfg = self.cfg
        edges = self.network.edges
        n_edges = len(edges)
        ds = float(cfg.sample_ds_m)

        lengths = np.array([e.length for e in edges], dtype=float)
        counts = np.maximum(2, np.ceil(lengths / ds).astype(np.int64) + 1)
        offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
        total = int(counts.sum())

        s_flat = np.empty(total, dtype=float)
        heading_flat = np.empty(total, dtype=float)
        kappa_flat = np.empty(total, dtype=float)
        grad_flat = np.empty(total, dtype=float)
        sigma_flat = np.empty(total, dtype=float)

        kernel = _gaussian_kernel(max(cfg.heading_smooth_m / ds, 0.5))

        for i, edge in enumerate(edges):
            n = int(counts[i])
            lo = int(offsets[i])
            s = np.linspace(0.0, edge.length, n)
            step = s[1] - s[0] if n > 1 else edge.length

            # Raw heading at each sample: the bearing of the polyline segment
            # the sample falls in. searchsorted on the edge's own cumulative
            # distances is exact and needs no interpolation of an angle.
            seg = np.clip(
                np.searchsorted(edge.cumulative, s, side="right") - 1,
                0,
                len(edge.bearings) - 1,
            )
            raw = edge.bearings[seg]
            # Unwrap so smoothing and differentiation see a continuous curve.
            unwrapped = np.unwrap(raw)
            smooth = _smooth_reflect(unwrapped, kernel) if n > 2 else unwrapped

            if n > 2 and step > 1e-9:
                kappa = np.gradient(smooth, step)
                grad = np.gradient(kappa, step)
            else:
                kappa = np.zeros(n)
                grad = np.zeros(n)

            # How badly the smoothed profile represents the polyline locally.
            # A gentle arc has a tiny residual; a staircase of survey noise has
            # a large one and its curvature deserves to be distrusted.
            resid = np.abs(unwrapped - smooth)
            rough = _smooth_reflect(resid, kernel) if n > 2 else resid
            rough[rough < CURVATURE_EPS] = 0.0
            sigma = cfg.curvature_sigma_floor + cfg.curvature_sigma_scale * rough / max(
                cfg.heading_smooth_m, 1e-6
            )

            # Convolution leaves ~1e-12 rad/m of floating-point residue on a
            # perfectly straight road. A curvature that small is a radius of
            # 10^9 km; it is numerical noise, and leaving it in would let a
            # straight road slowly separate hypotheses that are by construction
            # indistinguishable. Snap it to exactly zero.
            kappa[np.abs(kappa) < CURVATURE_EPS] = 0.0
            grad[np.abs(grad) < CURVATURE_EPS] = 0.0

            s_flat[lo : lo + n] = s
            heading_flat[lo : lo + n] = smooth
            kappa_flat[lo : lo + n] = kappa
            grad_flat[lo : lo + n] = grad
            sigma_flat[lo : lo + n] = sigma

        self.n_edges = n_edges
        self.lengths = lengths
        self._counts = counts
        self._offsets = offsets
        self._ds = ds
        self.s_flat = s_flat
        self.heading_flat = heading_flat
        self.kappa_flat = kappa_flat
        self.kappa_grad_flat = grad_flat
        self.sigma_kappa_flat = sigma_flat

        # Sharp-corner detector, used only to report where geometry is poor.
        self.max_abs_kappa = np.array(
            [np.abs(kappa_flat[offsets[i] : offsets[i] + counts[i]]).max() for i in range(n_edges)]
        )
        self._build_topology()

    def _build_topology(self) -> None:
        """Predecessors and per-edge end/start bearings for junction turns."""
        net = self.network
        self.start_bearing = np.array([float(e.bearings[0]) for e in net.edges])
        self.end_bearing = np.array([float(e.bearings[-1]) for e in net.edges])
        # An edge P precedes edge C when P.v == C.u.
        self.in_edges: dict[int, list[int]] = {}
        by_head: dict[object, list[int]] = {}
        for edge in net.edges:
            by_head.setdefault(edge.v, []).append(edge.index)
        for edge in net.edges:
            self.in_edges[edge.index] = list(by_head.get(edge.u, ()))

    # ---------------------------------------------------------------- queries

    def profile(self, edge_index: int) -> EdgeProfile:
        lo = int(self._offsets[edge_index])
        n = int(self._counts[edge_index])
        sl = slice(lo, lo + n)
        return EdgeProfile(
            index=int(edge_index),
            length=float(self.lengths[edge_index]),
            s=self.s_flat[sl],
            heading=self.heading_flat[sl],
            kappa=self.kappa_flat[sl],
            kappa_grad=self.kappa_grad_flat[sl],
            sigma_kappa=self.sigma_kappa_flat[sl],
        )

    def _sample_index(self, edge_idx: np.ndarray, s: np.ndarray) -> np.ndarray:
        edge_idx = np.asarray(edge_idx, dtype=np.int64)
        s = np.asarray(s, dtype=float)
        n = self._counts[edge_idx]
        step = np.where(n > 1, self.lengths[edge_idx] / np.maximum(n - 1, 1), 1.0)
        k = np.rint(np.clip(s, 0.0, self.lengths[edge_idx]) / np.maximum(step, 1e-9))
        k = np.clip(k, 0, n - 1).astype(np.int64)
        return self._offsets[edge_idx] + k

    def curvature(self, edge_idx: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised (edge, s) -> (kappa, dkappa/ds, sigma_kappa)."""
        idx = self._sample_index(edge_idx, s)
        return (
            self.kappa_flat[idx],
            self.kappa_grad_flat[idx],
            self.sigma_kappa_flat[idx],
        )

    def heading(self, edge_idx: np.ndarray, s: np.ndarray) -> np.ndarray:
        idx = self._sample_index(edge_idx, s)
        return wrap_angle(self.heading_flat[idx])

    def junction_turn(self, from_edge: int, to_edge: int) -> float:
        """Heading step taken when leaving ``from_edge`` for ``to_edge``.

        This is signed and wrapped: positive is a left turn in the local frame
        (headings are CCW from +E). It exists only at the node, which is why it
        is not part of either edge's sampled profile.
        """
        return float(wrap_angle(self.start_bearing[to_edge] - self.end_bearing[from_edge]))

    def successors(self, edge_index: int, history: tuple[int, ...] = ()) -> list[int]:
        hist = history or (int(edge_index),)
        return [int(i) for i in self.network.allowed_successors(int(edge_index), hist)]

    def predecessors(self, edge_index: int) -> list[int]:
        return list(self.in_edges.get(int(edge_index), ()))
