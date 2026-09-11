"""Display corridors built straight from the along-road uncertainty.

A corridor is ``[s - k*sigma_s, s + k*sigma_s]`` walked along the hypothesis's
own route and buffered by a few metres sideways. That is the whole construction.

What it deliberately is *not*: a circle of radius ``k*sigma`` in the plane. A
free-space covariance ellipse drawn around a road-locked belief is dishonest in
both directions at once - it claims the car might be in the courtyard of the
block next door, which the model never believed, while smearing away the one
thing the model does know, which is *which street*. When the along-road
uncertainty genuinely becomes too large to be useful, the answer is to say
``LOW_CONFIDENCE`` or ``AMBIGUOUS`` and show several narrow ribbons, not to
inflate one polygon until it covers the answer by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence

import numpy as np
from shapely.geometry import LineString
from shapely.ops import unary_union

from geotrace.coordinates import LocalFrame
from geotrace.pacman_tracker.config import CorridorConfig
from geotrace.pacman_tracker.roadmap import RoadGeometry
from geotrace.pacman_tracker.state import HypothesisSet


class Confidence(str, Enum):
    CONFIDENT = "CONFIDENT"
    AMBIGUOUS = "AMBIGUOUS"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


@dataclass
class Corridor:
    """One narrow ribbon of road the car may be on."""

    edges: list[int]
    centerline: np.ndarray
    """(M, 2) local-frame polyline covering [s-k*sigma, s+k*sigma]."""

    mass: float
    representative_edge: int
    s: float
    sigma_s: float
    length_m: float
    half_width_m: float

    def polygon(self):
        if len(self.centerline) < 2:
            return None
        return LineString(self.centerline).buffer(self.half_width_m, cap_style=2)

    def to_json(self) -> dict[str, Any]:
        return {
            "edges": self.edges,
            "mass": round(self.mass, 4),
            "representative_edge": self.representative_edge,
            "distance_along_edge_m": round(self.s, 2),
            "sigma_s_m": round(self.sigma_s, 2),
            "length_m": round(self.length_m, 1),
            "half_width_m": self.half_width_m,
        }


@dataclass
class CorridorSet:
    t: float
    corridors: list[Corridor]
    confidence: Confidence
    top_mass: float
    sigma_s: float
    reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def area_m2(self) -> float:
        polys = [c.polygon() for c in self.corridors]
        polys = [p for p in polys if p is not None]
        return float(unary_union(polys).area) if polys else 0.0

    def contains(self, point: Sequence[float]) -> bool:
        from shapely.geometry import Point

        p = Point(float(point[0]), float(point[1]))
        return any((poly := c.polygon()) is not None and poly.contains(p) for c in self.corridors)

    def to_geojson(self, frame: LocalFrame) -> dict[str, Any]:
        features = []
        for corridor in self.corridors:
            poly = corridor.polygon()
            if poly is None or poly.is_empty:
                continue
            if poly.geom_type == "Polygon":
                geometry = {
                    "type": "Polygon",
                    "coordinates": [
                        frame.coords_to_geojson(np.asarray(poly.exterior.coords))
                    ],
                }
            else:
                geometry = {
                    "type": "MultiPolygon",
                    "coordinates": [[
                        frame.coords_to_geojson(np.asarray(part.exterior.coords))
                    ] for part in poly.geoms],
                }
            features.append(
                {
                    "type": "Feature",
                    "geometry": geometry,
                    "properties": {**corridor.to_json(), "confidence": self.confidence.value},
                }
            )
        return {"type": "FeatureCollection", "features": features}

    def to_json(self) -> dict[str, Any]:
        return {
            "t": round(self.t, 2),
            "confidence": self.confidence.value,
            "top_mass": round(self.top_mass, 4),
            "sigma_s_m": round(self.sigma_s, 2),
            "reason": self.reason,
            "area_m2": round(self.area_m2(), 1),
            "corridors": [c.to_json() for c in self.corridors],
            **self.diagnostics,
        }


class CorridorBuilder:
    def __init__(self, geometry: RoadGeometry, cfg: CorridorConfig) -> None:
        self.geometry = geometry
        self.cfg = cfg

    def build(self, hs: HypothesisSet, t: float, distance: float,
              sigma_s: float, map_sigma: Optional[np.ndarray] = None) -> CorridorSet:
        """Corridors from odometry plus route-map alignment uncertainty.

        Every hypothesis shares the car's ``sigma_s``; ``map_sigma`` differs by
        route because a recent turn may have re-anchored one history.
        """
        cfg = self.cfg
        if len(hs) == 0:
            return CorridorSet(t=t, corridors=[], confidence=Confidence.LOW_CONFIDENCE,
                               top_mass=0.0, sigma_s=float("inf"), reason="no_hypotheses")
        weights = hs.weights()
        map_sigma = (np.zeros(len(hs)) if map_sigma is None
                     else np.asarray(map_sigma, dtype=float))
        order = hs.order()
        s_values_all = hs.s(distance)

        # Group by edge: one corridor per street the belief occupies.
        by_edge: dict[int, list[int]] = {}
        for i in order:
            by_edge.setdefault(int(hs.edge[i]), []).append(int(i))
        ranked = sorted(by_edge.items(), key=lambda kv: -float(weights[kv[1]].sum()))

        corridors: list[Corridor] = []
        for edge, members in ranked[: cfg.max_corridors]:
            mass = float(weights[members].sum())
            if corridors and mass < cfg.min_branch_mass:
                break
            lead = members[0]
            s_values = s_values_all[members]
            # Spread of the group counts as uncertainty too, not just the one
            # global sigma: two hypotheses on the same edge via routes of
            # different length really are at different places.
            spread = float(np.sqrt(np.average(
                (s_values - s_values[np.argmax(weights[members])]) ** 2,
                weights=np.maximum(weights[members], 1e-12))))
            route_sigma = float(np.sqrt(np.average(
                map_sigma[members] ** 2, weights=np.maximum(weights[members], 1e-12))))
            sigma = float(np.sqrt(sigma_s ** 2 + route_sigma ** 2 + spread ** 2))
            corridor = self._corridor_for(hs, lead, float(s_values_all[lead]), sigma, mass)
            if corridor is not None:
                corridors.append(corridor)

        top_mass = corridors[0].mass if corridors else 0.0
        sigma_top = corridors[0].sigma_s if corridors else float("inf")
        # How many streets the belief really occupies, not how many corridors
        # were drawn: exp(H) over the weight each edge carries.
        edge_mass = np.array([float(weights[m].sum()) for m in by_edge.values()])
        edge_mass = edge_mass[edge_mass > 0]
        effective = (
            float(np.exp(-np.sum(edge_mass * np.log(edge_mass)))) if edge_mass.size else 0.0
        )
        confidence, reason = self._classify(corridors, top_mass, sigma_top, effective)
        return CorridorSet(
            t=t, corridors=corridors, confidence=confidence,
            top_mass=top_mass, sigma_s=sigma_top, reason=reason,
            diagnostics={
                "hypotheses": len(hs),
                "distinct_edges": len(by_edge),
                "effective_streets": round(effective, 2),
            },
        )

    def _classify(
        self, corridors, top_mass: float, sigma_s: float, effective_streets: float
    ) -> tuple[Confidence, str]:
        cfg = self.cfg
        if not corridors:
            return Confidence.LOW_CONFIDENCE, "no_hypotheses"
        if sigma_s > cfg.low_confidence_sigma_s_m:
            return Confidence.LOW_CONFIDENCE, f"sigma_s={sigma_s:.0f}m"
        if effective_streets > cfg.confident_effective_streets:
            return (
                Confidence.AMBIGUOUS,
                f"belief occupies {effective_streets:.1f} effective streets",
            )
        competing = sum(1 for c in corridors if c.mass >= cfg.min_branch_mass)
        if top_mass < cfg.confident_mass and competing > 1:
            return Confidence.AMBIGUOUS, f"top_mass={top_mass:.2f} over {competing} corridors"
        if sigma_s > cfg.confident_sigma_s_m:
            return Confidence.LOW_CONFIDENCE, f"sigma_s={sigma_s:.0f}m"
        return Confidence.CONFIDENT, ""

    def _corridor_for(
        self, hs: HypothesisSet, index: int, s: float, sigma: float, mass: float
    ) -> Optional[Corridor]:
        """Walk the route back and the graph forward to cover +-k*sigma."""
        geo = self.geometry
        cfg = self.cfg
        edge = int(hs.edge[index])
        s = float(np.clip(s, 0.0, geo.lengths[edge]))
        reach = cfg.k_sigma * sigma

        back_pieces: list[np.ndarray] = []
        edges = [edge]
        remaining = reach - s
        node = hs.routes[index].parent
        lo = max(0.0, s - reach)
        while remaining > 0 and node is not None:
            prev = int(node.edge)
            length = float(geo.lengths[prev])
            start = max(0.0, length - remaining)
            back_pieces.append(_slice(geo, prev, start, length))
            edges.insert(0, prev)
            remaining -= length - start
            node = node.parent

        forward_pieces: list[np.ndarray] = []
        length = float(geo.lengths[edge])
        hi = min(length, s + reach)
        remaining = (s + reach) - length
        cursor = edge
        history = hs.routes[index].tail(4)
        guard = 0
        while remaining > 0 and guard < 8:
            guard += 1
            successors = geo.successors(cursor, history)
            if len(successors) != 1:
                # A fork ahead is real ambiguity: stop rather than pick one.
                break
            cursor = successors[0]
            seg_len = float(geo.lengths[cursor])
            end = min(seg_len, remaining)
            forward_pieces.append(_slice(geo, cursor, 0.0, end))
            edges.append(cursor)
            remaining -= end
            history = tuple(list(history)[-3:] + [cursor])

        middle = _slice(geo, edge, lo, hi)
        parts = list(reversed(back_pieces)) + [middle] + forward_pieces
        parts = [p for p in parts if len(p) >= 2]
        if not parts:
            return None
        line = np.concatenate(parts, axis=0)
        length_m = float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())
        return Corridor(
            edges=edges, centerline=line, mass=mass, representative_edge=edge,
            s=s, sigma_s=sigma, length_m=length_m, half_width_m=cfg.lateral_buffer_m,
        )


def _slice(geo: RoadGeometry, edge: int, lo: float, hi: float) -> np.ndarray:
    """Polyline of ``edge`` between two distances along it."""
    e = geo.network.edges[edge]
    lo = max(0.0, min(lo, e.length))
    hi = max(lo, min(hi, e.length))
    inner = e.coords[(e.cumulative > lo) & (e.cumulative < hi)]
    pts = [np.asarray(e.position(lo))]
    if len(inner):
        pts.append(inner)
    pts.append(np.asarray(e.position(hi)))
    stacked = np.vstack([p.reshape(-1, 2) for p in pts])
    keep = np.ones(len(stacked), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(stacked, axis=0), axis=1) > 1e-6
    return stacked[keep]
