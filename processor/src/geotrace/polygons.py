"""Probabilistic position polygons.

Given the particle cloud, take the smallest set of particles carrying at least
gamma (0.95 by default) of the probability mass:

    sum_{i=1..K} w_(i) >= gamma        (weights sorted descending)

then group those particles by *connected branch of the road graph* and build one
corridor per branch as a union of buffers around the occupied road segments:

    P_0.95 = union_i Buffer(segment_i, r_i),    r_i = r_min + k * sigma_perp

Deliberately NOT a convex hull. A hull over a post-junction belief would swallow
the courtyards, the buildings and the empty space between two independent roads
and would claim the car might be inside a block of flats. Two branches that have
diverged produce two separate polygons with separate probabilities; while the
cloud still straddles the junction they are genuinely one connected Y-shaped
region and are reported as such.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import substring, unary_union

from geotrace.config import PolygonConfig
from geotrace.coordinates import LocalFrame
from geotrace.road_graph import RoadNetwork


@dataclass
class BranchComponent:
    """One connected corridor of the belief."""

    component_id: str
    probability: float
    geometry: Polygon | MultiPolygon
    particle_indices: np.ndarray
    edge_indices: list[int]
    representative_xy: tuple[float, float]
    area_m2: float
    street_names: list[str] = field(default_factory=list)

    def to_geojson_feature(self, frame: LocalFrame, extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        props = {
            "component_id": self.component_id,
            "probability": round(self.probability, 6),
            "area_m2": round(self.area_m2, 1),
            "street_names": self.street_names,
        }
        if extra:
            props.update(extra)
        return {
            "type": "Feature",
            "properties": props,
            "geometry": _geometry_to_wgs84(self.geometry, frame),
        }


@dataclass
class UncertaintySet:
    """All branches of the belief at one timestamp."""

    t: float
    confidence: float
    components: list[BranchComponent]
    gps_state: str = "UNKNOWN"
    seconds_since_trusted: float = 0.0
    n_selected: int = 0
    n_particles: int = 0

    @property
    def total_area_m2(self) -> float:
        return float(sum(c.area_m2 for c in self.components))

    @property
    def best(self) -> Optional[BranchComponent]:
        return max(self.components, key=lambda c: c.probability, default=None)

    def contains(self, xy: Sequence[float]) -> bool:
        point = shapely.points(float(xy[0]), float(xy[1]))
        return any(shapely.contains(c.geometry, point) for c in self.components)

    def to_json(self, frame: LocalFrame) -> dict[str, Any]:
        return {
            "t": round(self.t, 3),
            "confidence": self.confidence,
            "gps_state": self.gps_state,
            "seconds_since_trusted": round(self.seconds_since_trusted, 2),
            "particles_selected": self.n_selected,
            "particles_total": self.n_particles,
            "components": [
                {
                    "component_id": c.component_id,
                    "probability": round(c.probability, 6),
                    "area_m2": round(c.area_m2, 1),
                    "street_names": c.street_names,
                    "geometry": _geometry_to_wgs84(c.geometry, frame),
                }
                for c in self.components
            ],
        }


def _geometry_to_wgs84(geometry: Polygon | MultiPolygon, frame: LocalFrame) -> dict[str, Any]:
    def ring(coords: Sequence[Sequence[float]]) -> list[list[float]]:
        return frame.coords_to_geojson(coords)

    if isinstance(geometry, Polygon):
        return {
            "type": "Polygon",
            "coordinates": [ring(geometry.exterior.coords)]
            + [ring(hole.coords) for hole in geometry.interiors],
        }
    polygons = []
    for part in geometry.geoms:
        polygons.append(
            [ring(part.exterior.coords)] + [ring(hole.coords) for hole in part.interiors]
        )
    return {"type": "MultiPolygon", "coordinates": polygons}


def select_high_probability_set(weights: np.ndarray, gamma: float) -> np.ndarray:
    """Smallest index set whose weights sum to at least ``gamma``.

    Returns indices into the original array, ordered by descending weight.
    """
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(w)[::-1]
    cumulative = np.cumsum(w[order])
    total = cumulative[-1]
    target = gamma * (total if total > 0 else 1.0)
    k = int(np.searchsorted(cumulative, target, side="left")) + 1
    return order[: min(k, w.size)].astype(np.int64)


def cross_track_sigma(seconds_since_trusted: float, cfg: PolygonConfig) -> float:
    """Lateral uncertainty of a particle that sits on the road centreline.

    A particle has no lateral spread by construction, but the car does: it is in
    one of the lanes, offset from the centreline, and the longer GPS has been
    unavailable the less certain even the lane-level along-track position is.
    """
    return cfg.cross_track_sigma_base_m + cfg.cross_track_sigma_per_s * max(
        0.0, seconds_since_trusted
    )


def build_uncertainty_set(
    network: RoadNetwork,
    edge_idx: np.ndarray,
    s_values: np.ndarray,
    weights: np.ndarray,
    cfg: PolygonConfig,
    t: float = 0.0,
    gps_state: str = "UNKNOWN",
    seconds_since_trusted: float = 0.0,
) -> UncertaintySet:
    """Build the gamma-mass corridor set for one particle cloud."""
    weights = np.asarray(weights, dtype=float)
    edge_idx = np.asarray(edge_idx, dtype=np.int64)
    s_values = np.asarray(s_values, dtype=float)
    selected = select_high_probability_set(weights, cfg.confidence)
    if selected.size == 0:
        return UncertaintySet(t, cfg.confidence, [], gps_state, seconds_since_trusted, 0, len(weights))

    sigma = cross_track_sigma(seconds_since_trusted, cfg)
    radius = min(cfg.r_min_m + cfg.k_sigma * sigma, cfg.max_radius_m)

    # One buffered sub-segment per occupied edge, spanning the particles on it.
    buffers = []
    edges_used: list[int] = []
    for edge in np.unique(edge_idx[selected]):
        on_edge = selected[edge_idx[selected] == edge]
        s_on = s_values[on_edge]
        line = network.edges[int(edge)].line
        lo = max(0.0, float(s_on.min()) - radius * 0.5)
        hi = min(line.length, float(s_on.max()) + radius * 0.5)
        piece = substring(line, lo, max(hi, lo + 0.5))
        if piece.is_empty:
            continue
        if isinstance(piece, LineString) and piece.length < 1e-6:
            piece = piece.buffer(0.5)
        buffers.append(piece.buffer(radius, cap_style="round", join_style="round"))
        edges_used.append(int(edge))

    if not buffers:
        return UncertaintySet(t, cfg.confidence, [], gps_state, seconds_since_trusted, len(selected), len(weights))

    merged = unary_union(buffers)
    parts = list(merged.geoms) if isinstance(merged, MultiPolygon) else [merged]

    positions = network.positions_fast(edge_idx[selected], s_values[selected])
    points = shapely.points(positions[:, 0], positions[:, 1])
    total_selected_weight = float(weights[selected].sum())
    if total_selected_weight <= 0:
        total_selected_weight = 1.0

    components: list[BranchComponent] = []
    for order, part in enumerate(sorted(parts, key=lambda p: -p.area)):
        inside = shapely.contains(part, points)
        if not np.any(inside):
            continue
        member = selected[inside]
        probability = float(weights[member].sum())
        if probability / total_selected_weight < cfg.min_component_probability:
            continue
        member_positions = positions[inside]
        member_weights = weights[member]
        if member_weights.sum() <= 0:
            member_weights = np.ones_like(member_weights)
        representative = np.average(member_positions, axis=0, weights=member_weights)
        member_edges = sorted({int(e) for e in edge_idx[member]})
        names = sorted(
            {
                str(network.edges[e].name)
                for e in member_edges
                if network.edges[e].name
            }
        )
        geometry = part
        if cfg.simplify_tolerance_m > 0:
            geometry = part.simplify(cfg.simplify_tolerance_m, preserve_topology=True)
        components.append(
            BranchComponent(
                component_id=f"branch-{order + 1:02d}",
                probability=probability,
                geometry=geometry,
                particle_indices=member,
                edge_indices=member_edges,
                representative_xy=(float(representative[0]), float(representative[1])),
                area_m2=float(geometry.area),
                street_names=names[:6],
            )
        )

    components.sort(key=lambda c: -c.probability)
    for rank, component in enumerate(components, start=1):
        component.component_id = f"branch-{rank:02d}"
    return UncertaintySet(
        t=t,
        confidence=cfg.confidence,
        components=components,
        gps_state=gps_state,
        seconds_since_trusted=seconds_since_trusted,
        n_selected=int(selected.size),
        n_particles=int(weights.size),
    )


def branch_aware_estimate(
    network: RoadNetwork,
    uncertainty: UncertaintySet,
    edge_idx: np.ndarray,
    s_values: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    """Point estimate that never falls between two branches.

    Takes the highest-probability branch and returns the weighted mean of the
    particles inside it, snapped onto the road.
    """
    best = uncertainty.best
    if best is None or best.particle_indices.size == 0:
        positions = network.positions_fast(edge_idx, s_values)
        return tuple(np.average(positions, axis=0, weights=weights))  # type: ignore[return-value]

    member = best.particle_indices
    member_weights = np.asarray(weights, dtype=float)[member]
    if member_weights.sum() <= 0:
        member_weights = np.ones_like(member_weights)
    # Weighted mean along-edge on the single dominant edge of the branch, so the
    # estimate stays on the carriageway when no parking offset exists.
    edges_here = np.asarray(edge_idx, dtype=np.int64)[member]
    unique, inverse = np.unique(edges_here, return_inverse=True)
    per_edge = np.bincount(inverse, weights=member_weights, minlength=unique.size)
    dominant = int(unique[int(np.argmax(per_edge))])
    on_dominant = edges_here == dominant
    s_mean = float(
        np.average(np.asarray(s_values, dtype=float)[member][on_dominant],
                   weights=member_weights[on_dominant])
    )
    return network.edges[dominant].position(s_mean)


def convex_hull_area(points: np.ndarray) -> float:
    """Only used by the tests, to demonstrate how much space a hull would claim
    compared with the corridor union we actually produce."""
    from shapely.geometry import MultiPoint

    if len(points) < 3:
        return 0.0
    return float(MultiPoint([tuple(p) for p in np.asarray(points, dtype=float)]).convex_hull.area)


def uncertainty_to_geojson(
    sets: Sequence[UncertaintySet], frame: LocalFrame
) -> dict[str, Any]:
    features = []
    for item in sets:
        for component in item.components:
            features.append(
                component.to_geojson_feature(
                    frame,
                    extra={
                        "t": round(item.t, 3),
                        "confidence": item.confidence,
                        "gps_state": item.gps_state,
                    },
                )
            )
    return {"type": "FeatureCollection", "features": features}
