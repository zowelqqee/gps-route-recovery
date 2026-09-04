"""Drivable road network in the trip's local metric frame.

The graph itself comes from OpenStreetMap through OSMnx and is cached on disk as
GraphML - it is downloaded once, never on every run. For tests (and when the
network is unavailable) small synthetic graphs can be built in code.

:class:`RoadNetwork` projects every node and edge geometry into the local frame
once, then answers the three questions the particle filter asks millions of
times: where is a given (edge, distance-along) in metres, which edges leave a
node, and which edges are near a point.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from geotrace.coordinates import LocalFrame

DRIVABLE_HIGHWAYS = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
    "residential", "living_street", "service", "road",
}

DEFAULT_SPEED_KMH = {
    "motorway": 110, "motorway_link": 60, "trunk": 90, "trunk_link": 50,
    "primary": 60, "primary_link": 40, "secondary": 60, "secondary_link": 40,
    "tertiary": 60, "tertiary_link": 40, "unclassified": 60, "residential": 40,
    "living_street": 20, "service": 20, "road": 40,
}

EdgeId = tuple[Any, Any, int]


class RoadGraphError(RuntimeError):
    pass


@dataclass
class Edge:
    """One directed road segment, pre-projected into the local frame."""

    edge_id: EdgeId
    index: int
    u: Any
    v: Any
    coords: np.ndarray
    """(M, 2) polyline in local E/N metres, M >= 2."""

    cumulative: np.ndarray
    """(M,) cumulative distance along ``coords``, cumulative[0] == 0."""

    length: float
    bearings: np.ndarray
    """(M-1,) heading of each sub-segment, radians CCW from +E."""

    name: Optional[str] = None
    highway: Optional[str] = None
    speed_limit_ms: float = 16.7
    oneway: bool = False
    reverse_index: Optional[int] = None
    """Index of the edge that is this one driven backwards, if it exists.
    Used to forbid U-turns."""

    line: LineString = field(repr=False, default=None)  # type: ignore[assignment]

    def position(self, s: float) -> tuple[float, float]:
        """Point at distance ``s`` from the edge start, clamped to the edge."""
        s = min(max(s, 0.0), self.length)
        i = int(np.searchsorted(self.cumulative, s, side="right")) - 1
        i = min(max(i, 0), len(self.coords) - 2)
        seg_len = self.cumulative[i + 1] - self.cumulative[i]
        frac = 0.0 if seg_len <= 0 else (s - self.cumulative[i]) / seg_len
        p0, p1 = self.coords[i], self.coords[i + 1]
        return float(p0[0] + frac * (p1[0] - p0[0])), float(p0[1] + frac * (p1[1] - p0[1]))

    def bearing(self, s: float) -> float:
        s = min(max(s, 0.0), self.length)
        i = int(np.searchsorted(self.cumulative, s, side="right")) - 1
        i = min(max(i, 0), len(self.bearings) - 1)
        return float(self.bearings[i])

    @property
    def start_bearing(self) -> float:
        return float(self.bearings[0])


def _segment_bearings(coords: np.ndarray) -> np.ndarray:
    deltas = np.diff(coords, axis=0)
    return np.arctan2(deltas[:, 1], deltas[:, 0])


def _parse_speed(value: Any, highway: Any) -> float:
    """OSM `maxspeed` is free-form text; fall back to a per-class default."""
    candidates = value if isinstance(value, (list, tuple)) else [value]
    for item in candidates:
        if item is None:
            continue
        text = str(item).strip().lower()
        mph = "mph" in text
        digits = "".join(ch for ch in text if ch.isdigit())
        if digits:
            speed = float(digits)
            return speed * (0.44704 if mph else 1000.0 / 3600.0)
    tag = highway[0] if isinstance(highway, (list, tuple)) and highway else highway
    return DEFAULT_SPEED_KMH.get(str(tag), 40) * 1000.0 / 3600.0


def _first(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


class RoadNetwork:
    """Local-frame view of a drivable road graph."""

    _BASE = 1.0e7
    """Per-edge offset used by the flattened fast index; larger than any
    conceivable single edge length."""

    def __init__(self, graph: nx.MultiDiGraph, frame: LocalFrame) -> None:
        self.graph = graph
        self.frame = frame
        self.edges: list[Edge] = []
        self.edge_index: dict[EdgeId, int] = {}
        self.node_xy: dict[Any, tuple[float, float]] = {}
        self.out_edges: dict[Any, list[int]] = {}
        self._build()

    # ---------------------------------------------------------------- build

    def _build(self) -> None:
        for node, data in self.graph.nodes(data=True):
            if "x" not in data or "y" not in data:
                raise RoadGraphError(f"node {node} has no x/y (lon/lat) attributes")
            self.node_xy[node] = self.frame.to_local(float(data["y"]), float(data["x"]))

        for u, v, k, data in self.graph.edges(keys=True, data=True):
            coords = self._edge_coords(u, v, data)
            if coords is None or len(coords) < 2:
                continue
            cumulative = np.concatenate(
                [[0.0], np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))]
            )
            length = float(cumulative[-1])
            if length < 1e-3:
                continue
            index = len(self.edges)
            edge = Edge(
                edge_id=(u, v, k),
                index=index,
                u=u,
                v=v,
                coords=coords,
                cumulative=cumulative,
                length=length,
                bearings=_segment_bearings(coords),
                name=_first(data.get("name")),
                highway=_first(data.get("highway")),
                speed_limit_ms=_parse_speed(data.get("maxspeed"), data.get("highway")),
                oneway=bool(data.get("oneway", False)),
                line=LineString(coords),
            )
            self.edges.append(edge)
            self.edge_index[(u, v, k)] = index
            self.out_edges.setdefault(u, []).append(index)
            self.out_edges.setdefault(v, [])

        # Pair each edge with its reverse twin so U-turns can be forbidden.
        by_pair: dict[tuple[Any, Any], list[int]] = {}
        for edge in self.edges:
            by_pair.setdefault((edge.u, edge.v), []).append(edge.index)
        for edge in self.edges:
            twins = by_pair.get((edge.v, edge.u))
            if twins:
                edge.reverse_index = twins[0]

        if not self.edges:
            raise RoadGraphError("road graph contains no usable edges")

        self._lines = [e.line for e in self.edges]
        self._tree = STRtree(self._lines)
        self._build_fast_index()
        allx = np.concatenate([e.coords for e in self.edges])
        self.bounds = (
            float(allx[:, 0].min()), float(allx[:, 1].min()),
            float(allx[:, 0].max()), float(allx[:, 1].max()),
        )

    def _edge_coords(self, u: Any, v: Any, data: dict[str, Any]) -> Optional[np.ndarray]:
        geometry = data.get("geometry")
        if geometry is not None and hasattr(geometry, "coords"):
            lonlat = np.asarray(geometry.coords, dtype=float)
            return self.frame.to_local_array(lonlat[:, 1], lonlat[:, 0])
        if u not in self.node_xy or v not in self.node_xy:
            return None
        return np.array([self.node_xy[u], self.node_xy[v]], dtype=float)

    # ------------------------------------------------------------- queries

    def __len__(self) -> int:
        return len(self.edges)

    def edge_by_id(self, edge_id: EdgeId) -> Edge:
        return self.edges[self.edge_index[edge_id]]

    def successors(self, edge_index: int, allow_uturn: bool = False) -> list[int]:
        """Edges leaving the far node of ``edge_index``.

        Direction of travel is already encoded in the directed graph, so an
        edge simply does not exist against a one-way street. The reverse twin
        of the current edge is excluded unless a U-turn is explicitly allowed.
        """
        edge = self.edges[edge_index]
        candidates = list(self.out_edges.get(edge.v, ()))
        if not allow_uturn:
            candidates = [
                c for c in candidates if not (self.edges[c].v == edge.u and self.edges[c].u == edge.v)
            ]
        return candidates

    def nearest_edges(self, point: Sequence[float], k: int = 5, radius: float = 250.0) -> list[int]:
        """Indices of the ``k`` nearest edges to a local-frame point."""
        p = Point(float(point[0]), float(point[1]))
        found = self._tree.query(p.buffer(radius))
        indices = [int(i) for i in np.atleast_1d(found)]
        if not indices:
            found = self._tree.query_nearest(p)
            indices = [int(i) for i in np.atleast_1d(found)]
        indices.sort(key=lambda i: self._lines[i].distance(p))
        return indices[:k]

    def distance_to_road(self, point: Sequence[float]) -> float:
        """Metres from a point to the closest drivable edge."""
        p = Point(float(point[0]), float(point[1]))
        nearest = self._tree.query_nearest(p)
        indices = [int(i) for i in np.atleast_1d(nearest)]
        if not indices:
            return float("inf")
        return float(min(self._lines[i].distance(p) for i in indices))

    def project(self, point: Sequence[float], edge_index: int) -> tuple[float, float]:
        """Project a point onto an edge -> (distance along, offset from line)."""
        p = Point(float(point[0]), float(point[1]))
        line = self._lines[edge_index]
        return float(line.project(p)), float(line.distance(p))

    def snap(self, point: Sequence[float], k: int = 5) -> Optional[tuple[int, float, float]]:
        """Best (edge_index, s, offset) for a point, or None if nothing is near."""
        best: Optional[tuple[int, float, float]] = None
        for index in self.nearest_edges(point, k=k):
            s, offset = self.project(point, index)
            if best is None or offset < best[2]:
                best = (index, s, offset)
        return best

    def positions(self, edge_indices: np.ndarray, s_values: np.ndarray) -> np.ndarray:
        """Vectorised (edge, s) -> (N, 2) local positions."""
        out = np.empty((len(edge_indices), 2))
        for i, (ei, s) in enumerate(zip(edge_indices, s_values)):
            out[i] = self.edges[int(ei)].position(float(s))
        return out

    def bearings_at(self, edge_indices: np.ndarray, s_values: np.ndarray) -> np.ndarray:
        out = np.empty(len(edge_indices))
        for i, (ei, s) in enumerate(zip(edge_indices, s_values)):
            out[i] = self.edges[int(ei)].bearing(float(s))
        return out


    # ------------------------------------------------- vectorised fast index

    def _build_fast_index(self) -> None:
        """Flatten every edge polyline into one array so that (edge, s) -> point
        is a single vectorised lookup.

        The particle filter evaluates this for thousands of particles at every
        step; a Python loop over ``Edge.position`` is two orders of magnitude too
        slow. Each edge's cumulative distances are shifted by ``i * _BASE`` so
        that the concatenation stays globally monotonic and one ``searchsorted``
        finds the right sub-segment of the right edge.
        """
        counts = np.array([len(e.coords) for e in self.edges], dtype=np.int64)
        self._pt_offset = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
        self._pt_counts = counts
        self._coords_flat = np.concatenate([e.coords for e in self.edges]).astype(float)
        cum = np.concatenate([e.cumulative for e in self.edges]).astype(float)
        self._base = np.arange(len(self.edges), dtype=float) * self._BASE
        edge_of_point = np.repeat(np.arange(len(self.edges)), counts)
        self._cum_global = cum + self._base[edge_of_point]
        self._bear_flat = np.concatenate([e.bearings for e in self.edges]).astype(float)
        seg_counts = counts - 1
        self._seg_offset = np.concatenate([[0], np.cumsum(seg_counts)[:-1]]).astype(np.int64)
        self._lengths = np.array([e.length for e in self.edges], dtype=float)
        self._speed_limits = np.array([e.speed_limit_ms for e in self.edges], dtype=float)

    def _segment_lookup(self, edge_idx: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (global point index of the sub-segment start, clamped s)."""
        edge_idx = np.asarray(edge_idx, dtype=np.int64)
        s = np.clip(np.asarray(s, dtype=float), 0.0, self._lengths[edge_idx])
        query = s + self._base[edge_idx]
        idx = np.searchsorted(self._cum_global, query, side="right") - 1
        low = self._pt_offset[edge_idx]
        high = low + self._pt_counts[edge_idx] - 2
        return np.clip(idx, low, high), s

    def positions_fast(self, edge_idx: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Vectorised (edge, s) -> (N, 2) local positions."""
        idx, s = self._segment_lookup(edge_idx, s)
        seg_start = self._cum_global[idx]
        seg_end = self._cum_global[idx + 1]
        seg_len = seg_end - seg_start
        frac = np.where(seg_len > 1e-9, (s + self._base[np.asarray(edge_idx, dtype=np.int64)] - seg_start) / np.where(seg_len > 1e-9, seg_len, 1.0), 0.0)
        p0 = self._coords_flat[idx]
        p1 = self._coords_flat[idx + 1]
        return p0 + frac[:, None] * (p1 - p0)

    def bearings_fast(self, edge_idx: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Vectorised (edge, s) -> heading of the road at that point, radians."""
        idx, _ = self._segment_lookup(edge_idx, s)
        edge_idx = np.asarray(edge_idx, dtype=np.int64)
        seg = self._seg_offset[edge_idx] + (idx - self._pt_offset[edge_idx])
        return self._bear_flat[np.clip(seg, 0, len(self._bear_flat) - 1)]

    @property
    def lengths(self) -> np.ndarray:
        return self._lengths

    @property
    def speed_limits(self) -> np.ndarray:
        return self._speed_limits

    def successor_table(self, allow_uturn: bool = False) -> list[np.ndarray]:
        """Pre-computed successor lists, indexed by edge. Built once per run."""
        key = "_succ_uturn" if allow_uturn else "_succ_no_uturn"
        cached = getattr(self, key, None)
        if cached is None:
            cached = [
                np.array(self.successors(i, allow_uturn=allow_uturn), dtype=np.int64)
                for i in range(len(self.edges))
            ]
            setattr(self, key, cached)
        return cached

    @property
    def edge_lengths(self) -> np.ndarray:
        return np.array([e.length for e in self.edges], dtype=float)

    @property
    def edge_speed_limits(self) -> np.ndarray:
        return np.array([e.speed_limit_ms for e in self.edges], dtype=float)


# --------------------------------------------------------------- OSM access


def download_graph(place: str, output: str | Path, network_type: str = "drive") -> Path:
    """Download a drivable graph from OpenStreetMap and cache it as GraphML.

    Run once. ``load_graph`` reads the cache and never touches the network.
    """
    import osmnx as ox

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    graph = ox.graph_from_place(place, network_type=network_type, simplify=True)
    ox.save_graphml(graph, out)
    return out


def download_graph_bbox(
    north: float, south: float, east: float, west: float,
    output: str | Path, network_type: str = "drive",
) -> Path:
    """Same, for a bounding box. Much smaller and faster than a whole city."""
    import osmnx as ox

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:  # OSMnx 2.x takes a bbox tuple, 1.x took keyword arguments.
        graph = ox.graph_from_bbox(bbox=(west, south, east, north), network_type=network_type)
    except TypeError:  # pragma: no cover - depends on the installed OSMnx
        graph = ox.graph_from_bbox(north, south, east, west, network_type=network_type)
    ox.save_graphml(graph, out)
    return out


def load_graph(path: str | Path) -> nx.MultiDiGraph:
    """Load a cached GraphML graph. Never downloads."""
    p = Path(path)
    if not p.exists():
        raise RoadGraphError(
            f"road graph cache not found: {p}\n"
            "Download it once with:\n"
            f'  geotrace download-map --place "Saint Petersburg, Russia" --output {p}'
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            import osmnx as ox

            graph = ox.load_graphml(p)
        except Exception:  # pragma: no cover - fall back to plain networkx
            graph = nx.read_graphml(p)
            graph = nx.MultiDiGraph(graph)
            for _node, data in graph.nodes(data=True):
                for key in ("x", "y"):
                    if key in data:
                        data[key] = float(data[key])
    return graph


def clip_graph(graph: nx.MultiDiGraph, lat: float, lon: float, radius_m: float) -> nx.MultiDiGraph:
    """Keep only the part of a city graph near the trip.

    A whole-city GraphML has hundreds of thousands of edges; projecting all of
    them for a 10-minute trip is pure waste.
    """
    frame = LocalFrame(lat, lon)
    keep = []
    for node, data in graph.nodes(data=True):
        if "x" not in data or "y" not in data:
            continue
        e, n = frame.to_local(float(data["y"]), float(data["x"]))
        if math.hypot(e, n) <= radius_m:
            keep.append(node)
    if not keep:
        raise RoadGraphError(
            f"no road graph nodes within {radius_m:.0f} m of {lat:.5f}, {lon:.5f}. "
            "The cached graph probably covers a different area than the trip."
        )
    return graph.subgraph(keep).copy()


# ------------------------------------------------------------ synthetic graphs


def build_graph_from_segments(
    segments: Iterable[tuple[str, Sequence[tuple[float, float]], dict[str, Any]]],
    origin_lat: float = 59.9343,
    origin_lon: float = 30.3351,
) -> nx.MultiDiGraph:
    """Build an OSMnx-shaped graph from local-metre polylines.

    ``segments`` is a sequence of (name, [(E, N), ...], attrs). ``attrs`` may
    contain ``oneway`` (default False, i.e. both directions are created),
    ``highway`` and ``maxspeed``.

    Used by the tests and by ``geotrace simulate`` so the whole pipeline can run
    with no network access at all.
    """
    frame = LocalFrame(origin_lat, origin_lon)
    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "epsg:4326"
    node_of: dict[tuple[int, int], int] = {}

    def node_for(point: Sequence[float]) -> int:
        key = (int(round(point[0] * 100)), int(round(point[1] * 100)))
        if key not in node_of:
            node_id = len(node_of) + 1
            lat, lon = frame.to_geo(float(point[0]), float(point[1]))
            graph.add_node(node_id, x=lon, y=lat)
            node_of[key] = node_id
        return node_of[key]

    for name, points, attrs in segments:
        pts = [tuple(float(c) for c in p) for p in points]
        oneway = bool(attrs.get("oneway", False))
        for a, b in zip(pts[:-1], pts[1:]):
            ua, ub = node_for(a), node_for(b)
            if ua == ub:
                continue
            length = math.dist(a, b)
            base = {
                "name": name,
                "highway": attrs.get("highway", "residential"),
                "maxspeed": attrs.get("maxspeed"),
                "oneway": oneway,
                "length": length,
            }
            graph.add_edge(ua, ub, **dict(base))
            if not oneway:
                graph.add_edge(ub, ua, **dict(base))
    return graph, frame  # type: ignore[return-value]
