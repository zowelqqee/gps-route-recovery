"""Hypothesis state - a route, and nothing else.

One hypothesis ("Pacman") is a claim about **which roads the car drove along**,
and that is the whole of it. It does not carry a speed, a distance, or an
accelerometer bias, because there is one car and those are properties of the
car, not of a road. They live in
:class:`~geotrace.pacman_tracker.speed.GlobalSpeedTracker`.

    hypothesis  =  ( edge, route, route_offset, alignment mean/variance,
                      log-weight )

``route_offset`` is the length of the route up to the *start* of ``edge``, so
position along the current edge is

    s_i  =  D  -  route_offset_i - alignment_bias_i

with ``D`` the one global distance travelled. ``sigma_D`` is shared by everyone;
the additional alignment uncertainty belongs to a route and grows with distance
since its last reliable turn anchor. A hypothesis is still not entitled to a
different opinion about how fast the car was going.

Why this matters, from the run that motivated it: with a per-hypothesis speed,
the population held hypotheses at 7, 13 and 2 m/s simultaneously, each pulled
towards whatever its own road's curvature implied, and the map ended up setting
the speed. It also meant the along-road uncertainty had to be represented by
*spawning siblings* along the road, which multiplied through every junction -
571 000 branches and 310 000 hypotheses culled by the beam limit on one 20
minute trip. With one global ``D``, along-road uncertainty is one number, the
siblings are unnecessary, and the population is the set of distinct road
states rather than a product of roads and speeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class RouteNode:
    """One link of a route, stored as a parent pointer so hypotheses share
    their common past."""

    edge: int
    parent: Optional["RouteNode"]
    depth: int
    entered_t: float
    offset: float
    """Route length up to the start of this edge."""

    @staticmethod
    def root(edge: int, t: float, offset: float = 0.0) -> "RouteNode":
        return RouteNode(edge=int(edge), parent=None, depth=0,
                         entered_t=float(t), offset=float(offset))

    def child(self, edge: int, t: float, offset: float) -> "RouteNode":
        return RouteNode(edge=int(edge), parent=self, depth=self.depth + 1,
                         entered_t=float(t), offset=float(offset))

    def edges(self) -> list[int]:
        out: list[int] = []
        node: Optional[RouteNode] = self
        while node is not None:
            out.append(node.edge)
            node = node.parent
        out.reverse()
        return out

    def tail(self, n: int) -> tuple[int, ...]:
        """Last ``n`` edges, oldest first. Feeds turn-restriction lookup."""
        out: list[int] = []
        node: Optional[RouteNode] = self
        while node is not None and len(out) < n:
            out.append(node.edge)
            node = node.parent
        out.reverse()
        return tuple(out)


@dataclass
class PacmanState:
    """Scalar view of one hypothesis, for diagnostics and reporting."""

    edge: int
    s: float
    route_offset: float
    weight: float
    log_weight: float
    rank: int
    route: list[int]
    normalized_rms: float
    hypothesis_id: int
    sigma_s: float

    def to_json(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "edge": self.edge,
            "distance_along_edge_m": round(self.s, 2),
            "route_offset_m": round(self.route_offset, 1),
            "sigma_s_m": round(self.sigma_s, 2),
            "weight": round(self.weight, 6),
            "log_weight": round(self.log_weight, 3),
            "rank": self.rank,
            "normalized_rms": round(self.normalized_rms, 3),
            "route_len": len(self.route),
            "route_tail": self.route[-6:],
        }


class HypothesisSet:
    """Structure-of-arrays population of route hypotheses."""

    __slots__ = ("edge", "route_offset", "offset_bias", "map_sigma_anchor",
                 "map_anchor_distance", "anchor_turn_angle", "anchor_turn_t",
                 "anchor_turn_window", "anchor_turn_map_offset",
                 "logw", "routes", "ids",
                 "strikes", "ewm_z2", "ewm_w", "turn_angle", "turn_t",
                 "turn_window", "spawned", "born_t", "_next_id")

    def __init__(self) -> None:
        self.edge = np.zeros(0, dtype=np.int64)
        self.route_offset = np.zeros(0, dtype=float)
        # Accumulated disagreement between the odometer's distance and this
        # route's length on the map. They are not the same quantity: the car
        # does not drive down the polyline centreline, and over kilometres the
        # difference is tens of metres. Corrected from the measured timing of
        # each junction turn.
        self.offset_bias = np.zeros(0, dtype=float)
        # Route-distance alignment is a local, per-route estimate. The mean is
        # ``offset_bias``; uncertainty grows with physical distance since the
        # last informative junction and is reset by a reliable turn anchor.
        self.map_sigma_anchor = np.zeros(0, dtype=float)
        self.map_anchor_distance = np.zeros(0, dtype=float)
        # A strong junction can mature after a short following edge has already
        # been crossed. Keep its alignment event separate from the turn-score
        # slot so descendants cannot overwrite the anchor before it resolves.
        self.anchor_turn_angle = np.zeros(0, dtype=float)
        self.anchor_turn_t = np.full(0, np.nan, dtype=float)
        self.anchor_turn_window = np.zeros(0, dtype=float)
        self.anchor_turn_map_offset = np.zeros(0, dtype=float)
        self.logw = np.zeros(0, dtype=float)
        self.routes: list[RouteNode] = []
        self.ids = np.zeros(0, dtype=np.int64)
        self.strikes = np.zeros(0, dtype=np.int32)
        self.ewm_z2 = np.zeros(0, dtype=float)
        self.ewm_w = np.zeros(0, dtype=float)
        # A turn taken at a junction, waiting for the gyro evidence that spans
        # it to mature. NaN when there is none outstanding.
        self.turn_angle = np.zeros(0, dtype=float)
        self.turn_t = np.full(0, np.nan, dtype=float)
        self.turn_window = np.zeros(0, dtype=float)
        # Whether this hypothesis has already spawned successors for the node
        # ahead of it. The crossing is not an instant - it is a stretch of road
        # over which we do not know whether it has happened - so successors are
        # spawned once on entering that stretch and the parent lives on beside
        # them until it is definitely past.
        self.spawned = np.zeros(0, dtype=bool)
        self.born_t = np.zeros(0, dtype=float)
        self._next_id = 0

    def __len__(self) -> int:
        return int(self.edge.shape[0])

    def allocate_ids(self, n: int) -> np.ndarray:
        ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
        self._next_id += n
        return ids

    def s(self, distance: float) -> np.ndarray:
        """Position along each hypothesis's current edge, from the global D."""
        return float(distance) - (self.route_offset + self.offset_bias)

    def map_distance_sigma(self, distance: float, relative_sigma: float,
                           floor_m: float, cap_m: float) -> np.ndarray:
        """Uncertainty in physical-distance <-> mapped-route alignment.

        Between geometric anchors the dominant error is an uncertain local
        path-length scale, so its standard deviation grows linearly with
        distance, rather than as an unbounded random walk over tracker ticks.
        """
        travelled = np.maximum(0.0, np.abs(float(distance) - self.map_anchor_distance))
        anchored = np.maximum(self.map_sigma_anchor, float(floor_m))
        sigma = np.hypot(anchored, float(relative_sigma) * travelled)
        return np.minimum(sigma, float(cap_m))

    def take(self, index: np.ndarray) -> "HypothesisSet":
        index = np.asarray(index, dtype=np.int64)
        out = HypothesisSet()
        out.edge = self.edge[index].copy()
        out.route_offset = self.route_offset[index].copy()
        out.offset_bias = self.offset_bias[index].copy()
        out.map_sigma_anchor = self.map_sigma_anchor[index].copy()
        out.map_anchor_distance = self.map_anchor_distance[index].copy()
        out.anchor_turn_angle = self.anchor_turn_angle[index].copy()
        out.anchor_turn_t = self.anchor_turn_t[index].copy()
        out.anchor_turn_window = self.anchor_turn_window[index].copy()
        out.anchor_turn_map_offset = self.anchor_turn_map_offset[index].copy()
        out.logw = self.logw[index].copy()
        out.routes = [self.routes[i] for i in index]
        out.ids = self.ids[index].copy()
        out.strikes = self.strikes[index].copy()
        out.ewm_z2 = self.ewm_z2[index].copy()
        out.ewm_w = self.ewm_w[index].copy()
        out.turn_angle = self.turn_angle[index].copy()
        out.turn_t = self.turn_t[index].copy()
        out.turn_window = self.turn_window[index].copy()
        out.spawned = self.spawned[index].copy()
        out.born_t = self.born_t[index].copy()
        out._next_id = self._next_id
        return out

    def weights(self) -> np.ndarray:
        if len(self) == 0:
            return np.zeros(0)
        m = self.logw.max()
        w = np.exp(self.logw - m)
        total = w.sum()
        return w / total if total > 0 else np.full(len(self), 1.0 / len(self))

    def normalized_rms(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.sqrt(np.where(self.ewm_w > 1e-9,
                                    self.ewm_z2 / np.maximum(self.ewm_w, 1e-9), 0.0))

    def order(self) -> np.ndarray:
        return np.argsort(-self.logw, kind="stable")

    def snapshot(self, distance: float, sigma_s: float | np.ndarray,
                 limit: Optional[int] = None) -> list[PacmanState]:
        w = self.weights()
        rms = self.normalized_rms()
        s = self.s(distance)
        sigma_values = np.broadcast_to(np.asarray(sigma_s, dtype=float), (len(self),))
        order = self.order()
        if limit is not None:
            order = order[:limit]
        return [
            PacmanState(
                edge=int(self.edge[i]), s=float(s[i]),
                route_offset=float(self.route_offset[i] + self.offset_bias[i]),
                weight=float(w[i]), log_weight=float(self.logw[i]), rank=rank,
                route=self.routes[i].edges(), normalized_rms=float(rms[i]),
                hypothesis_id=int(self.ids[i]), sigma_s=float(sigma_values[i]),
            )
            for rank, i in enumerate(order)
        ]

    def mass_on_edges(self, edges: Iterable[int]) -> float:
        wanted = np.fromiter((int(e) for e in edges), dtype=np.int64)
        if wanted.size == 0 or len(self) == 0:
            return 0.0
        return float(self.weights()[np.isin(self.edge, wanted)].sum())


def make_set(edges: Sequence[int], route_offsets: Sequence[float],
             log_weights: Sequence[float], routes: Sequence[RouteNode],
             t0: float) -> HypothesisSet:
    n = len(edges)
    hs = HypothesisSet()
    hs.edge = np.asarray(edges, dtype=np.int64).copy()
    hs.route_offset = np.asarray(route_offsets, dtype=float).copy()
    hs.offset_bias = np.zeros(n, dtype=float)
    hs.map_sigma_anchor = np.zeros(n, dtype=float)
    hs.map_anchor_distance = np.zeros(n, dtype=float)
    hs.anchor_turn_angle = np.zeros(n, dtype=float)
    hs.anchor_turn_t = np.full(n, np.nan, dtype=float)
    hs.anchor_turn_window = np.zeros(n, dtype=float)
    hs.anchor_turn_map_offset = np.zeros(n, dtype=float)
    hs.logw = np.asarray(log_weights, dtype=float).copy()
    hs.routes = list(routes)
    hs.strikes = np.zeros(n, dtype=np.int32)
    hs.ewm_z2 = np.zeros(n, dtype=float)
    hs.ewm_w = np.zeros(n, dtype=float)
    hs.turn_angle = np.zeros(n, dtype=float)
    hs.turn_t = np.full(n, np.nan, dtype=float)
    hs.turn_window = np.zeros(n, dtype=float)
    hs.spawned = np.zeros(n, dtype=bool)
    hs.born_t = np.full(n, float(t0), dtype=float)
    hs._next_id = 0
    hs.ids = hs.allocate_ids(n)
    return hs


def concat(a: HypothesisSet, b: HypothesisSet) -> HypothesisSet:
    if len(a) == 0:
        return b
    if len(b) == 0:
        return a
    out = HypothesisSet()
    out.edge = np.concatenate([a.edge, b.edge])
    out.route_offset = np.concatenate([a.route_offset, b.route_offset])
    out.offset_bias = np.concatenate([a.offset_bias, b.offset_bias])
    out.map_sigma_anchor = np.concatenate([a.map_sigma_anchor, b.map_sigma_anchor])
    out.map_anchor_distance = np.concatenate([a.map_anchor_distance, b.map_anchor_distance])
    out.anchor_turn_angle = np.concatenate([a.anchor_turn_angle, b.anchor_turn_angle])
    out.anchor_turn_t = np.concatenate([a.anchor_turn_t, b.anchor_turn_t])
    out.anchor_turn_window = np.concatenate([a.anchor_turn_window, b.anchor_turn_window])
    out.anchor_turn_map_offset = np.concatenate(
        [a.anchor_turn_map_offset, b.anchor_turn_map_offset])
    out.logw = np.concatenate([a.logw, b.logw])
    out.routes = a.routes + b.routes
    out.ids = np.concatenate([a.ids, b.ids])
    out.strikes = np.concatenate([a.strikes, b.strikes])
    out.ewm_z2 = np.concatenate([a.ewm_z2, b.ewm_z2])
    out.ewm_w = np.concatenate([a.ewm_w, b.ewm_w])
    out.turn_angle = np.concatenate([a.turn_angle, b.turn_angle])
    out.turn_t = np.concatenate([a.turn_t, b.turn_t])
    out.turn_window = np.concatenate([a.turn_window, b.turn_window])
    out.spawned = np.concatenate([a.spawned, b.spawned])
    out.born_t = np.concatenate([a.born_t, b.born_t])
    out._next_id = max(a._next_id, b._next_id)
    return out
