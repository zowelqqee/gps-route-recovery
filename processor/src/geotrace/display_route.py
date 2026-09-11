"""The displayed route during a GPS outage.

Without this, the drawn route during an outage is the free inertial estimate,
which is constrained by nothing at all: on
trip-32c24d86-06af-45fd-a3a0-79354f2cbb70 it left the Kantemirovsky bridge
sideways and ended 48 m out, in the river, right before trust returned.

The walker holds one piece of state across output ticks - which edge the
display is on, and how far along it - and moves it by the distance the EKF
itself moved. The particle cloud is consulted for exactly two things: which
successor to take at a junction, and a bounded along-track correction on the
edge the walker is already on.

That division of labour is what makes the route continuous. The alternative -
asking the cloud for a position every tick - reproduces the churn in
`branch_aware_estimate`, whose two nested argmaxes carry no memory of the
previous tick, so two near-equal branches swap freely. Here, once the walker
has committed to an edge at a junction, the sibling branch is no longer one of
that edge's successors, so it *cannot* be flipped back onto. The hysteresis
falls out of the topology and costs no tuning constant.

This is a display heuristic, not the particle posterior or a MAP trajectory.
The posterior is exported and evaluated separately as road_posterior.

The consequence: a wrong turn at a junction is not
recoverable until GPS trust returns, because re-syncing is exactly the jump
this class exists to forbid. `off_best_branch_ticks` makes that measurable
instead of invisible.

This class MUST NOT mutate the particle filter or the EKF, and MUST NOT draw a
random number - `test_the_run_is_reproducible` and
`test_the_walker_leaves_the_cloud_untouched` both depend on it, and being
RNG-free is what keeps the on-device Swift port tractable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from geotrace.config import Config
from geotrace.coordinates import wrap_angle
from geotrace.road_graph import RoadNetwork

_MAX_JUNCTIONS_PER_TICK = 20
"""An output tick is a second and city edges are often 10-30 m, so crossing
several in one tick is ordinary. The cap only stops a pathological graph from
spinning forever."""


@dataclass
class DisplayRouteWalker:
    """A position walked along connected road edges, one output tick at a time."""

    net: RoadNetwork
    cfg: Config
    edge: int
    s: float
    history: tuple[int, ...] = ()
    ticks: int = 0
    junctions_crossed: int = 0
    dead_ends: int = 0
    corrections_accepted: int = 0
    corrections_refused: int = 0
    off_best_branch_ticks: int = 0
    distance_m: float = 0.0
    """Distance represented by this live road walk during the current outage."""
    stalled: bool = False
    """The route ran out under a car that is still moving - see `advance`."""

    # ------------------------------------------------------------- lifecycle

    @classmethod
    def start(
        cls, net: RoadNetwork, pf: Any, ekf: Any, cfg: Config
    ) -> Optional["DisplayRouteWalker"]:
        """Snap the EKF's current position onto the graph, or refuse.

        Refusing matters as much as starting. A car in a courtyard, a private
        drive, or any road the extract does not have is genuinely off-graph,
        and snapping it would lock the whole outage onto a street it was never
        on - the same false confidence `PolygonConfig.off_road_distance_m`
        exists to keep out of the uncertainty, reappearing in the position.
        """
        xy = np.asarray(ekf.position, dtype=float)
        max_snap = cfg.pf.display_route_max_snap_m
        if net.distance_to_road(xy) > max_snap:
            return None
        candidates = net.nearest_edges(
            xy, k=cfg.pf.init_candidate_edges, radius=max(2.0 * max_snap, 1.0)
        )
        if not candidates:
            return None

        mass = _mass_by_edge(pf)
        heading = float(ekf.heading)
        best_index: Optional[int] = None
        best_score = -1.0
        best_s = 0.0
        # Sorted so an exact tie always resolves to the lowest edge index.
        for index in sorted(int(c) for c in candidates):
            s_on, offset = net.project(xy, index)
            if offset > max_snap:
                continue
            bearing_error = wrap_angle(net.edges[index].bearing(s_on) - heading)
            score = (
                math.exp(-(offset**2) / (2.0 * max(1.0, cfg.pf.init_radius_m) ** 2))
                * math.exp(-(bearing_error**2) / (2.0 * cfg.pf.sigma_heading_rad**2))
                * (float(mass.get(index, 0.0)) + cfg.pf.display_route_min_branch_mass)
            )
            if score > best_score:
                best_score, best_index, best_s = score, index, float(s_on)
        if best_index is None:
            return None
        return cls(
            net=net, cfg=cfg, edge=best_index, s=best_s, history=(best_index,)
        )

    # -------------------------------------------------------------- stepping

    def advance(
        self,
        pf: Any,
        ekf: Any,
        ds: float,
        dt: float,
        best_edges: Sequence[int] = (),
    ) -> tuple[float, float]:
        """One output tick. ``ds`` is how far the EKF itself moved."""
        self.ticks += 1
        travelled = max(0.0, float(ds))
        self.distance_m += travelled
        s_before = self.s

        budget = travelled + self.cfg.pf.display_route_budget_margin_m
        reach = self.net.reachable_within_distance(
            self.edge, self.s, budget, self.history
        )
        # First-hop labels are relative to the edge the tick started on, so
        # they answer the first junction. Should the tick span a second one,
        # every option scores the same floor and the choice falls to heading
        # alone - which is the honest reading: the cloud expressed an opinion
        # about where this tick's travel goes, not about every junction inside
        # it.
        mass_by_hop = _mass_by_first_hop(pf, reach)

        self.s += travelled
        for _ in range(_MAX_JUNCTIONS_PER_TICK):
            length = float(self.net.edges[self.edge].length)
            if self.s <= length:
                break
            remainder = self.s - length
            successor = self._choose_successor(mass_by_hop, float(ekf.heading))
            if successor is None:
                # The route ended and the car did not. Freezing here would peg
                # the drawn position in place for the rest of the outage while
                # the car drives on - worse than the unconstrained estimate
                # this replaced, and silently so. Only about 0.1% of edges in
                # a real extract are terminal even counting a U-turn, but one
                # of them stalled a walker for 619 ticks on
                # trip-b4faeae0-a941-4a87-9b18-de7aaa84f721. Give up instead
                # and let the caller fall back; the honest signal is a broken
                # line, not a stopped one.
                self.s = length
                self.dead_ends += 1
                self.stalled = travelled > 0.0
                break
            self.edge = successor
            self.s = min(remainder, float(self.net.edges[successor].length))
            self.history = (self.history + (successor,))[
                -max(1, self.net.restriction_history_limit) :
            ]
            self.junctions_crossed += 1
        self.s = float(np.clip(self.s, 0.0, float(self.net.edges[self.edge].length)))

        self._apply_along_track_correction(pf, ekf, s_floor=min(s_before, self.s))

        if best_edges and self.edge not in {int(e) for e in best_edges}:
            self.off_best_branch_ticks += 1
        return self.position()

    def _choose_successor(
        self, mass_by_hop: dict[int, float], heading: float
    ) -> Optional[int]:
        """Which way out of this junction, by cloud weight and gyro heading.

        Both terms are needed. Weight alone loses the fork the moment the cloud
        is undecided - which is precisely when a junction is hardest and when
        `heading_consensus` is documented to sit at 0.7-0.85. Heading alone
        ignores everything the map filter has learned. Using `ekf.heading` here
        is the IMU answering "which way did we turn"; it flows road -> display
        only, never back into the filter.
        """
        options = self.net.allowed_successors(
            self.edge, self.history, allow_uturn=False
        )
        if len(options) == 0:
            # A car at a dead end really does turn around.
            options = self.net.allowed_successors(
                self.edge, self.history, allow_uturn=True
            )
        if len(options) == 0:
            return None

        floor = self.cfg.pf.display_route_min_branch_mass
        sigma_turn = self.cfg.pf.sigma_turn_rad
        best_option: Optional[int] = None
        best_score = -1.0
        for option in sorted(int(o) for o in options):
            bearing_error = wrap_angle(self.net.edges[option].bearing(0.0) - heading)
            score = (mass_by_hop.get(option, 0.0) + floor) * math.exp(
                -(bearing_error**2) / (2.0 * sigma_turn**2)
            )
            if score > best_score:
                best_score, best_option = score, option
        return best_option

    def _apply_along_track_correction(self, pf: Any, ekf: Any, s_floor: float) -> None:
        """Let the cloud pull the walker along its current edge, if the EKF agrees.

        This is the only place the cloud may move the display other than by
        choosing a branch, and it is bounded twice: it never draws backwards,
        and it is refused outright when the result would sit further from the
        EKF's own position than that filter's own admitted uncertainty allows
        (`display_route_ekf_sigma_k`). That test is self-scaling - metres of
        sigma seconds into an outage, hundreds of metres deep into one - which
        is the whole point: late in a long outage unaided inertial position
        carries no information and the road should win.
        """
        edge_idx = np.asarray(pf.edge_idx)
        weights = np.asarray(pf.w, dtype=float)
        on_edge = edge_idx == self.edge
        mass_here = float(weights[on_edge].sum())
        if mass_here < self.cfg.pf.display_route_min_branch_mass:
            return

        s_cloud = float(
            np.average(np.asarray(pf.s, dtype=float)[on_edge], weights=weights[on_edge])
        )
        gain = self.cfg.pf.display_route_along_gain
        length = float(self.net.edges[self.edge].length)
        s_new = float(np.clip(self.s + gain * (s_cloud - self.s), s_floor, length))

        candidate = self.net.edges[self.edge].position(s_new)
        sigma = math.sqrt(max(1e-6, float(ekf.P[0, 0]) + float(ekf.P[1, 1])))
        if math.dist(candidate, ekf.position) <= self.cfg.pf.display_route_ekf_sigma_k * sigma:
            self.s = s_new
            self.corrections_accepted += 1
        else:
            self.corrections_refused += 1

    # ------------------------------------------------------------- accessors

    def position(self) -> tuple[float, float]:
        return self.net.edges[self.edge].position(self.s)

    def to_json(self) -> dict[str, Any]:
        return {
            "ticks": self.ticks,
            "junctions_crossed": self.junctions_crossed,
            "dead_ends": self.dead_ends,
            "corrections_accepted": self.corrections_accepted,
            "corrections_refused": self.corrections_refused,
            "off_best_branch_ticks": self.off_best_branch_ticks,
            "distance_m": self.distance_m,
        }


def _mass_by_edge(pf: Any) -> dict[int, float]:
    """Particle weight per occupied edge, accumulated in sorted edge order."""
    edge_idx = np.asarray(pf.edge_idx)
    weights = np.asarray(pf.w, dtype=float)
    if edge_idx.size == 0:
        return {}
    unique, inverse = np.unique(edge_idx, return_inverse=True)
    totals = np.bincount(inverse, weights=weights, minlength=unique.size)
    return {int(e): float(m) for e, m in zip(unique, totals)}


def _mass_by_first_hop(pf: Any, reach: dict[int, tuple[float, int]]) -> dict[int, float]:
    """Particle weight sorted into "which way out of here reaches it" buckets."""
    out: dict[int, float] = {}
    for edge, mass in sorted(_mass_by_edge(pf).items()):
        entry = reach.get(edge)
        if entry is None:
            continue
        out[entry[1]] = out.get(entry[1], 0.0) + mass
    return out
