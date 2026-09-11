"""Read-only turn-event instrumentation for beam combinatorics."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from geotrace.pacman_tracker.single_path import turn_kind
from geotrace.pacman_tracker.turns import TurnEvent


class CompositeObserver:
    def __init__(self, *observers) -> None:
        self.observers = observers

    def observe(self, *args, **kwargs) -> None:
        for observer in self.observers:
            observer.observe(*args, **kwargs)


class BeamTurnObserver:
    """Capture population structure before/after each strong IMU turn.

    The optional ground truth is read only for rank/mass diagnostics and never
    returned to the tracker.
    """

    def __init__(self, events: Sequence[TurnEvent], truth, settle_s: float = 6.5,
                 top_n: int = 20) -> None:
        self.events = list(events)
        self.truth = truth
        self.settle_s = float(settle_s)
        self.top_n = int(top_n)
        self.before: dict[int, dict[str, Any]] = {}
        self.after: dict[int, dict[str, Any]] = {}

    def observe(self, t, hs, sample, match, manager, distance=0.0,
                sigma_s=0.0, speed=0.0) -> None:
        for i, event in enumerate(self.events):
            if i not in self.before and t >= event.t_start:
                self.before[i] = self._snapshot(
                    hs, manager, distance, event, t, compact=True)
            if i not in self.after and t >= event.t_end + self.settle_s:
                self.after[i] = self._snapshot(
                    hs, manager, distance, event, t, compact=False)

    def _snapshot(self, hs, manager, distance, event, t,
                  compact: bool) -> dict[str, Any]:
        n = len(hs)
        if n == 0:
            return {"t": round(float(t), 2), "population": 0}
        weights = hs.weights()
        order = hs.order()
        histories = {tuple(route.edges()) for route in hs.routes}
        edge_mass: dict[int, float] = {}
        for edge, weight in zip(hs.edge, weights):
            edge_mass[int(edge)] = edge_mass.get(int(edge), 0.0) + float(weight)
        masses = np.array(list(edge_mass.values()), dtype=float)
        entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1e-300))))
        edge_entropy = float(-np.sum(masses * np.log(np.maximum(masses, 1e-300))))
        turns = np.zeros(n)
        for j, route in enumerate(hs.routes):
            if route.parent is not None:
                turns[j] = manager.geometry.junction_turn(route.parent.edge, route.edge)
        measured_kind = turn_kind(event.delta_psi)
        kinds = np.array([turn_kind(x) for x in turns])
        cells = np.floor((hs.route_offset + hs.offset_bias)
                         / max(manager.cfg.merge_s_tol_m, 1e-6)).astype(np.int64)
        dedup_keys = set(zip((int(x) for x in hs.edge), (int(x) for x in cells)))
        gt_edge = self.truth.edge_at(event.t_end + 1.0)
        hits = np.nonzero(hs.edge == gt_edge)[0] if gt_edge is not None else np.zeros(0, int)
        ranks = np.empty(n, dtype=np.int64)
        ranks[order] = np.arange(n)
        result = {
            "t": round(float(t), 2),
            "population": n,
            "unique_physical_histories": len(histories),
            "unique_current_edges": len(edge_mass),
            "deduplicatable_states": n - len(dedup_keys),
            "turn_kind_counts": {kind: int(np.sum(kinds == kind))
                                 for kind in ("left", "straight", "right")},
            "compatible_count": int(np.sum(kinds == measured_kind)),
            "incompatible_count": int(np.sum(kinds != measured_kind)),
            "top1_mass": round(float(weights[order[:1]].sum()), 6),
            "top5_mass": round(float(weights[order[:5]].sum()), 6),
            "top20_mass": round(float(weights[order[:20]].sum()), 6),
            "entropy_nats": round(entropy, 4),
            "effective_hypotheses": round(math.exp(entropy), 3),
            "edge_entropy_nats": round(edge_entropy, 4),
            "effective_current_edges": round(math.exp(edge_entropy), 3),
            "truth_edge": gt_edge,
            "truth_rank": (int(np.min(ranks[hits])) if hits.size else None),
            "truth_mass": round(float(weights[hits].sum()), 8) if hits.size else 0.0,
        }
        if not compact:
            result["top_routes"] = [
                {
                    "rank": rank + 1,
                    "edge": int(hs.edge[j]),
                    "raw_log_score": round(float(hs.logw[j]), 4),
                    "mass": round(float(weights[j]), 8),
                    "last_turn_deg": round(math.degrees(float(turns[j])), 2),
                    "route_tail": hs.routes[j].tail(10),
                }
                for rank, j in enumerate(order[:self.top_n])
            ]
        return result

    def to_json(self) -> list[dict[str, Any]]:
        rows = []
        for i, event in enumerate(self.events):
            rows.append({
                **event.to_json(),
                "before": self.before.get(i),
                "after": self.after.get(i),
            })
        return rows
