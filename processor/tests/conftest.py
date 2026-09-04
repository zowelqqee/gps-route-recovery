"""Shared fixtures.

The synthetic graphs here are deliberately tiny and hand-drawn so that the
expected answer can be worked out on paper.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from geotrace.config import Config
from geotrace.road_graph import RoadNetwork, build_graph_from_segments

ORIGIN_LAT = 59.9311
ORIGIN_LON = 30.3609


@pytest.fixture
def config() -> Config:
    cfg = Config()
    cfg.pf.n_particles = 400
    return cfg


@pytest.fixture
def fork_network() -> RoadNetwork:
    """The integration graph from the specification::

                    branch A
                   /
        start -- junction
                   \\
                    branch B

    The stem runs due east for 500 m; branch A leaves the junction heading
    north-east, branch B south-east. Both are 700 m long.
    """
    segments = [
        ("Stem", [(0.0, 0.0), (500.0, 0.0)], {"highway": "secondary"}),
        ("Branch A", [(500.0, 0.0), (1000.0, 500.0), (1400.0, 900.0)], {"highway": "secondary"}),
        ("Branch B", [(500.0, 0.0), (1000.0, -500.0), (1400.0, -900.0)], {"highway": "secondary"}),
    ]
    graph, frame = build_graph_from_segments(segments, ORIGIN_LAT, ORIGIN_LON)
    return RoadNetwork(graph, frame)


@pytest.fixture(scope="session")
def grid_network() -> RoadNetwork:
    """The larger built-in street grid, with loops and a one-way street.

    The fork fixture is deliberately tiny so that branch behaviour can be read
    off by hand; it is too small for a long drive, because the car reaches a
    dead end and a large injected offset lands on the other branch by accident.
    Anything that needs room to drive uses this instead.
    """
    from geotrace.simulate import spb_synthetic_network

    return spb_synthetic_network(ORIGIN_LAT, ORIGIN_LON)


@pytest.fixture
def oneway_network() -> RoadNetwork:
    """A stem feeding a one-way street that may only be driven eastwards."""
    segments = [
        ("Stem", [(0.0, 0.0), (300.0, 0.0)], {"highway": "residential"}),
        ("One way east", [(300.0, 0.0), (900.0, 0.0)], {"highway": "residential", "oneway": True}),
        ("Side", [(300.0, 0.0), (300.0, 400.0)], {"highway": "residential"}),
    ]
    graph, frame = build_graph_from_segments(segments, ORIGIN_LAT, ORIGIN_LON)
    return RoadNetwork(graph, frame)


def edge_named(network: RoadNetwork, name: str, u_at: tuple[float, float]) -> int:
    """Index of the edge called ``name`` that starts at local point ``u_at``."""
    for edge in network.edges:
        if str(edge.name) != name:
            continue
        if math.dist(edge.coords[0], u_at) < 1.0:
            return edge.index
    raise AssertionError(f"no edge {name!r} starting near {u_at}")
