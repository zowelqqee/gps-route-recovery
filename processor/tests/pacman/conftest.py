import numpy as np
import pytest

from geotrace.pacman_tracker.config import PacmanConfig
from geotrace.pacman_tracker.roadmap import RoadGeometry
from geotrace.pacman_tracker.synthetic import arc, build_network, straight


@pytest.fixture
def straight_network():
    network, frame = build_network([("Long Straight", straight(600.0), {})])
    return network, frame


@pytest.fixture
def straight_geometry(straight_network):
    network, _ = straight_network
    return RoadGeometry(network, PacmanConfig().geometry)


@pytest.fixture
def junction_network():
    """A --+-- C with B branching left and D branching right, as in the brief.

                  B
                 /
    A ----------+--------- C
                 \\
                  D
    """
    node = (300.0, 0.0)
    ways = [
        ("A", straight(300.0, start=(0.0, 0.0), heading_rad=0.0), {}),
        ("C", straight(300.0, start=node, heading_rad=0.0), {}),
        ("B", straight(300.0, start=node, heading_rad=np.pi / 2), {}),
        ("D", straight(300.0, start=node, heading_rad=-np.pi / 2), {}),
    ]
    return build_network(ways)


@pytest.fixture
def curved_network():
    """Straight, then a 90-degree left arc, then straight again."""
    R = 50.0
    a = straight(200.0)
    b = arc(R, np.pi / 2, start=(200.0, 0.0))
    c = straight(200.0, start=(200.0 + R, R), heading_rad=np.pi / 2)
    return build_network([("A", a, {}), ("Bend", b, {}), ("C", c, {})])
