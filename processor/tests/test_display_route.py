"""The displayed route walker: connectivity, junction choice, sigma gating."""

from __future__ import annotations

import math

import numpy as np
import pytest

from geotrace.config import Config
from geotrace.display_route import DisplayRouteWalker
from geotrace.ekf import ExtendedKalmanFilter
from geotrace.particle_filter import RoadParticleFilter
from geotrace.road_graph import RoadNetwork, build_graph_from_segments

from conftest import ORIGIN_LAT, ORIGIN_LON, edge_named


def make_pf(network: RoadNetwork, cfg: Config, n: int = 200) -> RoadParticleFilter:
    cfg.pf.n_particles = n
    return RoadParticleFilter(network, cfg, rng=np.random.default_rng(cfg.seed))


def make_ekf(cfg: Config, xy=(0.0, 0.0), heading=0.0, speed=10.0, sigma=5.0):
    ekf = ExtendedKalmanFilter(
        cfg.motion, initial_state=[xy[0], xy[1], speed, heading, 0.0, 0.0]
    )
    ekf.P[0, 0] = sigma**2
    ekf.P[1, 1] = sigma**2
    return ekf


def put_cloud_on(pf: RoadParticleFilter, edge: int, s: float = 10.0) -> None:
    """Move the whole cloud onto one edge without touching anything else."""
    pf.edge_idx[:] = edge
    pf.s[:] = s
    pf.w[:] = 1.0 / len(pf.w)


@pytest.fixture
def two_islands() -> RoadNetwork:
    """Two streets that never connect, 20 m apart - a river between them."""
    segments = [
        ("Island A", [(0.0, 0.0), (0.0, 400.0)], {"highway": "secondary"}),
        ("Island B", [(20.0, 0.0), (20.0, 400.0)], {"highway": "secondary"}),
    ]
    graph, _frame = build_graph_from_segments(segments, ORIGIN_LAT, ORIGIN_LON)
    return RoadNetwork(graph, _frame)


# ------------------------------------------------------------------ starting


def test_the_walker_refuses_to_start_far_from_every_edge(fork_network, config) -> None:
    """A courtyard or an unmapped drive must not be snapped onto a street:
    that would lock the whole outage onto a road the car was never on."""
    pf = make_pf(fork_network, config)
    pf.initialize((100.0, 0.0), heading=0.0, speed=10.0)
    far = config.pf.display_route_max_snap_m + 50.0
    ekf = make_ekf(config, xy=(100.0, far))
    assert DisplayRouteWalker.start(fork_network, pf, ekf, config) is None


def test_the_walker_starts_on_the_edge_under_the_car(fork_network, config) -> None:
    pf = make_pf(fork_network, config)
    pf.initialize((100.0, 0.0), heading=0.0, speed=10.0)
    ekf = make_ekf(config, xy=(100.0, 2.0))
    walker = DisplayRouteWalker.start(fork_network, pf, ekf, config)
    assert walker is not None
    assert walker.edge == edge_named(fork_network, "Stem", (0.0, 0.0))
    assert walker.s == pytest.approx(100.0, abs=1.0)
    assert fork_network.distance_to_road(walker.position()) < 1e-6


# -------------------------------------------------------------- connectivity


def test_the_walker_never_jumps_to_a_disconnected_street(two_islands, config) -> None:
    """Constraint A: the whole cloud sitting on an unreachable street is not a
    reason to teleport there - the car cannot have driven across the water."""
    island_a = edge_named(two_islands, "Island A", (0.0, 0.0))
    island_b = edge_named(two_islands, "Island B", (20.0, 0.0))
    pf = make_pf(two_islands, config)
    pf.initialize((0.0, 50.0), heading=math.pi / 2, speed=10.0)
    ekf = make_ekf(config, xy=(0.0, 50.0), heading=math.pi / 2)
    walker = DisplayRouteWalker.start(two_islands, pf, ekf, config)
    assert walker is not None and walker.edge == island_a

    put_cloud_on(pf, island_b, s=60.0)
    for _ in range(30):
        walker.advance(pf, ekf, ds=5.0, dt=1.0)
        assert walker.edge == island_a, "crossed to a street with no connection"


def test_the_walker_respects_a_one_way_street(oneway_network, config) -> None:
    one_way = edge_named(oneway_network, "One way east", (300.0, 0.0))
    stem = edge_named(oneway_network, "Stem", (0.0, 0.0))
    pf = make_pf(oneway_network, config)
    pf.initialize((400.0, 0.0), heading=0.0, speed=10.0)
    ekf = make_ekf(config, xy=(400.0, 0.0))
    walker = DisplayRouteWalker.start(oneway_network, pf, ekf, config)
    assert walker is not None and walker.edge == one_way

    put_cloud_on(pf, stem, s=100.0)
    for _ in range(40):
        walker.advance(pf, ekf, ds=20.0, dt=1.0)
        assert walker.edge != stem, "walked the wrong way down a one-way street"


# ----------------------------------------------------------- junction choice


def _fork_walker_at_junction(network, config, heading):
    stem = edge_named(network, "Stem", (0.0, 0.0))
    pf = make_pf(network, config, n=400)
    pf.initialize((480.0, 0.0), heading=0.0, speed=10.0)
    ekf = make_ekf(config, xy=(480.0, 0.0), heading=heading)
    walker = DisplayRouteWalker.start(network, pf, ekf, config)
    assert walker is not None and walker.edge == stem
    return pf, ekf, walker


def test_the_walker_follows_the_gyro_at_a_fork_when_the_cloud_is_undecided(
    fork_network, config
) -> None:
    """The undecided cloud is exactly the case a fork is hardest, and the only
    thing that knows about the turn is the gyro."""
    branch_a = edge_named(fork_network, "Branch A", (500.0, 0.0))
    branch_b = edge_named(fork_network, "Branch B", (500.0, 0.0))
    pf, ekf, walker = _fork_walker_at_junction(
        fork_network, config, heading=math.radians(45.0)
    )
    half = len(pf.edge_idx) // 2
    pf.edge_idx[:half] = branch_a
    pf.edge_idx[half:] = branch_b
    pf.s[:] = 20.0
    pf.w[:] = 1.0 / len(pf.w)

    walker.advance(pf, ekf, ds=40.0, dt=1.0)
    assert walker.edge == branch_a, "north-east heading must pick the north-east branch"


def test_the_walker_follows_the_cloud_at_a_fork_when_the_cloud_is_decided(
    fork_network, config
) -> None:
    branch_b = edge_named(fork_network, "Branch B", (500.0, 0.0))
    pf, ekf, walker = _fork_walker_at_junction(fork_network, config, heading=0.0)
    put_cloud_on(pf, branch_b, s=20.0)
    walker.advance(pf, ekf, ds=40.0, dt=1.0)
    assert walker.edge == branch_b


def test_the_walker_does_not_flip_back_after_committing_to_a_branch(
    fork_network, config
) -> None:
    """The regression this design exists to prevent: `branch_aware_estimate`
    re-runs two unmemoised argmaxes every tick, so two near-equal branches
    swap freely. Once the walker is on a branch the sibling is no longer a
    successor of it, so no amount of cloud weight can pull it back."""
    branch_a = edge_named(fork_network, "Branch A", (500.0, 0.0))
    branch_b = edge_named(fork_network, "Branch B", (500.0, 0.0))
    pf, ekf, walker = _fork_walker_at_junction(
        fork_network, config, heading=math.radians(45.0)
    )
    put_cloud_on(pf, branch_a, s=20.0)
    walker.advance(pf, ekf, ds=40.0, dt=1.0)
    assert walker.edge == branch_a

    put_cloud_on(pf, branch_b, s=200.0)
    for _ in range(25):
        walker.advance(pf, ekf, ds=10.0, dt=1.0)
        assert walker.edge != branch_b


# --------------------------------------------------- the EKF sigma arbiter


def test_the_along_track_correction_is_refused_beyond_k_sigma(
    fork_network, config
) -> None:
    """Constraint B: a confident EKF refuses a road answer that contradicts it."""
    stem = edge_named(fork_network, "Stem", (0.0, 0.0))
    pf = make_pf(fork_network, config)
    pf.initialize((100.0, 0.0), heading=0.0, speed=10.0)
    ekf = make_ekf(config, xy=(100.0, 0.0), sigma=1.0)  # very sure of itself
    walker = DisplayRouteWalker.start(fork_network, pf, ekf, config)
    assert walker is not None and walker.edge == stem

    put_cloud_on(pf, stem, s=400.0)  # the cloud is 300 m further along
    walker.advance(pf, ekf, ds=0.0, dt=1.0)
    assert walker.corrections_refused == 1
    assert walker.corrections_accepted == 0
    assert walker.s == pytest.approx(100.0, abs=1.0)


def test_the_along_track_correction_is_accepted_once_sigma_is_large(
    fork_network, config
) -> None:
    """The same disagreement, once the EKF admits it no longer knows: deep in
    an outage the road is the better answer and must be allowed to win."""
    stem = edge_named(fork_network, "Stem", (0.0, 0.0))
    pf = make_pf(fork_network, config)
    pf.initialize((100.0, 0.0), heading=0.0, speed=10.0)
    ekf = make_ekf(config, xy=(100.0, 0.0), sigma=400.0)  # minutes of blind drift
    walker = DisplayRouteWalker.start(fork_network, pf, ekf, config)
    assert walker is not None

    put_cloud_on(pf, stem, s=400.0)
    walker.advance(pf, ekf, ds=0.0, dt=1.0)
    assert walker.corrections_accepted == 1
    assert walker.s > 150.0


# ------------------------------------------------------------- invariants


def test_the_walker_leaves_the_cloud_untouched(fork_network, config) -> None:
    """It is a reader of the filter, never a writer - the reproducibility of
    the whole run depends on it drawing no random numbers either."""
    pf = make_pf(fork_network, config)
    pf.initialize((100.0, 0.0), heading=0.0, speed=10.0)
    ekf = make_ekf(config, xy=(100.0, 0.0))
    walker = DisplayRouteWalker.start(fork_network, pf, ekf, config)
    assert walker is not None

    before = (
        pf.edge_idx.copy(), pf.s.copy(), pf.w.copy(), pf.psi.copy(), pf.v.copy(),
    )
    ekf_before = (ekf.x.copy(), ekf.P.copy())
    for _ in range(50):
        walker.advance(pf, ekf, ds=8.0, dt=1.0)
    assert np.array_equal(pf.edge_idx, before[0])
    assert np.array_equal(pf.s, before[1])
    assert np.array_equal(pf.w, before[2])
    assert np.array_equal(pf.psi, before[3])
    assert np.array_equal(pf.v, before[4])
    assert np.array_equal(ekf.x, ekf_before[0])
    assert np.array_equal(ekf.P, ekf_before[1])


def test_the_walker_never_draws_backwards(fork_network, config) -> None:
    stem = edge_named(fork_network, "Stem", (0.0, 0.0))
    pf = make_pf(fork_network, config)
    pf.initialize((200.0, 0.0), heading=0.0, speed=10.0)
    ekf = make_ekf(config, xy=(200.0, 0.0), sigma=300.0)
    walker = DisplayRouteWalker.start(fork_network, pf, ekf, config)
    assert walker is not None

    put_cloud_on(pf, stem, s=20.0)  # the cloud is well behind the walker
    seen = [walker.s]
    for _ in range(20):
        walker.advance(pf, ekf, ds=1.0, dt=1.0)
        seen.append(walker.s)
    assert all(b >= a - 1e-9 for a, b in zip(seen, seen[1:])), seen


def test_the_walker_crosses_several_short_edges_in_one_tick(
    grid_network, config
) -> None:
    """A second at city speed can span several OSM fragments; the walker must
    keep going rather than stopping at the first junction."""
    pf = make_pf(grid_network, config)
    first = grid_network.edges[0]
    start = np.array(first.position(0.0))
    pf.initialize(start, heading=first.bearing(0.0), speed=15.0)
    ekf = make_ekf(config, xy=start, heading=first.bearing(0.0), sigma=50.0)
    walker = DisplayRouteWalker.start(grid_network, pf, ekf, config)
    assert walker is not None

    # Enough to run off the end of the starting edge whatever its length.
    walker.advance(pf, ekf, ds=2.5 * float(grid_network.edges[walker.edge].length), dt=1.0)
    assert walker.junctions_crossed >= 1
    assert grid_network.distance_to_road(walker.position()) < 1e-6


def test_a_dead_end_turns_the_walker_round_rather_than_teleporting_it(
    two_islands, config
) -> None:
    """At the end of an isolated street a car really does turn around. What it
    cannot do is reappear on the street across the water, and the walker must
    stay inside its own connected component however long it is driven."""
    island_a = edge_named(two_islands, "Island A", (0.0, 0.0))
    island_b = edge_named(two_islands, "Island B", (20.0, 0.0))
    component = {island_a, edge_named(two_islands, "Island A", (0.0, 400.0))}
    pf = make_pf(two_islands, config)
    pf.initialize((0.0, 380.0), heading=math.pi / 2, speed=10.0)
    ekf = make_ekf(config, xy=(0.0, 380.0), heading=math.pi / 2)
    walker = DisplayRouteWalker.start(two_islands, pf, ekf, config)
    assert walker is not None and walker.edge == island_a

    put_cloud_on(pf, island_b, s=200.0)
    for _ in range(10):
        point = walker.advance(pf, ekf, ds=50.0, dt=1.0)
        assert walker.edge in component, "left its own connected component"
        assert abs(point[0]) < 1.0, "drifted across to the other bank"


def test_a_route_that_runs_out_under_a_moving_car_stalls_rather_than_freezing(
    config,
) -> None:
    """A terminal edge must not peg the drawn position in place for the rest of
    the outage while the car drives on - that is worse, and quieter, than the
    unconstrained estimate this replaced. Real extracts are only about 0.1%
    such edges, but one of them held a walker still for 619 ticks on
    trip-b4faeae0-a941-4a87-9b18-de7aaa84f721."""
    segments = [
        ("Spur", [(0.0, 0.0), (0.0, 60.0)], {"highway": "service", "oneway": True})
    ]
    graph, frame = build_graph_from_segments(segments, ORIGIN_LAT, ORIGIN_LON)
    net = RoadNetwork(graph, frame)
    spur = edge_named(net, "Spur", (0.0, 0.0))
    pf = make_pf(net, config)
    pf.initialize((0.0, 10.0), heading=math.pi / 2, speed=10.0)
    ekf = make_ekf(config, xy=(0.0, 10.0), heading=math.pi / 2)
    walker = DisplayRouteWalker.start(net, pf, ekf, config)
    assert walker is not None and walker.edge == spur
    assert not walker.stalled

    walker.advance(pf, ekf, ds=200.0, dt=1.0)  # straight off the end
    assert walker.stalled, "a moving car with no route left must give up"


def test_a_stationary_car_at_the_end_of_a_route_does_not_stall(config) -> None:
    """Sitting still where a street ends is not that failure: there is nothing
    left to follow, but nothing is going wrong either."""
    segments = [
        ("Spur", [(0.0, 0.0), (0.0, 60.0)], {"highway": "service", "oneway": True})
    ]
    graph, frame = build_graph_from_segments(segments, ORIGIN_LAT, ORIGIN_LON)
    net = RoadNetwork(graph, frame)
    pf = make_pf(net, config)
    pf.initialize((0.0, 55.0), heading=math.pi / 2, speed=0.0)
    ekf = make_ekf(config, xy=(0.0, 55.0), heading=math.pi / 2, speed=0.0)
    walker = DisplayRouteWalker.start(net, pf, ekf, config)
    assert walker is not None

    for _ in range(5):
        walker.advance(pf, ekf, ds=0.0, dt=1.0)
    assert not walker.stalled


def test_the_walker_stays_on_the_graph_every_tick(fork_network, config) -> None:
    pf = make_pf(fork_network, config)
    pf.initialize((100.0, 0.0), heading=0.0, speed=10.0)
    ekf = make_ekf(config, xy=(100.0, 0.0), sigma=100.0)
    walker = DisplayRouteWalker.start(fork_network, pf, ekf, config)
    assert walker is not None
    for _ in range(60):
        point = walker.advance(pf, ekf, ds=9.0, dt=1.0)
        assert fork_network.distance_to_road(point) < 1e-6
