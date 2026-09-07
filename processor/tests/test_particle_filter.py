"""Particle filter mechanics: resampling, weights, junctions, one-way streets."""

from __future__ import annotations

import math

import numpy as np
import pytest

from geotrace.config import Config
from geotrace.coordinates import wrap_angle
from geotrace.ekf import ExtendedKalmanFilter
from geotrace.particle_filter import (
    RoadParticleFilter,
    effective_sample_size,
    normalize_weights,
    systematic_resample,
)
from geotrace.pipeline import _outage_heading_assist
from geotrace.road_graph import RoadNetwork
from geotrace.road_graph import TurnRestriction

from conftest import edge_named


# ------------------------------------------------------------ weight algebra


def test_normalize_weights_sums_to_one() -> None:
    w = normalize_weights(np.array([1.0, 3.0, 6.0]))
    assert w.sum() == pytest.approx(1.0)
    assert w == pytest.approx([0.1, 0.3, 0.6])


def test_normalize_weights_preserves_ratios() -> None:
    raw = np.array([2.0, 8.0, 10.0])
    w = normalize_weights(raw)
    assert w[1] / w[0] == pytest.approx(4.0)


def test_normalize_weights_handles_a_total_collapse() -> None:
    """All-zero weights mean the filter knows nothing; uniform says exactly that
    and is far better than a division by zero."""
    w = normalize_weights(np.zeros(5))
    assert w == pytest.approx(np.full(5, 0.2))


def test_normalize_weights_ignores_nan_and_negatives() -> None:
    w = normalize_weights(np.array([1.0, np.nan, -2.0, 3.0]))
    assert w.sum() == pytest.approx(1.0)
    assert w[1] == 0.0 and w[2] == 0.0


def test_effective_sample_size_of_uniform_weights_is_n() -> None:
    assert effective_sample_size(np.full(100, 0.01)) == pytest.approx(100.0)


def test_effective_sample_size_of_a_degenerate_set_is_one() -> None:
    w = np.zeros(100)
    w[0] = 1.0
    assert effective_sample_size(w) == pytest.approx(1.0)


# ---------------------------------------------------------------- resampling


def test_systematic_resample_returns_n_indices() -> None:
    rng = np.random.default_rng(0)
    picks = systematic_resample(normalize_weights(rng.random(500)), rng)
    assert len(picks) == 500
    assert picks.min() >= 0 and picks.max() < 500


def test_systematic_resample_follows_the_weights() -> None:
    """A particle holding 70% of the mass must get roughly 70% of the copies."""
    rng = np.random.default_rng(7)
    w = np.array([0.7, 0.2, 0.1])
    counts = np.bincount(systematic_resample(w, rng), minlength=3)
    assert counts.sum() == 3
    picks = np.bincount(systematic_resample(np.repeat(w / 100, 100), rng), minlength=300)
    assert picks[:100].sum() == pytest.approx(210, abs=3)


def test_systematic_resample_keeps_only_the_surviving_particle() -> None:
    rng = np.random.default_rng(1)
    w = np.zeros(50)
    w[17] = 1.0
    assert np.all(systematic_resample(w, rng) == 17)


def test_systematic_resample_is_low_variance() -> None:
    """Systematic resampling must reproduce the expected counts far more tightly
    than multinomial resampling of the same weights."""
    rng = np.random.default_rng(3)
    n = 1000
    w = normalize_weights(np.ones(n))
    counts = np.bincount(systematic_resample(w, rng), minlength=n)
    assert counts.max() <= 2, "uniform weights must give an almost exact 1:1 copy"


def test_systematic_resample_is_deterministic_for_a_seed() -> None:
    w = normalize_weights(np.random.default_rng(5).random(200))
    a = systematic_resample(w, np.random.default_rng(99))
    b = systematic_resample(w, np.random.default_rng(99))
    assert np.array_equal(a, b)


def test_systematic_resample_of_an_empty_set() -> None:
    assert len(systematic_resample(np.zeros(0), np.random.default_rng(0))) == 0


# ------------------------------------------------------------- graph queries


def test_successors_at_a_fork(fork_network: RoadNetwork) -> None:
    """The stem leads to exactly two branches, and not back down itself."""
    stem = edge_named(fork_network, "Stem", (0.0, 0.0))
    successors = fork_network.successors(stem)
    assert len(successors) == 2
    names = {str(fork_network.edges[i].name) for i in successors}
    assert names == {"Branch A", "Branch B"}


def test_uturn_is_forbidden_by_default(fork_network: RoadNetwork) -> None:
    stem = edge_named(fork_network, "Stem", (0.0, 0.0))
    reverse = edge_named(fork_network, "Stem", (500.0, 0.0))
    assert reverse not in fork_network.successors(stem, allow_uturn=False)
    assert reverse in fork_network.successors(stem, allow_uturn=True)


def test_a_oneway_street_has_no_reverse_edge(oneway_network: RoadNetwork) -> None:
    """Driving against a one-way street must be structurally impossible, not
    merely penalised."""
    forward = edge_named(oneway_network, "One way east", (300.0, 0.0))
    assert oneway_network.edges[forward].oneway
    assert all(
        not (e.u == oneway_network.edges[forward].v and e.v == oneway_network.edges[forward].u)
        for e in oneway_network.edges
    ), "no edge may run back down the one-way street"


def test_no_route_enters_a_oneway_street_from_its_far_end(oneway_network: RoadNetwork) -> None:
    far_node = oneway_network.edges[edge_named(oneway_network, "One way east", (300.0, 0.0))].v
    outgoing = oneway_network.out_edges.get(far_node, [])
    assert outgoing == [], "the east end of the one-way street is a dead end"


# ------------------------------------------------------------ filter dynamics


def make_filter(network: RoadNetwork, cfg: Config, n: int = 400) -> RoadParticleFilter:
    cfg.pf.n_particles = n
    return RoadParticleFilter(network, cfg, rng=np.random.default_rng(cfg.seed))


def test_initialisation_places_every_particle_on_a_road(fork_network, config) -> None:
    pf = make_filter(fork_network, config)
    pf.initialize((100.0, 3.0), heading=0.0, speed=10.0)
    assert pf.initialized
    positions = pf.positions()
    assert len(positions) == config.pf.n_particles
    for point in positions:
        assert fork_network.distance_to_road(point) < 1e-6


def test_initialisation_prefers_edges_pointing_the_right_way(fork_network, config) -> None:
    """At 15 m accuracy the nearest edge is often the opposite carriageway; the
    heading has to break the tie."""
    pf = make_filter(fork_network, config, n=2000)
    pf.initialize((250.0, 0.0), heading=0.0, speed=10.0)
    bearings = pf.road_bearings()
    assert np.mean(np.abs(wrap_angle(bearings - 0.0)) < 0.2) > 0.9


def test_particles_advance_along_the_edge(fork_network, config) -> None:
    pf = make_filter(fork_network, config)
    pf.initialize((100.0, 0.0), heading=0.0, speed=10.0)
    before = pf.positions()[:, 0].mean()
    for _ in range(10):
        pf.predict((0.0, 0.0, 0.0), 0.0, 0.1)
    after = pf.positions()[:, 0].mean()
    assert after - before == pytest.approx(10.0, rel=0.15)


def test_acceleration_is_projected_on_the_road_not_free_heading(fork_network, config) -> None:
    """A wrong gyro heading must not move a route particle off its road model."""
    pf = make_filter(fork_network, config, n=200)
    pf.initialize((100.0, 0.0), heading=0.0, speed=0.0)
    pf.v[:] = 0.0
    pf.b_a[:] = 0.0
    pf.psi[:] = math.pi / 2.0  # deliberately wrong: road points east
    pf.pf.sigma_s = 0.0
    pf.pf.sigma_v = 0.0
    before = pf.positions()[:, 0].copy()
    for _ in range(10):
        pf.predict((2.0, 0.0, 0.0), 0.0, 0.1)
    after = pf.positions()[:, 0]
    assert np.all(after > before + 0.5)


# ------------------------------------------------------- heading consensus


def test_heading_consensus_is_none_before_initialisation(fork_network, config) -> None:
    pf = make_filter(fork_network, config)
    assert pf.heading_consensus() is None


def test_heading_consensus_is_confident_on_a_single_road(fork_network, config) -> None:
    """Every particle sits on the Stem, which runs due east: bearing 0."""
    pf = make_filter(fork_network, config, n=200)
    pf.initialize((100.0, 0.0), heading=0.0, speed=10.0)
    bearing, resultant = pf.heading_consensus()
    assert resultant > 0.99
    assert bearing == pytest.approx(0.0, abs=0.05)


def test_heading_consensus_is_uncertain_when_the_cloud_splits_at_a_fork(
    fork_network, config
) -> None:
    """Branch A and B leave the junction in genuinely different directions;
    an even split must not read as a confident, single direction."""
    branch_a = edge_named(fork_network, "Branch A", (500.0, 0.0))
    branch_b = edge_named(fork_network, "Branch B", (500.0, 0.0))
    pf = make_filter(fork_network, config, n=200)
    pf.initialize((100.0, 0.0), heading=0.0, speed=10.0)
    half = len(pf.edge_idx) // 2
    pf.edge_idx[:half] = branch_a
    pf.edge_idx[half:] = branch_b
    pf.s[:] = 50.0
    pf.w[:] = 1.0 / len(pf.w)
    _bearing, resultant = pf.heading_consensus()
    assert resultant < config.pf.outage_heading_assist_min_resultant


# --------------------------------------------------- outage heading assist


def _ekf(config: Config, heading: float = 0.0) -> ExtendedKalmanFilter:
    return ExtendedKalmanFilter(config.motion, initial_state=[0.0, 0.0, 10.0, heading, 0.0, 0.0])


def test_outage_heading_assist_corrects_a_small_plausible_drift(fork_network, config) -> None:
    """A confident consensus close to the EKF's own heading is exactly the
    case this exists for: a fine correction of a small, plausible drift."""
    pf = make_filter(fork_network, config, n=200)
    pf.initialize((100.0, 0.0), heading=0.0, speed=10.0)  # Stem, bearing 0
    ekf = _ekf(config, heading=math.radians(10.0))  # a small, plausible drift
    bearing = _outage_heading_assist(pf, ekf, config)
    assert bearing is not None
    assert bearing == pytest.approx(0.0, abs=0.05)


def test_outage_heading_assist_is_none_when_the_cloud_is_undecided(fork_network, config) -> None:
    branch_a = edge_named(fork_network, "Branch A", (500.0, 0.0))
    branch_b = edge_named(fork_network, "Branch B", (500.0, 0.0))
    pf = make_filter(fork_network, config, n=200)
    pf.initialize((100.0, 0.0), heading=0.0, speed=10.0)
    half = len(pf.edge_idx) // 2
    pf.edge_idx[:half] = branch_a
    pf.edge_idx[half:] = branch_b
    pf.s[:] = 50.0
    pf.w[:] = 1.0 / len(pf.w)
    ekf = _ekf(config, heading=0.0)
    assert _outage_heading_assist(pf, ekf, config) is None


def test_outage_heading_assist_never_overrides_a_confidently_wrong_branch(
    fork_network, config
) -> None:
    """The failure this exists to fix (trip-b4faeae0-a941-4a87-9b18-de7aaa84f721,
    a 373 s outage): the cloud can be confidently, unanimously wrong after a
    long enough blind stretch. Agreement alone (resultant) must not be enough
    - it must also already roughly agree with the EKF's own independent
    heading, or it is refused outright rather than redirecting the estimate
    wholesale."""
    pf = make_filter(fork_network, config, n=200)
    pf.initialize((100.0, 0.0), heading=0.0, speed=10.0)  # Stem, bearing 0
    ekf = _ekf(config, heading=math.radians(170.0))  # a wholesale disagreement
    assert _outage_heading_assist(pf, ekf, config) is None


def test_no_turn_restriction_removes_a_successor(fork_network, config) -> None:
    stem = edge_named(fork_network, "Stem", (0.0, 0.0))
    branch_a = edge_named(fork_network, "Branch A", (500.0, 0.0))
    branch_b = edge_named(fork_network, "Branch B", (500.0, 0.0))
    fork_network.turn_restrictions = (
        TurnRestriction(
            kind="no",
            from_edge=fork_network.edges[stem].edge_id,
            via_edges=(),
            to_edge=fork_network.edges[branch_a].edge_id,
        ),
    )
    fork_network.restriction_history_limit = 2
    fork_network._restrictions_by_trigger_edge = fork_network._index_turn_restrictions(
        fork_network.turn_restrictions
    )
    allowed = fork_network.allowed_successors(stem, history=(stem,))
    assert branch_a not in allowed
    assert branch_b in allowed


def test_only_turn_restriction_keeps_a_single_successor(fork_network, config) -> None:
    """`only_*` must narrow the choice down to its `to` edge, not just add it."""
    stem = edge_named(fork_network, "Stem", (0.0, 0.0))
    branch_a = edge_named(fork_network, "Branch A", (500.0, 0.0))
    branch_b = edge_named(fork_network, "Branch B", (500.0, 0.0))
    fork_network.turn_restrictions = (
        TurnRestriction(
            kind="only",
            from_edge=fork_network.edges[stem].edge_id,
            via_edges=(),
            to_edge=fork_network.edges[branch_a].edge_id,
        ),
    )
    fork_network.restriction_history_limit = 2
    fork_network._restrictions_by_trigger_edge = fork_network._index_turn_restrictions(
        fork_network.turn_restrictions
    )
    allowed = fork_network.allowed_successors(stem, history=(stem,))
    assert set(allowed) == {branch_a}
    assert branch_b not in allowed


def test_a_stationary_cloud_stays_put(fork_network, config) -> None:
    pf = make_filter(fork_network, config)
    pf.initialize((100.0, 0.0), heading=0.0, speed=0.0)
    before = pf.positions().mean(axis=0)
    for _ in range(50):
        pf.predict((0.0, 0.0, 0.0), 0.0, 0.1)
    assert np.linalg.norm(pf.positions().mean(axis=0) - before) < 5.0



def test_particles_pass_through_the_junction_onto_both_branches(fork_network, config) -> None:
    """Crossing a fork must split the belief, not pick one road arbitrarily."""
    pf = make_filter(fork_network, config, n=2000)
    pf.initialize((450.0, 0.0), heading=0.0, speed=12.0)
    for _ in range(120):  # ~144 m, well past the junction at 500 m
        pf.predict((0.0, 0.0, 0.0), 0.0, 0.1)

    names = [str(fork_network.edges[int(i)].name) for i in pf.edge_idx]
    on_a = names.count("Branch A")
    on_b = names.count("Branch B")
    assert on_a > 0 and on_b > 0, "both branches must be represented"
    assert on_a + on_b > 0.95 * len(names), "almost everything should have crossed"
    # Straight ahead with no gyro evidence: neither branch may be discarded.
    assert 0.2 < on_a / (on_a + on_b) < 0.8


def test_the_gyro_decides_which_branch_gets_the_weight(fork_network, config) -> None:
    """A left turn on the gyro must favour the branch that turns left. This is
    the core claim of the whole approach."""
    pf = make_filter(fork_network, config, n=3000)
    pf.initialize((450.0, 0.0), heading=0.0, speed=12.0)
    # Branch A leaves at +45 degrees; turn at a rate that reaches it.
    yaw = math.radians(45.0) / 3.0  # 45 degrees over three seconds
    for _ in range(60):
        pf.predict((0.0, 0.0, 0.0), yaw, 0.1)
    pf.update_weights()

    weight_a = sum(
        w for w, i in zip(pf.w, pf.edge_idx) if str(fork_network.edges[int(i)].name) == "Branch A"
    )
    weight_b = sum(
        w for w, i in zip(pf.w, pf.edge_idx) if str(fork_network.edges[int(i)].name) == "Branch B"
    )
    assert weight_a > weight_b
    assert weight_a > 0.7


def test_a_particle_never_enters_a_oneway_street_backwards(oneway_network, config) -> None:
    pf = make_filter(oneway_network, config, n=800)
    forward = edge_named(oneway_network, "One way east", (300.0, 0.0))
    # Seed everything at the far (illegal) end and drive: there is nowhere to go.
    pf.initialize((880.0, 0.0), heading=0.0, speed=8.0)
    for _ in range(80):
        pf.predict((0.0, 0.0, 0.0), 0.0, 0.1)
    for index in pf.edge_idx:
        edge = oneway_network.edges[int(index)]
        assert not (edge.u == oneway_network.edges[forward].v), (
            "no particle may leave the far end of a one-way street"
        )


def test_dead_end_particles_are_killed_by_the_weight_update(oneway_network, config) -> None:
    pf = make_filter(oneway_network, config, n=500)
    pf.initialize((880.0, 0.0), heading=0.0, speed=8.0)
    for _ in range(60):
        pf.predict((0.0, 0.0, 0.0), 0.0, 0.1)
    assert pf.result.dead_ends > 0


def test_resampling_triggers_when_neff_collapses(fork_network, config) -> None:
    pf = make_filter(fork_network, config, n=500)
    pf.initialize((100.0, 0.0), heading=0.0, speed=10.0)
    pf.w = normalize_weights(np.concatenate([[1000.0], np.ones(499) * 1e-6]))
    assert pf.n_eff < config.pf.resample_threshold * 500
    assert pf.maybe_resample()
    assert pf.w == pytest.approx(np.full(500, 1.0 / 500))


def test_resampling_does_not_trigger_when_the_cloud_is_healthy(fork_network, config) -> None:
    pf = make_filter(fork_network, config, n=500)
    pf.initialize((100.0, 0.0), heading=0.0, speed=10.0)
    assert not pf.maybe_resample()


def test_gps_weight_favours_the_particles_near_the_fix(fork_network, config) -> None:
    """L_GPS = exp(-|z - pos|^2 / 2 sigma^2)."""
    pf = make_filter(fork_network, config, n=1000)
    pf.initialize((250.0, 0.0), heading=0.0, speed=10.0)
    target = np.array([300.0, 0.0])
    pf.update_weights(gps_xy=target, gps_sigma=8.0)
    distances = np.linalg.norm(pf.positions() - target, axis=1)
    near = pf.w[distances < 20].sum()
    far = pf.w[distances > 60].sum()
    assert near > far


def test_divergence_is_detected(fork_network, config) -> None:
    pf = make_filter(fork_network, config, n=500)
    pf.initialize((100.0, 0.0), heading=0.0, speed=10.0)
    pf.update_weights(gps_xy=np.array([1300.0, 800.0]), gps_sigma=8.0)
    assert pf.has_diverged()


def test_reinitialisation_moves_the_cloud_and_keeps_the_biases(fork_network, config) -> None:
    pf = make_filter(fork_network, config, n=500)
    pf.initialize((100.0, 0.0), heading=0.0, speed=10.0, accel_bias=0.2)
    before = float(np.average(pf.b_a, weights=pf.w))
    pf.reinitialize((1200.0, 700.0), heading=math.pi / 4, speed=10.0, sigma=10.0)
    assert pf.positions()[:, 0].mean() > 900
    assert float(np.average(pf.b_a, weights=pf.w)) == pytest.approx(before, abs=0.05)


def test_zero_velocity_update_stops_slow_particles(fork_network, config) -> None:
    pf = make_filter(fork_network, config, n=200)
    pf.initialize((100.0, 0.0), heading=0.0, speed=0.4)
    pf.v[:] = 0.4
    pf.zero_velocity_update(max_speed_ms=1.5)
    assert np.all(pf.v == 0.0)


def test_zero_velocity_update_leaves_a_fast_hypothesis_alone(fork_network, config) -> None:
    """A particle convinced it is doing 40 km/h is a different hypothesis, not a
    bias error; the weights, not a hard reset, should judge it."""
    pf = make_filter(fork_network, config, n=200)
    pf.initialize((100.0, 0.0), heading=0.0, speed=11.0)
    pf.v[:] = 11.0
    pf.zero_velocity_update(max_speed_ms=1.5)
    assert np.all(pf.v == 11.0)


def test_the_filter_refuses_to_integrate_across_a_large_gap(fork_network, config) -> None:
    pf = make_filter(fork_network, config, n=200)
    pf.initialize((100.0, 0.0), heading=0.0, speed=10.0)
    pf.predict((0.0, 0.0, 0.0), 0.0, config.motion.max_gap_s + 1.0)
    assert pf.result.skipped_gaps == 1


# ---------------------------------------------------------- reproducibility


def _run(network: RoadNetwork, seed: int) -> np.ndarray:
    cfg = Config()
    cfg.seed = seed
    cfg.pf.n_particles = 300
    pf = RoadParticleFilter(network, cfg, rng=np.random.default_rng(cfg.seed))
    pf.initialize((100.0, 0.0), heading=0.0, speed=10.0)
    for _ in range(80):
        pf.predict((0.2, 0.0, 0.0), 0.02, 0.1)
        pf.update_weights()
        pf.maybe_resample()
    return pf.positions()


def test_the_same_seed_gives_bit_identical_results(fork_network: RoadNetwork) -> None:
    assert np.array_equal(_run(fork_network, 42), _run(fork_network, 42))


def test_a_different_seed_gives_a_different_run(fork_network: RoadNetwork) -> None:
    assert not np.array_equal(_run(fork_network, 42), _run(fork_network, 43))


# --------------------------------------------------------- reinit continuity


@pytest.fixture
def river_banks_network():
    """Two parallel streets 20 m apart that never connect - a narrow river
    between them. A fix at (10, 50) is exactly equidistant from both with
    identical bearing, so any preference for one over the other in a re-seed
    can only come from route continuity, not geometry."""
    from geotrace.road_graph import build_graph_from_segments

    segments = [
        ("Bank A", [(0.0, 0.0), (0.0, 100.0)], {"highway": "secondary"}),
        ("Bank B", [(20.0, 0.0), (20.0, 100.0)], {"highway": "secondary"}),
    ]
    graph, frame = build_graph_from_segments(segments, 59.9311, 30.3609)
    return RoadNetwork(graph, frame)


def test_reinit_prefers_the_edge_connected_to_where_the_cloud_already_was(
    river_banks_network, config
) -> None:
    """The bug this guards: a geometric tie between two disconnected roads
    either side of a river must break towards the bank the car was already
    on, not lock onto the wrong one forever."""
    net = river_banks_network
    bank_a = edge_named(net, "Bank A", (0.0, 0.0))
    bank_b = edge_named(net, "Bank B", (20.0, 0.0))
    pf = make_filter(net, config, n=2000)

    pf.initialize((10.0, 50.0), heading=math.pi / 2, speed=0.0, previous_edges=[bank_a])
    on_a = int(np.sum(pf.edge_idx == bank_a))
    on_b = int(np.sum(pf.edge_idx == bank_b))
    assert on_a > on_b, "a geometric tie must break towards the connected bank"
    assert on_a > 0.7 * (on_a + on_b)


def test_reinit_continuity_is_a_penalty_not_a_hard_filter(
    river_banks_network, config
) -> None:
    """A real route change (or an outage long enough that the car could be
    anywhere) must still be reachable, just less preferred: a fix squarely on
    the disconnected bank still lands there, it is just not exclusive."""
    net = river_banks_network
    bank_a = edge_named(net, "Bank A", (0.0, 0.0))
    bank_b = edge_named(net, "Bank B", (20.0, 0.0))
    pf = make_filter(net, config, n=2000)

    pf.initialize(
        (20.0, 50.0), heading=math.pi / 2, speed=0.0,
        position_sigma=5.0, previous_edges=[bank_a],
    )
    on_a = int(np.sum(pf.edge_idx == bank_a))
    on_b = int(np.sum(pf.edge_idx == bank_b))
    assert on_b > 0.9 * (on_a + on_b), "geometry must still dominate a clear-cut fix"
    assert on_a > 0, "the penalised bank must remain reachable, not impossible"


def test_reinitialize_passes_the_current_cloud_as_previous_edges(
    river_banks_network, config
) -> None:
    net = river_banks_network
    bank_a = edge_named(net, "Bank A", (0.0, 0.0))
    bank_b = edge_named(net, "Bank B", (20.0, 0.0))
    pf = make_filter(net, config, n=2000)
    pf.initialize((0.0, 50.0), heading=math.pi / 2, speed=10.0, position_sigma=5.0)
    # A tiny floor probability (see `initialize`'s `max(score, 1e-6)`) means an
    # occasional particle on the far bank is expected, not a bug.
    assert int(np.sum(pf.edge_idx == bank_a)) > 0.99 * len(pf.edge_idx)

    pf.reinitialize((10.0, 50.0), heading=math.pi / 2, speed=10.0, sigma=10.0)
    on_a = int(np.sum(pf.edge_idx == bank_a))
    on_b = int(np.sum(pf.edge_idx == bank_b))
    assert on_a > on_b, "reinitialize must carry the pre-divergence cloud forward as continuity"
