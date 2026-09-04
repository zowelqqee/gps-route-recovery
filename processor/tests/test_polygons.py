"""Probability-mass selection, branch clustering and corridor construction."""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon

from geotrace.config import PolygonConfig
from geotrace.polygons import (
    branch_aware_estimate,
    build_uncertainty_set,
    convex_hull_area,
    cross_track_sigma,
    select_high_probability_set,
    uncertainty_to_geojson,
)
from geotrace.road_graph import RoadNetwork

from conftest import edge_named

CFG = PolygonConfig()


# --------------------------------------------------- probability mass selection


def test_selection_takes_the_smallest_set_reaching_the_mass() -> None:
    """sum_{i=1..K} w_(i) >= gamma, weights sorted descending."""
    w = np.array([0.5, 0.3, 0.15, 0.05])
    picked = select_high_probability_set(w, 0.95)
    assert list(picked) == [0, 1, 2]


def test_selection_stops_as_soon_as_the_mass_is_reached() -> None:
    w = np.array([0.96, 0.02, 0.02])
    assert list(select_high_probability_set(w, 0.95)) == [0]


def test_selection_of_uniform_weights_takes_gamma_of_them() -> None:
    w = np.full(1000, 0.001)
    assert len(select_high_probability_set(w, 0.95)) == 950


def test_selection_is_ordered_by_descending_weight() -> None:
    w = np.array([0.1, 0.6, 0.3])
    assert list(select_high_probability_set(w, 0.99)) == [1, 2, 0]


def test_selection_handles_unnormalised_weights() -> None:
    w = np.array([50.0, 30.0, 15.0, 5.0])
    assert list(select_high_probability_set(w, 0.95)) == [0, 1, 2]


def test_selection_of_an_empty_set() -> None:
    assert len(select_high_probability_set(np.zeros(0), 0.95)) == 0


def test_cross_track_sigma_grows_while_gps_is_missing() -> None:
    assert cross_track_sigma(0.0, CFG) == CFG.cross_track_sigma_base_m
    assert cross_track_sigma(60.0, CFG) > cross_track_sigma(0.0, CFG)


# -------------------------------------------------------------- one branch


def test_a_single_road_gives_one_component(fork_network: RoadNetwork) -> None:
    stem = edge_named(fork_network, "Stem", (0.0, 0.0))
    n = 500
    edge_idx = np.full(n, stem)
    s = np.random.default_rng(0).uniform(100, 200, n)
    result = build_uncertainty_set(fork_network, edge_idx, s, np.full(n, 1 / n), CFG)
    assert len(result.components) == 1
    assert result.components[0].probability == pytest.approx(0.95, abs=0.02)


def test_the_corridor_actually_contains_the_particles(fork_network: RoadNetwork) -> None:
    stem = edge_named(fork_network, "Stem", (0.0, 0.0))
    n = 300
    edge_idx = np.full(n, stem)
    s = np.linspace(100, 300, n)
    result = build_uncertainty_set(fork_network, edge_idx, s, np.full(n, 1 / n), CFG)
    for point in fork_network.positions_fast(edge_idx, s):
        assert result.contains(point)


def test_the_corridor_is_no_wider_than_the_configured_radius(fork_network: RoadNetwork) -> None:
    stem = edge_named(fork_network, "Stem", (0.0, 0.0))
    n = 200
    edge_idx = np.full(n, stem)
    s = np.linspace(100, 300, n)
    result = build_uncertainty_set(fork_network, edge_idx, s, np.full(n, 1 / n), CFG,
                                   seconds_since_trusted=0.0)
    radius = CFG.r_min_m + CFG.k_sigma * cross_track_sigma(0.0, CFG)
    # A point well off to the side of the road must not be inside.
    assert not result.contains((200.0, radius + 20.0))


def test_the_corridor_widens_during_a_long_outage(fork_network: RoadNetwork) -> None:
    stem = edge_named(fork_network, "Stem", (0.0, 0.0))
    n = 200
    edge_idx = np.full(n, stem)
    s = np.linspace(100, 300, n)
    tight = build_uncertainty_set(fork_network, edge_idx, s, np.full(n, 1 / n), CFG,
                                  seconds_since_trusted=0.0)
    wide = build_uncertainty_set(fork_network, edge_idx, s, np.full(n, 1 / n), CFG,
                                 seconds_since_trusted=120.0)
    assert wide.total_area_m2 > tight.total_area_m2


# ------------------------------------------------------ several branches


def _fork_cloud(network: RoadNetwork, weight_a: float = 0.7, n: int = 1000, s_lo=400.0, s_hi=600.0):
    """A cloud that has already travelled well down both branches."""
    branch_a = edge_named(network, "Branch A", (500.0, 0.0))
    branch_b = edge_named(network, "Branch B", (500.0, 0.0))
    n_a = int(n * weight_a)
    edge_idx = np.array([branch_a] * n_a + [branch_b] * (n - n_a))
    rng = np.random.default_rng(4)
    s = rng.uniform(s_lo, s_hi, n)
    weights = np.full(n, 1.0 / n)
    return edge_idx, s, weights


def test_a_fork_produces_two_separate_polygons(fork_network: RoadNetwork) -> None:
    """After the branches have diverged, the belief is two disjoint corridors."""
    edge_idx, s, weights = _fork_cloud(fork_network)
    result = build_uncertainty_set(fork_network, edge_idx, s, weights, CFG,
                                   seconds_since_trusted=30.0)
    assert len(result.components) == 2
    a, b = result.components
    assert a.geometry.disjoint(b.geometry)


def test_branch_probabilities_reflect_the_particle_weights(fork_network: RoadNetwork) -> None:
    edge_idx, s, weights = _fork_cloud(fork_network, weight_a=0.7)
    result = build_uncertainty_set(fork_network, edge_idx, s, weights, CFG,
                                   seconds_since_trusted=30.0)
    probabilities = sorted((c.probability for c in result.components), reverse=True)
    assert probabilities[0] / sum(probabilities) == pytest.approx(0.7, abs=0.03)


def test_components_are_ranked_by_probability(fork_network: RoadNetwork) -> None:
    edge_idx, s, weights = _fork_cloud(fork_network, weight_a=0.3)
    result = build_uncertainty_set(fork_network, edge_idx, s, weights, CFG,
                                   seconds_since_trusted=30.0)
    ids = [c.component_id for c in result.components]
    assert ids == ["branch-01", "branch-02"]
    assert result.components[0].probability >= result.components[1].probability
    assert result.best is result.components[0]


def test_no_convex_hull_over_independent_branches(fork_network: RoadNetwork) -> None:
    """The specification's hard rule. A hull over a post-fork belief claims the
    car might be in the courtyards between the two roads."""
    edge_idx, s, weights = _fork_cloud(fork_network)
    result = build_uncertainty_set(fork_network, edge_idx, s, weights, CFG,
                                   seconds_since_trusted=30.0)
    points = fork_network.positions_fast(edge_idx, s)
    hull = convex_hull_area(points)
    assert result.total_area_m2 < 0.35 * hull


def test_the_space_between_the_branches_is_excluded(fork_network: RoadNetwork) -> None:
    """The midpoint between the two branches is open ground; nothing may claim
    the car could be there."""
    edge_idx, s, weights = _fork_cloud(fork_network)
    result = build_uncertainty_set(fork_network, edge_idx, s, weights, CFG,
                                   seconds_since_trusted=30.0)
    branch_a = edge_named(fork_network, "Branch A", (500.0, 0.0))
    branch_b = edge_named(fork_network, "Branch B", (500.0, 0.0))
    point_a = fork_network.edges[branch_a].position(500.0)
    point_b = fork_network.edges[branch_b].position(500.0)
    midpoint = ((point_a[0] + point_b[0]) / 2, (point_a[1] + point_b[1]) / 2)

    assert not result.contains(midpoint)
    hull = MultiPoint([tuple(p) for p in fork_network.positions_fast(edge_idx, s)]).convex_hull
    assert hull.contains(Point(*midpoint)), "a hull would have swallowed it"


def test_a_cloud_still_straddling_the_junction_is_one_connected_region(
    fork_network: RoadNetwork,
) -> None:
    """Honest behaviour: while the belief covers the fork itself it really is
    one connected Y-shaped region and is reported as one component."""
    edge_idx, s, weights = _fork_cloud(fork_network, s_lo=0.0, s_hi=40.0)
    result = build_uncertainty_set(fork_network, edge_idx, s, weights, CFG,
                                   seconds_since_trusted=5.0)
    assert len(result.components) == 1


def test_tiny_components_are_dropped(fork_network: RoadNetwork) -> None:
    edge_idx, s, weights = _fork_cloud(fork_network, weight_a=0.999, n=2000)
    cfg = PolygonConfig(min_component_probability=0.05)
    result = build_uncertainty_set(fork_network, edge_idx, s, weights, cfg,
                                   seconds_since_trusted=30.0)
    assert len(result.components) == 1


# ------------------------------------------------------------ point estimate


def test_the_point_estimate_never_lands_between_two_branches(
    fork_network: RoadNetwork,
) -> None:
    """Averaging the branches would put the car in the block between them."""
    edge_idx, s, weights = _fork_cloud(fork_network, weight_a=0.55)
    result = build_uncertainty_set(fork_network, edge_idx, s, weights, CFG,
                                   seconds_since_trusted=30.0)
    estimate = branch_aware_estimate(fork_network, result, edge_idx, s, weights)

    assert fork_network.distance_to_road(estimate) < 1.0, "the estimate must be on a road"
    naive = np.average(fork_network.positions_fast(edge_idx, s), axis=0, weights=weights)
    assert fork_network.distance_to_road(naive) > 20.0, "the naive mean is off-road"


def test_the_point_estimate_follows_the_dominant_branch(fork_network: RoadNetwork) -> None:
    branch_b = edge_named(fork_network, "Branch B", (500.0, 0.0))
    edge_idx, s, weights = _fork_cloud(fork_network, weight_a=0.05)
    result = build_uncertainty_set(fork_network, edge_idx, s, weights, CFG,
                                   seconds_since_trusted=30.0)
    estimate = branch_aware_estimate(fork_network, result, edge_idx, s, weights)
    assert estimate[1] < 0, "Branch B runs south-east, so N must be negative"


# ------------------------------------------------------------------ GeoJSON


def test_geojson_output_is_wgs84_and_well_formed(fork_network: RoadNetwork) -> None:
    edge_idx, s, weights = _fork_cloud(fork_network)
    result = build_uncertainty_set(fork_network, edge_idx, s, weights, CFG,
                                   seconds_since_trusted=30.0)
    collection = uncertainty_to_geojson([result], fork_network.frame)
    assert collection["type"] == "FeatureCollection"
    assert len(collection["features"]) == 2
    for feature in collection["features"]:
        assert feature["geometry"]["type"] in {"Polygon", "MultiPolygon"}
        assert set(feature["properties"]) >= {"component_id", "probability", "area_m2"}
        ring = feature["geometry"]["coordinates"][0]
        lon, lat = ring[0][:2] if isinstance(ring[0][0], float) else ring[0][0][:2]
        assert 29.0 < lon < 32.0 and 59.0 < lat < 61.0


def test_component_json_matches_the_documented_shape(fork_network: RoadNetwork) -> None:
    edge_idx, s, weights = _fork_cloud(fork_network)
    result = build_uncertainty_set(fork_network, edge_idx, s, weights, CFG,
                                   seconds_since_trusted=30.0)
    payload = result.to_json(fork_network.frame)
    first = payload["components"][0]
    assert set(first) >= {"component_id", "probability", "geometry"}
    assert first["geometry"]["type"] in {"Polygon", "MultiPolygon"}
    assert 0.0 < first["probability"] <= 1.0
