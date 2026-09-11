"""The heading/curvature index must describe the road, nothing else."""

import math

import numpy as np

from geotrace.pacman_tracker.config import GeometryConfig, PacmanConfig
from geotrace.pacman_tracker.roadmap import RoadGeometry
from geotrace.pacman_tracker.synthetic import arc, build_network, straight


def test_straight_road_has_zero_curvature(straight_geometry):
    profile = straight_geometry.profile(0)
    assert np.allclose(profile.kappa, 0.0, atol=1e-9)
    assert np.allclose(profile.kappa_grad, 0.0, atol=1e-9)


def test_arc_curvature_matches_one_over_radius():
    radius = 60.0
    network, _ = build_network([("Arc", arc(radius, math.pi / 2), {})])
    geometry = RoadGeometry(network, PacmanConfig().geometry)
    profile = geometry.profile(0)
    # Ends are affected by the smoothing kernel's reflection; judge the middle.
    middle = profile.kappa[len(profile.kappa) // 4 : -len(profile.kappa) // 4]
    assert abs(float(np.median(middle)) - 1.0 / radius) < 0.002


def test_total_turn_is_preserved_by_smoothing():
    """Smoothing may spread a corner but must not change how much it turns."""
    corner = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)]
    network, _ = build_network([("Corner", corner, {})])
    geometry = RoadGeometry(network, PacmanConfig().geometry)
    profile = geometry.profile(0)
    integral = float(np.trapezoid(profile.kappa, profile.s))
    assert abs(integral - math.pi / 2) < 0.05


def test_junction_turn_is_signed_and_wrapped(junction_network):
    network, _ = junction_network
    geometry = RoadGeometry(network, PacmanConfig().geometry)
    by_name = {e.name: e.index for e in network.edges}
    assert abs(geometry.junction_turn(by_name["A"], by_name["C"])) < 1e-6
    assert abs(geometry.junction_turn(by_name["A"], by_name["B"]) - math.pi / 2) < 1e-6
    assert abs(geometry.junction_turn(by_name["A"], by_name["D"]) + math.pi / 2) < 1e-6


def test_curvature_lookup_is_vectorised_and_clamped(curved_network):
    network, _ = curved_network
    geometry = RoadGeometry(network, PacmanConfig().geometry)
    edges = np.array([0, 1, 1, 2])
    s = np.array([-50.0, 10.0, 1e6, 5.0])
    kappa, grad, sigma = geometry.curvature(edges, s)
    assert kappa.shape == (4,)
    assert np.all(np.isfinite(kappa)) and np.all(sigma > 0)


def test_sampling_step_does_not_change_the_turn_integral():
    corner = [(0.0, 0.0), (80.0, 0.0), (80.0, 80.0)]
    network, _ = build_network([("Corner", corner, {})])
    integrals = []
    for ds in (1.0, 2.0, 4.0):
        geometry = RoadGeometry(network, GeometryConfig(sample_ds_m=ds))
        profile = geometry.profile(0)
        integrals.append(float(np.trapezoid(profile.kappa, profile.s)))
    assert max(integrals) - min(integrals) < 0.05
