"""Coordinate conversion and angle handling."""

from __future__ import annotations

import math

import numpy as np
import pytest

from geotrace.coordinates import (
    LocalFrame,
    SphericalFrame,
    course_to_heading,
    haversine_m,
    heading_to_course,
    wrap_angle,
    wrap_angle_deg,
)

SPB = (59.93431, 30.32574)


@pytest.mark.parametrize(
    "lat,lon",
    [
        (59.93431, 30.32574),   # the origin itself
        (59.9400, 30.3400),
        (59.8500, 30.2000),
        (60.0500, 30.5500),
        (59.93431, 30.42574),   # ~5.6 km east
    ],
)
def test_local_frame_round_trip(lat: float, lon: float) -> None:
    """to_geo(to_local(p)) must return p to well under a millimetre."""
    frame = LocalFrame(*SPB)
    east, north = frame.to_local(lat, lon)
    back_lat, back_lon = frame.to_geo(east, north)
    assert haversine_m(lat, lon, back_lat, back_lon) < 1e-6


def test_local_frame_origin_is_zero() -> None:
    frame = LocalFrame(*SPB)
    east, north = frame.to_local(*SPB)
    assert abs(east) < 1e-6 and abs(north) < 1e-6


def test_local_frame_axes_point_east_and_north() -> None:
    frame = LocalFrame(*SPB)
    east, north = frame.to_local(SPB[0], SPB[1] + 0.01)
    assert east > 0 and abs(north) < 1.0
    east, north = frame.to_local(SPB[0] + 0.01, SPB[1])
    assert north > 0 and abs(east) < 1.0


def test_local_frame_distance_matches_the_geodesic() -> None:
    """Planar distance in the local frame == true distance on the ellipsoid.

    Compared against the WGS84 geodesic, not against the spherical haversine:
    LocalFrame projects on the ellipsoid, so the two agree to machine precision,
    while the R = 6371 km sphere is itself ~0.3% off at this latitude.
    """
    from pyproj import Geod

    frame = LocalFrame(*SPB)
    other = (59.9500, 30.3600)
    east, north = frame.to_local(*other)
    geodesic = Geod(ellps="WGS84").inv(SPB[1], SPB[0], other[1], other[0])[2]
    assert math.hypot(east, north) == pytest.approx(geodesic, rel=1e-9)
    # The simple spherical model is close, but only to a few parts per thousand.
    assert math.hypot(east, north) == pytest.approx(haversine_m(*SPB, *other), rel=5e-3)


def test_spherical_frame_matches_specification_formula() -> None:
    """E = R cos(phi0) (lambda - lambda0), N = R (phi - phi0)."""
    frame = SphericalFrame(*SPB)
    lat, lon = 59.9400, 30.3400
    east, north = frame.to_local(lat, lon)
    r = 6371000.0
    assert east == pytest.approx(r * math.cos(math.radians(SPB[0])) * math.radians(lon - SPB[1]))
    assert north == pytest.approx(r * math.radians(lat - SPB[0]))


def test_spherical_frame_round_trip() -> None:
    frame = SphericalFrame(*SPB)
    east, north = frame.to_local(59.9400, 30.3400)
    lat, lon = frame.to_geo(east, north)
    assert (lat, lon) == pytest.approx((59.9400, 30.3400), abs=1e-9)


def test_vectorised_conversion_matches_scalar() -> None:
    frame = LocalFrame(*SPB)
    lats = [59.93, 59.94, 59.95]
    lons = [30.32, 30.33, 30.34]
    arr = frame.to_local_array(lats, lons)
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        assert arr[i] == pytest.approx(frame.to_local(lat, lon))
    back = frame.to_geo_array(arr)
    assert back[:, 0] == pytest.approx(lats)
    assert back[:, 1] == pytest.approx(lons)


def test_local_frame_rejects_impossible_origin() -> None:
    with pytest.raises(ValueError):
        LocalFrame(120.0, 0.0)


# ------------------------------------------------------------- angle wrapping


@pytest.mark.parametrize(
    "angle,expected",
    [
        (0.0, 0.0),
        (math.pi / 2, math.pi / 2),
        (math.pi, math.pi),
        (-math.pi, math.pi),
        (3 * math.pi / 2, -math.pi / 2),
        (2 * math.pi, 0.0),
        (-3 * math.pi, math.pi),
        (10 * math.pi + 0.3, 0.3),
    ],
)
def test_wrap_angle(angle: float, expected: float) -> None:
    assert wrap_angle(angle) == pytest.approx(expected)


def test_wrap_angle_always_in_range() -> None:
    values = np.linspace(-40.0, 40.0, 2001)
    wrapped = wrap_angle(values)
    assert np.all(wrapped > -math.pi - 1e-12)
    assert np.all(wrapped <= math.pi + 1e-12)


def test_wrap_angle_preserves_the_angle_modulo_two_pi() -> None:
    values = np.linspace(-40.0, 40.0, 501)
    residual = (values - wrap_angle(values)) / (2 * math.pi)
    assert np.allclose(residual, np.round(residual))


def test_wrap_angle_is_vectorised() -> None:
    out = wrap_angle(np.array([0.0, 3 * math.pi]))
    assert isinstance(out, np.ndarray)
    assert out == pytest.approx([0.0, math.pi])


def test_wrap_angle_deg() -> None:
    assert wrap_angle_deg(370.0) == pytest.approx(10.0)
    assert wrap_angle_deg(-350.0) == pytest.approx(10.0)
    assert wrap_angle_deg(180.0) == pytest.approx(180.0)


@pytest.mark.parametrize("course", [0.0, 45.0, 90.0, 180.0, 270.0, 359.0])
def test_course_heading_round_trip(course: float) -> None:
    """CoreLocation course (clockwise from north) <-> filter psi (CCW from east)."""
    assert heading_to_course(course_to_heading(course)) == pytest.approx(course, abs=1e-9)


def test_course_north_is_ninety_degrees_of_heading() -> None:
    assert course_to_heading(0.0) == pytest.approx(math.pi / 2)     # north
    assert course_to_heading(90.0) == pytest.approx(0.0)            # east
    assert course_to_heading(180.0) == pytest.approx(-math.pi / 2)  # south
