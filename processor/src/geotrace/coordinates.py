"""Geodetic <-> local metric frame.

Two implementations are provided on purpose:

* :class:`SphericalFrame` is the flat tangent-plane approximation from the
  specification, kept because it is trivial to reason about and to test:

      E = R * cos(phi0) * (lambda - lambda0)
      N = R * (phi - phi0)

* :class:`LocalFrame` is what the production code uses. It wraps ``pyproj`` with
  an azimuthal-equidistant projection centred on the trip origin, which keeps
  distances correct over a whole city instead of only a few kilometres.

Both expose the same ``to_local`` / ``to_geo`` pair and both are covered by a
round-trip test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from pyproj import CRS, Transformer

from geotrace.config import EARTH_RADIUS_M

TWO_PI = 2.0 * math.pi


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap an angle in radians into (-pi, pi].

    Used everywhere a heading difference is computed. The half-open convention
    means wrap(pi) == pi and wrap(-pi) == pi.
    """
    wrapped = np.mod(np.asarray(angle, dtype=float) + math.pi, TWO_PI) - math.pi
    wrapped = np.where(wrapped <= -math.pi, wrapped + TWO_PI, wrapped)
    if np.isscalar(angle) or np.ndim(angle) == 0:
        return float(wrapped)
    return wrapped


def wrap_angle_deg(angle: float | np.ndarray) -> float | np.ndarray:
    wrapped = np.mod(np.asarray(angle, dtype=float) + 180.0, 360.0) - 180.0
    wrapped = np.where(wrapped <= -180.0, wrapped + 360.0, wrapped)
    if np.isscalar(angle) or np.ndim(angle) == 0:
        return float(wrapped)
    return wrapped


def course_to_heading(course_deg: float) -> float:
    """CoreLocation course is degrees clockwise from true north.

    The filter heading psi is the mathematical convention: radians
    counter-clockwise from east (+E axis), matching the E/N state layout.
    """
    return float(wrap_angle(math.radians(90.0 - course_deg)))


def heading_to_course(psi_rad: float) -> float:
    """Inverse of :func:`course_to_heading`, returned in [0, 360)."""
    return float(np.mod(90.0 - math.degrees(psi_rad), 360.0))


@dataclass
class SphericalFrame:
    """Flat tangent-plane model, exactly as written in the specification."""

    lat0: float
    lon0: float
    radius: float = EARTH_RADIUS_M

    def to_local(self, lat: float, lon: float) -> tuple[float, float]:
        phi0 = math.radians(self.lat0)
        east = self.radius * math.cos(phi0) * math.radians(lon - self.lon0)
        north = self.radius * math.radians(lat - self.lat0)
        return east, north

    def to_geo(self, east: float, north: float) -> tuple[float, float]:
        phi0 = math.radians(self.lat0)
        lat = self.lat0 + math.degrees(north / self.radius)
        lon = self.lon0 + math.degrees(east / (self.radius * math.cos(phi0)))
        return lat, lon


class LocalFrame:
    """Metric working frame for one trip.

    The origin is the first *trusted* GPS fix of the trip. Everything downstream
    (filters, road graph, polygons) works in metres in this frame; conversion
    back to WGS84 happens only when GeoJSON is written.
    """

    def __init__(self, lat0: float, lon0: float) -> None:
        if not (-90.0 <= lat0 <= 90.0 and -180.0 <= lon0 <= 180.0):
            raise ValueError(f"origin out of range: {lat0}, {lon0}")
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        self.crs = CRS.from_proj4(
            f"+proj=aeqd +lat_0={self.lat0} +lon_0={self.lon0} "
            "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )
        self._fwd = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        self._inv = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)

    def to_local(self, lat: float, lon: float) -> tuple[float, float]:
        east, north = self._fwd.transform(lon, lat)
        return float(east), float(north)

    def to_geo(self, east: float, north: float) -> tuple[float, float]:
        lon, lat = self._inv.transform(east, north)
        return float(lat), float(lon)

    def to_local_array(self, lats: Sequence[float], lons: Sequence[float]) -> np.ndarray:
        """Vectorised ``to_local``. Returns an (N, 2) array of [E, N]."""
        if len(lats) == 0:
            return np.zeros((0, 2))
        east, north = self._fwd.transform(np.asarray(lons, dtype=float), np.asarray(lats, dtype=float))
        return np.column_stack([np.asarray(east, dtype=float), np.asarray(north, dtype=float)])

    def to_geo_array(self, points: np.ndarray) -> np.ndarray:
        """Vectorised ``to_geo``. Takes (N, 2) [E, N], returns (N, 2) [lat, lon]."""
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        if points.size == 0:
            return np.zeros((0, 2))
        lon, lat = self._inv.transform(points[:, 0], points[:, 1])
        return np.column_stack([np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)])

    def coords_to_geojson(self, points: Iterable[Sequence[float]]) -> list[list[float]]:
        """GeoJSON wants [lon, lat] pairs."""
        arr = np.asarray(list(points), dtype=float).reshape(-1, 2)
        geo = self.to_geo_array(arr)
        return [[float(lon), float(lat)] for lat, lon in geo]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"LocalFrame(lat0={self.lat0:.6f}, lon0={self.lon0:.6f})"


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres, used for sanity checks and tests."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))
