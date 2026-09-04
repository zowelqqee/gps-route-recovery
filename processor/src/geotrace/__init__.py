"""geotrace - probabilistic car route recovery from degraded GPS.

The package reads a trip recorded by the GeoTraceLab iOS app (or a synthetic
trip produced by ``geotrace simulate``), detects GPS failures, and reconstructs
the most likely driven route using IMU dead reckoning constrained by an
OpenStreetMap road graph.

The reconstructed route is always a probabilistic estimate, never a measured
ground truth.
"""

__version__ = "0.1.0"

from geotrace.config import Config  # noqa: F401
