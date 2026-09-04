"""Road-only tracker boundary.

The road tracker deliberately has no parking state: it is a thin named boundary
around the established graph particle filter so callers cannot accidentally put
free-space terminal logic back into the road model.
"""

from geotrace.particle_filter import RoadParticleFilter


class RoadTracker(RoadParticleFilter):
    """OSM-constrained route and road-corridor tracker."""

