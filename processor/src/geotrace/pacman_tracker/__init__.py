"""Road-locked multi-hypothesis route recovery ("Pacman").

A parallel, independent implementation of the GPS-outage reconstruction core.
It shares the project's loaders, coordinate frame, IMU front end and road graph,
and shares nothing at all with `geotrace.road_ekf` / `geotrace.particle_filter`,
which remain in place as the reference to benchmark against.

Two ideas carry it. A car is on a road, so track *which road and how far along
it* rather than a free (x, y). And there is only one car, so speed and distance
are estimated **once**, globally, from the IMU - never per hypothesis, and never
from the map. See ``docs/PACMAN_TRACKER.md``.
"""

from geotrace.pacman_tracker.config import PacmanConfig
from geotrace.pacman_tracker.corridor import Confidence, Corridor, CorridorSet
from geotrace.pacman_tracker.roadmap import RoadGeometry
from geotrace.pacman_tracker.spectral import SpectralSpeedModel
from geotrace.pacman_tracker.speed import GlobalSpeedTracker, SpeedConfig, SpeedSample
from geotrace.pacman_tracker.state import HypothesisSet, PacmanState, RouteNode
from geotrace.pacman_tracker.tracker import (
    PacmanTracker,
    TrackerInputs,
    TrackerResult,
    build_inputs,
)
from geotrace.pacman_tracker.turns import HeadingIntegrator, TurnEvent, detect_turns

__all__ = [
    "Confidence",
    "Corridor",
    "CorridorSet",
    "GlobalSpeedTracker",
    "HeadingIntegrator",
    "HypothesisSet",
    "PacmanConfig",
    "PacmanState",
    "PacmanTracker",
    "RoadGeometry",
    "RouteNode",
    "SpectralSpeedModel",
    "SpeedConfig",
    "SpeedSample",
    "TrackerInputs",
    "TrackerResult",
    "TurnEvent",
    "build_inputs",
    "detect_turns",
]
