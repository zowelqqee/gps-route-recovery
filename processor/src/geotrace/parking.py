"""Parking zone probability.

    P(Z_j) = sum_i w_i * 1[pos(p_i) in Z_j]

The zones are read from GeoJSON through :class:`ParkingZoneProvider`. The
repository ships *test* polygons in ``sample-data/parking-zones.geojson``. They
are hand-drawn for the pipeline to have something to chew on and are explicitly
not the official Saint Petersburg paid-parking boundaries.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import geopandas as gpd
import numpy as np
import shapely
from geopandas import GeoDataFrame

from geotrace.config import ParkingConfig
from geotrace.coordinates import LocalFrame


class ParkingZoneProvider(ABC):
    """Source of parking-zone polygons."""

    @abstractmethod
    def load_zones(self) -> GeoDataFrame:
        """Return zones in EPSG:4326 with at least an ``id`` column."""

    @property
    def attribution(self) -> str:
        return "unspecified"


class GeoJSONParkingZoneProvider(ParkingZoneProvider):
    """Loads zones from any GeoJSON file with polygon features."""

    def __init__(self, path: str | Path, id_field: str = "id", official: bool = False) -> None:
        self.path = Path(path)
        self.id_field = id_field
        self.official = official

    def load_zones(self) -> GeoDataFrame:
        if not self.path.exists():
            raise FileNotFoundError(
                f"parking zones file not found: {self.path}\n"
                "Pass --parking-zones with a GeoJSON of polygons, or use "
                "sample-data/parking-zones.geojson (test polygons, not official)."
            )
        frame = gpd.read_file(self.path)
        if frame.empty:
            raise ValueError(f"{self.path} contains no features")
        if self.id_field not in frame.columns:
            frame[self.id_field] = [f"zone-{i:02d}" for i in range(len(frame))]
        if frame.crs is None:
            frame = frame.set_crs("EPSG:4326")
        else:
            frame = frame.to_crs("EPSG:4326")
        return frame

    @property
    def attribution(self) -> str:
        if self.official:
            return f"official source declared by the operator: {self.path}"
        return (
            f"UNOFFICIAL test polygons from {self.path}. These are not the "
            "official paid-parking boundaries of Saint Petersburg."
        )


@dataclass
class ZoneProbability:
    zone_id: str
    probability: float
    name: Optional[str] = None


@dataclass
class ParkingDecision:
    """What the processor is willing to claim about the parking zone."""

    selected_zone: Optional[str]
    probability: float
    second_probability: float
    decision: str
    """"confident" | "ambiguous" | "outside_known_zones"."""

    ranking: list[ZoneProbability]
    attribution: str = "unspecified"

    def to_json(self) -> dict[str, Any]:
        return {
            "selected_zone": self.selected_zone,
            "probability": round(self.probability, 6),
            "second_probability": round(self.second_probability, 6),
            "decision": self.decision,
            "ranking": [
                {"zone_id": z.zone_id, "name": z.name, "probability": round(z.probability, 6)}
                for z in self.ranking[:5]
            ],
            "zone_source": self.attribution,
        }


def zone_probabilities(
    zones: GeoDataFrame,
    positions_latlon: np.ndarray,
    weights: np.ndarray,
    id_field: str = "id",
    name_field: str = "name",
) -> list[ZoneProbability]:
    """P(Z_j) = sum of the weights of the particles falling inside Z_j."""
    weights = np.asarray(weights, dtype=float)
    positions_latlon = np.asarray(positions_latlon, dtype=float).reshape(-1, 2)
    if positions_latlon.size == 0 or zones.empty:
        return []
    points = shapely.points(positions_latlon[:, 1], positions_latlon[:, 0])
    out: list[ZoneProbability] = []
    for _, row in zones.iterrows():
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            continue
        inside = shapely.contains(geometry, points)
        out.append(
            ZoneProbability(
                zone_id=str(row[id_field]),
                probability=float(weights[inside].sum()),
                name=str(row[name_field]) if name_field in zones.columns else None,
            )
        )
    out.sort(key=lambda z: -z.probability)
    return out


def decide_zone(
    ranking: Sequence[ZoneProbability], cfg: ParkingConfig, attribution: str = "unspecified"
) -> ParkingDecision:
    """Apply the two configured thresholds and refuse to over-claim.

    A zone is only "confident" when it holds at least ``selected_probability``
    of the mass AND beats the runner-up by ``margin_to_second``.
    """
    if not ranking:
        return ParkingDecision(None, 0.0, 0.0, "outside_known_zones", [], attribution)
    first = ranking[0]
    second = ranking[1].probability if len(ranking) > 1 else 0.0
    if first.probability <= 0:
        return ParkingDecision(None, 0.0, 0.0, "outside_known_zones", list(ranking), attribution)
    confident = (
        first.probability >= cfg.selected_probability
        and (first.probability - second) >= cfg.margin_to_second
    )
    return ParkingDecision(
        selected_zone=first.zone_id,
        probability=first.probability,
        second_probability=second,
        decision="confident" if confident else "ambiguous",
        ranking=list(ranking),
        attribution=attribution,
    )


def write_test_zones(path: str | Path, frame: LocalFrame, boxes: Sequence[tuple[str, float, float, float, float]]) -> Path:
    """Write a GeoJSON of rectangular test zones given local-metre boxes."""
    features = []
    for zone_id, e0, n0, e1, n1 in boxes:
        ring = [(e0, n0), (e1, n0), (e1, n1), (e0, n1), (e0, n0)]
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "id": zone_id,
                    "name": f"Test zone {zone_id}",
                    "official": False,
                },
                "geometry": {"type": "Polygon", "coordinates": [frame.coords_to_geojson(ring)]},
            }
        )
    payload = {
        "type": "FeatureCollection",
        "note": "Synthetic test polygons. NOT the official Saint Petersburg paid-parking zones.",
        "features": features,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
