"""Photographs as an extra localisation signal.

What this module actually does today:

  * parses the OCR lines the iPhone produced with Vision;
  * pulls out house numbers, street names and parking-zone numbers;
  * matches those against the street names already present in the road graph,
    i.e. against the hypotheses the particle filter is holding;
  * re-weights particles that sit on a matching street.

What it deliberately does NOT do: claim an exact address from a single photo.
There is no reference image database in this repository, so there is nothing to
match a facade against. :class:`VisualPlaceRecognizer` defines the interface and
the expected database format for when a legally usable geo-referenced image set
exists, and the stub returns no candidates rather than inventing matches.

    e_I = Encoder(I)
    similarity(I, I_j) = e_I . e_j / (|e_I| |e_j|)
    P(l_j | I) ~ exp(similarity(I, I_j) / tau) * P_prior(l_j)
    w'_i ~ w_i * L_image(p_i)^alpha * L_OCR(p_i)^beta
"""

from __future__ import annotations

import math
import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from geotrace.config import PhotoConfig
from geotrace.models import LocationCandidate, OCRResult, Photo, WeightedRegion
from geotrace.road_graph import RoadNetwork

# Russian street-type words, in the forms that actually appear on signs.
STREET_TYPES = {
    "улица": "ул", "ул": "ул", "проспект": "пр", "пр-кт": "пр", "просп": "пр", "пр": "пр",
    "переулок": "пер", "пер": "пер", "набережная": "наб", "наб": "наб",
    "бульвар": "бул", "бул": "бул", "шоссе": "ш", "ш": "ш",
    "площадь": "пл", "пл": "пл", "линия": "линия", "аллея": "аллея", "проезд": "проезд",
    "street": "ул", "st": "ул", "avenue": "пр", "ave": "пр", "embankment": "наб",
}

# A number that is part of a clock time is never a house number or a zone:
# "Платная парковка 08:00-21:00" is opening hours, not address 8 in zone 8.
_NOT_A_TIME = r"(?!\s*[:.]\s*\d)"

HOUSE_RE = re.compile(
    r"(?:^|[^\d\w:])(?:д\.?|дом|house|№)?\s*(\d{1,3})" + _NOT_A_TIME
    + r"\s*(?:к(?:орп)?\.?\s*\d{1,2})?"
    r"\s*(?:лит(?:ера)?\.?\s*[а-яa-z])?(?:$|[^\d:])",
    re.IGNORECASE,
)
# Saint Petersburg parking zones are four digits (e.g. 7812). Requiring at least
# three keeps "парковка 08:00" from being read as zone 8.
ZONE_RE = re.compile(
    r"(?:зона|zone|парковочная\s+зона|парковка)\D{0,12}(\d{3,5})" + _NOT_A_TIME,
    re.IGNORECASE,
)
ZONE_BARE_RE = re.compile(r"\b(\d{4})\b" + _NOT_A_TIME)
TIME_RANGE_RE = re.compile(r"\d{1,2}\s*[:.]\s*\d{2}")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("ё", "е")
    return re.sub(r"[^0-9a-zа-я]+", " ", text).strip()


@dataclass
class ParsedSign:
    """Structured content extracted from the OCR of one photo."""

    street_names: list[str] = field(default_factory=list)
    house_numbers: list[str] = field(default_factory=list)
    parking_zones: list[str] = field(default_factory=list)
    raw_lines: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "street_names": self.street_names,
            "house_numbers": self.house_numbers,
            "parking_zones": self.parking_zones,
            "mean_confidence": round(self.confidence, 3),
            "raw_lines": self.raw_lines,
        }

    @property
    def is_empty(self) -> bool:
        return not (self.street_names or self.house_numbers or self.parking_zones)


def parse_ocr(results: Sequence[OCRResult], cfg: Optional[PhotoConfig] = None) -> ParsedSign:
    """Extract street / house / zone tokens from Vision output."""
    cfg = cfg or PhotoConfig()
    parsed = ParsedSign()
    confidences: list[float] = []

    for item in results:
        if item.confidence < cfg.min_ocr_confidence:
            continue
        raw = item.text.strip()
        if not raw:
            continue
        parsed.raw_lines.append(raw)
        confidences.append(item.confidence)
        # Opening hours contribute nothing but false house numbers.
        is_opening_hours = bool(TIME_RANGE_RE.search(raw))
        text = normalize(raw)
        tokens = text.split()

        # Street name: a street-type word plus the words around it.
        for i, token in enumerate(tokens):
            if token in STREET_TYPES:
                before = [t for t in tokens[max(0, i - 3) : i] if not t.isdigit()]
                after = [t for t in tokens[i + 1 : i + 4] if not t.isdigit()]
                name = " ".join(before) if before else " ".join(after)
                name = name.strip()
                if name and name not in parsed.street_names:
                    parsed.street_names.append(name)

        if not is_opening_hours:
            match = HOUSE_RE.search(raw)
            if match and match.group(1) not in parsed.house_numbers:
                parsed.house_numbers.append(match.group(1))

        zone = ZONE_RE.search(raw)
        if zone:
            if zone.group(1) not in parsed.parking_zones:
                parsed.parking_zones.append(zone.group(1))
        elif "зона" in text or "парк" in text or "zone" in text:
            bare = ZONE_BARE_RE.search(raw)
            if bare and bare.group(1) not in parsed.parking_zones:
                parsed.parking_zones.append(bare.group(1))

    parsed.confidence = float(np.mean(confidences)) if confidences else 0.0
    return parsed


class PhotoLocalizer(ABC):
    """Interface for turning a photo into location candidates."""

    @abstractmethod
    def localize(
        self,
        image_path: Path,
        prior_regions: list[WeightedRegion],
        ocr_results: list[OCRResult],
    ) -> list[LocationCandidate]:
        """Return candidate locations, best first. May return an empty list."""


class OCRStreetMatcher(PhotoLocalizer):
    """The implemented localiser: OCR text matched against road-graph street names.

    This never geocodes against an external service. It only asks: "of the
    streets the particle filter currently believes the car could be on, which
    one is named on this sign?" That is a legitimate, self-contained constraint,
    and it is honest about being nothing more than that.
    """

    def __init__(self, network: RoadNetwork, cfg: Optional[PhotoConfig] = None) -> None:
        self.net = network
        self.cfg = cfg or PhotoConfig()
        self._by_name: dict[str, list[int]] = {}
        for edge in self.net.edges:
            if not edge.name:
                continue
            key = self._street_key(str(edge.name))
            if key:
                self._by_name.setdefault(key, []).append(edge.index)

    @staticmethod
    def _street_key(name: str) -> str:
        """Reduce a street name to its distinctive words, dropping the type."""
        tokens = [t for t in normalize(name).split() if t not in STREET_TYPES]
        return " ".join(tokens)

    def match_edges(self, parsed: ParsedSign) -> dict[int, float]:
        """Edge index -> match score in [0, 1] for the streets named on the sign."""
        scores: dict[int, float] = {}
        for candidate in parsed.street_names:
            key = self._street_key(candidate)
            if not key:
                continue
            for name, indices in self._by_name.items():
                score = _name_similarity(key, name)
                if score < 0.6:
                    continue
                for index in indices:
                    scores[index] = max(scores.get(index, 0.0), score)
        return scores

    def localize(
        self,
        image_path: Path,
        prior_regions: list[WeightedRegion],
        ocr_results: list[OCRResult],
    ) -> list[LocationCandidate]:
        parsed = parse_ocr(ocr_results, self.cfg)
        if parsed.is_empty:
            return []
        matches = self.match_edges(parsed)
        candidates: list[LocationCandidate] = []
        for index, score in sorted(matches.items(), key=lambda kv: -kv[1]):
            edge = self.net.edges[index]
            east, north = edge.position(edge.length * 0.5)
            lat, lon = self.net.frame.to_geo(east, north)
            prior = _prior_at(prior_regions, lat, lon, self.cfg.address_boost_radius_m)
            if prior <= 0:
                continue
            candidates.append(
                LocationCandidate(
                    latitude=lat,
                    longitude=lon,
                    score=score * prior,
                    source="ocr_street_match",
                    label=str(edge.name),
                    radius_m=max(edge.length * 0.5, 40.0),
                    diagnostics={
                        "edge_index": index,
                        "name_match_score": round(score, 3),
                        "prior_weight": round(prior, 4),
                        "house_numbers": parsed.house_numbers,
                        "parking_zones": parsed.parking_zones,
                        "note": (
                            "Street-name match against the road graph only. This "
                            "is not a verified address and not a visual match."
                        ),
                    },
                )
            )
        candidates.sort(key=lambda c: -c.score)
        return candidates[:5]

    def particle_likelihood(
        self, parsed: ParsedSign, edge_idx: np.ndarray, alpha: Optional[float] = None
    ) -> np.ndarray:
        """L_OCR(p_i): boost particles sitting on a street named on the sign.

        Applied as ``w'_i ~ w_i * L_OCR(p_i)^beta``; particles elsewhere keep a
        weight of 1 rather than 0, because a sign in a photograph can easily be
        the cross street.
        """
        beta = self.cfg.ocr_boost_beta if alpha is None else alpha
        matches = self.match_edges(parsed)
        if not matches:
            return np.ones(len(edge_idx), dtype=float)
        boost = np.ones(len(self.net.edges), dtype=float)
        for index, score in matches.items():
            boost[index] = 1.0 + self.cfg.address_boost_alpha * score
        return boost[np.asarray(edge_idx, dtype=np.int64)] ** beta


def _name_similarity(a: str, b: str) -> float:
    """Token-overlap similarity, tolerant of OCR noise and word order."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    exact = len(ta & tb)
    fuzzy = 0.0
    for wa in ta - tb:
        best = max((_ratio(wa, wb) for wb in tb - ta), default=0.0)
        if best >= 0.8:
            fuzzy += best
    return (exact + fuzzy) / max(len(ta), len(tb))


def _ratio(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b).ratio()


def _prior_at(regions: Sequence[WeightedRegion], lat: float, lon: float, radius_m: float) -> float:
    """How much prior mass sits near a candidate point."""
    if not regions:
        return 1.0
    total = 0.0
    for region in regions:
        # Small-angle metric distance is fine at these scales.
        dlat = (lat - region.latitude) * 111320.0
        dlon = (lon - region.longitude) * 111320.0 * math.cos(math.radians(lat))
        distance = math.hypot(dlat, dlon)
        if distance <= radius_m + region.radius_m:
            total += region.probability
    return total


# ---------------------------------------------------------------------------
# Visual place recognition - interface only, deliberately not implemented.
# ---------------------------------------------------------------------------

VPR_DATABASE_COLUMNS = (
    "image_path",   # str,   path to the reference image
    "latitude",     # float, WGS84
    "longitude",    # float, WGS84
    "heading",      # float, degrees from true north the camera was facing
    "captured_at",  # ISO-8601 timestamp
    "embedding",    # float32[D], L2-normalised descriptor from Encoder(I)
)


@dataclass
class VPRMatch:
    image_path: str
    latitude: float
    longitude: float
    similarity: float
    heading: Optional[float] = None


class VisualPlaceRecognizer(ABC):
    """Interface for image-to-place matching.

    Expected reference database, one row per geo-referenced image::

        image_path, latitude, longitude, heading, captured_at, embedding

    Enable a real implementation only with a legally usable, geo-referenced
    image set. Downloading or scraping Google or Yandex panoramas is out of
    scope and is not supported here.
    """

    @abstractmethod
    def encode(self, image_path: Path) -> np.ndarray:
        """e_I = Encoder(I), L2-normalised."""

    @abstractmethod
    def query(self, embedding: np.ndarray, top_k: int = 10) -> list[VPRMatch]:
        """Nearest reference images by cosine similarity."""

    def posterior(
        self, matches: Sequence[VPRMatch], priors: Sequence[float], temperature: float
    ) -> np.ndarray:
        """P(l_j | I) ~ exp(similarity / tau) * P_prior(l_j)."""
        if not matches:
            return np.zeros(0)
        sims = np.array([m.similarity for m in matches], dtype=float)
        prior = np.asarray(priors, dtype=float) if len(priors) == len(matches) else np.ones(len(matches))
        logits = sims / max(temperature, 1e-6)
        weights = np.exp(logits - logits.max()) * prior
        total = weights.sum()
        return weights / total if total > 0 else np.full(len(matches), 1.0 / len(matches))


class NullVisualPlaceRecognizer(VisualPlaceRecognizer):
    """The shipped stub. Returns nothing, and says why.

    It exists so the pipeline has a real seam to plug a recogniser into, not to
    pretend there is one. Fabricating matches would be worse than returning
    nothing.
    """

    reason = (
        "No geo-referenced reference image database is bundled with this "
        "repository, so visual place recognition is unavailable. Provide a "
        "database with columns "
        + ", ".join(VPR_DATABASE_COLUMNS)
        + " and implement VisualPlaceRecognizer to enable it."
    )

    def encode(self, image_path: Path) -> np.ndarray:
        raise NotImplementedError(self.reason)

    def query(self, embedding: np.ndarray, top_k: int = 10) -> list[VPRMatch]:
        return []

    def available(self) -> bool:
        return False


def photo_diagnostics(photo: Photo, parsed: ParsedSign, candidates: Sequence[LocationCandidate]) -> dict[str, Any]:
    """Everything worth keeping about one photo, for diagnostics.json."""
    return {
        "image_path": Path(photo.image_path).name if photo.image_path else None,
        "captured_at": photo.captured_at.isoformat() if photo.captured_at else None,
        "monotonic_time": photo.monotonic_time,
        "camera_orientation": photo.camera_orientation,
        "device_heading_deg": photo.device_heading_deg,
        "assumed_position": (
            None
            if photo.assumed_latitude is None
            else {
                "latitude": photo.assumed_latitude,
                "longitude": photo.assumed_longitude,
                "accuracy_m": photo.assumed_accuracy,
                "note": "last position the app believed in, not a verified address",
            }
        ),
        "exif_position": (
            None
            if photo.exif_latitude is None
            else {"latitude": photo.exif_latitude, "longitude": photo.exif_longitude}
        ),
        "ocr": parsed.to_json(),
        "candidates": [c.to_json() for c in candidates],
        "vpr": {"available": False, "reason": NullVisualPlaceRecognizer.reason},
    }
