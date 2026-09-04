"""Parking-zone probability and photo/OCR handling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from geotrace.config import ParkingConfig, PhotoConfig
from geotrace.coordinates import LocalFrame
from geotrace.models import LocationCandidate, OCRResult, Photo, WeightedRegion
from geotrace.parking import (
    GeoJSONParkingZoneProvider,
    ParkingZoneProvider,
    decide_zone,
    write_test_zones,
    zone_probabilities,
)
from geotrace.photo import (
    NullVisualPlaceRecognizer,
    OCRStreetMatcher,
    VPR_DATABASE_COLUMNS,
    parse_ocr,
    photo_diagnostics,
)
from geotrace.road_graph import RoadNetwork

FRAME = LocalFrame(59.9311, 30.3609)
CFG = ParkingConfig()


@pytest.fixture
def zones_file(tmp_path: Path) -> Path:
    return write_test_zones(
        tmp_path / "zones.geojson",
        FRAME,
        [
            ("test-zone-01", -50.0, -50.0, 250.0, 150.0),
            ("test-zone-02", 400.0, -50.0, 700.0, 150.0),
        ],
    )


def test_provider_loads_zones_in_wgs84(zones_file: Path) -> None:
    frame = GeoJSONParkingZoneProvider(zones_file).load_zones()
    assert len(frame) == 2
    assert frame.crs.to_epsg() == 4326
    assert set(frame["id"]) == {"test-zone-01", "test-zone-02"}


def test_provider_is_explicit_that_the_test_zones_are_not_official(zones_file: Path) -> None:
    attribution = GeoJSONParkingZoneProvider(zones_file).attribution
    assert "not the official" in attribution.lower()


def test_provider_reports_a_missing_file_usefully(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="parking zones file not found"):
        GeoJSONParkingZoneProvider(tmp_path / "nope.geojson").load_zones()


def test_provider_interface_is_abstract() -> None:
    with pytest.raises(TypeError):
        ParkingZoneProvider()  # type: ignore[abstract]


def _positions(points: list[tuple[float, float]]) -> np.ndarray:
    return np.array([FRAME.to_geo(e, n) for e, n in points])


def test_zone_probability_is_the_weight_of_the_particles_inside(zones_file: Path) -> None:
    """P(Z_j) = sum_i w_i * 1[pos(p_i) in Z_j]."""
    zones = GeoJSONParkingZoneProvider(zones_file).load_zones()
    positions = _positions([(100.0, 50.0)] * 8 + [(500.0, 50.0)] * 2)
    weights = np.full(10, 0.1)
    ranking = zone_probabilities(zones, positions, weights)
    by_id = {z.zone_id: z.probability for z in ranking}
    assert by_id["test-zone-01"] == pytest.approx(0.8)
    assert by_id["test-zone-02"] == pytest.approx(0.2)


def test_particles_outside_every_zone_count_for_nothing(zones_file: Path) -> None:
    zones = GeoJSONParkingZoneProvider(zones_file).load_zones()
    positions = _positions([(5000.0, 5000.0)] * 5)
    ranking = zone_probabilities(zones, positions, np.full(5, 0.2))
    assert all(z.probability == 0.0 for z in ranking)


def test_zone_ranking_is_sorted(zones_file: Path) -> None:
    zones = GeoJSONParkingZoneProvider(zones_file).load_zones()
    positions = _positions([(500.0, 50.0)] * 9 + [(100.0, 50.0)])
    ranking = zone_probabilities(zones, positions, np.full(10, 0.1))
    assert ranking[0].zone_id == "test-zone-02"
    assert ranking[0].probability >= ranking[1].probability


# ------------------------------------------------------------- the decision


def test_a_dominant_zone_is_confident() -> None:
    from geotrace.parking import ZoneProbability

    ranking = [ZoneProbability("test-zone-01", 0.982), ZoneProbability("test-zone-02", 0.011)]
    decision = decide_zone(ranking, CFG)
    assert decision.selected_zone == "test-zone-01"
    assert decision.decision == "confident"
    assert decision.probability == pytest.approx(0.982)
    assert decision.second_probability == pytest.approx(0.011)


def test_a_close_call_is_ambiguous_not_confident() -> None:
    """Two adjacent zones at 0.5 each must never be reported as a decision."""
    from geotrace.parking import ZoneProbability

    ranking = [ZoneProbability("a", 0.52), ZoneProbability("b", 0.47)]
    assert decide_zone(ranking, CFG).decision == "ambiguous"


def test_a_high_probability_with_a_thin_margin_is_ambiguous() -> None:
    from geotrace.parking import ZoneProbability

    ranking = [ZoneProbability("a", 0.99), ZoneProbability("b", 0.90)]
    assert decide_zone(ranking, CFG).decision == "ambiguous"


def test_thresholds_are_configurable() -> None:
    from geotrace.parking import ZoneProbability

    ranking = [ZoneProbability("a", 0.80), ZoneProbability("b", 0.05)]
    assert decide_zone(ranking, ParkingConfig()).decision == "ambiguous"
    assert decide_zone(
        ranking, ParkingConfig(selected_probability=0.75, margin_to_second=0.5)
    ).decision == "confident"


def test_no_zone_at_all_is_reported_honestly() -> None:
    decision = decide_zone([], CFG)
    assert decision.selected_zone is None
    assert decision.decision == "outside_known_zones"


def test_all_zones_empty_is_reported_as_outside() -> None:
    from geotrace.parking import ZoneProbability

    decision = decide_zone([ZoneProbability("a", 0.0)], CFG)
    assert decision.decision == "outside_known_zones"


def test_decision_json_carries_the_zone_source() -> None:
    from geotrace.parking import ZoneProbability

    payload = decide_zone([ZoneProbability("a", 0.99)], CFG, attribution="UNOFFICIAL test").to_json()
    assert payload["zone_source"] == "UNOFFICIAL test"
    assert set(payload) >= {"selected_zone", "probability", "second_probability", "decision"}


# ------------------------------------------------------------------- OCR


def test_ocr_extracts_a_street_a_house_and_a_zone() -> None:
    parsed = parse_ocr(
        [
            OCRResult("Лиговский проспект", 0.95),
            OCRResult("д. 45 к.2", 0.81),
            OCRResult("Парковочная зона 7812", 0.9),
        ]
    )
    assert "лиговский" in parsed.street_names
    assert "45" in parsed.house_numbers
    assert "7812" in parsed.parking_zones


def test_ocr_handles_an_english_sign() -> None:
    parsed = parse_ocr([OCRResult("Nevsky Avenue", 0.9), OCRResult("house 28", 0.8)])
    assert parsed.street_names and "28" in parsed.house_numbers


def test_low_confidence_lines_are_discarded() -> None:
    parsed = parse_ocr(
        [OCRResult("Садовая улица", 0.9), OCRResult("гтрщб", 0.05)],
        PhotoConfig(min_ocr_confidence=0.3),
    )
    assert parsed.raw_lines == ["Садовая улица"]


def test_ocr_of_nothing_is_empty_not_a_guess() -> None:
    assert parse_ocr([]).is_empty
    assert parse_ocr([OCRResult("...", 0.9)]).is_empty


def test_ocr_mean_confidence_is_reported() -> None:
    parsed = parse_ocr([OCRResult("Садовая улица", 0.8), OCRResult("дом 12", 0.6)])
    assert parsed.confidence == pytest.approx(0.7)


# ------------------------------------------------------ OCR to road matching


def test_ocr_boosts_particles_on_the_named_street(fork_network: RoadNetwork) -> None:
    matcher = OCRStreetMatcher(fork_network)
    parsed = parse_ocr([OCRResult("улица Branch A", 0.9)])
    edge_idx = np.array([e.index for e in fork_network.edges])
    likelihood = matcher.particle_likelihood(parsed, edge_idx)

    on_a = [e.index for e in fork_network.edges if str(e.name) == "Branch A"]
    on_b = [e.index for e in fork_network.edges if str(e.name) == "Branch B"]
    assert likelihood[on_a].mean() > likelihood[on_b].mean()


def test_particles_elsewhere_keep_a_neutral_weight(fork_network: RoadNetwork) -> None:
    """A sign in a photo may well be the cross street, so nothing is zeroed."""
    matcher = OCRStreetMatcher(fork_network)
    parsed = parse_ocr([OCRResult("улица Branch A", 0.9)])
    edge_idx = np.array([e.index for e in fork_network.edges])
    likelihood = matcher.particle_likelihood(parsed, edge_idx)
    assert likelihood.min() >= 1.0


def test_an_unrecognised_street_leaves_every_weight_alone(fork_network: RoadNetwork) -> None:
    matcher = OCRStreetMatcher(fork_network)
    parsed = parse_ocr([OCRResult("улица Несуществующая", 0.9)])
    edge_idx = np.array([e.index for e in fork_network.edges])
    assert np.all(matcher.particle_likelihood(parsed, edge_idx) == 1.0)


def test_localize_returns_candidates_within_the_prior(fork_network: RoadNetwork) -> None:
    matcher = OCRStreetMatcher(fork_network)
    edge = next(e for e in fork_network.edges if str(e.name) == "Branch A")
    lat, lon = fork_network.frame.to_geo(*edge.position(edge.length / 2))
    regions = [WeightedRegion("branch-01", 0.9, lat, lon, 200.0)]
    candidates = matcher.localize(
        Path("nonexistent.jpg"), regions, [OCRResult("улица Branch A", 0.95)]
    )
    assert candidates
    assert candidates[0].source == "ocr_street_match"
    assert "not a verified address" in candidates[0].diagnostics["note"]


def test_localize_returns_nothing_when_there_is_no_text(fork_network: RoadNetwork) -> None:
    matcher = OCRStreetMatcher(fork_network)
    assert matcher.localize(Path("x.jpg"), [], []) == []


# ------------------------------------------------- visual place recognition


def test_the_vpr_stub_finds_nothing_and_says_why() -> None:
    """Fabricating photo matches would be worse than returning nothing."""
    vpr = NullVisualPlaceRecognizer()
    assert vpr.query(np.zeros(8)) == []
    assert not vpr.available()
    assert "no geo-referenced reference image database" in vpr.reason.lower()


def test_the_vpr_stub_refuses_to_encode() -> None:
    with pytest.raises(NotImplementedError):
        NullVisualPlaceRecognizer().encode(Path("x.jpg"))


def test_the_expected_vpr_database_format_is_documented() -> None:
    assert VPR_DATABASE_COLUMNS == (
        "image_path", "latitude", "longitude", "heading", "captured_at", "embedding",
    )


def test_vpr_posterior_matches_the_softmax_formula() -> None:
    """P(l_j | I) ~ exp(similarity / tau) * P_prior(l_j)."""
    from geotrace.photo import VPRMatch

    vpr = NullVisualPlaceRecognizer()
    matches = [VPRMatch("a.jpg", 0.0, 0.0, 0.9), VPRMatch("b.jpg", 0.0, 0.0, 0.5)]
    posterior = vpr.posterior(matches, [1.0, 1.0], temperature=0.1)
    assert posterior.sum() == pytest.approx(1.0)
    assert posterior[0] > posterior[1]

    expected = np.exp(np.array([0.9, 0.5]) / 0.1)
    assert posterior == pytest.approx(expected / expected.sum())


def test_the_prior_shifts_the_posterior() -> None:
    from geotrace.photo import VPRMatch

    vpr = NullVisualPlaceRecognizer()
    matches = [VPRMatch("a.jpg", 0.0, 0.0, 0.6), VPRMatch("b.jpg", 0.0, 0.0, 0.6)]
    posterior = vpr.posterior(matches, [0.9, 0.1], temperature=0.1)
    assert posterior[0] == pytest.approx(0.9)


def test_photo_diagnostics_never_claim_a_verified_address() -> None:
    photo = Photo(
        image_path="photo-001.jpg",
        ocr=[OCRResult("Садовая улица 12", 0.9)],
        assumed_latitude=59.93,
        assumed_longitude=30.36,
    )
    payload = photo_diagnostics(photo, parse_ocr(photo.ocr), [])
    assert payload["vpr"]["available"] is False
    assert "not a verified address" in payload["assumed_position"]["note"]
    json.dumps(payload)


def test_opening_hours_are_not_read_as_a_house_number_or_a_zone() -> None:
    """A parking sign carries its hours next to its zone number. Reading
    "Платная парковка 08:00-21:00" as house 8 in zone 8 poisons both the address
    extraction and the zone decision."""
    parsed = parse_ocr(
        [
            OCRResult("Воскресенская набережная", 0.94),
            OCRResult("д. 12", 0.86),
            OCRResult("Парковочная зона 7812", 0.81),
            OCRResult("Платная парковка 08:00-21:00", 0.64),
        ]
    )
    assert parsed.house_numbers == ["12"]
    assert parsed.parking_zones == ["7812"]
    # The line is still kept verbatim for diagnostics.
    assert any("08:00" in line for line in parsed.raw_lines)


def test_a_bare_time_yields_nothing() -> None:
    parsed = parse_ocr([OCRResult("09:00 - 18:00", 0.9)])
    assert parsed.house_numbers == []
    assert parsed.parking_zones == []


def test_a_two_digit_zone_is_not_invented_from_a_time() -> None:
    parsed = parse_ocr([OCRResult("парковка 07:30", 0.9)])
    assert parsed.parking_zones == []
