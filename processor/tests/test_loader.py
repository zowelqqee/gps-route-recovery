"""Reading and writing trip directories."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geotrace.loader import TripLoadError, load_trip, write_trip
from geotrace.models import LocationSample, MotionSample, MountCalibration, Trip, TripMetadata


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


GPS_LINE = json.dumps(
    {
        "type": "location",
        "wall_time": "2026-09-04T12:30:05.520Z",
        "monotonic_time": 13842.52,
        "latitude": 59.93431,
        "longitude": 30.32574,
        "horizontal_accuracy": 11.2,
        "speed": 12.4,
        "course": 274.0,
    }
)

MOTION_LINE = json.dumps(
    {
        "type": "motion",
        "monotonic_time": 13842.54,
        "user_acceleration_g": [0.12, -0.03, 0.01],
        "user_acceleration_ms2": [1.1768, -0.2942, 0.0981],
        "rotation_rate": [0.01, -0.02, 0.18],
        "gravity": [0.02, -0.12, -0.99],
        "quaternion": [0.91, 0.02, 0.03, 0.41],
    }
)


def test_the_documented_sample_lines_parse(tmp_path: Path) -> None:
    """Exactly the two examples from the specification."""
    write_lines(tmp_path / "trip" / "samples.jsonl", [GPS_LINE, MOTION_LINE])
    trip, report = load_trip(tmp_path / "trip")
    assert len(trip.locations) == 1 and len(trip.motions) == 1
    assert report.malformed_lines == 0

    gps = trip.locations[0]
    assert gps.latitude == pytest.approx(59.93431)
    assert gps.horizontal_accuracy == pytest.approx(11.2)
    assert gps.wall_time is not None and gps.wall_time.year == 2026

    motion = trip.motions[0]
    assert motion.user_acceleration_ms2 == pytest.approx((1.1768, -0.2942, 0.0981), abs=1e-4)


def test_metadata_is_read(tmp_path: Path) -> None:
    root = tmp_path / "trip-abc"
    write_lines(root / "samples.jsonl", [GPS_LINE])
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "trip_id": "ABC-123",
                "started_at": "2026-09-04T12:30:00Z",
                "device_model": "iPhone15,3",
                "calibration": {
                    "reference_quaternion": [1, 0, 0, 0],
                    "gravity_device": [0, 0, -1],
                    "heading_source": "gps_course",
                    "still_duration_s": 6.0,
                },
            }
        )
    )
    trip, _ = load_trip(root)
    assert trip.metadata.trip_id == "ABC-123"
    assert trip.metadata.device_model == "iPhone15,3"
    assert trip.metadata.calibration is not None
    assert trip.metadata.calibration.heading_source == "gps_course"


def test_a_missing_metadata_file_falls_back_to_the_directory_name(tmp_path: Path) -> None:
    write_lines(tmp_path / "trip-xyz" / "samples.jsonl", [GPS_LINE])
    trip, _ = load_trip(tmp_path / "trip-xyz")
    assert trip.metadata.trip_id == "trip-xyz"


def test_a_truncated_last_line_is_skipped_not_fatal(tmp_path: Path) -> None:
    """samples.jsonl is written incrementally; the app can be killed mid-write."""
    write_lines(
        tmp_path / "trip" / "samples.jsonl",
        [GPS_LINE, MOTION_LINE, '{"type": "location", "monotonic_ti'],
    )
    trip, report = load_trip(tmp_path / "trip")
    assert len(trip.locations) == 1
    assert report.malformed_lines == 1


def test_unknown_sample_types_are_counted_and_ignored(tmp_path: Path) -> None:
    write_lines(
        tmp_path / "trip" / "samples.jsonl",
        [GPS_LINE, json.dumps({"type": "barometer", "monotonic_time": 1.0})],
    )
    _trip, report = load_trip(tmp_path / "trip")
    assert report.unknown_type_lines == 1


def test_fixes_with_negative_accuracy_are_dropped(tmp_path: Path) -> None:
    """CoreLocation marks an invalid fix with a negative accuracy."""
    bad = json.dumps(
        {"type": "location", "monotonic_time": 1.0, "latitude": 59.9,
         "longitude": 30.3, "horizontal_accuracy": -1.0}
    )
    write_lines(tmp_path / "trip" / "samples.jsonl", [GPS_LINE, bad])
    trip, report = load_trip(tmp_path / "trip")
    assert len(trip.locations) == 1
    assert report.rejected_reasons["negative_accuracy"] == 1


def test_fixes_with_impossible_coordinates_are_dropped(tmp_path: Path) -> None:
    bad = json.dumps(
        {"type": "location", "monotonic_time": 1.0, "latitude": 200.0,
         "longitude": 30.3, "horizontal_accuracy": 5.0}
    )
    write_lines(tmp_path / "trip" / "samples.jsonl", [GPS_LINE, bad])
    trip, report = load_trip(tmp_path / "trip")
    assert len(trip.locations) == 1
    assert report.rejected_locations == 1


def test_samples_are_sorted_by_time(tmp_path: Path) -> None:
    lines = [
        json.dumps({"type": "location", "monotonic_time": t, "latitude": 59.9,
                    "longitude": 30.3, "horizontal_accuracy": 5.0})
        for t in (30.0, 10.0, 20.0)
    ]
    write_lines(tmp_path / "trip" / "samples.jsonl", lines)
    trip, _ = load_trip(tmp_path / "trip")
    assert [s.monotonic_time for s in trip.locations] == [10.0, 20.0, 30.0]


def test_blank_lines_are_tolerated(tmp_path: Path) -> None:
    (tmp_path / "trip").mkdir(parents=True)
    (tmp_path / "trip" / "samples.jsonl").write_text(f"\n{GPS_LINE}\n\n{MOTION_LINE}\n\n")
    trip, report = load_trip(tmp_path / "trip")
    assert len(trip.locations) == 1 and report.malformed_lines == 0


def test_a_directory_without_samples_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(TripLoadError, match="samples.jsonl"):
        load_trip(tmp_path / "empty")


def test_a_missing_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(TripLoadError, match="not a trip directory"):
        load_trip(tmp_path / "nope")


def test_photos_and_their_ocr_are_loaded(tmp_path: Path) -> None:
    root = tmp_path / "trip"
    write_lines(root / "samples.jsonl", [GPS_LINE])
    photos = root / "photos"
    photos.mkdir()
    (photos / "photo-001.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (photos / "photo-001.json").write_text(
        json.dumps(
            {
                "image_path": "photo-001.jpg",
                "captured_at": "2026-09-04T12:40:00Z",
                "ocr": [{"text": "Садовая улица", "confidence": 0.92}],
                "assumed_latitude": 59.93,
                "assumed_longitude": 30.33,
            }
        ),
        encoding="utf-8",
    )
    trip, _ = load_trip(root)
    assert len(trip.photos) == 1
    assert trip.photos[0].ocr[0].text == "Садовая улица"
    assert Path(trip.photos[0].image_path).exists()


def test_round_trip_through_write_and_load(tmp_path: Path) -> None:
    trip = Trip(
        metadata=TripMetadata(
            trip_id="round-trip",
            calibration=MountCalibration(heading_source="gps_course", still_duration_s=6.0),
        ),
        locations=[
            LocationSample(monotonic_time=1.0, latitude=59.93, longitude=30.33,
                           horizontal_accuracy=9.0, speed=11.0, course=90.0)
        ],
        motions=[
            MotionSample(monotonic_time=1.02, user_acceleration_g=(0.1, 0.0, 0.0),
                         rotation_rate=(0.0, 0.0, 0.05))
        ],
    )
    out = write_trip(trip, tmp_path / "written")
    loaded, report = load_trip(out)
    assert report.malformed_lines == 0
    assert loaded.metadata.trip_id == "round-trip"
    assert loaded.locations[0].latitude == pytest.approx(59.93)
    assert loaded.motions[0].user_acceleration_g == pytest.approx((0.1, 0.0, 0.0))
    assert loaded.metadata.calibration.heading_source == "gps_course"


def test_the_written_stream_is_one_merged_timeline(tmp_path: Path) -> None:
    """Each line of samples.jsonl is a GPS or a motion sample, in time order."""
    trip = Trip(
        metadata=TripMetadata(trip_id="merged"),
        locations=[LocationSample(monotonic_time=float(t), latitude=59.93, longitude=30.33,
                                  horizontal_accuracy=9.0) for t in (1, 3)],
        motions=[MotionSample(monotonic_time=float(t)) for t in (2, 4)],
    )
    out = write_trip(trip, tmp_path / "merged")
    rows = [json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines()]
    assert [r["type"] for r in rows] == ["location", "motion", "location", "motion"]
    assert [r["monotonic_time"] for r in rows] == [1.0, 2.0, 3.0, 4.0]


def test_the_reference_track_survives_a_round_trip(tmp_path: Path) -> None:
    trip = Trip(
        metadata=TripMetadata(trip_id="ref"),
        locations=[LocationSample(monotonic_time=1.0, latitude=59.93, longitude=30.33,
                                  horizontal_accuracy=9.0)],
        reference_locations=[LocationSample(monotonic_time=1.0, latitude=59.94, longitude=30.34,
                                            horizontal_accuracy=9.0)],
    )
    out = write_trip(trip, tmp_path / "ref")
    loaded, _ = load_trip(out)
    assert loaded.reference_locations[0].latitude == pytest.approx(59.94)


def test_an_exported_zip_package_is_accepted(tmp_path: Path) -> None:
    """The app shares a zip; the processor must open it without manual unpacking."""
    import zipfile

    root = tmp_path / "trip-zip"
    write_lines(root / "samples.jsonl", [GPS_LINE, MOTION_LINE])
    (root / "metadata.json").write_text(json.dumps({"trip_id": "ZIP-1"}))
    archive = tmp_path / "trip.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in root.rglob("*"):
            zf.write(path, Path("trip-zip") / path.relative_to(root))

    trip, _ = load_trip(archive)
    assert trip.metadata.trip_id == "ZIP-1"
    assert len(trip.locations) == 1


def test_a_zip_with_a_traversal_path_is_refused(tmp_path: Path) -> None:
    import zipfile

    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escaped.txt", "nope")
    with pytest.raises(TripLoadError, match="unsafe path"):
        load_trip(archive)


def test_extra_unknown_keys_do_not_break_parsing(tmp_path: Path) -> None:
    """A newer app version may add fields the processor has never seen."""
    line = json.dumps(
        {"type": "location", "monotonic_time": 1.0, "latitude": 59.9, "longitude": 30.3,
         "horizontal_accuracy": 5.0, "future_field": {"nested": True}}
    )
    write_lines(tmp_path / "trip" / "samples.jsonl", [line])
    trip, report = load_trip(tmp_path / "trip")
    assert len(trip.locations) == 1 and report.malformed_lines == 0
