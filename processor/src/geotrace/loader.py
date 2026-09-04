"""Read and write trip directories.

The loader is deliberately forgiving: `samples.jsonl` is written incrementally
on the phone, so the last line may be truncated if the app was killed. A bad
line is counted and skipped, never fatal.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from geotrace.models import (
    LocationSample,
    MotionSample,
    Photo,
    Trip,
    TripMetadata,
    sort_samples,
)

SAMPLES_FILE = "samples.jsonl"
METADATA_FILE = "metadata.json"
FAULTS_FILE = "faults.json"
REFERENCE_FILE = "reference-samples.jsonl"
PHOTOS_DIR = "photos"
RESULTS_DIR = "results"


class TripLoadError(RuntimeError):
    """Raised when a directory cannot be interpreted as a trip at all."""


@dataclass
class LoadReport:
    """What the loader had to skip. Ends up in diagnostics.json."""

    total_lines: int = 0
    malformed_lines: int = 0
    unknown_type_lines: int = 0
    rejected_locations: int = 0
    rejected_reasons: dict[str, int] = field(default_factory=dict)

    def note_rejection(self, reason: str) -> None:
        self.rejected_locations += 1
        self.rejected_reasons[reason] = self.rejected_reasons.get(reason, 0) + 1

    def to_json(self) -> dict[str, Any]:
        return {
            "total_lines": self.total_lines,
            "malformed_lines": self.malformed_lines,
            "unknown_type_lines": self.unknown_type_lines,
            "rejected_locations": self.rejected_locations,
            "rejected_reasons": self.rejected_reasons,
        }


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any] | None]]:
    """Yield (line_number, parsed-or-None) for every non-empty line."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for number, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield number, json.loads(raw)
            except json.JSONDecodeError:
                yield number, None


def _read_samples(
    path: Path, report: LoadReport, drop_invalid_locations: bool
) -> tuple[list[LocationSample], list[MotionSample]]:
    locations: list[LocationSample] = []
    motions: list[MotionSample] = []
    for _number, obj in iter_jsonl(path):
        report.total_lines += 1
        if obj is None:
            report.malformed_lines += 1
            continue
        kind = obj.get("type")
        try:
            if kind == "location":
                sample = LocationSample.from_json(obj)
                if drop_invalid_locations and not sample.is_usable:
                    reason = (
                        "negative_accuracy"
                        if (sample.horizontal_accuracy or -1) < 0
                        else "invalid_coordinates_or_time"
                    )
                    report.note_rejection(reason)
                    continue
                locations.append(sample)
            elif kind == "motion":
                motions.append(MotionSample.from_json(obj))
            else:
                report.unknown_type_lines += 1
        except (KeyError, TypeError, ValueError):
            report.malformed_lines += 1
    return sort_samples(locations), sort_samples(motions)


def load_trip(
    path: str | Path, drop_invalid_locations: bool = True
) -> tuple[Trip, LoadReport]:
    """Load a trip directory (or a .zip / .geotrace package exported by the app)."""
    root = Path(path)
    if root.is_file() and root.suffix.lower() in {".zip", ".geotrace"}:
        root = _extract_package(root)
    if not root.is_dir():
        raise TripLoadError(f"not a trip directory: {path}")

    root = _descend_to_trip_root(root)
    samples_path = root / SAMPLES_FILE
    if not samples_path.exists():
        raise TripLoadError(
            f"{root} contains no {SAMPLES_FILE}. "
            "Expected a trip directory exported by GeoTraceLab."
        )

    report = LoadReport()
    metadata_path = root / METADATA_FILE
    if metadata_path.exists():
        metadata = TripMetadata.from_json(json.loads(metadata_path.read_text("utf-8")))
    else:
        metadata = TripMetadata(trip_id=root.name)

    locations, motions = _read_samples(samples_path, report, drop_invalid_locations)

    photos: list[Photo] = []
    photos_dir = root / PHOTOS_DIR
    if photos_dir.is_dir():
        for sidecar in sorted(photos_dir.glob("*.json")):
            try:
                photo = Photo.from_json(json.loads(sidecar.read_text("utf-8")))
            except json.JSONDecodeError:
                report.malformed_lines += 1
                continue
            if not photo.image_path:
                photo.image_path = str(sidecar.with_suffix(".jpg").name)
            candidate = photos_dir / Path(photo.image_path).name
            if candidate.exists():
                photo.image_path = str(candidate)
            photos.append(photo)

    faults = None
    faults_path = root / FAULTS_FILE
    if faults_path.exists():
        faults = json.loads(faults_path.read_text("utf-8"))

    reference: list[LocationSample] = []
    reference_path = root / REFERENCE_FILE
    if reference_path.exists():
        ref_report = LoadReport()
        reference, _ = _read_samples(reference_path, ref_report, drop_invalid_locations)

    trip = Trip(
        metadata=metadata,
        locations=locations,
        motions=motions,
        photos=photos,
        root=str(root),
        faults=faults,
        reference_locations=reference,
    )
    return trip, report


def _descend_to_trip_root(root: Path) -> Path:
    """A zip may wrap the trip in one extra directory level; look one down."""
    if (root / SAMPLES_FILE).exists():
        return root
    children = [c for c in root.iterdir() if c.is_dir()]
    for child in children:
        if (child / SAMPLES_FILE).exists():
            return child
    return root


def _extract_package(archive: Path) -> Path:
    target = archive.parent / f".{archive.stem}-extracted"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            # Refuse absolute paths and traversal, we are unpacking user data.
            if member.startswith("/") or ".." in Path(member).parts:
                raise TripLoadError(f"unsafe path in archive: {member}")
        zf.extractall(target)
    return target


def write_trip(trip: Trip, out_dir: str | Path, copy_photos: bool = True) -> Path:
    """Write a trip back out in exactly the format `load_trip` reads."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / RESULTS_DIR).mkdir(exist_ok=True)

    metadata = trip.metadata
    metadata.location_sample_count = len(trip.locations)
    metadata.motion_sample_count = len(trip.motions)
    (out / METADATA_FILE).write_text(
        json.dumps(metadata.to_json(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    write_samples(out / SAMPLES_FILE, trip.locations, trip.motions)

    if trip.reference_locations:
        write_samples(out / REFERENCE_FILE, trip.reference_locations, [])

    if trip.faults is not None:
        (out / FAULTS_FILE).write_text(
            json.dumps(trip.faults, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    if trip.photos:
        photos_out = out / PHOTOS_DIR
        photos_out.mkdir(exist_ok=True)
        for index, photo in enumerate(trip.photos, start=1):
            source = Path(photo.image_path) if photo.image_path else None
            name = source.name if source else f"photo-{index:03d}.jpg"
            if copy_photos and source and source.exists():
                shutil.copy2(source, photos_out / name)
            stored = dict(photo.to_json())
            stored["image_path"] = name
            (photos_out / f"{Path(name).stem}.json").write_text(
                json.dumps(stored, indent=2, ensure_ascii=False), encoding="utf-8"
            )
    return out


def write_samples(
    path: str | Path,
    locations: Iterable[LocationSample],
    motions: Iterable[MotionSample],
) -> None:
    """Merge both streams onto one timeline and write samples.jsonl."""
    merged: list[Any] = list(locations) + list(motions)
    merged.sort(key=lambda s: (s.monotonic_time, 0 if isinstance(s, LocationSample) else 1))
    with Path(path).open("w", encoding="utf-8") as handle:
        for sample in merged:
            handle.write(json.dumps(sample.to_json(), ensure_ascii=False) + "\n")


def results_dir(trip_root: str | Path) -> Path:
    out = Path(trip_root) / RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
