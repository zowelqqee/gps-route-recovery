"""Data model shared by the iOS recorder, the loader and the algorithms.

The wire format is exactly what GeoTraceLab writes:

    trip-<UUID>/
      metadata.json
      samples.jsonl      one JSON object per line, "type": "location" | "motion"
      photos/photo-001.jpg + photo-001.json
      results/

`samples.jsonl` is append-only on the phone, so it must stay tolerant of a
truncated last line and of unknown extra keys.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from geotrace.config import G_TO_MS2


def _parse_iso(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _f(value: Any) -> float:
    """Plain Python float. numpy scalars are not JSON-serialisable."""
    return float(value)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class LocationSample:
    """One CoreLocation fix.

    `monotonic_time` is the phone's uptime clock and is the only timestamp the
    filters use: wall-clock can jump when the system clock is corrected.
    """

    monotonic_time: float
    latitude: float
    longitude: float
    wall_time: Optional[datetime] = None
    altitude: Optional[float] = None
    horizontal_accuracy: Optional[float] = None
    vertical_accuracy: Optional[float] = None
    speed: Optional[float] = None
    speed_accuracy: Optional[float] = None
    course: Optional[float] = None
    course_accuracy: Optional[float] = None
    source_information: Optional[dict[str, Any]] = None
    synthetic: bool = False
    """True when fault injection created or moved this fix."""

    @property
    def is_usable(self) -> bool:
        """CoreLocation marks an invalid fix with a negative accuracy."""
        if self.horizontal_accuracy is None or self.horizontal_accuracy < 0:
            return False
        if not math.isfinite(self.latitude) or not math.isfinite(self.longitude):
            return False
        if abs(self.latitude) > 90.0 or abs(self.longitude) > 180.0:
            return False
        if not math.isfinite(self.monotonic_time) or self.monotonic_time < 0:
            return False
        return True

    @property
    def has_valid_course(self) -> bool:
        return (
            self.course is not None
            and self.course >= 0.0
            and (self.course_accuracy is None or self.course_accuracy >= 0.0)
        )

    @property
    def has_valid_speed(self) -> bool:
        return self.speed is not None and self.speed >= 0.0

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "LocationSample":
        return cls(
            monotonic_time=float(obj["monotonic_time"]),
            latitude=float(obj["latitude"]),
            longitude=float(obj["longitude"]),
            wall_time=_parse_iso(obj.get("wall_time")),
            altitude=obj.get("altitude"),
            horizontal_accuracy=obj.get("horizontal_accuracy"),
            vertical_accuracy=obj.get("vertical_accuracy"),
            speed=obj.get("speed"),
            speed_accuracy=obj.get("speed_accuracy"),
            course=obj.get("course"),
            course_accuracy=obj.get("course_accuracy"),
            source_information=obj.get("source_information"),
            synthetic=bool(obj.get("synthetic", False)),
        )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "type": "location",
            "monotonic_time": round(_f(self.monotonic_time), 4),
            "latitude": _f(self.latitude),
            "longitude": _f(self.longitude),
        }
        if self.wall_time is not None:
            out["wall_time"] = _iso(self.wall_time)
        for key in (
            "altitude",
            "horizontal_accuracy",
            "vertical_accuracy",
            "speed",
            "speed_accuracy",
            "course",
            "course_accuracy",
            "source_information",
        ):
            value = getattr(self, key)
            if value is None:
                continue
            out[key] = value if isinstance(value, dict) else _f(value)
        if self.synthetic:
            out["synthetic"] = True
        return out


@dataclass
class MotionSample:
    """One CMDeviceMotion frame, device frame, 50 Hz on the phone."""

    monotonic_time: float
    user_acceleration_g: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_rate: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gravity: tuple[float, float, float] = (0.0, 0.0, -1.0)
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """(w, x, y, z), device -> reference frame, as CMAttitude reports it."""

    attitude: Optional[tuple[float, float, float]] = None
    """(roll, pitch, yaw) in radians, for diagnostics only."""

    magnetic_field: Optional[tuple[float, float, float]] = None
    magnetic_accuracy: Optional[int] = None
    wall_time: Optional[datetime] = None

    @property
    def user_acceleration_ms2(self) -> tuple[float, float, float]:
        """CoreMotion userAcceleration is in g; a_[m/s^2] = 9.80665 * a_[g]."""
        return tuple(G_TO_MS2 * c for c in self.user_acceleration_g)  # type: ignore[return-value]

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "MotionSample":
        acc_g = obj.get("user_acceleration_g")
        if acc_g is None and obj.get("user_acceleration_ms2") is not None:
            acc_g = [c / G_TO_MS2 for c in obj["user_acceleration_ms2"]]
        mag = obj.get("magnetic_field")
        att = obj.get("attitude")
        return cls(
            monotonic_time=float(obj["monotonic_time"]),
            user_acceleration_g=tuple(acc_g or (0.0, 0.0, 0.0)),  # type: ignore[arg-type]
            rotation_rate=tuple(obj.get("rotation_rate") or (0.0, 0.0, 0.0)),  # type: ignore[arg-type]
            gravity=tuple(obj.get("gravity") or (0.0, 0.0, -1.0)),  # type: ignore[arg-type]
            quaternion=tuple(obj.get("quaternion") or (1.0, 0.0, 0.0, 0.0)),  # type: ignore[arg-type]
            attitude=tuple(att) if att else None,  # type: ignore[arg-type]
            magnetic_field=tuple(mag) if mag else None,  # type: ignore[arg-type]
            magnetic_accuracy=obj.get("magnetic_accuracy"),
            wall_time=_parse_iso(obj.get("wall_time")),
        )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "type": "motion",
            "monotonic_time": round(_f(self.monotonic_time), 4),
            "user_acceleration_g": [round(_f(c), 6) for c in self.user_acceleration_g],
            "user_acceleration_ms2": [round(_f(c), 6) for c in self.user_acceleration_ms2],
            "rotation_rate": [round(_f(c), 6) for c in self.rotation_rate],
            "gravity": [round(_f(c), 6) for c in self.gravity],
            "quaternion": [round(_f(c), 6) for c in self.quaternion],
        }
        if self.attitude is not None:
            out["attitude"] = [round(_f(c), 6) for c in self.attitude]
        if self.magnetic_field is not None:
            out["magnetic_field"] = [round(_f(c), 4) for c in self.magnetic_field]
        if self.magnetic_accuracy is not None:
            out["magnetic_accuracy"] = self.magnetic_accuracy
        if self.wall_time is not None:
            out["wall_time"] = _iso(self.wall_time)
        return out


@dataclass
class MountCalibration:
    """How the phone sat in the holder.

    The phone is assumed rigidly mounted for the whole trip. The device->vehicle
    rotation is captured once, before the trip, and the vehicle heading is then
    driven by the gyro. The magnetometer is deliberately NOT trusted as the
    primary heading source: a car body distorts it badly.
    """

    reference_quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    gravity_device: tuple[float, float, float] = (0.0, 0.0, -1.0)
    forward_axis_device: Optional[tuple[float, float, float]] = None
    """Vehicle forward direction expressed in the device frame, if the
    straight-line calibration drive was performed."""

    initial_heading_deg: Optional[float] = None
    """Vehicle heading at t0, from a trusted GPS course if one was available."""

    heading_source: str = "unknown"
    """"gps_course" | "magnetometer" | "unknown"."""

    still_duration_s: float = 0.0
    captured_at: Optional[datetime] = None

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "MountCalibration":
        fwd = obj.get("forward_axis_device")
        return cls(
            reference_quaternion=tuple(obj.get("reference_quaternion") or (1.0, 0.0, 0.0, 0.0)),  # type: ignore[arg-type]
            gravity_device=tuple(obj.get("gravity_device") or (0.0, 0.0, -1.0)),  # type: ignore[arg-type]
            forward_axis_device=tuple(fwd) if fwd else None,  # type: ignore[arg-type]
            initial_heading_deg=obj.get("initial_heading_deg"),
            heading_source=obj.get("heading_source", "unknown"),
            still_duration_s=float(obj.get("still_duration_s", 0.0)),
            captured_at=_parse_iso(obj.get("captured_at")),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "reference_quaternion": [_f(c) for c in self.reference_quaternion],
            "gravity_device": [_f(c) for c in self.gravity_device],
            "forward_axis_device": (
                [_f(c) for c in self.forward_axis_device] if self.forward_axis_device else None
            ),
            "initial_heading_deg": (
                None if self.initial_heading_deg is None else _f(self.initial_heading_deg)
            ),
            "heading_source": self.heading_source,
            "still_duration_s": _f(self.still_duration_s),
            "captured_at": _iso(self.captured_at),
        }


@dataclass
class OCRResult:
    """One line recognised by Vision on the phone."""

    text: str
    confidence: float
    bounding_box: Optional[tuple[float, float, float, float]] = None

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "OCRResult":
        bbox = obj.get("bounding_box")
        return cls(
            text=str(obj.get("text", "")),
            confidence=float(obj.get("confidence", 0.0)),
            bounding_box=tuple(bbox) if bbox else None,  # type: ignore[arg-type]
        )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"text": self.text, "confidence": self.confidence}
        if self.bounding_box:
            out["bounding_box"] = list(self.bounding_box)
        return out


@dataclass
class Photo:
    """A stop photo plus everything the phone knew when it was taken."""

    image_path: str
    captured_at: Optional[datetime] = None
    monotonic_time: Optional[float] = None
    ocr: list[OCRResult] = field(default_factory=list)
    camera_orientation: Optional[str] = None
    device_heading_deg: Optional[float] = None
    assumed_latitude: Optional[float] = None
    assumed_longitude: Optional[float] = None
    assumed_accuracy: Optional[float] = None
    exif_latitude: Optional[float] = None
    exif_longitude: Optional[float] = None
    note: Optional[str] = None

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "Photo":
        return cls(
            image_path=str(obj.get("image_path") or obj.get("image") or ""),
            captured_at=_parse_iso(obj.get("captured_at")),
            monotonic_time=obj.get("monotonic_time"),
            ocr=[OCRResult.from_json(o) for o in obj.get("ocr", [])],
            camera_orientation=obj.get("camera_orientation"),
            device_heading_deg=obj.get("device_heading_deg"),
            assumed_latitude=obj.get("assumed_latitude"),
            assumed_longitude=obj.get("assumed_longitude"),
            assumed_accuracy=obj.get("assumed_accuracy"),
            exif_latitude=obj.get("exif_latitude"),
            exif_longitude=obj.get("exif_longitude"),
            note=obj.get("note"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "captured_at": _iso(self.captured_at),
            "monotonic_time": self.monotonic_time,
            "ocr": [o.to_json() for o in self.ocr],
            "camera_orientation": self.camera_orientation,
            "device_heading_deg": self.device_heading_deg,
            "assumed_latitude": self.assumed_latitude,
            "assumed_longitude": self.assumed_longitude,
            "assumed_accuracy": self.assumed_accuracy,
            "exif_latitude": self.exif_latitude,
            "exif_longitude": self.exif_longitude,
            "note": self.note,
        }


@dataclass
class TripMetadata:
    trip_id: str
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    device_model: Optional[str] = None
    os_version: Optional[str] = None
    app_version: Optional[str] = None
    calibration: Optional[MountCalibration] = None
    location_sample_count: int = 0
    motion_sample_count: int = 0
    notes: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "TripMetadata":
        known = {
            "trip_id",
            "started_at",
            "ended_at",
            "device_model",
            "os_version",
            "app_version",
            "calibration",
            "location_sample_count",
            "motion_sample_count",
            "notes",
        }
        cal = obj.get("calibration")
        return cls(
            trip_id=str(obj.get("trip_id") or obj.get("id") or "unknown"),
            started_at=_parse_iso(obj.get("started_at")),
            ended_at=_parse_iso(obj.get("ended_at")),
            device_model=obj.get("device_model"),
            os_version=obj.get("os_version"),
            app_version=obj.get("app_version"),
            calibration=MountCalibration.from_json(cal) if cal else None,
            location_sample_count=int(obj.get("location_sample_count", 0) or 0),
            motion_sample_count=int(obj.get("motion_sample_count", 0) or 0),
            notes=obj.get("notes"),
            extra={k: v for k, v in obj.items() if k not in known},
        )

    def to_json(self) -> dict[str, Any]:
        out = {
            "trip_id": self.trip_id,
            "started_at": _iso(self.started_at),
            "ended_at": _iso(self.ended_at),
            "device_model": self.device_model,
            "os_version": self.os_version,
            "app_version": self.app_version,
            "calibration": self.calibration.to_json() if self.calibration else None,
            "location_sample_count": self.location_sample_count,
            "motion_sample_count": self.motion_sample_count,
            "notes": self.notes,
        }
        out.update(self.extra)
        return out


@dataclass
class Trip:
    """A loaded trip: metadata + both sample streams + photos."""

    metadata: TripMetadata
    locations: list[LocationSample] = field(default_factory=list)
    motions: list[MotionSample] = field(default_factory=list)
    photos: list[Photo] = field(default_factory=list)
    root: Optional[str] = None
    faults: Optional[dict[str, Any]] = None
    reference_locations: list[LocationSample] = field(default_factory=list)
    """Only populated for a corrupted copy of a synthetic trip: the clean GPS
    track kept aside so that error metrics can be computed. On a real trip with
    a real outage there is no ground truth and this stays empty."""

    @property
    def usable_locations(self) -> list[LocationSample]:
        return [s for s in self.locations if s.is_usable]

    @property
    def duration_s(self) -> float:
        times = [s.monotonic_time for s in self.locations] + [
            s.monotonic_time for s in self.motions
        ]
        return (max(times) - min(times)) if times else 0.0

    @property
    def t0(self) -> float:
        times = [s.monotonic_time for s in self.locations] + [
            s.monotonic_time for s in self.motions
        ]
        return min(times) if times else 0.0

    def summary(self) -> dict[str, Any]:
        usable = self.usable_locations
        return {
            "trip_id": self.metadata.trip_id,
            "duration_s": round(self.duration_s, 2),
            "location_samples": len(self.locations),
            "usable_location_samples": len(usable),
            "motion_samples": len(self.motions),
            "photos": len(self.photos),
            "has_reference": bool(self.reference_locations),
            "faults": (self.faults or {}).get("faults") if self.faults else None,
        }


@dataclass
class WeightedRegion:
    """A prior region handed to a PhotoLocalizer: one branch of the belief."""

    component_id: str
    probability: float
    latitude: float
    longitude: float
    radius_m: float


@dataclass
class LocationCandidate:
    """A place proposed by a photo-based localiser."""

    latitude: float
    longitude: float
    score: float
    source: str
    label: Optional[str] = None
    radius_m: float = 50.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "score": self.score,
            "source": self.source,
            "label": self.label,
            "radius_m": self.radius_m,
            "diagnostics": self.diagnostics,
        }


def sort_samples(samples: Sequence[Any]) -> list[Any]:
    return sorted(samples, key=lambda s: s.monotonic_time)
