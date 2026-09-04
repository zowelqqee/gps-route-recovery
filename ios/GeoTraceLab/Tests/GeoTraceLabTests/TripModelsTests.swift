import CoreLocation
import XCTest

@testable import GeoTraceLab

/// The wire format is a contract with the Python processor: if these change,
/// `schemas/trip.schema.json` and `geotrace.loader` must change with them.
final class TripModelsTests: XCTestCase {

    func testAccelerationIsConvertedFromGToMetresPerSecondSquared() {
        let sample = MotionSample(
            monotonicTime: 13842.54,
            userAccelerationG: [0.12, -0.03, 0.01],
            rotationRate: [0.01, -0.02, 0.18],
            gravity: [0.02, -0.12, -0.99],
            quaternion: [0.91, 0.02, 0.03, 0.41]
        )
        // The exact values from the specification's worked example.
        XCTAssertEqual(sample.userAccelerationMS2[0], 1.1768, accuracy: 1e-4)
        XCTAssertEqual(sample.userAccelerationMS2[1], -0.2942, accuracy: 1e-4)
        XCTAssertEqual(sample.userAccelerationMS2[2], 0.0981, accuracy: 1e-4)
    }

    func testMotionSampleKeepsBothUnits() throws {
        let sample = MotionSample(
            monotonicTime: 1.0,
            userAccelerationG: [0.5, 0, 0],
            rotationRate: [0, 0, 0],
            gravity: [0, 0, -1],
            quaternion: [1, 0, 0, 0]
        )
        let json = try encodeToObject(sample)
        XCTAssertEqual(json["type"] as? String, "motion")
        XCTAssertNotNil(json["user_acceleration_g"])
        XCTAssertNotNil(json["user_acceleration_ms2"])
    }

    func testMotionSampleUsesSnakeCaseKeys() throws {
        let sample = MotionSample(
            monotonicTime: 12.5,
            userAccelerationG: [0, 0, 0],
            rotationRate: [0, 0, 0],
            gravity: [0, 0, -1],
            quaternion: [1, 0, 0, 0],
            attitude: [0.1, 0.2, 0.3],
            magneticField: [22, -8, 41],
            magneticAccuracy: 1
        )
        let json = try encodeToObject(sample)
        for key in [
            "type", "monotonic_time", "user_acceleration_g", "user_acceleration_ms2",
            "rotation_rate", "gravity", "quaternion", "attitude", "magnetic_field",
            "magnetic_accuracy",
        ] {
            XCTAssertNotNil(json[key], "missing key \(key)")
        }
    }

    func testLocationSampleUsesSnakeCaseKeys() throws {
        let sample = LocationSample(
            wallTime: Date(timeIntervalSince1970: 1_788_000_000),
            monotonicTime: 13842.52,
            latitude: 59.93431,
            longitude: 30.32574,
            altitude: 6.0,
            horizontalAccuracy: 11.2,
            verticalAccuracy: 18.0,
            speed: 12.4,
            speedAccuracy: 1.0,
            course: 274.0,
            courseAccuracy: 8.0
        )
        let json = try encodeToObject(sample)
        XCTAssertEqual(json["type"] as? String, "location")
        XCTAssertEqual(json["latitude"] as? Double, 59.93431)
        XCTAssertEqual(json["horizontal_accuracy"] as? Double, 11.2)
        XCTAssertNotNil(json["monotonic_time"])
        XCTAssertNotNil(json["wall_time"])
        XCTAssertNotNil(json["course_accuracy"])
    }

    func testTimestampsAreISO8601WithMillisecondsAndZ() throws {
        let sample = LocationSample(
            wallTime: Date(timeIntervalSince1970: 1_788_000_005.52),
            monotonicTime: 1.0, latitude: 59.9, longitude: 30.3, horizontalAccuracy: 5
        )
        let json = try encodeToObject(sample)
        let text = try XCTUnwrap(json["wall_time"] as? String)
        XCTAssertTrue(text.hasSuffix("Z"), "expected a Z-suffixed UTC timestamp, got \(text)")
        XCTAssertTrue(text.contains("."), "expected fractional seconds, got \(text)")
    }

    func testSamplesRoundTripThroughJSON() throws {
        let original = LocationSample(
            wallTime: Date(timeIntervalSince1970: 1_788_000_000),
            monotonicTime: 100.5, latitude: 59.93, longitude: 30.33,
            horizontalAccuracy: 9.0, speed: 11.0, course: 90.0
        )
        let data = try TripJSON.encoder().encode(original)
        let decoded = try TripJSON.decoder().decode(LocationSample.self, from: data)
        XCTAssertEqual(decoded.latitude, original.latitude)
        XCTAssertEqual(decoded.monotonicTime, original.monotonicTime)
        XCTAssertEqual(decoded.course, original.course)
    }

    // MARK: - Fix validity

    func testAFixWithNegativeAccuracyIsUnusable() {
        // CoreLocation marks an invalid fix this way; it must never be recorded.
        let sample = LocationSample(
            wallTime: Date(), monotonicTime: 10, latitude: 59.9, longitude: 30.3,
            horizontalAccuracy: -1
        )
        XCTAssertFalse(sample.isUsable)
    }

    func testAFixWithNoAccuracyIsUnusable() {
        let sample = LocationSample(
            wallTime: Date(), monotonicTime: 10, latitude: 59.9, longitude: 30.3)
        XCTAssertFalse(sample.isUsable)
    }

    func testAFixWithAnImpossibleCoordinateIsUnusable() {
        let sample = LocationSample(
            wallTime: Date(), monotonicTime: 10, latitude: 120, longitude: 30.3,
            horizontalAccuracy: 5)
        XCTAssertFalse(sample.isUsable)
    }

    func testAFixWithANonPositiveTimestampIsUnusable() {
        let sample = LocationSample(
            wallTime: Date(), monotonicTime: 0, latitude: 59.9, longitude: 30.3,
            horizontalAccuracy: 5)
        XCTAssertFalse(sample.isUsable)
    }

    func testAGoodFixIsUsable() {
        let sample = LocationSample(
            wallTime: Date(), monotonicTime: 13842.5, latitude: 59.93431,
            longitude: 30.32574, horizontalAccuracy: 11.2)
        XCTAssertTrue(sample.isUsable)
    }

    // MARK: - Timebase

    func testMonotonicAndWallClockConversionsAreInverses() {
        let timebase = Timebase(bootDate: Date(timeIntervalSince1970: 1_788_000_000))
        let date = Date(timeIntervalSince1970: 1_788_013_842.52)
        let monotonic = timebase.monotonic(from: date)
        XCTAssertEqual(monotonic, 13842.52, accuracy: 1e-6)
        XCTAssertEqual(
            timebase.wallClock(from: monotonic).timeIntervalSince1970,
            date.timeIntervalSince1970, accuracy: 1e-6
        )
    }

    func testTheTwoSensorStreamsShareOneTimeline() {
        // CMDeviceMotion.timestamp is already seconds since boot; a CLLocation
        // date converted through the same Timebase must land on the same scale.
        let timebase = Timebase(bootDate: Date(timeIntervalSince1970: 1_000_000))
        let locationDate = Date(timeIntervalSince1970: 1_000_500)
        XCTAssertEqual(timebase.monotonic(from: locationDate), 500, accuracy: 1e-9)
    }

    // MARK: - Metadata

    func testMetadataEncodesTheCalibration() throws {
        let metadata = TripMetadata(
            tripId: "trip-abc",
            startedAt: Date(timeIntervalSince1970: 1_788_000_000),
            calibration: MountCalibration(
                referenceQuaternion: [0.9, 0.1, 0.2, 0.3],
                gravityDevice: [0, -0.2, -0.97],
                forwardAxisDevice: [1, 0, 0],
                initialHeadingDeg: 274,
                headingSource: "gps_course",
                stillDurationS: 6
            ),
            locationSampleCount: 300,
            motionSampleCount: 15000
        )
        let json = try encodeToObject(metadata)
        XCTAssertEqual(json["trip_id"] as? String, "trip-abc")
        XCTAssertEqual(json["location_sample_count"] as? Int, 300)
        let calibration = try XCTUnwrap(json["calibration"] as? [String: Any])
        XCTAssertEqual(calibration["heading_source"] as? String, "gps_course")
        XCTAssertNotNil(calibration["forward_axis_device"])
        XCTAssertNotNil(calibration["reference_quaternion"])
    }

    func testPhotoRecordEncodesOCRAndTheAssumedPosition() throws {
        let record = PhotoRecord(
            imagePath: "photo-001.jpg",
            capturedAt: Date(timeIntervalSince1970: 1_788_000_000),
            ocr: [OCRLine(text: "Лиговский проспект", confidence: 0.94)],
            assumedLatitude: 59.93,
            assumedLongitude: 30.36,
            note: "Assumed position from the last GPS fix. Not a verified address."
        )
        let json = try encodeToObject(record)
        XCTAssertEqual(json["image_path"] as? String, "photo-001.jpg")
        XCTAssertNotNil(json["assumed_latitude"])
        let ocr = try XCTUnwrap(json["ocr"] as? [[String: Any]])
        XCTAssertEqual(ocr.first?["text"] as? String, "Лиговский проспект")
        XCTAssertEqual(ocr.first?["confidence"] as? Double, 0.94)
    }

    // MARK: - Helpers

    private func encodeToObject<T: Encodable>(_ value: T) throws -> [String: Any] {
        let data = try TripJSON.encoder().encode(value)
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}
