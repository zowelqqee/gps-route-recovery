import XCTest
@testable import LiveTracking

/// Reading recorded trips, including ones written before the live runtime
/// existed. The on-disk format is a contract with the Python processor and with
/// every trip already sitting on a phone.
final class TripReplayTests: XCTestCase {
    static var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    /// The real recorded drive: written by an earlier build of the app, with no
    /// calibration block and no live-tracking files. It must still load.
    func testAnOlderTripWithoutCalibrationStillLoads() throws {
        let url = Self.repositoryRoot
            .appendingPathComponent("trip-8cd9b15c-bac6-4b59-8a16-f1e217b01e47")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: url.appendingPathComponent("samples.jsonl").path),
            "the recorded drive is not present in this checkout"
        )
        let trip = try TripReplay.load(directory: url)
        XCTAssertEqual(trip.locations.count, 456)
        XCTAssertEqual(trip.motions.count, 23_147)
        XCTAssertEqual(trip.malformedLines, 0)
        XCTAssertNil(trip.calibration, "this trip predates mount calibration")
        XCTAssertGreaterThan(trip.endedAt - trip.startedAt, 460)
        XCTAssertTrue(trip.locations.allSatisfy { $0.isUsable })
    }

    func testSamplesAreReturnedInTimestampOrder() throws {
        let url = Self.repositoryRoot
            .appendingPathComponent("trip-8cd9b15c-bac6-4b59-8a16-f1e217b01e47")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: url.appendingPathComponent("samples.jsonl").path)
        )
        let trip = try TripReplay.load(directory: url)
        XCTAssertEqual(trip.locations.map(\.monotonicTime),
                       trip.locations.map(\.monotonicTime).sorted())
        XCTAssertEqual(trip.motions.map(\.monotonicTime),
                       trip.motions.map(\.monotonicTime).sorted())
    }

    func testALocationLineDecodesEveryDocumentedField() throws {
        let json = """
        {"type":"location","wall_time":"2026-09-04T12:30:05.520Z","monotonic_time":13842.52,
         "latitude":59.93431,"longitude":30.32574,"altitude":6.0,"horizontal_accuracy":11.2,
         "vertical_accuracy":18.0,"speed":12.4,"speed_accuracy":1.0,"course":274.0,
         "course_accuracy":8.0,"source_information":{"is_simulated_by_software":false}}
        """
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any]
        )
        let sample = try XCTUnwrap(TripReplay.decodeLocation(object))
        XCTAssertEqual(sample.monotonicTime, 13842.52)
        XCTAssertEqual(sample.latitude, 59.93431)
        XCTAssertEqual(sample.horizontalAccuracy, 11.2)
        XCTAssertEqual(sample.course, 274.0)
        XCTAssertFalse(sample.isSimulatedBySoftware)
        XCTAssertTrue(sample.isUsable)
    }

    func testAMotionLineDecodesAndConvertsUnits() throws {
        let json = """
        {"type":"motion","monotonic_time":13842.54,"user_acceleration_g":[0.12,-0.03,0.01],
         "user_acceleration_ms2":[1.1768,-0.2942,0.0981],"rotation_rate":[0.01,-0.02,0.18],
         "gravity":[0.02,-0.12,-0.99],"quaternion":[0.91,0.02,0.03,0.41]}
        """
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any]
        )
        let sample = try XCTUnwrap(TripReplay.decodeMotion(object))
        XCTAssertEqual(sample.userAccelerationMS2.x, 1.1768, accuracy: 1e-4)
        XCTAssertEqual(sample.userAccelerationMS2.y, -0.2942, accuracy: 1e-4)
        XCTAssertEqual(sample.rotationRate.z, 0.18, accuracy: 1e-9)
    }

    func testAFixWithNegativeAccuracyIsUnusable() throws {
        let json = """
        {"type":"location","monotonic_time":10,"latitude":59.9,"longitude":30.3,
         "horizontal_accuracy":-1}
        """
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any]
        )
        let sample = try XCTUnwrap(TripReplay.decodeLocation(object))
        XCTAssertFalse(sample.isUsable, "CoreLocation marks an invalid fix this way")
    }

    func testACalibrationBlockDecodes() throws {
        let json = """
        {"reference_quaternion":[0.9,0.1,0.2,0.3],"gravity_device":[0,-0.2,-0.97],
         "forward_axis_device":[1,0,0],"initial_heading_deg":274,"heading_source":"gps_course"}
        """
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any]
        )
        let calibration = try XCTUnwrap(TripReplay.decodeCalibration(object))
        XCTAssertEqual(calibration.headingSource, "gps_course")
        XCTAssertTrue(calibration.definesVehicleFrame)
    }

    func testACalibrationWithoutAForwardAxisDoesNotDefineTheVehicleFrame() throws {
        let json = """
        {"reference_quaternion":[1,0,0,0],"gravity_device":[0,0,-1],"heading_source":"unknown"}
        """
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any]
        )
        let calibration = try XCTUnwrap(TripReplay.decodeCalibration(object))
        XCTAssertFalse(calibration.definesVehicleFrame,
                       "without the straight-line step there is no exact alignment")
    }

    func testUnknownFieldsDoNotBreakParsing() throws {
        let json = """
        {"type":"location","monotonic_time":10,"latitude":59.9,"longitude":30.3,
         "horizontal_accuracy":5,"future_field":{"nested":true}}
        """
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any]
        )
        XCTAssertNotNil(TripReplay.decodeLocation(object))
    }
}
