import CoreLocation
import XCTest

@testable import GeoTraceLab

/// Heading conventions and the mount-calibration state machine.
final class CalibrationTests: XCTestCase {

    // MARK: - Heading conventions

    func testCourseNorthIsNinetyDegreesOfHeading() {
        // CoreLocation course is clockwise from north; the processor's psi is
        // counter-clockwise from east. Getting this wrong rotates the whole
        // reconstruction by 90 degrees.
        XCTAssertEqual(CalibrationManager.headingRadians(fromCourse: 0), .pi / 2, accuracy: 1e-9)
        XCTAssertEqual(CalibrationManager.headingRadians(fromCourse: 90), 0, accuracy: 1e-9)
        XCTAssertEqual(CalibrationManager.headingRadians(fromCourse: 180), -.pi / 2, accuracy: 1e-9)
    }

    func testHeadingIsWrappedIntoRange() {
        for course in stride(from: 0.0, to: 360.0, by: 7.0) {
            let heading = CalibrationManager.headingRadians(fromCourse: course)
            XCTAssertGreaterThan(heading, -Double.pi - 1e-9)
            XCTAssertLessThanOrEqual(heading, Double.pi + 1e-9)
        }
    }

    func testCourseAndHeadingConversionsAreInverses() {
        for course in stride(from: 0.0, to: 360.0, by: 11.0) {
            let heading = CalibrationManager.headingRadians(fromCourse: course)
            let roundTripped = CalibrationManager.courseDegrees(fromHeading: heading)
            XCTAssertEqual(roundTripped, course, accuracy: 1e-6)
        }
    }

    func testCourseDegreesAreAlwaysInZeroToThreeSixty() {
        for heading in stride(from: -4.0, to: 4.0, by: 0.13) {
            let course = CalibrationManager.courseDegrees(fromHeading: heading)
            XCTAssertGreaterThanOrEqual(course, 0)
            XCTAssertLessThan(course, 360)
        }
    }

    // MARK: - State machine

    func testCalibrationStartsIdle() {
        XCTAssertEqual(CalibrationManager().phase, .idle)
    }

    func testBeginMovesToHoldStill() {
        let manager = CalibrationManager()
        manager.begin()
        if case .holdStill(let remaining) = manager.phase {
            XCTAssertEqual(remaining, CalibrationManager.stillDuration, accuracy: 1e-9)
        } else {
            XCTFail("expected holdStill, got \(manager.phase)")
        }
    }

    func testFinishingWithoutEverHoldingStillFails() {
        let manager = CalibrationManager()
        manager.begin()
        manager.finish()
        guard case .failed(let message) = manager.phase else {
            return XCTFail("expected a failure, got \(manager.phase)")
        }
        XCTAssertTrue(message.lowercased().contains("still"))
        XCTAssertNil(manager.calibration)
    }

    func testResetReturnsToIdle() {
        let manager = CalibrationManager()
        manager.begin()
        manager.reset()
        XCTAssertEqual(manager.phase, .idle)
        XCTAssertNil(manager.calibration)
    }

    func testAFixDuringHoldStillDoesNotEndCalibration() {
        // The regression this guards: `ingest` returning false during holdStill
        // was read by the recorder as "calibration finished", so the very first
        // GPS fix skipped calibration entirely.
        let manager = CalibrationManager()
        manager.begin()
        let sample = LocationSample(
            wallTime: Date(), monotonicTime: 10, latitude: 59.93, longitude: 30.33,
            horizontalAccuracy: 8, speed: 12, course: 90
        )
        XCTAssertTrue(
            manager.ingest(location: sample, motion: nil),
            "calibration must still be running while the driver holds still"
        )
        if case .holdStill = manager.phase {} else {
            XCTFail("expected to still be in holdStill, got \(manager.phase)")
        }
    }

    func testFixesAreIgnoredOnceCalibrationIsOver() {
        let manager = CalibrationManager()
        let sample = LocationSample(
            wallTime: Date(), monotonicTime: 10, latitude: 59.93, longitude: 30.33,
            horizontalAccuracy: 8, speed: 12, course: 90
        )
        // Never started: nothing to feed.
        XCTAssertFalse(manager.ingest(location: sample, motion: nil))
    }
}
