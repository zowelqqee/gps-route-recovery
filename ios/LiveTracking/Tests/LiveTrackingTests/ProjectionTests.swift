import XCTest
@testable import LiveTracking

/// Cross-language fixtures. The expected values come from the Python baseline's
/// `geotrace.coordinates.LocalFrame` (pyproj `+proj=aeqd +datum=WGS84`), which
/// is the reference implementation. Regenerate with
/// `processor/.venv/bin/python tools/dump_projection_fixtures.py`.
final class ProjectionTests: XCTestCase {
    /// Trip origin used by every fixture: Ligovsky prospekt, Saint Petersburg.
    static let origin = (lat: 59.9311, lon: 30.3609)

    func testAngleWrapping() {
        XCTAssertEqual(Angles.wrap(0), 0, accuracy: 1e-12)
        XCTAssertEqual(Angles.wrap(.pi), .pi, accuracy: 1e-12)
        XCTAssertEqual(Angles.wrap(-.pi), .pi, accuracy: 1e-12, "the half-open convention")
        XCTAssertEqual(Angles.wrap(3 * .pi / 2), -.pi / 2, accuracy: 1e-12)
        XCTAssertEqual(Angles.wrap(10 * .pi + 0.3), 0.3, accuracy: 1e-9)
    }

    func testWrapAlwaysLandsInRange() {
        for i in stride(from: -40.0, through: 40.0, by: 0.017) {
            let w = Angles.wrap(i)
            XCTAssertGreaterThan(w, -Double.pi - 1e-12)
            XCTAssertLessThanOrEqual(w, Double.pi + 1e-12)
        }
    }

    func testCourseNorthIsNinetyDegreesOfHeading() {
        XCTAssertEqual(Angles.headingFromCourse(0), .pi / 2, accuracy: 1e-12)
        XCTAssertEqual(Angles.headingFromCourse(90), 0, accuracy: 1e-12)
        XCTAssertEqual(Angles.headingFromCourse(180), -.pi / 2, accuracy: 1e-12)
        // West: wrap(90 - 270) degrees = wrap(-pi) = +pi under the half-open rule.
        XCTAssertEqual(Angles.headingFromCourse(270), .pi, accuracy: 1e-12)
    }

    func testCourseHeadingRoundTrip() {
        for course in stride(from: 0.0, to: 360.0, by: 7.0) {
            let back = Angles.courseFromHeading(Angles.headingFromCourse(course))
            XCTAssertEqual(back, course, accuracy: 1e-9)
        }
    }

    func testFrameOriginProjectsToZero() {
        let frame = LocalFrame(latitude: Self.origin.lat, longitude: Self.origin.lon)
        let point = frame.toLocal(latitude: Self.origin.lat, longitude: Self.origin.lon)
        XCTAssertEqual(point.x, 0, accuracy: 1e-6)
        XCTAssertEqual(point.y, 0, accuracy: 1e-6)
    }

    func testAxesPointEastAndNorth() {
        let frame = LocalFrame(latitude: Self.origin.lat, longitude: Self.origin.lon)
        let east = frame.toLocal(latitude: Self.origin.lat, longitude: Self.origin.lon + 0.01)
        XCTAssertGreaterThan(east.x, 0)
        XCTAssertLessThan(abs(east.y), 1.0)
        let north = frame.toLocal(latitude: Self.origin.lat + 0.01, longitude: Self.origin.lon)
        XCTAssertGreaterThan(north.y, 0)
        XCTAssertLessThan(abs(north.x), 1.0)
    }

    func testRoundTripThroughWGS84() {
        let frame = LocalFrame(latitude: Self.origin.lat, longitude: Self.origin.lon)
        for (lat, lon) in [(59.94, 30.37), (59.85, 30.20), (60.05, 30.55), (59.70, 29.90)] {
            let point = frame.toLocal(latitude: lat, longitude: lon)
            let back = frame.toGeo(point)
            let error = frame.toLocal(back).distance(to: point)
            XCTAssertLessThan(error, 1e-6, "round trip at \(lat),\(lon)")
        }
    }
}

extension ProjectionTests {
    /// The parity test that matters: Swift's Vincenty-based AEQD must reproduce
    /// pyproj's ellipsoidal AEQD, because every downstream comparison between
    /// the two implementations is expressed in this frame.
    func testENUMatchesThePythonBaseline() {
        let frame = LocalFrame(
            latitude: ProjectionFixtures.originLatitude,
            longitude: ProjectionFixtures.originLongitude
        )
        var worst = 0.0
        for fixture in ProjectionFixtures.points {
            let point = frame.toLocal(latitude: fixture.lat, longitude: fixture.lon)
            let error = point.distance(to: Point(x: fixture.east, y: fixture.north))
            worst = max(worst, error)
            XCTAssertLessThan(
                error, 1e-3,
                "ENU mismatch at \(fixture.lat),\(fixture.lon): "
                + "swift=(\(point.x),\(point.y)) python=(\(fixture.east),\(fixture.north))"
            )
        }
        XCTAssertLessThan(worst, 1e-3, "worst ENU disagreement was \(worst) m")
    }

    func testInverseProjectionMatchesThePythonBaseline() {
        let frame = LocalFrame(
            latitude: ProjectionFixtures.originLatitude,
            longitude: ProjectionFixtures.originLongitude
        )
        for fixture in ProjectionFixtures.points {
            let geo = frame.toGeo(Point(x: fixture.east, y: fixture.north))
            // Compare in metres rather than degrees so the tolerance means something.
            let reprojected = frame.toLocal(geo)
            XCTAssertLessThan(
                reprojected.distance(to: Point(x: fixture.east, y: fixture.north)), 1e-3
            )
            XCTAssertEqual(geo.latitude, fixture.lat, accuracy: 1e-8)
            XCTAssertEqual(geo.longitude, fixture.lon, accuracy: 1e-8)
        }
    }
}
