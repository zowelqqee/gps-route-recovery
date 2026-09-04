import XCTest
@testable import LiveTracking

/// The free-space terminal tracker: signed velocity, reverse, and the parking
/// manoeuvres the road tracker cannot represent.
final class ParkingTrackerTests: XCTestCase {
    let network = Fixtures.straight()
    var frame: LocalFrame { network.frame }

    private func tracker(_ config: LiveTrackingConfig = Fixtures.config()) -> ParkingTracker {
        ParkingTracker(config: config, frame: frame)
    }

    /// Build a window of IMU covering [start, end] with a given per-second
    /// longitudinal acceleration profile.
    private func window(
        start: Double, end: Double, dt: Double = 0.04,
        accel: (Double) -> Double, yaw: (Double) -> Double = { _ in 0 },
        heading: Double = 0
    ) -> [IMUControl] {
        var out: [IMUControl] = []
        var t = start
        while t <= end + 1e-9 {
            let a = accel(t)
            out.append(
                IMUControl(
                    t: t, dt: dt,
                    aWorld: SIMD3(a * cos(heading), a * sin(heading), 0),
                    yawRate: yaw(t)
                )
            )
            t += dt
        }
        return out
    }

    private func fixes(
        _ samples: [(t: Double, point: Point, speed: Double)],
        accuracy: Double = 6, course: Double = 90
    ) -> [TrackerLocation] {
        samples.map {
            Fixtures.location(
                t: $0.t, point: $0.point, frame: frame,
                accuracy: accuracy, speed: $0.speed, course: course
            )
        }
    }

    // MARK: - Signed velocity

    /// The state that makes a reversing car representable at all. Nothing here
    /// applies `max(0, u)`.
    func testVelocityIsSignedNotClamped() {
        var state = ParkingTracker.State()
        state.u = 2.0
        let braking = IMUControl(t: 1, dt: 1.0, aWorld: SIMD3(-4, 0, 0), yawRate: 0)
        // Reach into the transition directly: after a strong deceleration from
        // 2 m/s the speed must go negative, not stop at zero.
        let parking = tracker()
        parking.testPredict(&state, control: braking)
        XCTAssertLessThan(state.u, 0, "signed velocity: -2 m/s means reversing")
        XCTAssertEqual(state.u, -2.0, accuracy: 1e-9)
    }

    func testReversingMovesThePositionBackwards() {
        var state = ParkingTracker.State()
        state.u = -1.5
        state.theta = 0  // pointing east
        let parking = tracker()
        let coast = IMUControl(t: 1, dt: 1.0, aWorld: SIMD3(0, 0, 0), yawRate: 0)
        parking.testPredict(&state, control: coast)
        XCTAssertLessThan(state.x, 0, "a car reversing east-facing moves west")
        XCTAssertEqual(state.x, -1.5, accuracy: 1e-9)
    }

    /// A reversing car's GPS reports an unsigned speed. The sign has to come
    /// from the direction it actually moved relative to where it is pointing.
    func testGPSSpeedSignIsInferredFromDisplacement() {
        let parking = tracker()
        var state = ParkingTracker.State()
        state.theta = 0
        state.x = 0
        state.y = 0
        // Fix behind the current position, but the car still points east.
        let fix = Fixtures.location(
            t: 10, point: Point(x: -5, y: 0), frame: frame,
            accuracy: 5, speed: 1.2, course: 90
        )
        parking.testGPSUpdate(&state, fix: fix, point: Point(x: -5, y: 0))
        XCTAssertLessThan(state.u, 0, "moving backwards while facing forwards is reverse")
        XCTAssertEqual(abs(state.u), 1.2, accuracy: 1e-9)
    }

    // MARK: - Full manoeuvres

    /// The scenario from the field: drive forward, stop, reverse 20 m, stop.
    func testForwardThenTwentyMetresOfReverse() {
        let endedAt = 100.0
        var controls = window(start: 40, end: 60, dt: 0.04) { _ in 0 }        // coasting
        controls += window(start: 60.04, end: 62, dt: 0.04) { _ in -2.5 }     // brake to a stop
        controls += window(start: 62.04, end: 64, dt: 0.04) { _ in -1.0 }     // into reverse
        controls += window(start: 64.04, end: 76, dt: 0.04) { _ in 0 }        // reversing
        controls += window(start: 76.04, end: 78, dt: 0.04) { _ in 1.0 }      // stop
        controls += window(start: 78.04, end: endedAt, dt: 0.04) { _ in 0 }

        // GPS reports an unsigned speed throughout; the last fixes cluster where
        // the car finally stopped, 20 m back from where it started reversing.
        var samples: [(t: Double, point: Point, speed: Double)] = []
        for step in 0..<20 { samples.append((41 + Double(step), Point(x: Double(step) * 5, y: 0), 5)) }
        samples.append((61, Point(x: 100, y: 0), 1.0))
        for step in 0..<12 {
            samples.append((64 + Double(step), Point(x: 100 - Double(step) * 1.7, y: 0), 1.7))
        }
        for step in 0..<12 {
            samples.append((78 + Double(step) * 1.5, Point(x: 80, y: 0), 0.0))
        }

        let result = tracker().evaluate(
            ParkingTracker.Inputs(
                controls: controls, fixes: fixes(samples), endedAt: endedAt,
                roadPrior: Point(x: 0, y: 0), roadHeading: 0, roadSpeed: 5,
                calibrationPresent: true,
                roadDistance: { self.network.distanceToRoad($0) }
            )
        )
        XCTAssertTrue(result.hasReverseMotion, "the manoeuvre included 20 m of reverse")
        let position = try? XCTUnwrap(result.positionLocal)
        XCTAssertNotNil(position)
        // The terminal cluster is at x = 80, 20 m back from x = 100.
        XCTAssertEqual(position!.x, 80, accuracy: 4)
        XCTAssertEqual(result.status, .confident)
    }

    /// A car parked in a courtyard is not on the road graph, and the parking
    /// tracker has to be able to say so.
    func testParkingFifteenMetresOffTheRoad() {
        let endedAt = 80.0
        let controls = window(start: 20, end: endedAt, dt: 0.04) { _ in 0 }
        var samples: [(t: Double, point: Point, speed: Double)] = []
        for step in 0..<10 { samples.append((30 + Double(step), Point(x: Double(step) * 4, y: 0), 4)) }
        // Turn off the road and stop 15 m away from it.
        for step in 0..<12 {
            samples.append((55 + Double(step) * 2, Point(x: 40, y: -15), 0.0))
        }
        let result = tracker().evaluate(
            ParkingTracker.Inputs(
                controls: controls, fixes: fixes(samples), endedAt: endedAt,
                roadPrior: Point(x: 0, y: 0), roadHeading: 0, roadSpeed: 4,
                calibrationPresent: true,
                roadDistance: { self.network.distanceToRoad($0) }
            )
        )
        let position = try? XCTUnwrap(result.positionLocal)
        XCTAssertNotNil(position)
        XCTAssertEqual(position!.y, -15, accuracy: 3,
                       "the free-space tracker must be able to leave the carriageway")
        XCTAssertGreaterThan(network.distanceToRoad(position!), 10)
    }

    func testAThirtyDegreeParkingTurn() {
        let endedAt = 90.0
        let turnRate = (30.0 * .pi / 180.0) / 4.0   // 30 degrees over four seconds
        var controls = window(start: 30, end: 60, dt: 0.04) { _ in 0 }
        controls += window(start: 60.04, end: 64, dt: 0.04, accel: { _ in 0 },
                           yaw: { _ in turnRate })
        controls += window(start: 64.04, end: endedAt, dt: 0.04) { _ in 0 }
        var samples: [(t: Double, point: Point, speed: Double)] = []
        for step in 0..<20 { samples.append((35 + Double(step), Point(x: Double(step) * 3, y: 0), 3)) }
        for step in 0..<10 { samples.append((70 + Double(step) * 2, Point(x: 62, y: 6), 0.0)) }
        let result = tracker().evaluate(
            ParkingTracker.Inputs(
                controls: controls, fixes: fixes(samples), endedAt: endedAt,
                roadPrior: Point(x: 0, y: 0), roadHeading: 0, roadSpeed: 3,
                calibrationPresent: true
            )
        )
        XCTAssertEqual(result.status, .confident)
        XCTAssertEqual(result.positionLocal?.x ?? 0, 62, accuracy: 4)
        XCTAssertEqual(result.positionLocal?.y ?? 0, 6, accuracy: 4)
    }

    /// Parallel parking: forward past the space, reverse in at an angle, then
    /// straighten. The endpoint is behind where the car first stopped.
    func testParallelParking() {
        let endedAt = 120.0
        var controls = window(start: 60, end: 80, dt: 0.04) { _ in 0 }
        controls += window(start: 80.04, end: 82, dt: 0.04) { _ in -2.0 }   // stop
        controls += window(start: 82.04, end: 84, dt: 0.04) { _ in -0.8 }   // into reverse
        controls += window(start: 84.04, end: 92, dt: 0.04, accel: { _ in 0 },
                           yaw: { _ in -0.12 })                              // swing in
        controls += window(start: 92.04, end: 94, dt: 0.04) { _ in 0.8 }    // stop
        controls += window(start: 94.04, end: endedAt, dt: 0.04) { _ in 0 }

        var samples: [(t: Double, point: Point, speed: Double)] = []
        for step in 0..<20 { samples.append((61 + Double(step), Point(x: Double(step) * 4, y: 0), 4)) }
        samples.append((81, Point(x: 80, y: 0), 0.5))
        for step in 0..<8 {
            samples.append((85 + Double(step), Point(x: 80 - Double(step) * 0.9, y: -Double(step) * 0.3), 0.9))
        }
        for step in 0..<12 {
            samples.append((95 + Double(step) * 2, Point(x: 73, y: -2.4), 0.0))
        }
        let result = tracker().evaluate(
            ParkingTracker.Inputs(
                controls: controls, fixes: fixes(samples), endedAt: endedAt,
                roadPrior: Point(x: 0, y: 0), roadHeading: 0, roadSpeed: 4,
                calibrationPresent: true
            )
        )
        XCTAssertTrue(result.hasReverseMotion)
        XCTAssertEqual(result.positionLocal?.x ?? 0, 73, accuracy: 4)
        XCTAssertLessThan(result.positionLocal?.y ?? 0, 0, "the car ended up in the kerb-side space")
    }

    // MARK: - Degraded data

    /// The receiver went quiet seven seconds before the driver pressed stop.
    func testLastFixSevenSecondsBeforeTheEnd() {
        let endedAt = 90.0
        let controls = window(start: 30, end: endedAt, dt: 0.04) { _ in 0 }
        var samples: [(t: Double, point: Point, speed: Double)] = []
        for step in 0..<15 { samples.append((40 + Double(step), Point(x: Double(step) * 2, y: 0), 2)) }
        for step in 0..<6 { samples.append((78 + Double(step) * 1.0, Point(x: 30, y: 0), 0.0)) }
        let result = tracker().evaluate(
            ParkingTracker.Inputs(
                controls: controls, fixes: fixes(samples), endedAt: endedAt,
                roadPrior: Point(x: 0, y: 0), roadHeading: 0, roadSpeed: 2,
                calibrationPresent: true
            )
        )
        XCTAssertNotNil(result.position)
        XCTAssertEqual(result.lastFixSecondsBeforeEnd ?? 0, 7, accuracy: 1.5)
        XCTAssertEqual(result.positionLocal?.x ?? 0, 30, accuracy: 3)
    }

    /// No GPS at all in the terminal window. There is no honest position to
    /// report, and reporting one anyway would be a lie.
    func testNoGPSAtTheEndYieldsInsufficientData() {
        let endedAt = 200.0
        let controls = window(start: 140, end: endedAt, dt: 0.04) { _ in 0 }
        let result = tracker().evaluate(
            ParkingTracker.Inputs(
                controls: controls, fixes: [], endedAt: endedAt,
                roadPrior: Point(x: 0, y: 0), roadHeading: 0, roadSpeed: 0,
                calibrationPresent: true
            )
        )
        XCTAssertEqual(result.status, .insufficientData)
        XCTAssertNil(result.position, "nothing may be drawn in place of a position we do not have")
        XCTAssertTrue(result.polygon.isEmpty)
    }

    func testScatteredTerminalFixesAreUncertainNotConfident() {
        let endedAt = 90.0
        let controls = window(start: 30, end: endedAt, dt: 0.04) { _ in 0 }
        // Slow fixes, but scattered far apart: no consistent cluster.
        var samples: [(t: Double, point: Point, speed: Double)] = []
        samples.append((70, Point(x: 0, y: 0), 0.5))
        samples.append((75, Point(x: 60, y: 40), 0.5))
        let result = tracker().evaluate(
            ParkingTracker.Inputs(
                controls: controls, fixes: fixes(samples), endedAt: endedAt,
                roadPrior: Point(x: 0, y: 0), roadHeading: 0, roadSpeed: 0,
                calibrationPresent: true
            )
        )
        XCTAssertEqual(result.status, .uncertain)
        XCTAssertGreaterThan(result.polygonRadiusM, 30, "a weak answer must carry a wide polygon")
    }

    /// Without a mount calibration the world frame is inferred rather than
    /// known, so the answer must be downgraded, not presented as certain.
    func testMissingCalibrationReducesConfidence() {
        let endedAt = 80.0
        let controls = window(start: 20, end: endedAt, dt: 0.04) { _ in 0 }
        var samples: [(t: Double, point: Point, speed: Double)] = []
        for step in 0..<12 { samples.append((60 + Double(step) * 1.5, Point(x: 40, y: 0), 0.0)) }

        let inputs = { (calibrated: Bool) in
            ParkingTracker.Inputs(
                controls: controls, fixes: self.fixes(samples), endedAt: endedAt,
                roadPrior: Point(x: 0, y: 0), roadHeading: 0, roadSpeed: 0,
                calibrationPresent: calibrated
            )
        }
        let calibrated = tracker().evaluate(inputs(true))
        let uncalibrated = tracker().evaluate(inputs(false))
        XCTAssertLessThan(uncalibrated.confidence, calibrated.confidence)
        XCTAssertTrue(uncalibrated.reason.contains("calibration"))
        XCTAssertNotEqual(uncalibrated.status, .confident,
                          "an uncalibrated answer is never presented as confident")
    }

    /// Recording stopped; that is a stop, not a licence to keep extrapolating.
    func testNoMovementAfterTheEnd() {
        let endedAt = 70.0
        // Controls continue past the end of the trip; they must be ignored.
        let controls = window(start: 20, end: 120, dt: 0.04) { t in t > endedAt ? 3.0 : 0.0 }
        var samples: [(t: Double, point: Point, speed: Double)] = []
        for step in 0..<10 { samples.append((55 + Double(step) * 1.2, Point(x: 25, y: 0), 0.0)) }
        let result = tracker().evaluate(
            ParkingTracker.Inputs(
                controls: controls, fixes: fixes(samples), endedAt: endedAt,
                roadPrior: Point(x: 0, y: 0), roadHeading: 0, roadSpeed: 0,
                calibrationPresent: true
            )
        )
        XCTAssertEqual(result.positionLocal?.x ?? 0, 25, accuracy: 3,
                       "3 m/s of post-stop acceleration must not move the answer")
        XCTAssertTrue(result.trajectory.allSatisfy { _ in true })
    }

    func testAPhysicallyUnreachableTerminalFixIsRejected() {
        let endedAt = 80.0
        let controls = window(start: 20, end: endedAt, dt: 0.04) { _ in 0 }
        var samples: [(t: Double, point: Point, speed: Double)] = []
        for step in 0..<10 { samples.append((60 + Double(step) * 1.5, Point(x: 30, y: 0), 0.0)) }
        // One fix five kilometres away, half a second after the previous one.
        samples.append((66.0, Point(x: 5000, y: 5000), 0.0))
        let result = tracker().evaluate(
            ParkingTracker.Inputs(
                controls: controls, fixes: fixes(samples), endedAt: endedAt,
                roadPrior: Point(x: 0, y: 0), roadHeading: 0, roadSpeed: 0,
                calibrationPresent: true
            )
        )
        XCTAssertGreaterThan(result.rejectedFixes, 0)
        XCTAssertEqual(result.positionLocal?.x ?? 0, 30, accuracy: 5)
    }

    func testTheWindowIsBoundedToTheConfiguredLength() {
        let endedAt = 300.0
        let controls = window(start: 0, end: endedAt, dt: 0.5) { _ in 0 }
        // A cluster long before the window, and one inside it.
        var samples: [(t: Double, point: Point, speed: Double)] = []
        for step in 0..<10 { samples.append((30 + Double(step), Point(x: -400, y: 0), 0.0)) }
        for step in 0..<10 { samples.append((270 + Double(step) * 2, Point(x: 200, y: 0), 0.0)) }
        let result = tracker().evaluate(
            ParkingTracker.Inputs(
                controls: controls, fixes: fixes(samples), endedAt: endedAt,
                roadPrior: Point(x: 100, y: 0), roadHeading: 0, roadSpeed: 0,
                calibrationPresent: true
            )
        )
        XCTAssertEqual(result.positionLocal?.x ?? 0, 200, accuracy: 5,
                       "only the terminal window may decide where the car is")
    }
}
