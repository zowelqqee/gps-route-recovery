import XCTest
@testable import LiveTracking

/// Gates and the TRUSTED / SUSPECT / LOST / RECOVERING machine.
final class GPSGateTests: XCTestCase {
    let config = GPSQualityConfig()
    let maxAccel = 6.0

    private func fix(
        t: Double, accuracy: Double = 10, speed: Double = 10, course: Double = 90
    ) -> TrackerLocation {
        TrackerLocation(
            monotonicTime: t, latitude: 59.9311, longitude: 30.3609,
            horizontalAccuracy: accuracy, speed: speed, speedAccuracy: 1,
            course: course, courseAccuracy: 5
        )
    }

    // MARK: - Physical gate

    func testPhysicalGateAcceptsAReachablePoint() {
        let gate = GPSGate.physical(
            distanceM: 12, dt: 1, previousSpeed: 10, config: config, maxAccel: maxAccel
        )
        XCTAssertTrue(gate.passed)
        XCTAssertEqual(gate.maxDistance, 10 + 0.5 * maxAccel + config.physicalMarginM, accuracy: 1e-9)
    }

    /// The classic Saint Petersburg failure: a four-kilometre jump in one
    /// second, reported with a confident accuracy.
    func testPhysicalGateRejectsATeleport() {
        XCTAssertFalse(
            GPSGate.physical(
                distanceM: 4000, dt: 1, previousSpeed: 10, config: config, maxAccel: maxAccel
            ).passed
        )
    }

    func testPhysicalGateIsExactAtTheBoundary() {
        let limit = GPSGate.physical(
            distanceM: 0, dt: 2, previousSpeed: 8, config: config, maxAccel: maxAccel
        ).maxDistance
        XCTAssertTrue(GPSGate.physical(distanceM: limit, dt: 2, previousSpeed: 8,
                                       config: config, maxAccel: maxAccel).passed)
        XCTAssertFalse(GPSGate.physical(distanceM: limit + 0.01, dt: 2, previousSpeed: 8,
                                        config: config, maxAccel: maxAccel).passed)
    }

    func testPhysicalGateAllowsMoreAfterALongerGap() {
        XCTAssertTrue(
            GPSGate.physical(
                distanceM: 900, dt: 60, previousSpeed: 14, config: config, maxAccel: maxAccel
            ).passed,
            "after a minute the car really could be a kilometre away"
        )
    }

    // MARK: - Mahalanobis gate

    func testMahalanobisOfAZeroResidualIsZero() {
        let gate = GPSGate.mahalanobis(
            residual: Point(x: 0, y: 0), covariance: .diagonal(100), threshold: 9.21
        )
        XCTAssertTrue(gate.passed)
        XCTAssertEqual(gate.distanceSquared, 0, accuracy: 1e-12)
    }

    func testMahalanobisMatchesTheDefinition() {
        // D^2 = r^T S^-1 r
        let gate = GPSGate.mahalanobis(
            residual: Point(x: 2, y: 5),
            covariance: Matrix2(a: 4, b: 0, c: 0, d: 25),
            threshold: 100
        )
        XCTAssertEqual(gate.distanceSquared, 2, accuracy: 1e-12)
    }

    func testMahalanobisThresholdIsTheTwoDOFChiSquare() {
        // chi^2(2) at p = 0.99 is 9.21.
        XCTAssertEqual(config.mahalanobisThreshold, 9.21, accuracy: 0.01)
        XCTAssertTrue(GPSGate.mahalanobis(residual: Point(x: 3, y: 0),
                                          covariance: .diagonal(1), threshold: 9.21).passed)
        XCTAssertFalse(GPSGate.mahalanobis(residual: Point(x: 3.2, y: 0),
                                           covariance: .diagonal(1), threshold: 9.21).passed)
    }

    /// The whole point of using it rather than a fixed radius: the same residual
    /// is fine when the filter is uncertain and not when it is confident.
    func testMahalanobisScalesWithTheCovariance() {
        let residual = Point(x: 50, y: 0)
        XCTAssertFalse(GPSGate.mahalanobis(residual: residual,
                                           covariance: .diagonal(100), threshold: 9.21).passed)
        XCTAssertTrue(GPSGate.mahalanobis(residual: residual,
                                          covariance: .diagonal(10000), threshold: 9.21).passed)
    }

    func testMahalanobisOnASingularCovarianceFailsClosed() {
        let gate = GPSGate.mahalanobis(
            residual: Point(x: 1, y: 1),
            covariance: Matrix2(a: 0, b: 0, c: 0, d: 0), threshold: 9.21
        )
        XCTAssertFalse(gate.passed)
        XCTAssertEqual(gate.distanceSquared, .infinity)
    }

    func testMeasurementSigmaHasAFloor() {
        XCTAssertEqual(GPSGate.measurementSigma(fix(t: 0, accuracy: 0.5), config: config),
                       config.minAccuracySigmaM)
        XCTAssertEqual(GPSGate.measurementSigma(fix(t: 0, accuracy: 30), config: config), 30)
        XCTAssertEqual(GPSGate.measurementSigma(fix(t: 0, accuracy: -1), config: config),
                       config.maxHorizontalAccuracyM)
    }

    // MARK: - State machine

    /// Feed enough consistent fixes to reach TRUSTED, and return the time of
    /// the last one. Times start at 1, because a monotonic timestamp of zero is
    /// not a valid fix and is rejected on purpose.
    @discardableResult
    private func promote(_ machine: GPSStateMachine) -> Double {
        var t = 0.0
        for step in 0..<config.recoverCount {
            t = 1.0 + Double(step)
            _ = machine.update(fix(t: t), measured: Point(x: 10 * Double(step), y: 0),
                               predictedSpeed: 10)
        }
        XCTAssertEqual(machine.state, .trusted)
        return t
    }

    func testATripStartsUntrusted() {
        XCTAssertEqual(GPSStateMachine(config: config, maxAccelMS2: maxAccel).state, .lost)
    }

    func testPromotionNeedsSeveralConsecutiveGoodFixes() {
        let machine = GPSStateMachine(config: config, maxAccelMS2: maxAccel)
        for step in 0..<(config.recoverCount - 1) {
            _ = machine.update(fix(t: 1 + Double(step)),
                               measured: Point(x: 10 * Double(step), y: 0), predictedSpeed: 10)
            XCTAssertNotEqual(machine.state, .trusted)
        }
        let last = config.recoverCount - 1
        _ = machine.update(fix(t: 1 + Double(last)), measured: Point(x: 10 * Double(last), y: 0),
                           predictedSpeed: 10)
        XCTAssertEqual(machine.state, .trusted)
    }

    func testTrustedToSuspectOnOneBadFix() {
        let machine = GPSStateMachine(config: config, maxAccelMS2: maxAccel)
        let t = promote(machine)
        _ = machine.update(fix(t: t + 1), measured: Point(x: 4000, y: 0), predictedSpeed: 10)
        XCTAssertEqual(machine.state, .suspect)
    }

    func testSuspectToLostAfterRepeatedBadFixes() {
        let machine = GPSStateMachine(config: config, maxAccelMS2: maxAccel)
        let t = promote(machine)
        for step in 0..<config.suspectToLostCount {
            _ = machine.update(fix(t: t + 1 + Double(step)),
                               measured: Point(x: 4000 + Double(step), y: 0), predictedSpeed: 10)
        }
        XCTAssertEqual(machine.state, .lost)
    }

    func testDropoutMovesToLost() {
        let machine = GPSStateMachine(config: config, maxAccelMS2: maxAccel)
        let t = promote(machine)
        XCTAssertTrue(machine.noteGap(now: t + config.lostGapSeconds + 1))
        XCTAssertEqual(machine.state, .lost)
    }

    func testAShortGapDoesNotTriggerLost() {
        let machine = GPSStateMachine(config: config, maxAccelMS2: maxAccel)
        let t = promote(machine)
        XCTAssertFalse(machine.noteGap(now: t + config.lostGapSeconds - 0.5))
        XCTAssertEqual(machine.state, .trusted)
    }

    /// Recovery must not jump straight to the first fix that comes back.
    func testRecoveryNeedsAConsistentChain() {
        let machine = GPSStateMachine(config: config, maxAccelMS2: maxAccel)
        promote(machine)
        _ = machine.noteGap(now: 200)
        XCTAssertEqual(machine.state, .lost)

        _ = machine.update(fix(t: 201), measured: Point(x: 0, y: 0), predictedSpeed: 10)
        XCTAssertEqual(machine.state, .recovering)
        for step in 1..<(config.recoverCount - 1) {
            _ = machine.update(fix(t: 201 + Double(step)),
                               measured: Point(x: 10 * Double(step), y: 0), predictedSpeed: 10)
            XCTAssertEqual(machine.state, .recovering, "trust must not return early")
        }
        _ = machine.update(fix(t: 201 + Double(config.recoverCount)),
                           measured: Point(x: 10 * Double(config.recoverCount), y: 0),
                           predictedSpeed: 10)
        XCTAssertEqual(machine.state, .trusted)
    }

    /// The specification's exact failure mode: several wrong points right after
    /// the signal returns. Individually plausible, mutually inconsistent.
    func testScatteredFalseRecoveryFixesNeverRestoreTrust() {
        let machine = GPSStateMachine(config: config, maxAccelMS2: maxAccel)
        promote(machine)
        _ = machine.noteGap(now: 200)
        let scatter = [Point(x: 0, y: 0), Point(x: 0, y: 40), Point(x: 30, y: -35),
                       Point(x: -20, y: 25), Point(x: 45, y: 30)]
        for (step, point) in scatter.enumerated() {
            _ = machine.update(fix(t: 201 + Double(step)), measured: point, predictedSpeed: 10)
        }
        XCTAssertNotEqual(machine.state, .trusted)
    }

    /// Recovery keeps the statistical gate active, but expands its covariance
    /// so a plausible return does not have to agree with stale inertial state.
    func testRecoveryMahalanobisGateExpandsWithDeadReckoningAge() {
        let machine = GPSStateMachine(config: config, maxAccelMS2: maxAccel)
        promote(machine)
        _ = machine.noteGap(now: 200)
        let decision = machine.update(
            fix(t: 201), measured: Point(x: 80, y: 0),
            predicted: Point(x: 0, y: 0), predictedSpeed: 10,
            covariance: .diagonal(100)
        )
        XCTAssertNotNil(decision.mahalanobis)
        XCTAssertTrue(decision.accepted)
    }

    func testWhileLostAFixIsCheckedAgainstThePreviousFix() {
        let machine = GPSStateMachine(config: config, maxAccelMS2: maxAccel)
        promote(machine)
        _ = machine.noteGap(now: 200)
        _ = machine.update(fix(t: 201, course: 90), measured: Point(x: 0, y: 0), predictedSpeed: 10)
        // Claims to be driving east at 10 m/s but actually moved 10 m north.
        let decision = machine.update(
            fix(t: 202, course: 90), measured: Point(x: 0, y: 10), predictedSpeed: 10
        )
        XCTAssertFalse(decision.accepted)
        XCTAssertTrue(decision.reasons.contains("course_inconsistent_with_previous_fix"))
    }

    /// A single false fix a kilometre away must be rejected outright.
    func testASingleFalsePointAKilometreAwayIsRejected() {
        let machine = GPSStateMachine(config: config, maxAccelMS2: maxAccel)
        let t = promote(machine)
        let decision = machine.update(
            fix(t: t + 1, accuracy: 5), measured: Point(x: 1000, y: 0), predictedSpeed: 10
        )
        XCTAssertFalse(decision.accepted)
        XCTAssertTrue(decision.reasons.contains("physical_gate"))
        XCTAssertEqual(decision.sigmaM, 5, "the fix claimed to be accurate; that is the point")
    }

    /// A whole cluster of consistent-looking but physically unreachable fixes
    /// must not rebuild trust either.
    func testAPhysicallyUnreachableFalseClusterNeverRestoresTrust() {
        let machine = GPSStateMachine(config: config, maxAccelMS2: maxAccel)
        promote(machine)
        // The cluster is internally consistent but 3 km from the last known
        // position, reached in one second.
        for step in 0..<8 {
            _ = machine.update(
                fix(t: 10 + Double(step) * 0.2, accuracy: 6, speed: 10, course: 90),
                measured: Point(x: 3000 + Double(step) * 2, y: 0), predictedSpeed: 10
            )
        }
        XCTAssertNotEqual(machine.state, .trusted)
    }

    func testEveryFailingGateIsReportedNotJustTheFirst() {
        let machine = GPSStateMachine(config: config, maxAccelMS2: maxAccel)
        let t = promote(machine)
        let decision = machine.update(
            fix(t: t + 1, accuracy: 90, speed: 40, course: 270),
            measured: Point(x: 4000, y: 0),
            predicted: Point(x: 0, y: 0), predictedSpeed: 10, predictedHeading: 0,
            covariance: .diagonal(25), roadDistanceM: 500
        )
        XCTAssertFalse(decision.accepted)
        XCTAssertEqual(decision.sigmaM, 90,
                       "poor reported accuracy must widen R, not fabricate an outage")
        XCTAssertTrue(decision.reasons.contains("physical_gate"))
        XCTAssertFalse(decision.reasons.contains("far_from_road"))
        XCTAssertEqual(decision.roadDistanceM, 500)
        XCTAssertGreaterThanOrEqual(decision.reasons.count, 2,
                                    "a rejection has to be diagnosable afterwards")
    }

    /// A graph coverage gap is not a contradictory sensor observation. A
    /// physically coherent GPS fix must keep tracking alive and expose the
    /// graph distance only as a diagnostic.
    func testCoherentFixOutsideGraphCoverageIsAccepted() {
        let machine = GPSStateMachine(config: config, maxAccelMS2: maxAccel)
        let t = promote(machine)
        let roadDistance = config.maxDistanceToRoadM + 50

        let decision = machine.update(
            fix(t: t + 1, accuracy: 5), measured: Point(x: 35, y: 0),
            predictedSpeed: 10, roadDistanceM: roadDistance
        )

        XCTAssertTrue(decision.accepted)
        XCTAssertFalse(decision.reasons.contains("far_from_road"))
        XCTAssertEqual(decision.roadDistanceM, roadDistance)
        XCTAssertEqual(machine.state, .trusted)
    }

    func testSimulatedFixesAreRejectedUnlessExplicitlyAllowed() {
        var sample = fix(t: 1)
        sample.isSimulatedBySoftware = true
        let strict = GPSStateMachine(config: config, maxAccelMS2: maxAccel)
        XCTAssertFalse(strict.update(sample, measured: Point(x: 0, y: 0)).accepted)

        var permissive = config
        permissive.allowSimulatedFixes = true
        let relaxed = GPSStateMachine(config: permissive, maxAccelMS2: maxAccel)
        XCTAssertTrue(relaxed.update(sample, measured: Point(x: 0, y: 0)).accepted)
    }

    func testCourseIsIgnoredBelowTheSpeedThreshold() {
        let machine = GPSStateMachine(config: config, maxAccelMS2: maxAccel)
        promote(machine)
        // Crawling, course pointing the opposite way: CoreLocation course is
        // noise at this speed and must not be used.
        let decision = machine.update(
            fix(t: 10, speed: 1, course: 270), measured: Point(x: 35, y: 0),
            predictedSpeed: 1, predictedHeading: 0
        )
        XCTAssertFalse(decision.reasons.contains("course_mismatch"))
    }
}
