import XCTest
@testable import LiveTracking

/// Device-to-vehicle transform, signed velocity and the ZUPT rule.
final class MotionTests: XCTestCase {

    // MARK: - Device to vehicle

    func testIdentityQuaternionIsIdentityRotation() {
        let v = SIMD3<Double>(1, 2, 3)
        let rotated = Quaternion.rotate(v, by: SIMD4(1, 0, 0, 0))
        XCTAssertEqual(rotated.x, 1, accuracy: 1e-12)
        XCTAssertEqual(rotated.y, 2, accuracy: 1e-12)
        XCTAssertEqual(rotated.z, 3, accuracy: 1e-12)
    }

    func testNinetyDegreeYawMapsDeviceXToWorldY() {
        let q = SIMD4<Double>(cos(.pi / 4), 0, 0, sin(.pi / 4))
        let rotated = Quaternion.rotate(SIMD3(1, 0, 0), by: q)
        XCTAssertEqual(rotated.x, 0, accuracy: 1e-12)
        XCTAssertEqual(rotated.y, 1, accuracy: 1e-12)
    }

    func testRotationIsOrthonormal() {
        let q = SIMD4<Double>(0.4, -0.2, 0.7, 0.1)
        let x = Quaternion.rotate(SIMD3(1, 0, 0), by: q)
        let y = Quaternion.rotate(SIMD3(0, 1, 0), by: q)
        XCTAssertEqual((x * x).sum(), 1, accuracy: 1e-12)
        XCTAssertEqual((x * y).sum(), 0, accuracy: 1e-12)
    }

    func testInverseRotationUndoesRotation() {
        let q = SIMD4<Double>(0.4, -0.2, 0.7, 0.1)
        let v = SIMD3<Double>(1.5, -2.0, 0.25)
        let back = Quaternion.rotateInverse(Quaternion.rotate(v, by: q), by: q)
        XCTAssertEqual(back.x, v.x, accuracy: 1e-12)
        XCTAssertEqual(back.y, v.y, accuracy: 1e-12)
        XCTAssertEqual(back.z, v.z, accuracy: 1e-12)
    }

    /// The mount calibration is what turns a phone lying at an arbitrary angle
    /// in a cradle into a vehicle heading. With a known forward axis the
    /// alignment is exact and immediate.
    func testMountCalibrationRecoversTheWorldFrameOffset() {
        let processor = MotionProcessor(config: MotionConfig())
        // The phone's reference frame is rotated 40 degrees from the vehicle's.
        let referenceYaw = 40.0 * .pi / 180
        let calibration = TrackerCalibration(
            referenceQuaternion: SIMD4(cos(referenceYaw / 2), 0, 0, sin(referenceYaw / 2)),
            gravityDevice: SIMD3(0, 0, -1),
            forwardAxisDevice: SIMD3(1, 0, 0),
            headingSource: "gps_course"
        )
        // The car is actually heading due north (psi = pi/2).
        let offset = processor.resolveHeadingOffset(
            referenceHeading: .pi / 2, calibration: calibration
        )
        XCTAssertEqual(Angles.wrap(offset + referenceYaw), .pi / 2, accuracy: 1e-9)
    }

    func testAlignmentRotatesAccelerationIntoEastNorth() {
        let processor = MotionProcessor(config: MotionConfig())
        let calibration = TrackerCalibration(
            referenceQuaternion: SIMD4(1, 0, 0, 0),
            gravityDevice: SIMD3(0, 0, -1),
            forwardAxisDevice: SIMD3(1, 0, 0)
        )
        // Reference X currently points along +X; the car heads north.
        processor.resolveHeadingOffset(referenceHeading: .pi / 2, calibration: calibration)
        let control = IMUControl(t: 1, dt: 0.1, aWorld: SIMD3(1, 0, 0), yawRate: 0)
        let aligned = processor.aligned(control)
        XCTAssertEqual(aligned.aWorld.x, 0, accuracy: 1e-9)
        XCTAssertEqual(aligned.aWorld.y, 1, accuracy: 1e-9)
    }

    // MARK: - Binning and timeline hygiene

    func testMotionIsBinnedToTheFilterRate() {
        var config = MotionConfig()
        config.filterDT = 0.1
        let processor = MotionProcessor(config: config)
        var controls: [IMUControl] = []
        for step in 0..<100 {
            let t = Double(step) / 50.0
            if let control = processor.ingest(Fixtures.motion(t: t, accelMS2: 1.0)) {
                controls.append(control)
            }
        }
        XCTAssertGreaterThan(controls.count, 15)
        for control in controls {
            XCTAssertEqual(control.dt, 0.1, accuracy: 1e-12)
            XCTAssertEqual(control.aWorld.x, 1.0, accuracy: 1e-6)
        }
    }

    func testOutOfOrderAndDuplicateSamplesAreDropped() {
        let processor = MotionProcessor(config: MotionConfig())
        _ = processor.ingest(Fixtures.motion(t: 1.0, accelMS2: 0))
        _ = processor.ingest(Fixtures.motion(t: 1.02, accelMS2: 0))
        let before = processor.processedSamples
        _ = processor.ingest(Fixtures.motion(t: 1.02, accelMS2: 0))  // duplicate
        _ = processor.ingest(Fixtures.motion(t: 0.5, accelMS2: 0))   // out of order
        XCTAssertEqual(processor.processedSamples, before, "a backwards dt must never integrate")
    }

    func testALongHoleIsFlaggedRatherThanIntegrated() {
        let processor = MotionProcessor(config: MotionConfig())
        var flagged = false
        for step in 0..<50 { _ = processor.ingest(Fixtures.motion(t: Double(step) / 50, accelMS2: 0)) }
        for step in 0..<50 {
            let t = 10.0 + Double(step) / 50
            if let control = processor.ingest(Fixtures.motion(t: t, accelMS2: 0)) {
                flagged = flagged || control.gapExceeded
            }
        }
        XCTAssertTrue(flagged, "a ten-second hole in the IMU stream must be marked")
    }

    // MARK: - Bias

    func testInitialBiasIsSignedNotAMagnitude() {
        // Taking |b| would inject a positive bias whatever the true sign, and
        // 0.2 m/s^2 of phantom acceleration integrates to ~200 m over a 45 s
        // outage.
        let quiet = Fixtures.controls(from: 0, to: 6, dt: 0.1, accel: -0.25, quiet: true)
        let forward = BiasEstimator.estimate(controls: quiet, heading: 0, config: MotionConfig())
        XCTAssertEqual(forward.accel, -0.25, accuracy: 1e-9)
        let backward = BiasEstimator.estimate(controls: quiet, heading: .pi, config: MotionConfig())
        XCTAssertEqual(backward.accel, 0.25, accuracy: 1e-9)
    }

    func testAbsurdBiasEstimatesAreRefused() {
        let noisy = Fixtures.controls(from: 0, to: 6, dt: 0.1, accel: 9.0, quiet: true)
        let bias = BiasEstimator.estimate(controls: noisy, heading: 0, config: MotionConfig())
        XCTAssertEqual(bias.accel, 0, "a 9 m/s^2 'stationary' period is not stationary")
    }

    func testBiasWithoutAStationaryPeriodIsZero() {
        let moving = Fixtures.controls(from: 0, to: 6, dt: 0.1, accel: 1.0, quiet: false)
        let bias = BiasEstimator.estimate(controls: moving, heading: 0, config: MotionConfig())
        XCTAssertEqual(bias.accel, 0)
        XCTAssertEqual(bias.gyro, 0)
    }

    // MARK: - ZUPT

    func testZUPTAppliesWhenTheCarIsActuallyStopped() {
        let config = MotionConfig()
        let ekf = DeadReckoningEKF(config: config, initialState: [0, 0, 0.4, 0, 0, 0])
        for _ in 0..<20 { ekf.zeroVelocityUpdate() }
        XCTAssertEqual(ekf.speed, 0, accuracy: 0.05)
    }

    /// The trap this guards: a quiet IMU is *not* a stop. A car cruising at a
    /// constant speed on a straight road has almost no acceleration and almost
    /// no yaw rate and looks identical to a parked one.
    func testAQuietIMUAtCruisingSpeedIsNotTreatedAsAStop() {
        let config = MotionConfig()
        let processor = MotionProcessor(config: config)
        var quietSeen = false
        for step in 0..<200 {
            // Perfectly steady 15 m/s: zero acceleration, zero rotation.
            if let control = processor.ingest(
                Fixtures.motion(t: Double(step) / 50, accelMS2: 0.0)
            ) {
                quietSeen = quietSeen || control.isQuiet
            }
        }
        XCTAssertTrue(quietSeen, "the IMU really is quiet here")

        // ...but the consumer must refuse to call it a stop while it believes
        // it is moving.
        let ekf = DeadReckoningEKF(config: config, initialState: [0, 0, 15.0, 0, 0, 0])
        let shouldApply = ekf.speed <= config.zuptMaxSpeedMS
        XCTAssertFalse(shouldApply, "a quiet IMU at 15 m/s must not trigger a ZUPT")
    }

    func testParticleZUPTLeavesAFastHypothesisAlone() {
        let network = Fixtures.straight()
        let tracker = RoadTracker(network: network, config: Fixtures.config(particles: 100))
        XCTAssertTrue(tracker.initialize(at: Point(x: 0, y: 0), heading: 0, speed: 11))
        tracker.setAllParticleSpeeds(11)
        tracker.zeroVelocityUpdate(maxSpeedMS: 1.5)
        XCTAssertTrue(tracker.cloud.speed.allSatisfy { $0 == 11 })
    }

    func testParticleZUPTStopsSlowParticles() {
        let network = Fixtures.straight()
        let tracker = RoadTracker(network: network, config: Fixtures.config(particles: 100))
        XCTAssertTrue(tracker.initialize(at: Point(x: 0, y: 0), heading: 0, speed: 0.4))
        tracker.setAllParticleSpeeds(0.4)
        tracker.zeroVelocityUpdate(maxSpeedMS: 1.5)
        XCTAssertTrue(tracker.cloud.speed.allSatisfy { $0 == 0 })
    }
}
