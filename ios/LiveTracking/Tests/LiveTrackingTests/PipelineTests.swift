import XCTest
@testable import LiveTracking

/// End-to-end behaviour of the on-device runtime.
final class PipelineTests: XCTestCase {

    /// Drive east along a straight road, then stop. Returns the pipeline and the
    /// fixes that were fed to it.
    private func driveAndStop(
        graphURL: URL? = nil, calibration: TrackerCalibration? = Fixtures.calibration(),
        endAt: Double = 120, stopFrom: Double = 95
    ) async -> (LiveTrackingPipeline, LocalFrame, [TrackerLocation]) {
        var config = Fixtures.config(particles: 300)
        config.runtime.headingFallbackWindowSeconds = 1.0
        let pipeline = LiveTrackingPipeline(
            config: config, graphURL: graphURL, graphRadiusM: 4000, calibration: calibration
        )
        let frame = LocalFrame(
            latitude: Fixtures.origin.latitude, longitude: Fixtures.origin.longitude
        )
        await pipeline.start(at: 1.0)

        var fixes: [TrackerLocation] = []
        var t = 1.0
        var x = 0.0
        var motionTime = 1.0
        while t <= endAt {
            let moving = t < stopFrom
            if moving { x += 8.0 }
            let fix = Fixtures.location(
                t: t, point: Point(x: x, y: 0), frame: frame,
                accuracy: 6, speed: moving ? 8 : 0, course: moving ? 90 : nil
            )
            fixes.append(fix)
            // 50 Hz motion between fixes.
            while motionTime < t {
                await pipeline.processMotion(
                    Fixtures.motion(t: motionTime, accelMS2: 0, yawRate: 0)
                )
                motionTime += 0.02
            }
            await pipeline.processLocation(fix)
            t += 1.0
        }
        return (pipeline, frame, fixes)
    }

    /// The final position always comes from the parking tracker. The road
    /// endpoint is diagnostic and must never be presented as the car's place.
    func testFinalPositionComesFromTheParkingTracker() async {
        let (pipeline, _, _) = await driveAndStop()
        let result = await pipeline.finish(at: 120)
        XCTAssertEqual(result.finalVehiclePositionSource, "parking_tracker")
        XCTAssertEqual(result.finalVehiclePosition?.latitude,
                       result.parkingResult.position?.latitude)
        XCTAssertEqual(result.finalVehiclePosition?.longitude,
                       result.parkingResult.position?.longitude)
    }

    /// With no usable terminal data the answer is nil, and specifically *not*
    /// the road tracker's endpoint wearing a disguise.
    func testThereIsNoHiddenFallbackToTheRoadTracker() async {
        var config = Fixtures.config(particles: 200)
        config.runtime.headingFallbackWindowSeconds = 1.0
        let pipeline = LiveTrackingPipeline(
            config: config, graphURL: nil, calibration: Fixtures.calibration()
        )
        await pipeline.start(at: 1.0)
        // Motion only: never a single GPS fix.
        var t = 1.0
        while t < 60 {
            await pipeline.processMotion(Fixtures.motion(t: t, accelMS2: 0.2))
            t += 0.02
        }
        let result = await pipeline.finish(at: 60)
        XCTAssertEqual(result.parkingResult.status, .insufficientData)
        XCTAssertNil(result.finalVehiclePosition)
        XCTAssertEqual(result.finalVehiclePositionSource, "parking_tracker")
    }

    /// Recording stopped, so the answer is fixed. A fix that turns up late is
    /// kept as a diagnostic sample and changes nothing.
    func testNothingMovesAfterTheEnd() async {
        let (pipeline, frame, _) = await driveAndStop()
        let result = await pipeline.finish(at: 120)
        let before = result.parkingResult.position

        // A late fix a kilometre away.
        await pipeline.processLocation(
            Fixtures.location(t: 130, point: Point(x: 5000, y: 5000), frame: frame,
                              accuracy: 5, speed: 0, course: nil)
        )
        await pipeline.processMotion(Fixtures.motion(t: 131, accelMS2: 5))
        let again = await pipeline.finish(at: 120)
        XCTAssertEqual(again.parkingResult.position?.latitude, before?.latitude)
        XCTAssertEqual(again.parkingResult.position?.longitude, before?.longitude)
        let late = await pipeline.lateSampleCount()
        XCTAssertGreaterThan(late, 0, "the late data is recorded, just not acted on")
    }

    func testOutOfOrderFixesAreDroppedNotIntegrated() async {
        let (pipeline, frame, _) = await driveAndStop(endAt: 40, stopFrom: 35)
        let snapshotBefore = await pipeline.snapshot()
        await pipeline.processLocation(
            Fixtures.location(t: 5, point: Point(x: 0, y: 0), frame: frame)
        )
        let outOfOrder = await pipeline.outOfOrderSampleCount()
        _ = await pipeline.finish(at: 40)
        XCTAssertGreaterThan(outOfOrder, 0)
        XCTAssertNotNil(snapshotBefore.gpsState)
    }

    /// The app records at 50 Hz and only publishes a snapshot a couple of times
    /// a second; the snapshot has to be a complete, self-contained view.
    func testSnapshotsAreProducedForTheUI() async {
        let (pipeline, _, _) = await driveAndStop(endAt: 60, stopFrom: 50)
        let snapshot = await pipeline.snapshot()
        XCTAssertGreaterThan(snapshot.rawTrack.count, 10)
        XCTAssertFalse(snapshot.acceptedFixes.isEmpty)
        XCTAssertEqual(snapshot.gpsState, .trusted)
        XCTAssertGreaterThan(snapshot.elapsed, 30)
    }

    /// A trip must survive the app going to the background and coming back: the
    /// pipeline is a plain actor with no lifecycle assumptions, and the
    /// accumulated state has to still be there.
    func testBackgroundAndForegroundDoesNotResetTheTrip() async {
        let (pipeline, frame, _) = await driveAndStop(endAt: 60, stopFrom: 55)
        let mid = await pipeline.snapshot()
        XCTAssertGreaterThan(mid.rawTrack.count, 10)

        // Simulate the gap: no calls at all for a while, then resume.
        var t = 61.0
        var motionTime = 60.02
        while t <= 100 {
            while motionTime < t {
                await pipeline.processMotion(Fixtures.motion(t: motionTime, accelMS2: 0))
                motionTime += 0.02
            }
            await pipeline.processLocation(
                Fixtures.location(t: t, point: Point(x: 440, y: 0), frame: frame,
                                  accuracy: 6, speed: 0, course: nil)
            )
            t += 1.0
        }
        let after = await pipeline.snapshot()
        XCTAssertGreaterThan(after.rawTrack.count, mid.rawTrack.count,
                             "the trip continued rather than starting over")
        let result = await pipeline.finish(at: 100)
        XCTAssertNotNil(result.parkingResult.position)
    }

    func testWithoutAGraphTheParkingAnswerStillWorks() async {
        let (pipeline, _, _) = await driveAndStop(graphURL: nil)
        let result = await pipeline.finish(at: 120)
        XCTAssertEqual(result.graphCoverage, .missing)
        XCTAssertFalse(result.roadResult.initialized, "no graph means no road tracker")
        XCTAssertNotNil(result.parkingResult.position,
                        "the free-space tracker does not need the graph")
    }

    func testCalibrationPresenceIsRecorded() async {
        let (withCalibration, _, _) = await driveAndStop(calibration: Fixtures.calibration())
        let a = await withCalibration.finish(at: 120)
        XCTAssertTrue(a.calibrationPresent)

        let (without, _, _) = await driveAndStop(calibration: nil)
        let b = await without.finish(at: 120)
        XCTAssertFalse(b.calibrationPresent)
        XCTAssertLessThan(b.parkingResult.confidence, a.parkingResult.confidence)
    }

    func testFinishIsIdempotent() async {
        let (pipeline, _, _) = await driveAndStop(endAt: 60, stopFrom: 50)
        let first = await pipeline.finish(at: 60)
        let second = await pipeline.finish(at: 60)
        XCTAssertEqual(first.parkingResult.position?.latitude,
                       second.parkingResult.position?.latitude)
        XCTAssertEqual(first.roadResult.endpoint?.latitude, second.roadResult.endpoint?.latitude)
    }
}

/// The debug fault injector must never touch what gets recorded.
final class FaultInjectorTests: XCTestCase {
    let frame = LocalFrame(latitude: 59.9311, longitude: 30.3609)

    private func sample(t: Double) -> TrackerLocation {
        Fixtures.location(t: t, point: Point(x: 100, y: 50), frame: frame,
                          accuracy: 7, speed: 9, course: 90)
    }

    func testNoneIsAPassThrough() {
        var injector = FaultInjector()
        let original = sample(t: 10)
        let out = injector.apply(to: original, frame: frame)
        XCTAssertEqual(out, original)
    }

    func testDropoutSuppressesTheFix() {
        var injector = FaultInjector()
        injector.arm(.dropout, at: 5, duration: 45)
        XCTAssertNil(injector.apply(to: sample(t: 10), frame: frame))
        // ...and stops suppressing once the window is over.
        XCTAssertNotNil(injector.apply(to: sample(t: 60), frame: frame))
    }

    func testOffsetMovesTheFixByTheRequestedMetres() {
        var injector = FaultInjector()
        injector.offsetEast = 900
        injector.offsetNorth = -600
        injector.arm(.offset, at: 5, duration: 45)
        let out = injector.apply(to: sample(t: 10), frame: frame)
        let point = frame.toLocal(latitude: out!.latitude, longitude: out!.longitude)
        XCTAssertEqual(point.x, 100 + 900, accuracy: 0.5)
        XCTAssertEqual(point.y, 50 - 600, accuracy: 0.5)
    }

    func testDriftGrowsWithTime() {
        var injector = FaultInjector()
        injector.driftEastMS = 3
        injector.arm(.drift, at: 0, duration: 60)
        let early = injector.apply(to: sample(t: 5), frame: frame)!
        let late = injector.apply(to: sample(t: 25), frame: frame)!
        let earlyX = frame.toLocal(latitude: early.latitude, longitude: early.longitude).x
        let lateX = frame.toLocal(latitude: late.latitude, longitude: late.longitude).x
        XCTAssertEqual(earlyX - 100, 15, accuracy: 0.5)
        XCTAssertEqual(lateX - 100, 75, accuracy: 0.5)
    }

    /// A lying receiver still reports a confident accuracy; that is exactly why
    /// horizontalAccuracy alone can never be the test.
    func testACorruptedFixStillClaimsToBeAccurate() {
        var injector = FaultInjector()
        injector.reportedAccuracyM = 12
        injector.arm(.offset, at: 0, duration: 60)
        let out = injector.apply(to: sample(t: 10), frame: frame)!
        XCTAssertEqual(out.horizontalAccuracy, 12)
    }

    /// The raw stream is the recording. Injection must only ever affect the copy
    /// handed to the live pipeline.
    func testInjectionDoesNotMutateTheOriginalFix() {
        var injector = FaultInjector()
        injector.arm(.jumps, at: 0, duration: 60)
        let original = sample(t: 10)
        let copy = original
        _ = injector.apply(to: original, frame: frame)
        XCTAssertEqual(original, copy, "the recorded fix is a value and must be untouched")
    }

    func testCompositeRunsOffsetThenDropoutThenFalseRecovery() {
        var injector = FaultInjector()
        injector.arm(.composite, at: 0, duration: 30)
        let offsetPhase = injector.apply(to: sample(t: 5), frame: frame)
        XCTAssertNotNil(offsetPhase)
        let dropoutPhase = injector.apply(to: sample(t: 30), frame: frame)
        XCTAssertNil(dropoutPhase, "the middle of the composite scenario is a real outage")
        let recoveryPhase = injector.apply(to: sample(t: 50), frame: frame)
        XCTAssertNotNil(recoveryPhase)
    }

    func testTheManifestRecordsWhatWasApplied() {
        var injector = FaultInjector()
        injector.arm(.offset, at: 12, duration: 45)
        let manifest = injector.manifest()
        XCTAssertEqual(manifest["mode"], "Offset")
        XCTAssertEqual(manifest["armed_at_monotonic"], "12.0")
        XCTAssertTrue(manifest["note"]?.contains("untouched") ?? false)
    }

    func testDisarmRestoresThePassThrough() {
        var injector = FaultInjector()
        injector.arm(.dropout, at: 0, duration: 60)
        XCTAssertNil(injector.apply(to: sample(t: 10), frame: frame))
        injector.disarm()
        XCTAssertNotNil(injector.apply(to: sample(t: 10), frame: frame))
        XCTAssertFalse(injector.isActive)
    }
}
