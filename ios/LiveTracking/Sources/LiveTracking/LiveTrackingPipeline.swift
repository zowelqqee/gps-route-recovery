import Foundation

/// The on-device runtime.
///
/// A port of `geotrace.pipeline.TrackingPipeline`, restructured from a batch
/// coordinator into something that can consume sensors as they arrive. It owns
/// the road tracker and the parking tracker, and it is the only thing that
/// mutates them: every state change happens inside this actor, so MapKit and
/// SwiftUI only ever see immutable snapshots and no computation touches the
/// main thread.
///
/// Nothing here talks to a server, a Mac or a Python process. The whole
/// reconstruction runs on the phone.
public actor LiveTrackingPipeline {
    // MARK: - Configuration

    private var config: LiveTrackingConfig
    private let graphURL: URL?
    private let graphRadiusM: Double
    private let calibration: TrackerCalibration?

    // MARK: - Derived state

    private var frame: LocalFrame?
    private var network: RoadNetwork?
    private var roadTracker: RoadTracker?
    private var parkingTracker: ParkingTracker?
    private var ekf: DeadReckoningEKF?
    private var stateMachine: GPSStateMachine
    private var motion: MotionProcessor

    public private(set) var graphCoverage: GraphCoverage = .missing
    public private(set) var isFinished = false

    // MARK: - Timeline

    private var startedAt: Double?
    private var endedAt: Double?
    private var lastLocationTime: Double = -.infinity
    private var lastControlTime: Double?
    private var lastTrustedTime: Double = 0
    private var previousGPSState: GPSState = .lost

    // MARK: - Buffers

    /// Controls held back until the heading offset is known; released in order
    /// once the vehicle heading can be anchored to a real compass direction.
    private var pendingControls: [IMUControl] = []
    private var headingResolved = false
    private var pendingReferenceHeading: Double?
    private var firstMotionTime: Double?
    private var latestMotionTime: Double?
    private var initialHeading: Double?
    /// Position and speed of the first usable fix: the EKF is seeded from these
    /// once the world frame is known, exactly as the batch baseline does.
    private var originPoint: Point?
    private var originSpeed: Double = 0
    private var initialBiases: (accel: Double, gyro: Double) = (0, 0)
    private var earlyControlsForBias: [IMUControl] = []

    /// Rolling window for the parking tracker, trimmed to its own window plus a
    /// margin so memory does not grow with trip length.
    private var parkingControls: [IMUControl] = []
    private var parkingFixes: [TrackerLocation] = []

    /// Accumulated IMU for the road tracker, integrated in batches.
    private var roadBatch: [IMUControl] = []
    private var roadBatchStart: Double?

    /// Fixes waiting for the IMU control that covers them.
    ///
    /// A fix is tested against the filter's *prediction at the fix's own time*,
    /// so it must not be consumed until the motion up to that instant has been
    /// integrated. This mirrors the batch baseline, where the fix cursor only
    /// advances inside the control loop, and it is the correct thing to do
    /// anyway: gating a fix against a prediction that is up to a bin stale
    /// flips marginal Mahalanobis tests for no good reason.
    private var queuedFixes: [TrackerLocation] = []

    // MARK: - History for UI and export

    private var rawTrack: [Coordinate] = []
    private var injectedTrack: [Coordinate] = []
    private var acceptedFixes: [Coordinate] = []
    private var rejectedFixes: [Coordinate] = []
    private var roadRouteLocal: [(t: Double, point: Point)] = []
    private var lastRoadComponents: [ConfidenceComponent] = []
    private var lastPolygonTime: Double = -.infinity
    private var lastOutputTime: Double = -.infinity
    private var lastParkingResult: ParkingTrackingResult?
    private var lastParkingEvaluation: Double = -.infinity
    private var diagnostics = LiveDiagnostics()
    private var roadStepDurations: [Double] = []
    private var parkingStepDurations: [Double] = []
    private var pendingRecords = LiveRecordBatch()
    private var currentSnapshot = LiveSnapshot()
    private var lastSnapshotTime: Double = -.infinity
    private var snapshotCount = 0
    private var faultModeLabel = "None"
    /// Debug hook: when set, every aligned control is captured so the stream can
    /// be diffed against the Python baseline's.
    private var controlTrace: [IMUControl]?

    // MARK: - Init

    public init(
        config: LiveTrackingConfig = LiveTrackingConfig(),
        graphURL: URL?,
        graphRadiusM: Double = 12_000,
        calibration: TrackerCalibration? = nil,
        startedAt: Double? = nil
    ) {
        self.config = config
        self.graphURL = graphURL
        self.graphRadiusM = graphRadiusM
        self.calibration = calibration
        self.stateMachine = GPSStateMachine(
            config: config.gps, maxAccelMS2: config.motion.maxAccelMS2
        )
        self.motion = MotionProcessor(config: config.motion)
        self.startedAt = startedAt
        self.lastTrustedTime = startedAt ?? 0
    }

    public func start(at monotonicTime: Double) {
        startedAt = monotonicTime
        lastTrustedTime = monotonicTime
    }

    public func setFaultLabel(_ label: String) {
        faultModeLabel = label
    }

    /// Start capturing the aligned IMU control stream, for cross-language
    /// diffing. Not used in the app.
    public func enableControlTrace() { controlTrace = [] }

    public func drainControlTrace() -> [IMUControl] {
        let trace = controlTrace ?? []
        controlTrace = []
        return trace
    }

    // MARK: - Sensor intake

    /// Feed one motion frame.
    ///
    /// Called at 50 Hz. Everything expensive is deferred: this only bins the
    /// sample, and the trackers are advanced from the accumulated batches.
    public func processMotion(_ sample: TrackerMotion) {
        guard !isFinished else {
            diagnostics.droppedAfterEndSamples += 1
            return
        }
        diagnostics.motionSamples += 1
        if firstMotionTime == nil { firstMotionTime = sample.monotonicTime }
        latestMotionTime = sample.monotonicTime
        guard let control = motion.ingest(sample) else { return }
        route(control)
        // The uncalibrated heading fallback needs a window of IMU, so the
        // resolve has to be retried as motion accumulates, not only when a fix
        // arrives.
        tryResolveHeading()
    }

    /// Feed one GPS fix, plus optionally the untouched original when a debug
    /// fault is rewriting the live stream.
    public func processLocation(_ sample: TrackerLocation, raw: TrackerLocation? = nil) {
        let original = raw ?? sample
        if let endedAt, sample.monotonicTime > endedAt {
            // A fix that arrives after the trip ended is kept for diagnostics
            // but must not move an already-finished result.
            diagnostics.droppedAfterEndSamples += 1
            pendingRecords.lateFixes.append(original)
            return
        }
        guard !isFinished else {
            diagnostics.droppedAfterEndSamples += 1
            pendingRecords.lateFixes.append(original)
            return
        }
        guard sample.monotonicTime.isFinite else { return }
        if sample.monotonicTime <= lastLocationTime {
            diagnostics.droppedOutOfOrderSamples += 1
            return
        }
        lastLocationTime = sample.monotonicTime
        diagnostics.locationSamples += 1

        rawTrack.append(original.coordinate)
        if raw != nil { injectedTrack.append(sample.coordinate) }
        trim(&rawTrack)
        trim(&injectedTrack)

        guard sample.isUsable else { return }

        // The first usable fix anchors the local frame and loads the graph.
        if frame == nil { bootstrap(with: sample) }
        guard let frame else { return }

        parkingFixes.append(sample)
        trimParkingWindow(now: sample.monotonicTime)

        let point = frame.toLocal(latitude: sample.latitude, longitude: sample.longitude)
        resolveHeadingIfPossible(using: sample, point: point)

        // Hold the fix until the IMU has been integrated up to its timestamp.
        // If motion is unavailable or already ahead, it is consumed at once.
        if ekf == nil {
            // No world frame yet: hold the fix. Nothing can be gated against a
            // filter that has not been seeded.
            queuedFixes.append(sample)
        } else if let last = lastControlTime, sample.monotonicTime <= last {
            consume(sample)
        } else if firstMotionTime == nil {
            consume(sample)
        } else {
            queuedFixes.append(sample)
        }
        maybeEvaluateParking(now: sample.monotonicTime)
        maybePublishSnapshot(now: sample.monotonicTime)
    }

    /// A debug dropout still has a real raw fix behind it. Keep that evidence
    /// for the map/replay trail while deliberately withholding it from all GPS
    /// gates and both trackers.
    public func processDroppedLocation(raw: TrackerLocation, fault: String) {
        if isFinished || (endedAt.map { raw.monotonicTime > $0 } ?? false) {
            diagnostics.droppedAfterEndSamples += 1
            pendingRecords.lateFixes.append(raw)
            return
        }
        guard raw.monotonicTime.isFinite else { return }
        rawTrack.append(raw.coordinate)
        trim(&rawTrack)
        pendingRecords.gateDecisions.append(
            GateDecision(
                monotonicTime: raw.monotonicTime,
                accepted: false,
                state: stateMachine.state,
                reasons: ["debug_dropout", fault],
                latitude: raw.latitude,
                longitude: raw.longitude
            )
        )
        maybePublishSnapshot(now: raw.monotonicTime)
    }

    /// Drain fixes the IMU has now caught up with.
    private func drainQueuedFixes(upTo time: Double) {
        guard ekf != nil, !queuedFixes.isEmpty else { return }
        var remaining: [TrackerLocation] = []
        for fix in queuedFixes {
            if fix.monotonicTime <= time { consume(fix) } else { remaining.append(fix) }
        }
        queuedFixes = remaining
    }

    private func consume(_ sample: TrackerLocation) {
        guard let frame, ekf != nil else { return }
        let point = frame.toLocal(latitude: sample.latitude, longitude: sample.longitude)
        let roadDistance = network?.distanceToRoad(point)
        if let roadDistance {
            graphCoverage = roadDistance.isFinite ? .loaded : .outside
        }

        let decision = stateMachine.update(
            sample,
            measured: point,
            predicted: ekf?.position,
            predictedSpeed: ekf?.speed ?? 0,
            predictedHeading: ekf?.heading,
            covariance: ekf?.positionCovariance,
            roadDistanceM: roadDistance
        )
        pendingRecords.gateDecisions.append(decision)

        if decision.accepted {
            diagnostics.acceptedFixes += 1
            acceptedFixes.append(sample.coordinate)
            trim(&acceptedFixes)
            applyAcceptedFix(sample, point: point, decision: decision)
        } else {
            diagnostics.rejectedFixes += 1
            rejectedFixes.append(sample.coordinate)
            trim(&rejectedFixes)
            for reason in decision.reasons {
                diagnostics.rejectionReasons[reason, default: 0] += 1
            }
        }

        if decision.state != previousGPSState {
            diagnostics.stateTransitions.append(
                .init(monotonicTime: sample.monotonicTime, state: decision.state)
            )
            previousGPSState = decision.state
            adaptParticleBudget(to: decision.state)
        }
    }

    // MARK: - Bootstrap

    private func bootstrap(with sample: TrackerLocation) {
        let origin = Coordinate(latitude: sample.latitude, longitude: sample.longitude)
        if let graphURL {
            do {
                let loaded = try RoadGraphLoader.load(
                    contentsOf: graphURL, origin: origin, radiusM: graphRadiusM
                )
                network = loaded
                frame = loaded.frame
                graphCoverage = .loaded
            } catch {
                // No graph is a degraded but honest mode: the parking tracker
                // works in free space and still produces an answer, while the
                // road tracker simply does not run.
                frame = LocalFrame(latitude: origin.latitude, longitude: origin.longitude)
                graphCoverage = .missing
            }
        } else {
            frame = LocalFrame(latitude: origin.latitude, longitude: origin.longitude)
            graphCoverage = .missing
        }
        guard let frame else { return }
        parkingTracker = ParkingTracker(config: config, frame: frame)
        if let network {
            roadTracker = RoadTracker(network: network, config: config, seed: config.seed)
        }
        originPoint = frame.toLocal(latitude: sample.latitude, longitude: sample.longitude)
        originSpeed = sample.hasValidSpeed ? (sample.speed ?? 0) : 0
        // The EKF is deliberately not created here. Its heading and sensor
        // biases come from the world-frame alignment, and starting it at
        // heading zero and then correcting later would leave every prediction
        // before the alignment wrong - which is exactly what makes an
        // implementation drift away from the reference for no good reason.
    }

    /// Anchor the gyro-integrated heading to a real compass direction.
    ///
    /// Preference order matches `geotrace.pipeline.initial_heading`: a trusted
    /// GPS course while actually moving, then the bearing between the first fix
    /// and the first fix ~20 m away. The magnetometer is never used - inside a
    /// car it is not reliable enough to seed a heading.
    private func resolveHeadingIfPossible(using sample: TrackerLocation, point: Point) {
        guard !headingResolved, pendingReferenceHeading == nil else { return }
        if sample.hasValidCourse, sample.hasValidSpeed,
           (sample.speed ?? 0) >= config.gps.minSpeedForCourseMS {
            pendingReferenceHeading = Angles.headingFromCourse(sample.course ?? 0)
        } else if point.magnitude >= 20, let startedAt,
                  sample.monotonicTime - startedAt > config.runtime.headingFallbackWindowSeconds {
            // Only a last resort: a GPS course is a far better anchor than the
            // chord from the first fix, and on a moving car one arrives within
            // seconds. Committing to the chord early is what made an early
            // version disagree with the baseline by 27 degrees.
            pendingReferenceHeading = atan2(point.y, point.x)
        } else if let startedAt, sample.monotonicTime - startedAt > 120 {
            // Long enough without ever moving: proceed with an unaligned frame
            // rather than never starting, and say so in the diagnostics.
            pendingReferenceHeading = 0
        }
        tryResolveHeading()
    }

    /// Commit the world-frame alignment once there is enough evidence for it.
    ///
    /// With a mount calibration the vehicle forward axis is known, so this fires
    /// on the first heading. Without one, the alignment is inferred from the
    /// direction of the strongest accelerations and needs a window of IMU
    /// behind it; committing on the first fraction of a second produces an
    /// alignment tens of degrees out, and every downstream gate then disagrees.
    private func tryResolveHeading(force: Bool = false) {
        guard !headingResolved, let heading = pendingReferenceHeading else { return }
        let calibrated = calibration?.definesVehicleFrame ?? false
        var windowFilled = calibrated || force
        if !windowFilled, let first = firstMotionTime, let latest = latestMotionTime {
            windowFilled = latest - first >= config.runtime.headingFallbackWindowSeconds
        }
        guard windowFilled else { return }

        headingResolved = true
        initialHeading = heading
        motion.resolveHeadingOffset(referenceHeading: heading, calibration: calibration)
        diagnostics.initialHeadingDegrees = Angles.courseFromHeading(heading)

        // Bias estimation needs the aligned stream, so it happens here rather
        // than at the first sample.
        let aligned = earlyControlsForBias.map { motion.aligned($0) }
        initialBiases = BiasEstimator.estimate(
            controls: aligned, heading: heading, config: config.motion
        )
        diagnostics.initialAccelBias = initialBiases.accel
        diagnostics.initialGyroBias = initialBiases.gyro
        earlyControlsForBias.removeAll(keepingCapacity: false)

        if ekf == nil, let origin = originPoint {
            ekf = DeadReckoningEKF(
                config: config.motion,
                initialState: [
                    origin.x, origin.y, originSpeed, heading,
                    initialBiases.accel, initialBiases.gyro,
                ]
            )
        }

        // Release everything that was waiting for the alignment.
        let queued = pendingControls
        pendingControls.removeAll(keepingCapacity: false)
        for control in queued { advance(motion.aligned(control)) }
    }

    // MARK: - Control routing

    private func route(_ control: IMUControl) {
        guard !headingResolved else {
            advance(motion.aligned(control))
            return
        }
        // Hold back until the heading is anchored, but keep the buffer bounded:
        // a very long stationary start must not grow it without limit.
        pendingControls.append(control)
        earlyControlsForBias.append(control)
        if pendingControls.count > 4000 { pendingControls.removeFirst() }
        if earlyControlsForBias.count > 600 { earlyControlsForBias.removeFirst() }
    }

    private func advance(_ control: IMUControl) {
        if let endedAt, control.t > endedAt { return }
        if let last = lastControlTime, control.t <= last { return }
        lastControlTime = control.t
        if controlTrace != nil { controlTrace?.append(control) }

        // Operation order matters and is the baseline's, exactly: propagate both
        // filters, then decide whether the vehicle is stopped, then consume the
        // fixes this control has caught up with, then apply the motion-only
        // evidence. Reordering any of it changes which random draws land where
        // and quietly moves the answer.

        // The parking tracker runs off the rolling window, so it only needs the
        // control retained; the evaluation itself is a pure function of it.
        parkingControls.append(control)
        trimParkingWindow(now: control.t)

        // The EKF is cheap and is what the GPS gates test against, so it runs on
        // every control.
        ekf?.predict(aWorld: control.aWorld, yawRate: control.yawRate, dt: control.dt)

        // The particle filter is the expensive one and is stepped in batches.
        roadBatch.append(control)
        if roadBatchStart == nil { roadBatchStart = control.t }
        if control.t - (roadBatchStart ?? control.t) >= config.runtime.roadStepSeconds
            - config.motion.filterDT * 0.5 {
            flushRoadBatch(at: control.t)
        }

        // A quiet IMU only means a stop if the filter also believes it is barely
        // moving; otherwise it just means steady cruising on a straight road.
        if control.isQuiet, let ekf, ekf.speed <= config.motion.zuptMaxSpeedMS {
            ekf.zeroVelocityUpdate()
            roadTracker?.zeroVelocityUpdate(maxSpeedMS: config.motion.zuptMaxSpeedMS)
        }

        stateMachine.noteGap(now: control.t)
        drainQueuedFixes(upTo: control.t)

        maybeUpdateRoadWeights(now: control.t)
        maybeEvaluateParking(now: control.t)
        maybePublishSnapshot(now: control.t)
    }

    /// Motion-only weight update and resampling check, at the output rate.
    private func maybeUpdateRoadWeights(now: Double) {
        guard now - lastOutputTime >= config.runtime.outputSeconds else { return }
        lastOutputTime = now
        guard let tracker = roadTracker, tracker.isInitialized else { return }
        let started = DispatchTime.now().uptimeNanoseconds
        tracker.updateWeights()
        tracker.maybeResample()
        recordRoadPosition(at: now)
        roadStepDurations.append(
            Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
        )
        if roadStepDurations.count > 2000 { roadStepDurations.removeFirst(1000) }
    }

    /// Integrate the accumulated IMU into the particle filter as one step.
    ///
    /// Running a 5000-particle filter at the raw 50 Hz is pure waste: vehicle
    /// dynamics do not change meaningfully inside 200 ms, and the batch mean is
    /// what the offline baseline uses anyway.
    private func flushRoadBatch(at now: Double) {
        defer {
            roadBatch.removeAll(keepingCapacity: true)
            roadBatchStart = nil
        }
        guard let tracker = roadTracker, tracker.isInitialized, !roadBatch.isEmpty else { return }
        let started = DispatchTime.now().uptimeNanoseconds

        var accel = SIMD3<Double>(repeating: 0)
        var yaw = 0.0
        var dt = 0.0
        var gapExceeded = false
        for control in roadBatch {
            accel += control.aWorld * control.dt
            yaw += control.yawRate * control.dt
            dt += control.dt
            gapExceeded = gapExceeded || control.gapExceeded
        }
        guard dt > 0 else { return }
        accel /= dt
        yaw /= dt

        // A gap inside the batch propagates: the tracker must widen rather than
        // integrate across a hole.
        tracker.predict(
            aWorld: accel, yawRate: yaw,
            dt: gapExceeded ? config.motion.maxGapSeconds + 0.001 : dt
        )
        roadStepDurations.append(
            Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
        )
        if roadStepDurations.count > 2000 { roadStepDurations.removeFirst(1000) }
    }

    private func recordRoadPosition(at now: Double) {
        guard let tracker = roadTracker, tracker.isInitialized, let network else { return }
        let sinceTrusted = Swift.max(0, now - lastTrustedTime)

        // Rebuilding corridors is the expensive part and the map cannot show
        // more than a couple of updates a second anyway.
        if now - lastPolygonTime >= config.runtime.polygonUpdateSeconds {
            let builder = ConfidencePolygonBuilder(network: network, config: config.polygon)
            let output = builder.build(cloud: tracker.cloud, secondsSinceTrusted: sinceTrusted)
            lastRoadComponents = output.components
            lastPolygonTime = now
            if let estimate = output.estimate {
                appendRoute(t: now, point: estimate)
            }
        } else if let estimate = quickEstimate(tracker: tracker) {
            appendRoute(t: now, point: estimate)
        }
    }

    /// Cheap between-polygon estimate: the weighted mean along the single
    /// highest-weight edge, which is the same rule the branch-aware estimate
    /// uses, without rebuilding the corridors.
    private func quickEstimate(tracker: RoadTracker) -> Point? {
        guard let network, tracker.cloud.count > 0 else { return nil }
        var weightByEdge: [Int32: Double] = [:]
        for index in 0..<tracker.cloud.count {
            weightByEdge[tracker.cloud.edge[index], default: 0] += tracker.cloud.weight[index]
        }
        guard let dominant = weightByEdge.max(by: { $0.value < $1.value })?.key else { return nil }
        var weighted = 0.0, weightSum = 0.0
        for index in 0..<tracker.cloud.count where tracker.cloud.edge[index] == dominant {
            weighted += tracker.cloud.s[index] * tracker.cloud.weight[index]
            weightSum += tracker.cloud.weight[index]
        }
        guard weightSum > 0 else { return nil }
        return network.position(edge: Int(dominant), s: weighted / weightSum)
    }

    private func appendRoute(t: Double, point: Point) {
        roadRouteLocal.append((t, point))
        if roadRouteLocal.count > config.runtime.maxRouteSamples {
            roadRouteLocal.removeFirst(roadRouteLocal.count - config.runtime.maxRouteSamples)
        }
        pendingRecords.roadPositions.append(
            LiveRecordBatch.RoadPosition(
                monotonicTime: t,
                coordinate: frame?.toGeo(point) ?? Coordinate(latitude: 0, longitude: 0),
                gpsState: stateMachine.state,
                particleCount: roadTracker?.particleCount ?? 0,
                effectiveSampleSize: roadTracker?.effectiveSampleSize ?? 0
            )
        )
    }

    // MARK: - Accepted fixes

    private func applyAcceptedFix(
        _ sample: TrackerLocation, point: Point, decision: GateDecision
    ) {
        guard let ekf else { return }
        let sigma = GPSGate.measurementSigma(sample, config: config.gps)
        let courseRad: Double? = (
            sample.hasValidCourse && sample.hasValidSpeed
                && (sample.speed ?? 0) >= config.gps.minSpeedForCourseMS
        ) ? Angles.headingFromCourse(sample.course ?? 0) : nil

        // Trust has just come back after an outage: both filters are re-anchored
        // on the recovered fix rather than blended with a stale dead-reckoning
        // solution that is hundreds of metres out.
        let restored = decision.state == .trusted
            && previousGPSState != .trusted
            && stateMachine.hadRealOutage
        if restored {
            ekf.reanchor(
                to: point, sigma: sigma, heading: courseRad,
                speed: sample.hasValidSpeed ? sample.speed : nil
            )
            diagnostics.reanchors.append(sample.monotonicTime)
            if let tracker = roadTracker, tracker.isInitialized {
                tracker.reinitialize(
                    at: point, heading: courseRad ?? ekf.heading, speed: ekf.speed, sigma: sigma
                )
            }
        }

        ekf.updatePosition(point, sigma: sigma)
        if sample.hasValidSpeed {
            ekf.updateSpeed(sample.speed ?? 0, sigma: Swift.max(1, sample.speedAccuracy ?? 1))
        }
        if let courseRad { ekf.updateHeading(courseRad, sigma: 25 * .pi / 180) }

        if decision.state == .trusted { lastTrustedTime = sample.monotonicTime }

        guard let tracker = roadTracker else { return }
        if !tracker.isInitialized, decision.state == .trusted {
            _ = tracker.initialize(
                at: point, heading: ekf.heading, speed: ekf.speed,
                accelBias: initialBiases.accel, gyroBias: initialBiases.gyro,
                positionSigma: sigma,
                particleCount: budget(for: decision.state)
            )
        } else if tracker.isInitialized, decision.state == .trusted {
            tracker.updateWeights(
                gps: point, gpsSigma: sigma,
                gpsCourseRad: sample.hasValidCourse
                    ? Angles.headingFromCourse(sample.course ?? 0) : nil,
                gpsSpeed: sample.hasValidSpeed ? sample.speed : nil
            )
            if tracker.hasDiverged() {
                // Every particle is impossible under this fix: the filter
                // followed the wrong branch. Re-seed rather than carry a
                // confidently wrong belief forward.
                tracker.reinitialize(
                    at: point, heading: ekf.heading, speed: ekf.speed, sigma: sigma
                )
            } else {
                tracker.maybeResample()
                tracker.inject(from: point, heading: ekf.heading, speed: ekf.speed, sigma: sigma)
            }
        }
    }

    // MARK: - Adaptive particle budget

    private func budget(for state: GPSState) -> Int {
        guard config.runtime.adaptiveParticleCounts else {
            return config.particleFilter.particleCount
        }
        switch state {
        case .trusted: return config.runtime.particlesTrusted
        case .suspect: return config.runtime.particlesSuspect
        case .lost: return config.runtime.particlesLost
        case .recovering: return config.runtime.particlesRecovering
        }
    }

    /// The road filter is only expensive when it is actually uncertain. Resizing
    /// goes through systematic resampling, so the posterior is preserved: this
    /// changes how finely the belief is represented, never what it is.
    private func adaptParticleBudget(to state: GPSState) {
        guard config.runtime.adaptiveParticleCounts,
              let tracker = roadTracker, tracker.isInitialized else { return }
        tracker.resize(to: budget(for: state))
    }

    // MARK: - Parking tracker

    private func trimParkingWindow(now: Double) {
        let horizon = now - config.parkingTracker.windowSeconds - 5
        if let first = parkingControls.first, first.t < horizon {
            parkingControls.removeAll { $0.t < horizon }
        }
        if let first = parkingFixes.first, first.monotonicTime < horizon {
            parkingFixes.removeAll { $0.monotonicTime < horizon }
        }
    }

    /// Re-run the parking estimate over the rolling window.
    ///
    /// It is a pure function of the window, so re-running it is exactly the
    /// batch algorithm the Python baseline runs - there is no incremental
    /// approximation that could drift away from it - and at 60 s of 10 Hz
    /// controls it costs well under a millisecond.
    private func maybeEvaluateParking(now: Double) {
        guard now - lastParkingEvaluation >= 1.0 else { return }
        lastParkingEvaluation = now
        lastParkingResult = evaluateParking(endedAt: now)
        if let result = lastParkingResult {
            pendingRecords.parkingResults.append(
                LiveRecordBatch.ParkingSample(monotonicTime: now, result: result)
            )
        }
    }

    private func evaluateParking(endedAt: Double) -> ParkingTrackingResult? {
        guard let parkingTracker, let ekf else { return nil }
        let started = DispatchTime.now().uptimeNanoseconds
        let windowStart = endedAt - config.parkingTracker.windowSeconds
        let prior = roadPrior(at: windowStart)
        let result = parkingTracker.evaluate(
            ParkingTracker.Inputs(
                controls: parkingControls,
                fixes: parkingFixes,
                endedAt: endedAt,
                roadPrior: prior.point,
                roadHeading: prior.heading,
                roadSpeed: ekf.speed,
                calibrationPresent: calibration?.definesVehicleFrame ?? false,
                roadDistance: network.map { network in { network.distanceToRoad($0) } }
            )
        )
        parkingStepDurations.append(
            Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
        )
        if parkingStepDurations.count > 2000 { parkingStepDurations.removeFirst(1000) }
        return result
    }

    /// The road tracker's belief at the start of the parking window, which is
    /// what seeds the free-space estimate. This is the only direction
    /// information flows between the two trackers.
    private func roadPrior(at windowStart: Double) -> (point: Point, heading: Double) {
        guard !roadRouteLocal.isEmpty else {
            return (ekf?.position ?? .zero, ekf?.heading ?? initialHeading ?? 0)
        }
        var bestIndex = 0
        var bestDelta = Double.greatestFiniteMagnitude
        for (index, entry) in roadRouteLocal.enumerated() {
            let delta = abs(entry.t - windowStart)
            if delta < bestDelta { bestDelta = delta; bestIndex = index }
        }
        let prior = roadRouteLocal[bestIndex].point
        var heading = initialHeading ?? 0
        if roadRouteLocal.count > 1 {
            let next = roadRouteLocal[Swift.min(bestIndex + 1, roadRouteLocal.count - 1)].point
            let delta = next - prior
            if delta.magnitude > 0.1 { heading = atan2(delta.y, delta.x) }
        }
        return (prior, heading)
    }

    // MARK: - Snapshots

    private func maybePublishSnapshot(now: Double) {
        guard now - lastSnapshotTime >= config.runtime.uiUpdateSeconds else { return }
        let elapsedSinceLast = now - lastSnapshotTime
        lastSnapshotTime = now
        snapshotCount += 1
        currentSnapshot = buildSnapshot(now: now, updateInterval: elapsedSinceLast)
    }

    private func buildSnapshot(now: Double, updateInterval: Double) -> LiveSnapshot {
        var snapshot = LiveSnapshot()
        snapshot.monotonicTime = now
        snapshot.elapsed = now - (startedAt ?? now)
        snapshot.gpsState = stateMachine.state
        snapshot.horizontalAccuracy = stateMachine.lastAccepted?.horizontalAccuracy
        snapshot.speedMS = ekf?.speed
        snapshot.rawTrack = rawTrack
        snapshot.injectedTrack = injectedTrack
        snapshot.acceptedFixes = acceptedFixes
        snapshot.rejectedFixes = rejectedFixes
        if let frame {
            snapshot.roadRoute = roadRouteLocal.map { frame.toGeo($0.point) }
            snapshot.roadPosition = roadRouteLocal.last.map { frame.toGeo($0.point) }
        }
        snapshot.roadComponents = lastRoadComponents
        snapshot.roadConfidence = lastRoadComponents.first?.probability ?? 0
        if let parking = lastParkingResult {
            snapshot.parkingTrajectory = parking.trajectory
            snapshot.parkingPosition = parking.position
            snapshot.parkingPolygon = parking.polygon
            snapshot.parkingStatus = parking.status
            snapshot.parkingConfidence = parking.confidence
            snapshot.hasReverseMotion = parking.hasReverseMotion
        }
        snapshot.particleCount = roadTracker?.particleCount ?? 0
        snapshot.effectiveSampleSize = roadTracker?.effectiveSampleSize ?? 0
        snapshot.roadStepMS = percentile(roadStepDurations, 0.95)
        snapshot.parkingStepMS = percentile(parkingStepDurations, 0.95)
        snapshot.updateHz = updateInterval > 0 ? 1.0 / updateInterval : 0
        snapshot.graphCoverage = graphCoverage
        snapshot.calibrationApplied = calibration?.definesVehicleFrame ?? false
        if calibration == nil {
            snapshot.calibrationWarning =
                "No mount calibration: heading is inferred from motion, confidence is reduced."
        } else if calibration?.definesVehicleFrame == false {
            snapshot.calibrationWarning =
                "Calibration incomplete (no straight-line drive): heading alignment is approximate."
        }
        snapshot.faultMode = faultModeLabel
        return snapshot
    }

    public func snapshot() -> LiveSnapshot { currentSnapshot }

    /// Samples that arrived after the trip ended. They are recorded for
    /// diagnostics but deliberately cannot change a finished result, so they are
    /// reported here rather than inside `LiveTrackingResult`.
    public func lateSampleCount() -> Int { diagnostics.droppedAfterEndSamples }

    /// Samples dropped because their timestamp went backwards.
    public func outOfOrderSampleCount() -> Int { diagnostics.droppedOutOfOrderSamples }

    /// The resolved yaw offset between CMDeviceMotion's reference frame and
    /// local East/North, in degrees. Diagnostic only.
    public func headingOffsetDegrees() -> Double {
        (motion.headingOffset ?? 0) * 180 / .pi
    }

    /// Hand over everything accumulated since the last call, so the recorder can
    /// append it to disk without holding the actor.
    public func drainRecords() -> LiveRecordBatch {
        let batch = pendingRecords
        pendingRecords = LiveRecordBatch()
        return batch
    }

    // MARK: - Finish

    /// End the trip and produce the final result.
    ///
    /// After this, neither tracker is advanced again: a late fix is kept as a
    /// diagnostic sample but cannot move a position that has already been
    /// reported to the driver.
    public func finish(at endedAtTime: Double) -> LiveTrackingResult {
        if isFinished, let cached = finishedResult { return cached }
        endedAt = endedAtTime
        // A trip that never gathered enough evidence still has to produce an
        // answer, so the alignment is forced with whatever there is.
        tryResolveHeading(force: true)
        if let control = motion.flush(), headingResolved {
            advance(motion.aligned(control))
        }
        drainQueuedFixes(upTo: endedAtTime)
        flushRoadBatch(at: endedAtTime)
        maybeUpdateRoadWeights(now: endedAtTime)

        let parking = evaluateParking(endedAt: endedAtTime)
            ?? ParkingTrackingResult(
                status: .insufficientData,
                reason: "the trip produced no usable GPS or motion data"
            )
        lastParkingResult = parking

        var road = RoadTrackingResult()
        if let tracker = roadTracker, tracker.isInitialized, let network, let frame {
            let sinceTrusted = Swift.max(0, endedAtTime - lastTrustedTime)
            let builder = ConfidencePolygonBuilder(network: network, config: config.polygon)
            let output = builder.build(cloud: tracker.cloud, secondsSinceTrusted: sinceTrusted)
            lastRoadComponents = output.components
            let endpoint = output.estimate ?? roadRouteLocal.last?.point
            road = RoadTrackingResult(
                endpoint: endpoint.map { frame.toGeo($0) },
                endpointLocal: endpoint,
                heading: ekf?.heading,
                speed: ekf?.speed ?? 0,
                edgeIndex: output.estimateEdge,
                edgeName: output.estimateEdge.flatMap { network.name(edge: $0) },
                confidence: output.components.first?.probability ?? 0,
                effectiveSampleSize: tracker.effectiveSampleSize,
                particleCount: tracker.particleCount,
                branchCount: output.components.count,
                components: output.components,
                gpsState: stateMachine.state,
                initialized: true
            )
        }

        diagnostics.resampleCount = roadTracker?.resampleCount ?? 0
        diagnostics.reinitializations = roadTracker?.reinitializations ?? 0
        diagnostics.deadEndParticles = roadTracker?.deadEndParticles ?? 0
        diagnostics.junctionTransitions = roadTracker?.junctionTransitions ?? 0
        diagnostics.skippedGaps = (roadTracker?.skippedGaps ?? 0) + (ekf?.skippedGaps ?? 0)
        diagnostics.roadStepMeanMS = mean(roadStepDurations)
        diagnostics.roadStepP95MS = percentile(roadStepDurations, 0.95)
        diagnostics.parkingStepMeanMS = mean(parkingStepDurations)
        diagnostics.parkingStepP95MS = percentile(parkingStepDurations, 0.95)
        diagnostics.outageWindows = outageWindows()

        let result = LiveTrackingResult(
            roadResult: road,
            parkingResult: parking,
            endedAtMonotonic: endedAtTime,
            calibrationPresent: calibration?.definesVehicleFrame ?? false,
            graphCoverage: graphCoverage,
            diagnostics: diagnostics
        )
        isFinished = true
        finishedResult = result
        currentSnapshot = buildSnapshot(now: endedAtTime, updateInterval: 0)
        return result
    }

    private var finishedResult: LiveTrackingResult?

    private func outageWindows() -> [LiveDiagnostics.OutageWindow] {
        let intervals = stateMachine.stateIntervals()
        guard let start = startedAt else { return [] }
        return intervals
            .filter { $0.state != .trusted }
            .map {
                LiveDiagnostics.OutageWindow(
                    startSeconds: $0.startSeconds - start,
                    endSeconds: $0.endSeconds - start,
                    state: $0.state
                )
            }
            .filter { $0.endSeconds - $0.startSeconds > 1.0 && $0.startSeconds > 0.5 }
    }

    // MARK: - Helpers

    private func trim(_ track: inout [Coordinate]) {
        if track.count > config.runtime.maxRouteSamples {
            track.removeFirst(track.count - config.runtime.maxRouteSamples)
        }
    }

    private func mean(_ values: [Double]) -> Double {
        values.isEmpty ? 0 : values.reduce(0, +) / Double(values.count)
    }

    private func percentile(_ values: [Double], _ q: Double) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let index = Swift.min(sorted.count - 1, Swift.max(0, Int(Double(sorted.count - 1) * q)))
        return sorted[index]
    }
}

/// Records produced by the pipeline for the recorder to persist.
public struct LiveRecordBatch: Sendable {
    public struct RoadPosition: Sendable, Codable {
        public var monotonicTime: Double
        public var coordinate: Coordinate
        public var gpsState: GPSState
        public var particleCount: Int
        public var effectiveSampleSize: Double
    }

    public struct ParkingSample: Sendable, Codable {
        public var monotonicTime: Double
        public var result: ParkingTrackingResult
    }

    public var roadPositions: [RoadPosition] = []
    public var parkingResults: [ParkingSample] = []
    public var gateDecisions: [GateDecision] = []
    public var lateFixes: [TrackerLocation] = []

    public var isEmpty: Bool {
        roadPositions.isEmpty && parkingResults.isEmpty
            && gateDecisions.isEmpty && lateFixes.isEmpty
    }

    public init() {}
}
