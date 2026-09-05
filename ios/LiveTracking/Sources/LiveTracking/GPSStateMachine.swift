import Foundation

/// The TRUSTED / SUSPECT / LOST / RECOVERING machine.
///
/// A direct port of `geotrace.gps_quality.GPSQualityMonitor`, including the two
/// details that matter most:
///
/// * returning to TRUSTED needs several *consecutive consistent* fixes, which is
///   what rejects the scatter a receiver emits in the first seconds after it
///   re-acquires;
/// * the Mahalanobis gate remains active in LOST and RECOVERING, but its
///   innovation covariance grows with dead-reckoning age. This rejects a true
///   teleport without making the filter defend a stale inertial prediction.
public final class GPSStateMachine {
    public private(set) var state: GPSState = .lost
    public private(set) var history: [GateDecision] = []
    public private(set) var lastAccepted: TrackerLocation?
    public private(set) var lastAcceptedPoint: Point?
    public private(set) var lastSeenTime: Double?

    private let config: GPSQualityConfig
    private let maxAccel: Double
    private var consecutiveGood = 0
    private var consecutiveBad = 0
    private var bootstrapDone = false
    private var everTrusted = false
    private var lostAfterTrusted = false
    private var untrustedSince: Double?
    /// Bounded so a long trip cannot grow the history without limit; the full
    /// stream is persisted to disk instead.
    private let historyLimit: Int

    public init(config: GPSQualityConfig, maxAccelMS2: Double, historyLimit: Int = 5000) {
        self.config = config
        self.maxAccel = maxAccelMS2
        self.historyLimit = historyLimit
    }

    public var isTrusted: Bool { state == .trusted }

    /// True once the trip has lost GPS *after* having trusted it.
    ///
    /// Every trip begins in LOST because nothing is trusted until several
    /// consistent fixes have arrived. That opening period is bootstrap, not an
    /// outage, and the first promotion to TRUSTED must not be treated as a
    /// recovery: there is no stale dead-reckoning solution to re-anchor from.
    public var hadRealOutage: Bool { lostAfterTrusted }

    /// Call on every filter step. Returns true when a dropout pushed us to LOST.
    @discardableResult
    public func noteGap(now: Double) -> Bool {
        guard let last = lastSeenTime else { return false }
        guard now - last > config.lostGapSeconds, state != .lost else { return false }
        state = .lost
        if everTrusted { lostAfterTrusted = true }
        consecutiveGood = 0
        untrustedSince = now
        return true
    }

    /// Test one fix and advance the machine.
    public func update(
        _ sample: TrackerLocation,
        measured: Point,
        predicted: Point? = nil,
        predictedSpeed: Double = 0,
        predictedHeading: Double? = nil,
        covariance: Matrix2? = nil,
        roadDistanceM: Double? = nil
    ) -> GateDecision {
        var reasons: [String] = []
        let sigma = GPSGate.measurementSigma(sample, config: config)
        var distanceM: Double?
        var maxDistanceM: Double?
        var mahalanobis: Double?

        // Every gate is evaluated, even after one has already failed: the full
        // reason list is what makes a rejection diagnosable afterwards.
        if !sample.isUsable { reasons.append("invalid_fix") }

        if !config.allowSimulatedFixes && sample.isSimulatedBySoftware {
            reasons.append("simulated_by_software")
        }

        // Physical gate against the last accepted fix.
        if let previous = lastAccepted, let previousPoint = lastAcceptedPoint {
            let dt = sample.monotonicTime - previous.monotonicTime
            let distance = measured.distance(to: previousPoint)
            var speed = predictedSpeed
            if previous.hasValidSpeed { speed = Swift.max(speed, previous.speed ?? 0) }
            let gate = GPSGate.physical(
                distanceM: distance, dt: dt, previousSpeed: speed,
                config: config, maxAccel: maxAccel
            )
            distanceM = distance
            maxDistanceM = gate.maxDistance
            if !gate.passed { reasons.append("physical_gate") }
        }

        let tracking = state == .trusted || state == .suspect
        if let predicted, let covariance {
            let residual = measured - predicted
            var innovationCovariance = covariance
            if !tracking {
                let since = Swift.max(0, sample.monotonicTime - (untrustedSince ?? sample.monotonicTime))
                let driftSigma = config.recoveryPositionSigmaM
                    + config.recoveryPositionSigmaGrowthMPS * since
                innovationCovariance = innovationCovariance + Matrix2.diagonal(driftSigma * driftSigma)
            }
            let S = innovationCovariance + Matrix2.diagonal(sigma * sigma)
            let gate = GPSGate.mahalanobis(
                residual: residual, covariance: S, threshold: config.mahalanobisThreshold
            )
            mahalanobis = gate.distanceSquared
            if !gate.passed { reasons.append("mahalanobis_gate") }
        }

        if tracking {
            if sample.hasValidSpeed, bootstrapDone {
                if abs((sample.speed ?? 0) - predictedSpeed) > config.maxSpeedMismatchMS {
                    reasons.append("speed_mismatch")
                }
            }

            if let predictedHeading, sample.hasValidCourse, sample.hasValidSpeed,
               (sample.speed ?? 0) >= config.minSpeedForCourseMS,
               predictedSpeed >= config.minSpeedForCourseMS,
               bootstrapDone {
                let measuredHeading = Angles.headingFromCourse(sample.course ?? 0)
                let error = abs(Angles.wrap(measuredHeading - predictedHeading) * 180 / .pi)
                if error > config.maxCourseErrorDeg { reasons.append("course_mismatch") }
            }
        } else {
            reasons.append(contentsOf: fixToFixReasons(sample, measured: measured))
        }

        // The graph is a routing prior, not a GPS-quality sensor. Its coverage
        // may be clipped or stale, so roadDistanceM remains diagnostic only;
        // it must never turn a coherent live GPS stream into a fake outage.

        let accepted = reasons.isEmpty
        advance(accepted: accepted, sample: sample, measured: measured)

        let decision = GateDecision(
            monotonicTime: sample.monotonicTime,
            accepted: accepted,
            state: state,
            reasons: reasons,
            latitude: sample.latitude,
            longitude: sample.longitude,
            distanceM: distanceM,
            maxDistanceM: maxDistanceM,
            mahalanobis: mahalanobis,
            roadDistanceM: roadDistanceM,
            sigmaM: sigma
        )
        history.append(decision)
        if history.count > historyLimit { history.removeFirst(history.count - historyLimit) }
        lastSeenTime = sample.monotonicTime
        return decision
    }

    private func advance(accepted: Bool, sample: TrackerLocation, measured: Point) {
        if accepted {
            consecutiveBad = 0
            consecutiveGood += 1
            lastAccepted = sample
            lastAcceptedPoint = measured
            switch state {
            case .trusted:
                break
            case .lost:
                state = .recovering
            case .recovering, .suspect:
                if consecutiveGood >= config.recoverCount {
                    state = .trusted
                    bootstrapDone = true
                } else {
                    state = .recovering
                }
            }
            if consecutiveGood >= config.recoverCount {
                state = .trusted
                bootstrapDone = true
                untrustedSince = nil
            }
            if state == .trusted { everTrusted = true }
        } else {
            consecutiveGood = 0
            consecutiveBad += 1
            switch state {
            case .trusted:
                state = .suspect
                untrustedSince = sample.monotonicTime
            case .suspect where consecutiveBad >= config.suspectToLostCount:
                state = .lost
                if everTrusted { lostAfterTrusted = true }
            case .recovering:
                state = .lost
                if everTrusted { lostAfterTrusted = true }
            default:
                break
            }
        }
    }

    /// Consistency of this fix with the previous accepted one.
    ///
    /// Used while the filter has lost track, where the only trustworthy
    /// reference is the fix stream itself. A receiver that has just re-acquired
    /// emits points that scatter: their implied speed and bearing disagree
    /// wildly with what they report, so the chain of consecutive agreements that
    /// RECOVERING requires never forms.
    private func fixToFixReasons(_ sample: TrackerLocation, measured: Point) -> [String] {
        guard let previous = lastAccepted, let previousPoint = lastAcceptedPoint else { return [] }
        let dt = sample.monotonicTime - previous.monotonicTime
        if dt <= 0 { return ["out_of_order"] }
        // After a long hole the implied speed and bearing say nothing.
        if dt > config.lostGapSeconds { return [] }

        var reasons: [String] = []
        let delta = measured - previousPoint
        let step = delta.magnitude
        let impliedSpeed = step / dt

        if sample.hasValidSpeed,
           abs((sample.speed ?? 0) - impliedSpeed) > config.recoverySpeedMismatchMS {
            reasons.append("speed_inconsistent_with_previous_fix")
        }

        if sample.hasValidCourse, step >= config.recoveryMinStepM,
           impliedSpeed >= config.minSpeedForCourseMS {
            let impliedHeading = atan2(delta.y, delta.x)
            let measuredHeading = Angles.headingFromCourse(sample.course ?? 0)
            let error = abs(Angles.wrap(measuredHeading - impliedHeading) * 180 / .pi)
            if error > config.recoveryCourseErrorDeg {
                reasons.append("course_inconsistent_with_previous_fix")
            }
        }
        return reasons
    }

    /// Collapse the per-fix history into `[start, end]` intervals per state.
    public func stateIntervals() -> [LiveDiagnostics.OutageWindow] {
        var windows: [LiveDiagnostics.OutageWindow] = []
        for item in history {
            if var last = windows.last, last.state == item.state {
                last.endSeconds = item.monotonicTime
                windows[windows.count - 1] = last
            } else {
                windows.append(
                    LiveDiagnostics.OutageWindow(
                        startSeconds: item.monotonicTime,
                        endSeconds: item.monotonicTime,
                        state: item.state
                    )
                )
            }
        }
        return windows
    }
}
