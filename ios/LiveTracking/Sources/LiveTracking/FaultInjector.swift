import Foundation

/// Debug-only GPS corruption, applied *only* to the stream the live pipeline
/// sees.
///
/// The point is to be able to drive with a perfectly good receiver and still
/// watch what the algorithm does when GPS fails. The raw `CLLocation` stream is
/// never modified: the recorder writes it to disk untouched, and what was
/// injected is written alongside it as a manifest, so a later Python run
/// reproduces exactly the same experiment.
public struct FaultInjector: Sendable {
    public enum Mode: String, Sendable, Codable, CaseIterable {
        case none = "None"
        case dropout = "Dropout"
        case offset = "Offset"
        case drift = "Drift"
        case jumps = "Jumps"
        case composite = "Composite"
    }

    public var mode: Mode = .none
    /// When the operator armed the current mode, on the monotonic clock.
    public var armedAt: Double = 0
    public var durationSeconds: Double = 45
    public var offsetEast: Double = 900
    public var offsetNorth: Double = -600
    public var driftEastMS: Double = 3
    public var driftNorthMS: Double = 0
    public var jumpSigmaM: Double = 250
    public var reportedAccuracyM: Double = 12
    public var seed: UInt64 = 7

    private var rng: ParityRNG

    public init() {
        rng = ParityRNG(seed: 7)
    }

    public mutating func arm(_ mode: Mode, at time: Double, duration: Double = 45) {
        self.mode = mode
        self.armedAt = time
        self.durationSeconds = duration
        self.rng = ParityRNG(seed: seed)
    }

    public mutating func disarm() {
        mode = .none
    }

    public var isActive: Bool { mode != .none }

    /// Elapsed time inside the armed window, or nil when the fault is not
    /// currently applying.
    private func phase(at time: Double) -> Double? {
        guard mode != .none else { return nil }
        let elapsed = time - armedAt
        guard elapsed >= 0 else { return nil }
        switch mode {
        case .composite:
            // offset -> dropout -> false recovery, the failure a real receiver
            // in a city actually produces.
            return elapsed <= durationSeconds * 2.2 ? elapsed : nil
        default:
            return elapsed <= durationSeconds ? elapsed : nil
        }
    }

    /// Apply the armed fault to one fix.
    ///
    /// Returns nil when the fault suppresses the fix entirely (a dropout), which
    /// is what makes the live pipeline experience a real outage.
    public mutating func apply(
        to sample: TrackerLocation, frame: LocalFrame
    ) -> TrackerLocation? {
        guard let elapsed = phase(at: sample.monotonicTime) else { return sample }
        var point = frame.toLocal(latitude: sample.latitude, longitude: sample.longitude)
        var output = sample

        switch mode {
        case .none:
            return sample
        case .dropout:
            return nil
        case .offset:
            point.x += offsetEast
            point.y += offsetNorth
        case .drift:
            point.x += driftEastMS * elapsed
            point.y += driftNorthMS * elapsed
        case .jumps:
            point.x += rng.normal(sigma: jumpSigmaM)
            point.y += rng.normal(sigma: jumpSigmaM)
        case .composite:
            let offsetPhase = durationSeconds * 0.5
            let dropoutPhase = durationSeconds * 1.6
            if elapsed < offsetPhase {
                point.x += offsetEast
                point.y += offsetNorth
            } else if elapsed < dropoutPhase {
                return nil
            } else {
                // A receiver coming back emits a handful of confident-looking
                // but wrong fixes before it settles.
                let decay = Swift.max(0, 1 - (elapsed - dropoutPhase) / 6.0)
                point.x += rng.normal(sigma: Swift.max(20, jumpSigmaM * decay))
                point.y += rng.normal(sigma: Swift.max(20, jumpSigmaM * decay))
            }
        }

        let corrupted = frame.toGeo(point)
        output.latitude = corrupted.latitude
        output.longitude = corrupted.longitude
        // A lying receiver still reports a good accuracy; that is precisely why
        // horizontalAccuracy alone can never be the test.
        output.horizontalAccuracy = reportedAccuracyM
        return output
    }

    /// What was applied, for `fault-manifest.json`.
    public func manifest() -> [String: String] {
        [
            "mode": mode.rawValue,
            "armed_at_monotonic": String(armedAt),
            "duration_s": String(durationSeconds),
            "offset_east_m": String(offsetEast),
            "offset_north_m": String(offsetNorth),
            "drift_east_ms": String(driftEastMS),
            "drift_north_ms": String(driftNorthMS),
            "jump_sigma_m": String(jumpSigmaM),
            "reported_accuracy_m": String(reportedAccuracyM),
            "seed": String(seed),
            "note": "Applied only to the stream fed to the live pipeline. "
                + "The recorded raw GPS in samples.jsonl is untouched.",
        ]
    }
}
