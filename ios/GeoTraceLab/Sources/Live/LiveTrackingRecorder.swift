import CoreLocation
import CoreMotion
import Foundation
import LiveTracking
import os

/// Bridges the app's recording layer to the on-device tracking runtime.
///
/// Everything computational lives in the `LiveTracking` package, which has no
/// UIKit, CoreLocation or CoreMotion dependency so the identical code can be
/// replayed on a Mac and compared against the Python baseline. This type is the
/// only place the two worlds meet: it converts samples, applies the debug fault
/// injector to the *live copy only*, and streams results to disk.
///
/// No Python runs here, nothing is sent to a Mac and no network is required.
@MainActor
public final class LiveTrackingRecorder: ObservableObject {
    @Published public private(set) var snapshot = LiveSnapshot()
    @Published public private(set) var finalResult: LiveTrackingResult?
    @Published public private(set) var isRunning = false
    @Published public private(set) var graphStatus: GraphStatus = .notLoaded
    @Published public var faultMode: FaultInjector.Mode = .none {
        didSet { applyFaultMode() }
    }
    @Published public var faultDuration: Double = 45

    public enum GraphStatus: Equatable {
        case notLoaded
        case loaded(edges: Int)
        case missing(String)
        case outsideCoverage

        public var label: String {
            switch self {
            case .notLoaded: return "Road graph not loaded"
            case .loaded(let edges): return "Road graph loaded (\(edges) edges)"
            case .missing(let reason): return "Graph coverage missing: \(reason)"
            case .outsideCoverage: return "Outside graph coverage"
            }
        }

        public var isHealthy: Bool {
            if case .loaded = self { return true }
            return false
        }
    }

    private var pipeline: LiveTrackingPipeline?
    private var injector = FaultInjector()
    private var writer: LiveResultWriter?
    private var frame: LocalFrame?
    private var snapshotTimer: Timer?
    private var persistTimer: Timer?
    private var finalizingResult: LiveTrackingResult?
    private let logger = Logger(subsystem: "com.geotrace.GeoTraceLab", category: "LiveTracking")

    /// The bundled Saint Petersburg graph, converted offline from the same
    /// GraphML the Python baseline uses.
    public static var bundledGraphURL: URL? {
        Bundle.main.url(forResource: "spb", withExtension: "geograph")
    }

    public init() {}

    // MARK: - Lifecycle

    public func start(trip: TripDirectory, calibration: MountCalibration?, at monotonicTime: TimeInterval) {
        var config = LiveTrackingConfig()
        // On the phone the adaptive budget is what keeps a 5000-particle filter
        // affordable: it is only spent while GPS is actually failing.
        config.runtime.adaptiveParticleCounts = true

        let tracker = calibration.flatMap(Self.convert(calibration:))
        let pipeline = LiveTrackingPipeline(
            config: config,
            graphURL: Self.bundledGraphURL,
            graphRadiusM: 15_000,
            calibration: tracker,
            startedAt: monotonicTime
        )
        self.pipeline = pipeline
        self.writer = LiveResultWriter(trip: trip)
        self.finalResult = nil
        self.snapshot = LiveSnapshot()
        self.frame = nil
        self.finalizingResult = nil
        self.injector = FaultInjector()
        self.faultMode = .none
        self.isRunning = true

        if let graphURL = Self.bundledGraphURL,
                  let bounds = try? RoadGraphLoader.readBounds(at: graphURL) {
            graphStatus = .loaded(edges: bounds.edges)
        } else {
            graphStatus = .missing("spb.geograph is not in the app bundle")
        }
        writer?.writeGraphMetadata()

        startTimers()
    }

    public func finish(at monotonicTime: TimeInterval) async -> LiveTrackingResult? {
        if let finalizingResult { return finalizingResult }
        guard let pipeline else { return nil }
        stopTimers()
        isRunning = false
        let result = await pipeline.finish(at: monotonicTime)
        let records = await pipeline.drainRecords()
        writer?.append(records)
        writer?.writeFinal(result)
        if injector.isActive || injector.mode != .none {
            writer?.writeFaultManifest(injector.manifest())
        }
        writer?.close()
        finalResult = result
        finalizingResult = result
        snapshot = await pipeline.snapshot()
        updateGraphStatus(result.graphCoverage)
        return result
    }

    // MARK: - Sensor intake

    /// Feed a fix. The recorded stream is never modified; when a debug fault is
    /// armed, only the copy handed to the pipeline is corrupted.
    public func ingest(location sample: LocationSample) {
        guard let pipeline else { return }
        let raw = Self.convert(location: sample)
        if frame == nil {
            frame = LocalFrame(latitude: raw.latitude, longitude: raw.longitude)
        }
        guard let frame else { return }

        var injected: TrackerLocation?
        if injector.isActive {
            injected = injector.apply(to: raw, frame: frame)
            if injected == nil {
                // The raw point remains visible in the recorded trip and in
                // diagnostics, but the tracker experiences a genuine outage.
                Task { await pipeline.processDroppedLocation(raw: raw, fault: faultMode.rawValue) }
                return
            }
        }
        let forPipeline = injected ?? raw
        let original: TrackerLocation? = injected == raw ? nil : raw
        Task { await pipeline.processLocation(forPipeline, raw: original) }
    }

    public func ingest(motion sample: MotionSample) {
        guard let pipeline else { return }
        let converted = Self.convert(motion: sample)
        Task { await pipeline.processMotion(converted) }
    }

    // MARK: - Debug faults

    private func applyFaultMode() {
        guard isRunning else { return }
        if faultMode == .none {
            injector.disarm()
        } else {
            injector.arm(faultMode, at: ProcessInfo.processInfo.systemUptime, duration: faultDuration)
        }
        let label = faultMode.rawValue
        Task { await pipeline?.setFaultLabel(label) }
    }

    // MARK: - Timers

    private func startTimers() {
        snapshotTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor in await self?.refreshSnapshot() }
        }
        persistTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
            Task { @MainActor in await self?.persist() }
        }
    }

    private func stopTimers() {
        snapshotTimer?.invalidate(); snapshotTimer = nil
        persistTimer?.invalidate(); persistTimer = nil
    }

    private func refreshSnapshot() async {
        guard let pipeline else { return }
        snapshot = await pipeline.snapshot()
        updateGraphStatus(snapshot.graphCoverage)
    }

    /// Append what the pipeline has produced since last time.
    ///
    /// Done periodically rather than at the end so that a crash or a kill costs
    /// at most the last few seconds, not the whole trip.
    private func persist() async {
        guard let pipeline else { return }
        let records = await pipeline.drainRecords()
        guard !records.isEmpty else { return }
        writer?.append(records)
    }

    /// Used on background transition and immediately before finalization.
    /// It only drains completed records; it never advances either tracker.
    public func persistNow() async {
        await persist()
    }

    private func updateGraphStatus(_ coverage: GraphCoverage) {
        switch coverage {
        case .loaded:
            if case .loaded = graphStatus { return }
            graphStatus = .loaded(edges: 0)
        case .missing:
            graphStatus = .missing("no graph for this area")
        case .outside:
            graphStatus = .outsideCoverage
        }
    }

    // MARK: - Conversion

    static func convert(location sample: LocationSample) -> TrackerLocation {
        TrackerLocation(
            monotonicTime: sample.monotonicTime,
            latitude: sample.latitude,
            longitude: sample.longitude,
            horizontalAccuracy: sample.horizontalAccuracy,
            speed: sample.speed,
            speedAccuracy: sample.speedAccuracy,
            course: sample.course,
            courseAccuracy: sample.courseAccuracy,
            isSimulatedBySoftware: sample.sourceInformation?.isSimulatedBySoftware ?? false
        )
    }

    static func convert(motion sample: MotionSample) -> TrackerMotion {
        TrackerMotion(
            monotonicTime: sample.monotonicTime,
            userAccelerationG: SIMD3(
                sample.userAccelerationG[0], sample.userAccelerationG[1], sample.userAccelerationG[2]
            ),
            rotationRate: SIMD3(
                sample.rotationRate[0], sample.rotationRate[1], sample.rotationRate[2]
            ),
            gravity: SIMD3(sample.gravity[0], sample.gravity[1], sample.gravity[2]),
            quaternion: SIMD4(
                sample.quaternion[0], sample.quaternion[1],
                sample.quaternion[2], sample.quaternion[3]
            )
        )
    }

    static func convert(calibration: MountCalibration) -> TrackerCalibration? {
        guard calibration.referenceQuaternion.count == 4,
              calibration.gravityDevice.count == 3,
              calibration.forwardAxisDevice == nil || calibration.forwardAxisDevice?.count == 3
        else { return nil }
        return TrackerCalibration(
            referenceQuaternion: SIMD4(
                calibration.referenceQuaternion[0], calibration.referenceQuaternion[1],
                calibration.referenceQuaternion[2], calibration.referenceQuaternion[3]
            ),
            gravityDevice: SIMD3(
                calibration.gravityDevice[0], calibration.gravityDevice[1],
                calibration.gravityDevice[2]
            ),
            forwardAxisDevice: calibration.forwardAxisDevice.map {
                SIMD3($0[0], $0[1], $0[2])
            },
            initialHeadingDegrees: calibration.initialHeadingDeg,
            headingSource: calibration.headingSource
        )
    }
}
