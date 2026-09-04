import Combine
import CoreLocation
import Foundation
import LiveTracking
import SwiftUI
import UIKit

/// Drives one recording session and owns everything the main screen shows.
@MainActor
public final class TripRecorder: ObservableObject {
    public enum State: Equatable {
        case idle
        case calibrating
        case recording
        case stopped
    }

    @Published public private(set) var state: State = .idle
    @Published public private(set) var trip: TripDirectory?
    @Published public private(set) var route: [CLLocationCoordinate2D] = []
    @Published public private(set) var locationCount = 0
    @Published public private(set) var motionCount = 0
    @Published public private(set) var elapsed: TimeInterval = 0
    @Published public private(set) var horizontalAccuracy: Double?
    @Published public private(set) var currentSpeed: Double?
    @Published public private(set) var errorMessage: String?
    @Published public private(set) var savedTrips: [TripDirectory] = []
    @Published public private(set) var photoCount = 0
    @Published public private(set) var isFinishing = false

    /// True when the car has been at a standstill long enough that offering the
    /// camera is safe. The photo button is never shown while moving.
    @Published public private(set) var isStopped = false

    public let location: LocationRecorder
    public let motion: MotionRecorder
    public let calibration: CalibrationManager
    public let live: LiveTrackingRecorder
    private let store: TripStore
    private let timebase: Timebase

    private var writer: SampleWriter?
    private var startedAt: Date?
    private var startMonotonic: TimeInterval?
    private var timer: AnyCancellable?
    private var stationarySince: TimeInterval?
    private var liveChanges: AnyCancellable?
    private var lifecycleObservers: [NSObjectProtocol] = []

    /// How long the car must be below `stoppedSpeed` before the camera unlocks.
    public static let stoppedDwell: TimeInterval = 5.0
    public static let stoppedSpeed: CLLocationSpeed = 0.8

    public init(
        location: LocationRecorder? = nil,
        motion: MotionRecorder = MotionRecorder(),
        calibration: CalibrationManager = CalibrationManager(),
        store: TripStore = .shared,
        live: LiveTrackingRecorder = LiveTrackingRecorder(),
        timebase: Timebase = Timebase()
    ) {
        self.timebase = timebase
        self.location = location ?? LocationRecorder(timebase: timebase)
        self.motion = motion
        self.calibration = calibration
        self.store = store
        self.live = live
        liveChanges = live.objectWillChange.sink { [weak self] _ in
            self?.objectWillChange.send()
        }
        refreshSavedTrips()
        wireSensors()
        wireLifecycle()
    }

    private func wireSensors() {
        location.onSample = { [weak self] sample in
            Task { @MainActor in self?.handle(location: sample) }
        }
        motion.onSample = { [weak self] sample in
            // Called on the motion queue at 50 Hz: write straight through and
            // preserve raw input before attempting any live computation.
            self?.writer?.append(motion: sample)
            Task { @MainActor [weak self] in
                guard let self, self.state == .recording || self.isFinishing else { return }
                self.live.ingest(motion: sample)
            }
        }
    }

    private func wireLifecycle() {
        lifecycleObservers.append(NotificationCenter.default.addObserver(
            forName: UIApplication.didEnterBackgroundNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in
                self?.writer?.flush()
                await self?.live.persistNow()
            }
        })
    }

    // MARK: - Permissions

    public func requestPermissions() {
        location.requestAuthorization()
    }

    public func requestBackgroundPermission() {
        location.requestAlwaysAuthorization()
    }

    public var needsLocationPermission: Bool {
        location.authorization == .unknown || location.authorization == .denied
    }

    public var needsBackgroundPermission: Bool { location.authorization != .always }

    // MARK: - Calibration

    public func beginCalibration() {
        guard state == .idle || state == .stopped else { return }
        errorMessage = nil
        calibration.begin()
        state = .calibrating
        motion.start()
        location.start()
        startTicker()
    }

    public func skipCalibrationAndRecord() {
        // A failed/unfinished calibration used to fall through to recording and
        // write nil metadata.  That made an accidental tap indistinguishable
        // from a calibrated trip to the offline processor.
        guard calibration.calibration != nil, calibration.phase == .done else {
            errorMessage = "Calibration is required before recording. Keep the phone still, then drive straight until the green confirmation appears."
            return
        }
        startRecording()
    }

    public func cancelCalibration() {
        calibration.reset()
        motion.stop()
        location.stop()
        stopTicker()
        state = .idle
    }

    // MARK: - Recording

    public func startRecording() {
        guard state == .idle || state == .calibrating || state == .stopped else { return }
        guard calibration.phase == .done, let mountCalibration = calibration.calibration else {
            errorMessage = "Mount calibration must reach READY before recording can start."
            return
        }
        do {
            let directory = try store.createTrip()
            let writer = try SampleWriter(url: directory.samplesURL)
            self.trip = directory
            self.writer = writer
            self.route = []
            self.locationCount = 0
            self.motionCount = 0
            self.photoCount = 0
            self.elapsed = 0
            self.errorMessage = nil
            self.startedAt = Date()
            self.startMonotonic = timebase.now
            self.isFinishing = false
            self.isStopped = false
            self.stationarySince = nil

            // Write metadata up front so a trip cut short by a crash is still
            // identifiable rather than an anonymous directory.
            try store.writeMetadata(makeMetadata(), to: directory)

            // One pipeline is owned by one trip. It is created after a valid
            // mount calibration and before either sensor can deliver a sample.
            live.start(
                trip: directory,
                calibration: mountCalibration,
                at: startMonotonic ?? timebase.now
            )

            if !motion.isRecording { motion.start() }
            if !location.isRecording { location.start() }
            startTicker()
            state = .recording
        } catch {
            errorMessage = "Could not start the trip: \(error.localizedDescription)"
            state = .idle
        }
    }

    public func stopRecording() {
        guard state == .recording, !isFinishing else { return }
        isFinishing = true
        let endedAtMonotonic = timebase.now
        let endedAtWallClock = timebase.wallClock(from: endedAtMonotonic)
        location.stop()
        motion.stop()
        stopTicker()
        Task { @MainActor [weak self] in
            guard let self else { return }
            _ = await self.live.finish(at: endedAtMonotonic)
            self.completeStop(endedAt: endedAtWallClock)
        }
    }

    private func completeStop(endedAt: Date) {
        writer?.flush()
        let counts = writer?.counts() ?? (locations: 0, motions: 0)
        locationCount = counts.locations
        motionCount = counts.motions
        writer?.close()
        writer = nil
        if let trip {
            try? store.writeMetadata(makeMetadata(ended: endedAt), to: trip)
        }
        isFinishing = false
        isStopped = true
        state = .stopped
        refreshSavedTrips()
    }

    private func makeMetadata(ended: Date? = nil) -> TripMetadata {
        let counts = writer?.counts() ?? (locationCount, motionCount)
        return TripMetadata(
            tripId: trip?.id ?? UUID().uuidString,
            startedAt: startedAt,
            endedAt: ended,
            deviceModel: UIDevice.current.model,
            osVersion: UIDevice.current.systemVersion,
            appVersion: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString")
                as? String,
            calibration: calibration.calibration,
            locationSampleCount: counts.0,
            motionSampleCount: counts.1,
            notes: nil
        )
    }

    // MARK: - Sample handling

    private func handle(location sample: LocationSample) {
        horizontalAccuracy = sample.horizontalAccuracy
        currentSpeed = (sample.speed ?? -1) >= 0 ? sample.speed : nil

        if state == .calibrating {
            if !calibration.ingest(location: sample, motion: motion.currentMotion) {
                startRecording()
            }
            return
        }

        guard state == .recording else { return }
        writer?.append(location: sample)
        live.ingest(location: sample)
        route.append(sample.coordinate)
        locationCount += 1
        updateStopDetection(sample)
    }

    /// The car counts as stopped only after it has been slow for a while - a
    /// momentary reading of 0.3 m/s at a junction is not a parking event.
    private func updateStopDetection(_ sample: LocationSample) {
        let speed = sample.speed ?? -1
        if speed >= 0, speed < Self.stoppedSpeed {
            if stationarySince == nil { stationarySince = sample.monotonicTime }
            if let since = stationarySince,
               sample.monotonicTime - since >= Self.stoppedDwell {
                isStopped = true
            }
        } else if speed >= Self.stoppedSpeed {
            stationarySince = nil
            isStopped = false
        }
    }

    // MARK: - Timer

    private func startTicker() {
        timer = Timer.publish(every: 0.5, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in
                guard let self else { return }
                if let start = self.startMonotonic, self.state == .recording {
                    self.elapsed = self.timebase.now - start
                }
                if self.state == .calibrating, let motion = self.motion.currentMotion {
                    if !self.calibration.ingest(motion: motion, now: self.timebase.now),
                       self.calibration.phase == .done {
                        self.startRecording()
                    }
                }
                let counts = self.writer?.counts() ?? (self.locationCount, self.motionCount)
                self.motionCount = counts.1
            }
    }

    private func stopTicker() {
        timer?.cancel()
        timer = nil
    }

    // MARK: - Trips

    public func refreshSavedTrips() {
        savedTrips = store.listTrips()
        if let trip { photoCount = store.readPhotos(from: trip).count }
    }

    public func delete(_ trip: TripDirectory) {
        try? store.delete(trip)
        if trip.id == self.trip?.id { self.trip = nil }
        refreshSavedTrips()
    }

    public func metadata(for trip: TripDirectory) -> TripMetadata? {
        store.readMetadata(from: trip)
    }

    public func photos(for trip: TripDirectory) -> [PhotoRecord] {
        store.readPhotos(from: trip)
    }

    public func nextPhotoName() -> String {
        guard let trip else { return "photo-001" }
        return store.nextPhotoName(in: trip)
    }

    public func notePhotoSaved() {
        photoCount += 1
        refreshSavedTrips()
    }

    public var lastKnownCoordinate: CLLocationCoordinate2D? {
        location.latest?.coordinate ?? route.last
    }

    public var lastKnownAccuracy: Double? { location.latest?.horizontalAccuracy }
}
