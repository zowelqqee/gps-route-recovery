import CoreMotion
import Foundation
import os

/// `CMDeviceMotion` at 50 Hz.
///
/// The reference frame is `.xArbitraryCorrectedZVertical`: Z is gravity-aligned
/// and the yaw reference is corrected by the magnetometer *drift-wise* without
/// being anchored to magnetic north. Inside a car the absolute magnetic heading
/// is unreliable, so the processor recovers the true yaw offset once, from a
/// trusted GPS course, and integrates the gyro from there.
public final class MotionRecorder: ObservableObject {
    public static let sampleRateHz: Double = 50

    @Published public private(set) var isRecording = false
    @Published public private(set) var sampleCount = 0
    @Published public private(set) var latest: MotionSample?
    @Published public private(set) var isAvailable: Bool

    /// Called for every frame, on the motion queue - not the main thread.
    public var onSample: ((MotionSample) -> Void)?

    private let manager: CMMotionManager
    private let queue: OperationQueue
    private let logger = Logger(subsystem: "com.geotrace.GeoTraceLab", category: "Motion")

    public init(manager: CMMotionManager = CMMotionManager()) {
        self.manager = manager
        self.queue = OperationQueue()
        queue.name = "com.geotrace.motion"
        queue.maxConcurrentOperationCount = 1
        queue.qualityOfService = .userInitiated
        self.isAvailable = manager.isDeviceMotionAvailable
    }

    public func start() {
        guard manager.isDeviceMotionAvailable else {
            isAvailable = false
            logger.error("device motion unavailable (simulator or unsupported hardware)")
            return
        }
        guard !isRecording else { return }
        sampleCount = 0
        manager.deviceMotionUpdateInterval = 1.0 / Self.sampleRateHz
        manager.startDeviceMotionUpdates(
            using: .xArbitraryCorrectedZVertical, to: queue
        ) { [weak self] motion, error in
            guard let self else { return }
            if let error {
                self.logger.error("motion failure: \(error.localizedDescription, privacy: .public)")
                return
            }
            guard let motion else { return }
            let sample = MotionSample(motion: motion)
            self.onSample?(sample)
            // Publishing every frame at 50 Hz would thrash SwiftUI; the counter
            // is only for the on-screen readout.
            DispatchQueue.main.async {
                self.sampleCount += 1
                self.latest = sample
            }
        }
        isRecording = true
    }

    public func stop() {
        guard isRecording else { return }
        manager.stopDeviceMotionUpdates()
        isRecording = false
    }

    /// A single current frame, used by calibration.
    public var currentMotion: CMDeviceMotion? { manager.deviceMotion }
}
