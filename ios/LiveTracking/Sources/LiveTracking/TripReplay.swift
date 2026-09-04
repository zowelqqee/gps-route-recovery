import Foundation

/// Reads a recorded trip directory and replays it through the live pipeline.
///
/// Same code path as the phone: the samples are pushed in timestamp order
/// exactly as CoreLocation and CoreMotion would deliver them. That is what makes
/// a Mac-side replay a meaningful check of what the device will do, and what
/// lets the result be compared against the Python baseline.
public enum TripReplay {
    public struct Trip {
        public var metadata: [String: Any]
        public var locations: [TrackerLocation]
        public var motions: [TrackerMotion]
        public var calibration: TrackerCalibration?
        public var malformedLines: Int
        public var startedAt: Double
        public var endedAt: Double
    }

    public enum ReplayError: LocalizedError {
        case missingSamples(String)
        case noUsableFix

        public var errorDescription: String? {
            switch self {
            case .missingSamples(let path):
                return "\(path) contains no samples.jsonl"
            case .noUsableFix:
                return "the trip contains no usable GPS fix"
            }
        }
    }

    /// Parse `samples.jsonl`, tolerating a truncated final line: the phone
    /// appends to it while recording, so the last line can be cut short if the
    /// app was killed.
    public static func load(directory: URL) throws -> Trip {
        let samplesURL = directory.appendingPathComponent("samples.jsonl")
        guard let handle = FileHandle(forReadingAtPath: samplesURL.path) else {
            throw ReplayError.missingSamples(directory.path)
        }
        defer { try? handle.close() }

        var locations: [TrackerLocation] = []
        var motions: [TrackerMotion] = []
        var malformed = 0

        let data = try handle.readToEnd() ?? Data()
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            var lineStart = 0
            for index in 0..<raw.count where raw[index] == 0x0A {
                let slice = Data(raw[lineStart..<index])
                lineStart = index + 1
                guard !slice.isEmpty else { continue }
                guard let object = try? JSONSerialization.jsonObject(with: slice)
                        as? [String: Any] else {
                    malformed += 1
                    continue
                }
                switch object["type"] as? String {
                case "location":
                    if let sample = decodeLocation(object) { locations.append(sample) }
                    else { malformed += 1 }
                case "motion":
                    if let sample = decodeMotion(object) { motions.append(sample) }
                    else { malformed += 1 }
                default:
                    malformed += 1
                }
            }
            if lineStart < raw.count { malformed += 1 }
        }

        locations.sort { $0.monotonicTime < $1.monotonicTime }
        motions.sort { $0.monotonicTime < $1.monotonicTime }

        var metadata: [String: Any] = [:]
        var calibration: TrackerCalibration?
        let metadataURL = directory.appendingPathComponent("metadata.json")
        if let metadataData = try? Data(contentsOf: metadataURL),
           let object = try? JSONSerialization.jsonObject(with: metadataData) as? [String: Any] {
            metadata = object
            if let raw = object["calibration"] as? [String: Any] {
                calibration = decodeCalibration(raw)
            }
        }

        let times = locations.map(\.monotonicTime) + motions.map(\.monotonicTime)
        guard let first = times.min(), let last = times.max() else {
            throw ReplayError.noUsableFix
        }

        return Trip(
            metadata: metadata, locations: locations, motions: motions,
            calibration: calibration, malformedLines: malformed,
            startedAt: first, endedAt: last
        )
    }

    static func decodeLocation(_ object: [String: Any]) -> TrackerLocation? {
        guard let t = object["monotonic_time"] as? Double,
              let lat = object["latitude"] as? Double,
              let lon = object["longitude"] as? Double
        else { return nil }
        let source = object["source_information"] as? [String: Any]
        return TrackerLocation(
            monotonicTime: t,
            latitude: lat,
            longitude: lon,
            horizontalAccuracy: object["horizontal_accuracy"] as? Double,
            speed: object["speed"] as? Double,
            speedAccuracy: object["speed_accuracy"] as? Double,
            course: object["course"] as? Double,
            courseAccuracy: object["course_accuracy"] as? Double,
            isSimulatedBySoftware: (source?["is_simulated_by_software"] as? Bool) ?? false
        )
    }

    static func decodeMotion(_ object: [String: Any]) -> TrackerMotion? {
        guard let t = object["monotonic_time"] as? Double else { return nil }
        func vector3(_ key: String) -> SIMD3<Double>? {
            guard let values = object[key] as? [Double], values.count >= 3 else { return nil }
            return SIMD3(values[0], values[1], values[2])
        }
        guard let acceleration = vector3("user_acceleration_g"),
              let rotation = vector3("rotation_rate")
        else { return nil }
        var quaternion = SIMD4<Double>(1, 0, 0, 0)
        if let values = object["quaternion"] as? [Double], values.count >= 4 {
            quaternion = SIMD4(values[0], values[1], values[2], values[3])
        }
        return TrackerMotion(
            monotonicTime: t,
            userAccelerationG: acceleration,
            rotationRate: rotation,
            gravity: vector3("gravity") ?? SIMD3(0, 0, -1),
            quaternion: quaternion
        )
    }

    static func decodeCalibration(_ object: [String: Any]) -> TrackerCalibration? {
        func vector3(_ key: String) -> SIMD3<Double>? {
            guard let values = object[key] as? [Double], values.count >= 3 else { return nil }
            return SIMD3(values[0], values[1], values[2])
        }
        var quaternion = SIMD4<Double>(1, 0, 0, 0)
        if let values = object["reference_quaternion"] as? [Double], values.count >= 4 {
            quaternion = SIMD4(values[0], values[1], values[2], values[3])
        }
        return TrackerCalibration(
            referenceQuaternion: quaternion,
            gravityDevice: vector3("gravity_device") ?? SIMD3(0, 0, -1),
            forwardAxisDevice: vector3("forward_axis_device"),
            initialHeadingDegrees: object["initial_heading_deg"] as? Double,
            headingSource: (object["heading_source"] as? String) ?? "unknown"
        )
    }

    /// Push a whole trip through the pipeline in timestamp order.
    ///
    /// The two streams are merged rather than run one after the other, because
    /// the order in which a fix and the surrounding IMU arrive is exactly what
    /// determines the filter's behaviour.
    public static func run(
        trip: Trip, pipeline: LiveTrackingPipeline
    ) async -> LiveTrackingResult {
        await pipeline.start(at: trip.startedAt)
        var locationIndex = 0
        var motionIndex = 0
        while locationIndex < trip.locations.count || motionIndex < trip.motions.count {
            let nextLocation = locationIndex < trip.locations.count
                ? trip.locations[locationIndex].monotonicTime : Double.infinity
            let nextMotion = motionIndex < trip.motions.count
                ? trip.motions[motionIndex].monotonicTime : Double.infinity
            if nextMotion <= nextLocation {
                await pipeline.processMotion(trip.motions[motionIndex])
                motionIndex += 1
            } else {
                await pipeline.processLocation(trip.locations[locationIndex])
                locationIndex += 1
            }
        }
        return await pipeline.finish(at: trip.endedAt)
    }
}
