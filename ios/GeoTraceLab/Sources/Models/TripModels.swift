import CoreLocation
import CoreMotion
import Foundation

/// Conversion factor for CoreMotion's `userAcceleration`, which is reported in g.
public let gravityMetresPerSecondSquared = 9.80665

// MARK: - Timebase

/// Shared monotonic clock for both sensor streams.
///
/// `CMDeviceMotion.timestamp` is seconds since device boot, on the same base as
/// `ProcessInfo.systemUptime`. `CLLocation.timestamp` is a wall-clock `Date`,
/// which can jump when the system clock is corrected. Both are written out on
/// one monotonic timeline so the processor can merge them; the wall clock is
/// kept alongside for human readability only.
public struct Timebase: Sendable {
    /// The instant the device booted, captured once so every conversion agrees.
    public let bootDate: Date

    public init(bootDate: Date? = nil) {
        self.bootDate = bootDate ?? Date(timeIntervalSinceNow: -ProcessInfo.processInfo.systemUptime)
    }

    public func monotonic(from date: Date) -> TimeInterval {
        date.timeIntervalSince(bootDate)
    }

    public func wallClock(from monotonic: TimeInterval) -> Date {
        bootDate.addingTimeInterval(monotonic)
    }

    public var now: TimeInterval { ProcessInfo.processInfo.systemUptime }
}

// MARK: - Samples

/// One CoreLocation fix, in the exact shape `samples.jsonl` expects.
public struct LocationSample: Codable, Equatable, Sendable {
    public var type = "location"
    public var wallTime: Date
    public var monotonicTime: TimeInterval
    public var latitude: CLLocationDegrees
    public var longitude: CLLocationDegrees
    public var altitude: CLLocationDistance?
    public var horizontalAccuracy: CLLocationAccuracy?
    public var verticalAccuracy: CLLocationAccuracy?
    public var speed: CLLocationSpeed?
    public var speedAccuracy: CLLocationSpeedAccuracy?
    public var course: CLLocationDirection?
    public var courseAccuracy: CLLocationDirectionAccuracy?
    public var sourceInformation: SourceInformation?

    public struct SourceInformation: Codable, Equatable, Sendable {
        public var isSimulatedBySoftware: Bool
        public var isProducedByAccessory: Bool

        enum CodingKeys: String, CodingKey {
            case isSimulatedBySoftware = "is_simulated_by_software"
            case isProducedByAccessory = "is_produced_by_accessory"
        }
    }

    enum CodingKeys: String, CodingKey {
        case type
        case wallTime = "wall_time"
        case monotonicTime = "monotonic_time"
        case latitude, longitude, altitude, speed, course
        case horizontalAccuracy = "horizontal_accuracy"
        case verticalAccuracy = "vertical_accuracy"
        case speedAccuracy = "speed_accuracy"
        case courseAccuracy = "course_accuracy"
        case sourceInformation = "source_information"
    }

    /// A fix CoreLocation itself marks as invalid, or one whose timestamp makes
    /// no sense, must never reach the filters.
    public var isUsable: Bool {
        guard let horizontalAccuracy, horizontalAccuracy >= 0 else { return false }
        guard latitude.isFinite, longitude.isFinite else { return false }
        guard abs(latitude) <= 90, abs(longitude) <= 180 else { return false }
        guard monotonicTime.isFinite, monotonicTime > 0 else { return false }
        return true
    }

    public var coordinate: CLLocationCoordinate2D {
        CLLocationCoordinate2D(latitude: latitude, longitude: longitude)
    }

    public init(location: CLLocation, timebase: Timebase) {
        self.wallTime = location.timestamp
        self.monotonicTime = timebase.monotonic(from: location.timestamp)
        self.latitude = location.coordinate.latitude
        self.longitude = location.coordinate.longitude
        self.altitude = location.altitude
        self.horizontalAccuracy = location.horizontalAccuracy
        self.verticalAccuracy = location.verticalAccuracy
        self.speed = location.speed
        self.speedAccuracy = location.speedAccuracy
        self.course = location.course
        self.courseAccuracy = location.courseAccuracy
        if let info = location.sourceInformation {
            self.sourceInformation = SourceInformation(
                isSimulatedBySoftware: info.isSimulatedBySoftware,
                isProducedByAccessory: info.isProducedByAccessory
            )
        }
    }

    /// Memberwise initialiser used by the tests and by imported data.
    public init(
        wallTime: Date,
        monotonicTime: TimeInterval,
        latitude: CLLocationDegrees,
        longitude: CLLocationDegrees,
        altitude: CLLocationDistance? = nil,
        horizontalAccuracy: CLLocationAccuracy? = nil,
        verticalAccuracy: CLLocationAccuracy? = nil,
        speed: CLLocationSpeed? = nil,
        speedAccuracy: CLLocationSpeedAccuracy? = nil,
        course: CLLocationDirection? = nil,
        courseAccuracy: CLLocationDirectionAccuracy? = nil,
        sourceInformation: SourceInformation? = nil
    ) {
        self.wallTime = wallTime
        self.monotonicTime = monotonicTime
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.horizontalAccuracy = horizontalAccuracy
        self.verticalAccuracy = verticalAccuracy
        self.speed = speed
        self.speedAccuracy = speedAccuracy
        self.course = course
        self.courseAccuracy = courseAccuracy
        self.sourceInformation = sourceInformation
    }
}

/// One `CMDeviceMotion` frame, device frame, recorded at 50 Hz.
public struct MotionSample: Codable, Equatable, Sendable {
    public var type = "motion"
    public var monotonicTime: TimeInterval
    /// CoreMotion reports this in g.
    public var userAccelerationG: [Double]
    /// The same vector in m/s^2, so the processor never has to guess the units.
    public var userAccelerationMS2: [Double]
    public var rotationRate: [Double]
    public var gravity: [Double]
    /// (w, x, y, z), device -> reference frame.
    public var quaternion: [Double]
    /// (roll, pitch, yaw) in radians, diagnostics only.
    public var attitude: [Double]?
    public var magneticField: [Double]?
    public var magneticAccuracy: Int?

    enum CodingKeys: String, CodingKey {
        case type, gravity, quaternion, attitude
        case monotonicTime = "monotonic_time"
        case userAccelerationG = "user_acceleration_g"
        case userAccelerationMS2 = "user_acceleration_ms2"
        case rotationRate = "rotation_rate"
        case magneticField = "magnetic_field"
        case magneticAccuracy = "magnetic_accuracy"
    }

    public init(motion: CMDeviceMotion) {
        let acceleration = motion.userAcceleration
        self.monotonicTime = motion.timestamp
        self.userAccelerationG = [acceleration.x, acceleration.y, acceleration.z]
        self.userAccelerationMS2 = [
            acceleration.x * gravityMetresPerSecondSquared,
            acceleration.y * gravityMetresPerSecondSquared,
            acceleration.z * gravityMetresPerSecondSquared,
        ]
        self.rotationRate = [motion.rotationRate.x, motion.rotationRate.y, motion.rotationRate.z]
        self.gravity = [motion.gravity.x, motion.gravity.y, motion.gravity.z]
        let q = motion.attitude.quaternion
        self.quaternion = [q.w, q.x, q.y, q.z]
        self.attitude = [motion.attitude.roll, motion.attitude.pitch, motion.attitude.yaw]

        let field = motion.magneticField
        // A field with accuracy `.uncalibrated` carries no usable information.
        if field.accuracy != .uncalibrated {
            self.magneticField = [field.field.x, field.field.y, field.field.z]
        }
        self.magneticAccuracy = Int(field.accuracy.rawValue)
    }

    public init(
        monotonicTime: TimeInterval,
        userAccelerationG: [Double],
        rotationRate: [Double],
        gravity: [Double],
        quaternion: [Double],
        attitude: [Double]? = nil,
        magneticField: [Double]? = nil,
        magneticAccuracy: Int? = nil
    ) {
        self.monotonicTime = monotonicTime
        self.userAccelerationG = userAccelerationG
        self.userAccelerationMS2 = userAccelerationG.map { $0 * gravityMetresPerSecondSquared }
        self.rotationRate = rotationRate
        self.gravity = gravity
        self.quaternion = quaternion
        self.attitude = attitude
        self.magneticField = magneticField
        self.magneticAccuracy = magneticAccuracy
    }
}

// MARK: - Calibration

/// How the phone sat in its holder for this trip.
///
/// The phone is assumed rigidly mounted. Vehicle heading is driven by the gyro
/// and corrected from a trusted GPS course; the magnetometer is recorded but is
/// deliberately not the primary heading source, because a car body distorts it.
public struct MountCalibration: Codable, Equatable, Sendable {
    public var referenceQuaternion: [Double]
    public var gravityDevice: [Double]
    /// Vehicle forward axis expressed in the device frame, from the
    /// straight-line calibration drive. Nil if it was not performed.
    public var forwardAxisDevice: [Double]?
    public var initialHeadingDeg: Double?
    /// "gps_course", "magnetometer" or "unknown".
    public var headingSource: String
    public var stillDurationS: Double
    public var capturedAt: Date?

    enum CodingKeys: String, CodingKey {
        case referenceQuaternion = "reference_quaternion"
        case gravityDevice = "gravity_device"
        case forwardAxisDevice = "forward_axis_device"
        case initialHeadingDeg = "initial_heading_deg"
        case headingSource = "heading_source"
        case stillDurationS = "still_duration_s"
        case capturedAt = "captured_at"
    }

    public init(
        referenceQuaternion: [Double] = [1, 0, 0, 0],
        gravityDevice: [Double] = [0, 0, -1],
        forwardAxisDevice: [Double]? = nil,
        initialHeadingDeg: Double? = nil,
        headingSource: String = "unknown",
        stillDurationS: Double = 0,
        capturedAt: Date? = nil
    ) {
        self.referenceQuaternion = referenceQuaternion
        self.gravityDevice = gravityDevice
        self.forwardAxisDevice = forwardAxisDevice
        self.initialHeadingDeg = initialHeadingDeg
        self.headingSource = headingSource
        self.stillDurationS = stillDurationS
        self.capturedAt = capturedAt
    }
}

// MARK: - Photos

public struct OCRLine: Codable, Equatable, Sendable, Identifiable {
    public var id: String { "\(text)-\(confidence)" }
    public var text: String
    public var confidence: Double
    /// [x, y, width, height] in Vision's normalised image coordinates.
    public var boundingBox: [Double]?

    enum CodingKeys: String, CodingKey {
        case text, confidence
        case boundingBox = "bounding_box"
    }

    public init(text: String, confidence: Double, boundingBox: [Double]? = nil) {
        self.text = text
        self.confidence = confidence
        self.boundingBox = boundingBox
    }
}

/// Sidecar written next to each stop photo.
///
/// `assumedLatitude` / `assumedLongitude` are the last position the app believed
/// in. They are explicitly not a verified address: without a reference image
/// database a single photograph cannot identify a place.
public struct PhotoRecord: Codable, Equatable, Sendable {
    public var imagePath: String
    public var capturedAt: Date?
    public var monotonicTime: TimeInterval?
    public var ocr: [OCRLine]
    public var cameraOrientation: String?
    public var deviceHeadingDeg: Double?
    public var assumedLatitude: Double?
    public var assumedLongitude: Double?
    public var assumedAccuracy: Double?
    public var exifLatitude: Double?
    public var exifLongitude: Double?
    public var note: String?

    enum CodingKeys: String, CodingKey {
        case ocr, note
        case imagePath = "image_path"
        case capturedAt = "captured_at"
        case monotonicTime = "monotonic_time"
        case cameraOrientation = "camera_orientation"
        case deviceHeadingDeg = "device_heading_deg"
        case assumedLatitude = "assumed_latitude"
        case assumedLongitude = "assumed_longitude"
        case assumedAccuracy = "assumed_accuracy"
        case exifLatitude = "exif_latitude"
        case exifLongitude = "exif_longitude"
    }

    public init(
        imagePath: String,
        capturedAt: Date? = nil,
        monotonicTime: TimeInterval? = nil,
        ocr: [OCRLine] = [],
        cameraOrientation: String? = nil,
        deviceHeadingDeg: Double? = nil,
        assumedLatitude: Double? = nil,
        assumedLongitude: Double? = nil,
        assumedAccuracy: Double? = nil,
        exifLatitude: Double? = nil,
        exifLongitude: Double? = nil,
        note: String? = nil
    ) {
        self.imagePath = imagePath
        self.capturedAt = capturedAt
        self.monotonicTime = monotonicTime
        self.ocr = ocr
        self.cameraOrientation = cameraOrientation
        self.deviceHeadingDeg = deviceHeadingDeg
        self.assumedLatitude = assumedLatitude
        self.assumedLongitude = assumedLongitude
        self.assumedAccuracy = assumedAccuracy
        self.exifLatitude = exifLatitude
        self.exifLongitude = exifLongitude
        self.note = note
    }
}

// MARK: - Metadata

public struct TripMetadata: Codable, Equatable, Sendable {
    public var tripId: String
    public var startedAt: Date?
    public var endedAt: Date?
    public var deviceModel: String?
    public var osVersion: String?
    public var appVersion: String?
    public var calibration: MountCalibration?
    public var locationSampleCount: Int
    public var motionSampleCount: Int
    public var notes: String?

    enum CodingKeys: String, CodingKey {
        case calibration, notes
        case tripId = "trip_id"
        case startedAt = "started_at"
        case endedAt = "ended_at"
        case deviceModel = "device_model"
        case osVersion = "os_version"
        case appVersion = "app_version"
        case locationSampleCount = "location_sample_count"
        case motionSampleCount = "motion_sample_count"
    }

    public init(
        tripId: String,
        startedAt: Date? = nil,
        endedAt: Date? = nil,
        deviceModel: String? = nil,
        osVersion: String? = nil,
        appVersion: String? = nil,
        calibration: MountCalibration? = nil,
        locationSampleCount: Int = 0,
        motionSampleCount: Int = 0,
        notes: String? = nil
    ) {
        self.tripId = tripId
        self.startedAt = startedAt
        self.endedAt = endedAt
        self.deviceModel = deviceModel
        self.osVersion = osVersion
        self.appVersion = appVersion
        self.calibration = calibration
        self.locationSampleCount = locationSampleCount
        self.motionSampleCount = motionSampleCount
        self.notes = notes
    }
}

// MARK: - JSON coding

public enum TripJSON {
    /// ISO-8601 with milliseconds and a trailing Z, matching the schema.
    public static let dateFormatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        return formatter
    }()

    public static func encoder(prettyPrinted: Bool = false) -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .custom { date, encoder in
            var container = encoder.singleValueContainer()
            try container.encode(dateFormatter.string(from: date))
        }
        if prettyPrinted {
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        } else {
            encoder.outputFormatting = [.withoutEscapingSlashes]
        }
        return encoder
    }

    public static func decoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .custom { decoder in
            let text = try decoder.singleValueContainer().decode(String.self)
            if let date = dateFormatter.date(from: text) { return date }
            // Tolerate timestamps written without fractional seconds.
            let fallback = ISO8601DateFormatter()
            fallback.formatOptions = [.withInternetDateTime]
            if let date = fallback.date(from: text) { return date }
            throw DecodingError.dataCorruptedError(
                in: try decoder.singleValueContainer(),
                debugDescription: "unrecognised timestamp: \(text)"
            )
        }
        return decoder
    }
}
