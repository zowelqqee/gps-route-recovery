import Foundation

/// Input and output types for the tracking runtime.
///
/// Deliberately free of CoreLocation and CoreMotion: the same code runs inside
/// the app on a phone and inside the replay harness on a Mac, and it is
/// compared against a Python implementation. The app converts its
/// CoreLocation/CoreMotion samples into these on the way in.
///
/// All timestamps are seconds on one monotonic clock (the device uptime clock),
/// which is the only timeline the filters ever use.

// MARK: - Sensor input

public struct TrackerLocation: Sendable, Equatable, Codable {
    public var monotonicTime: Double
    public var latitude: Double
    public var longitude: Double
    public var horizontalAccuracy: Double?
    /// CoreLocation reports an unsigned magnitude, negative when invalid.
    public var speed: Double?
    public var speedAccuracy: Double?
    /// Degrees clockwise from true north, negative when invalid.
    public var course: Double?
    public var courseAccuracy: Double?
    public var isSimulatedBySoftware: Bool

    public init(
        monotonicTime: Double,
        latitude: Double,
        longitude: Double,
        horizontalAccuracy: Double? = nil,
        speed: Double? = nil,
        speedAccuracy: Double? = nil,
        course: Double? = nil,
        courseAccuracy: Double? = nil,
        isSimulatedBySoftware: Bool = false
    ) {
        self.monotonicTime = monotonicTime
        self.latitude = latitude
        self.longitude = longitude
        self.horizontalAccuracy = horizontalAccuracy
        self.speed = speed
        self.speedAccuracy = speedAccuracy
        self.course = course
        self.courseAccuracy = courseAccuracy
        self.isSimulatedBySoftware = isSimulatedBySoftware
    }

    /// Mirrors `LocationSample.is_usable`: CoreLocation marks an invalid fix
    /// with a negative accuracy, and those must never reach a filter.
    public var isUsable: Bool {
        guard let accuracy = horizontalAccuracy, accuracy >= 0 else { return false }
        guard latitude.isFinite, longitude.isFinite else { return false }
        guard abs(latitude) <= 90, abs(longitude) <= 180 else { return false }
        guard monotonicTime.isFinite, monotonicTime > 0 else { return false }
        return true
    }

    public var hasValidCourse: Bool {
        guard let course, course >= 0 else { return false }
        return true
    }

    public var hasValidSpeed: Bool {
        guard let speed, speed >= 0 else { return false }
        return true
    }

    public var coordinate: Coordinate {
        Coordinate(latitude: latitude, longitude: longitude)
    }
}

public struct TrackerMotion: Sendable, Equatable {
    public var monotonicTime: Double
    /// CoreMotion userAcceleration, in g, device frame.
    public var userAccelerationG: SIMD3<Double>
    /// Body rates, rad/s, device frame.
    public var rotationRate: SIMD3<Double>
    public var gravity: SIMD3<Double>
    /// (w, x, y, z), device -> CMDeviceMotion reference frame.
    public var quaternion: SIMD4<Double>

    public init(
        monotonicTime: Double,
        userAccelerationG: SIMD3<Double>,
        rotationRate: SIMD3<Double>,
        gravity: SIMD3<Double> = SIMD3(0, 0, -1),
        quaternion: SIMD4<Double> = SIMD4(1, 0, 0, 0)
    ) {
        self.monotonicTime = monotonicTime
        self.userAccelerationG = userAccelerationG
        self.rotationRate = rotationRate
        self.gravity = gravity
        self.quaternion = quaternion
    }

    public var userAccelerationMS2: SIMD3<Double> {
        userAccelerationG * gravityMetresPerSecondSquared
    }
}

/// How the phone sat in its holder. Mirrors the app's `MountCalibration` and
/// the Python `MountCalibration`.
public struct TrackerCalibration: Sendable, Equatable, Codable {
    public var referenceQuaternion: SIMD4<Double>
    public var gravityDevice: SIMD3<Double>
    /// Vehicle forward axis in the device frame, from the straight-line
    /// calibration drive. Nil when that step was skipped, in which case the
    /// heading alignment falls back to the acceleration-direction estimate and
    /// confidence is reduced.
    public var forwardAxisDevice: SIMD3<Double>?
    public var initialHeadingDegrees: Double?
    public var headingSource: String

    public init(
        referenceQuaternion: SIMD4<Double> = SIMD4(1, 0, 0, 0),
        gravityDevice: SIMD3<Double> = SIMD3(0, 0, -1),
        forwardAxisDevice: SIMD3<Double>? = nil,
        initialHeadingDegrees: Double? = nil,
        headingSource: String = "unknown"
    ) {
        self.referenceQuaternion = referenceQuaternion
        self.gravityDevice = gravityDevice
        self.forwardAxisDevice = forwardAxisDevice
        self.initialHeadingDegrees = initialHeadingDegrees
        self.headingSource = headingSource
    }

    /// A calibration is only usable for the device->vehicle transform when the
    /// straight-line step actually produced a forward axis.
    public var definesVehicleFrame: Bool { forwardAxisDevice != nil }
}

// MARK: - GPS state

public enum GPSState: String, Sendable, Codable, CaseIterable {
    case trusted = "TRUSTED"
    case suspect = "SUSPECT"
    case lost = "LOST"
    case recovering = "RECOVERING"
}

/// The outcome of testing one fix. Mirrors `GateResult`.
public struct GateDecision: Sendable, Codable, Equatable {
    public var monotonicTime: Double
    public var accepted: Bool
    public var state: GPSState
    /// Every gate that failed, not just the first: a rejection has to be
    /// diagnosable afterwards.
    public var reasons: [String]
    public var distanceM: Double?
    public var maxDistanceM: Double?
    public var mahalanobis: Double?
    public var roadDistanceM: Double?
    public var sigmaM: Double?
    public var latitude: Double
    public var longitude: Double

    public init(
        monotonicTime: Double, accepted: Bool, state: GPSState, reasons: [String],
        latitude: Double, longitude: Double,
        distanceM: Double? = nil, maxDistanceM: Double? = nil,
        mahalanobis: Double? = nil, roadDistanceM: Double? = nil, sigmaM: Double? = nil
    ) {
        self.monotonicTime = monotonicTime
        self.accepted = accepted
        self.state = state
        self.reasons = reasons
        self.latitude = latitude
        self.longitude = longitude
        self.distanceM = distanceM
        self.maxDistanceM = maxDistanceM
        self.mahalanobis = mahalanobis
        self.roadDistanceM = roadDistanceM
        self.sigmaM = sigmaM
    }
}

// MARK: - Results

public enum ParkingStatus: String, Sendable, Codable {
    case confident = "CONFIDENT"
    case probable = "PROBABLE"
    case uncertain = "UNCERTAIN"
    case insufficientData = "INSUFFICIENT_DATA"
}

/// One branch of the road belief: a corridor and the probability it holds.
public struct ConfidenceComponent: Sendable, Codable, Equatable {
    public var componentID: String
    public var probability: Double
    public var areaM2: Double
    /// Closed rings in WGS84. One branch is usually a single corridor, but a
    /// belief that still straddles a junction is a genuine Y and is reported as
    /// several rings in the same component rather than being hulled over.
    public var rings: [[Coordinate]]
    public var streetNames: [String]

    public init(
        componentID: String, probability: Double, areaM2: Double,
        rings: [[Coordinate]], streetNames: [String]
    ) {
        self.componentID = componentID
        self.probability = probability
        self.areaM2 = areaM2
        self.rings = rings
        self.streetNames = streetNames
    }
}

public struct RoadTrackingResult: Sendable, Codable, Equatable {
    public var endpoint: Coordinate?
    public var endpointLocal: Point?
    public var heading: Double?
    public var speed: Double
    public var edgeIndex: Int?
    public var edgeName: String?
    public var confidence: Double
    public var effectiveSampleSize: Double
    public var particleCount: Int
    public var branchCount: Int
    public var components: [ConfidenceComponent]
    public var gpsState: GPSState
    public var initialized: Bool

    public init(
        endpoint: Coordinate? = nil, endpointLocal: Point? = nil, heading: Double? = nil,
        speed: Double = 0, edgeIndex: Int? = nil, edgeName: String? = nil,
        confidence: Double = 0, effectiveSampleSize: Double = 0, particleCount: Int = 0,
        branchCount: Int = 0, components: [ConfidenceComponent] = [],
        gpsState: GPSState = .lost, initialized: Bool = false
    ) {
        self.endpoint = endpoint
        self.endpointLocal = endpointLocal
        self.heading = heading
        self.speed = speed
        self.edgeIndex = edgeIndex
        self.edgeName = edgeName
        self.confidence = confidence
        self.effectiveSampleSize = effectiveSampleSize
        self.particleCount = particleCount
        self.branchCount = branchCount
        self.components = components
        self.gpsState = gpsState
        self.initialized = initialized
    }
}

public struct ParkingTrackingResult: Sendable, Codable, Equatable {
    public var status: ParkingStatus
    /// Nil when there is genuinely not enough data. In that case nothing else
    /// may be drawn in its place - in particular not the road endpoint.
    public var position: Coordinate?
    public var positionLocal: Point?
    public var confidence: Double
    public var polygonRadiusM: Double
    /// Circular corridor around `position`, in WGS84.
    public var polygon: [Coordinate]
    public var terminalClusterFixes: Int
    public var rejectedFixes: Int
    public var hasReverseMotion: Bool
    public var reason: String
    public var maxDistanceFromRoadM: Double?
    public var trajectory: [Coordinate]
    public var lastFixSecondsBeforeEnd: Double?

    public init(
        status: ParkingStatus, position: Coordinate? = nil, positionLocal: Point? = nil,
        confidence: Double = 0, polygonRadiusM: Double = 0, polygon: [Coordinate] = [],
        terminalClusterFixes: Int = 0, rejectedFixes: Int = 0, hasReverseMotion: Bool = false,
        reason: String = "", maxDistanceFromRoadM: Double? = nil,
        trajectory: [Coordinate] = [], lastFixSecondsBeforeEnd: Double? = nil
    ) {
        self.status = status
        self.position = position
        self.positionLocal = positionLocal
        self.confidence = confidence
        self.polygonRadiusM = polygonRadiusM
        self.polygon = polygon
        self.terminalClusterFixes = terminalClusterFixes
        self.rejectedFixes = rejectedFixes
        self.hasReverseMotion = hasReverseMotion
        self.reason = reason
        self.maxDistanceFromRoadM = maxDistanceFromRoadM
        self.trajectory = trajectory
        self.lastFixSecondsBeforeEnd = lastFixSecondsBeforeEnd
    }
}

/// What the phone shows and exports when the trip ends.
public struct LiveTrackingResult: Sendable, Codable, Equatable {
    public var roadResult: RoadTrackingResult
    public var parkingResult: ParkingTrackingResult
    /// Always the parking tracker's position. There is deliberately no hidden
    /// fallback to the road endpoint: the road tracker is constrained to the
    /// carriageway and cannot represent a car parked in a courtyard.
    public var finalVehiclePosition: Coordinate?
    public var finalVehiclePositionSource: String
    public var endedAtMonotonic: Double
    public var calibrationPresent: Bool
    public var graphCoverage: GraphCoverage
    public var diagnostics: LiveDiagnostics

    public init(
        roadResult: RoadTrackingResult, parkingResult: ParkingTrackingResult,
        endedAtMonotonic: Double, calibrationPresent: Bool,
        graphCoverage: GraphCoverage, diagnostics: LiveDiagnostics
    ) {
        self.roadResult = roadResult
        self.parkingResult = parkingResult
        self.finalVehiclePosition = parkingResult.position
        self.finalVehiclePositionSource = "parking_tracker"
        self.endedAtMonotonic = endedAtMonotonic
        self.calibrationPresent = calibrationPresent
        self.graphCoverage = graphCoverage
        self.diagnostics = diagnostics
    }
}

public enum GraphCoverage: String, Sendable, Codable {
    case loaded = "ROAD_GRAPH_LOADED"
    case missing = "GRAPH_COVERAGE_MISSING"
    case outside = "OUTSIDE_GRAPH_COVERAGE"
}

public struct LiveDiagnostics: Sendable, Codable, Equatable {
    public var locationSamples: Int = 0
    public var motionSamples: Int = 0
    public var acceptedFixes: Int = 0
    public var rejectedFixes: Int = 0
    public var rejectionReasons: [String: Int] = [:]
    public var stateTransitions: [StateTransition] = []
    public var outageWindows: [OutageWindow] = []
    public var reanchors: [Double] = []
    public var resampleCount: Int = 0
    public var reinitializations: Int = 0
    public var deadEndParticles: Int = 0
    public var junctionTransitions: Int = 0
    public var skippedGaps: Int = 0
    public var droppedOutOfOrderSamples: Int = 0
    public var droppedAfterEndSamples: Int = 0
    public var roadStepMeanMS: Double = 0
    public var roadStepP95MS: Double = 0
    public var parkingStepMeanMS: Double = 0
    public var parkingStepP95MS: Double = 0
    public var medianTrackToRoadM: Double?
    public var initialHeadingDegrees: Double?
    public var initialAccelBias: Double = 0
    public var initialGyroBias: Double = 0

    public init() {}

    public struct StateTransition: Sendable, Codable, Equatable {
        public var monotonicTime: Double
        public var state: GPSState
        public init(monotonicTime: Double, state: GPSState) {
            self.monotonicTime = monotonicTime
            self.state = state
        }
    }

    public struct OutageWindow: Sendable, Codable, Equatable {
        public var startSeconds: Double
        public var endSeconds: Double
        public var state: GPSState
        public init(startSeconds: Double, endSeconds: Double, state: GPSState) {
            self.startSeconds = startSeconds
            self.endSeconds = endSeconds
            self.state = state
        }
    }
}

/// Immutable snapshot handed to SwiftUI / MapKit. Everything the live screen
/// needs, computed off the main thread and published at a fixed low rate.
public struct LiveSnapshot: Sendable, Equatable {
    public var monotonicTime: Double = 0
    public var elapsed: Double = 0
    public var gpsState: GPSState = .lost
    public var horizontalAccuracy: Double?
    public var speedMS: Double?
    public var rawTrack: [Coordinate] = []
    public var injectedTrack: [Coordinate] = []
    public var acceptedFixes: [Coordinate] = []
    public var rejectedFixes: [Coordinate] = []
    public var roadRoute: [Coordinate] = []
    public var roadPosition: Coordinate?
    public var roadComponents: [ConfidenceComponent] = []
    public var roadConfidence: Double = 0
    public var parkingTrajectory: [Coordinate] = []
    public var parkingPosition: Coordinate?
    public var parkingPolygon: [Coordinate] = []
    public var parkingStatus: ParkingStatus = .insufficientData
    public var parkingConfidence: Double = 0
    public var particleCount: Int = 0
    public var effectiveSampleSize: Double = 0
    public var roadStepMS: Double = 0
    public var parkingStepMS: Double = 0
    public var updateHz: Double = 0
    public var graphCoverage: GraphCoverage = .missing
    public var calibrationApplied: Bool = false
    public var calibrationWarning: String?
    public var hasReverseMotion: Bool = false
    public var faultMode: String = "None"

    public init() {}
}
