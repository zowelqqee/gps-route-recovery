import Foundation

/// Every threshold, sigma and probability limit, in one place.
///
/// A direct port of `processor/src/geotrace/config.py`. The defaults here are
/// the Python defaults; if the two ever diverge, the parity harness fails,
/// which is the point. Nothing downstream hard-codes a constant.
public struct LiveTrackingConfig: Codable, Sendable, Equatable {
    public var motion = MotionConfig()
    public var gps = GPSQualityConfig()
    public var particleFilter = ParticleFilterConfig()
    public var polygon = PolygonConfig()
    public var parkingTracker = ParkingTrackerConfig()
    public var runtime = RuntimeConfig()
    public var seed: UInt64 = 42

    public init() {}
}

public let gravityMetresPerSecondSquared = 9.80665

// MARK: - Motion

/// Strapdown / dead-reckoning limits. Mirrors `MotionConfig`.
public struct MotionConfig: Codable, Sendable, Equatable {
    /// Largest timestamp gap that may still be integrated. Beyond this the
    /// propagation is refused: integrating across a hole in the IMU stream
    /// produces silently wrong positions.
    public var maxGapSeconds: Double = 1.0
    /// Physically plausible longitudinal acceleration of a passenger car.
    public var maxAccelMS2: Double = 6.0
    /// ~162 km/h. Above this is a sensor or holder failure, not a car.
    public var maxSpeedMS: Double = 45.0
    /// |a| below this with low rotation means the IMU is *quiet*. Quiet is not
    /// the same as stopped: steady cruising on a straight road looks identical.
    public var zuptAccelMS2: Double = 0.25
    /// Filter speed below which a quiet IMU may be treated as a real stop.
    public var zuptMaxSpeedMS: Double = 1.5
    public var zuptGyroRadS: Double = 0.03
    public var zuptWindowSeconds: Double = 1.0
    /// Random-walk sigma of the accelerometer bias, m/s^2 per sqrt(s). Small on
    /// purpose: the bias is only observable while GPS speed is available, and if
    /// it wanders fast everything learned before an outage is forgotten.
    public var accelBiasRandomWalk: Double = 0.008
    public var gyroBiasRandomWalk: Double = 0.002
    public var accelNoise: Double = 0.35
    public var gyroNoise: Double = 0.02
    /// The 50 Hz IMU stream is binned to this step before it reaches a filter.
    public var filterDT: Double = 0.1

    public init() {}
}

// MARK: - GPS quality

/// The TRUSTED / SUSPECT / LOST / RECOVERING machine. Mirrors `GPSQualityConfig`.
public struct GPSQualityConfig: Codable, Sendable, Equatable {
    public var maxHorizontalAccuracyM: Double = 50.0
    /// `m` in `d_max = v*dt + 0.5*a_max*dt^2 + m`.
    public var physicalMarginM: Double = 25.0
    /// chi^2 with 2 dof at p = 0.99. Applied only while TRUSTED or SUSPECT: once
    /// dead reckoning has run free the prediction is not a valid reference.
    public var mahalanobisThreshold: Double = 9.21
    /// Below this speed CoreLocation course is noise and is ignored.
    public var minSpeedForCourseMS: Double = 3.0
    public var maxCourseErrorDeg: Double = 60.0
    public var maxSpeedMismatchMS: Double = 8.0
    /// LOST / RECOVERING: consistency against the *previous accepted fix*.
    public var recoveryCourseErrorDeg: Double = 55.0
    public var recoverySpeedMismatchMS: Double = 7.0
    public var recoveryMinStepM: Double = 8.0
    public var maxDistanceToRoadM: Double = 60.0
    /// Whole-track check: a larger median means the wrong graph.
    public var maxMedianTrackToRoadM: Double = 25.0
    public var suspectToLostCount: Int = 3
    /// Consecutive accepted, mutually consistent fixes required to return to
    /// TRUSTED. This is what filters out the false points a receiver emits
    /// right after re-acquiring.
    public var recoverCount: Int = 4
    /// No fix at all for this long => LOST.
    public var lostGapSeconds: Double = 5.0
    /// Floor for the measurement sigma. Receivers routinely under-report.
    public var minAccuracySigmaM: Double = 5.0
    public var accuracySigmaScale: Double = 1.0
    /// Accept fixes CoreLocation flagged as simulated by software. False in
    /// normal use; needed for the iOS Simulator, never for real data.
    public var allowSimulatedFixes: Bool = false

    public init() {}
}

// MARK: - Particle filter

/// Mirrors `ParticleFilterConfig`.
public struct ParticleFilterConfig: Codable, Sendable, Equatable {
    /// Upper bound. The live runtime scales down from here per GPS state; see
    /// `RuntimeConfig.adaptiveParticleCounts`.
    public var particleCount: Int = 5000
    public var resampleThreshold: Double = 0.5
    public var initRadiusM: Double = 40.0
    public var initCandidateEdges: Int = 6
    /// Per-step process noise. These accumulate as sigma * sqrt(steps) over an
    /// outage, so they are set to real sensor noise, not to "add some spread".
    public var sigmaS: Double = 0.15
    public var sigmaV: Double = 0.05
    public var sigmaPsiRad: Double = 0.008
    public var sigmaGPSM: Double = 12.0
    public var sigmaHeadingRad: Double = 0.60
    public var sigmaTurnRad: Double = 0.70
    /// Loose: it only has to catch a particle doing 90 km/h through a courtyard.
    public var sigmaSpeedMS: Double = 4.0
    /// Tight: the only measurement that makes the per-particle accelerometer
    /// bias observable, and therefore the only thing bounding along-track drift.
    public var sigmaGPSSpeedMS: Double = 1.5
    public var zuptSpeedSigmaMS: Double = 0.3
    public var allowUTurn: Bool = false
    public var uturnMaxSpeedMS: Double = 1.5
    public var deadEndWeight: Double = 1e-9
    /// While TRUSTED, this fraction of the worst particles is replaced with
    /// fresh ones drawn around the fix, so a filter that committed to the wrong
    /// branch during an outage can climb back out.
    public var injectFraction: Double = 0.02
    public var divergenceLikelihood: Double = 1e-6
    public var headingSnapGain: Double = 0.35

    public init() {}
}

// MARK: - Polygons

/// Mirrors `PolygonConfig`.
public struct PolygonConfig: Codable, Sendable, Equatable {
    /// gamma: probability mass the polygon set must contain.
    public var confidence: Double = 0.95
    /// Half a carriageway: a particle sits on the centreline, the car does not.
    public var minRadiusM: Double = 8.0
    public var kSigma: Double = 2.0
    public var crossTrackSigmaBaseM: Double = 4.0
    public var crossTrackSigmaPerSecond: Double = 0.15
    public var maxRadiusM: Double = 60.0
    public var minComponentProbability: Double = 0.01

    public init() {}
}

// MARK: - Parking tracker

/// Mirrors `ParkingTrackerConfig`. Free-space terminal manoeuvre estimator,
/// intentionally separate from the road particle filter.
public struct ParkingTrackerConfig: Codable, Sendable, Equatable {
    public var windowSeconds: Double = 60.0
    public var terminalClusterWindowSeconds: Double = 30.0
    public var lowSpeedMS: Double = 3.0
    public var minClusterFixes: Int = 3
    public var maxAccuracyM: Double = 80.0
    public var physicalMarginM: Double = 25.0
    public var processPositionSigmaM: Double = 3.0
    public var uncertainRadiusM: Double = 55.0
    /// Signed speed below which the tracker considers itself reversing. Used
    /// only for the `hasReverseMotion` diagnostic flag.
    public var reverseSpeedThresholdMS: Double = -0.25
    /// Terminal cluster scatter rejection radius.
    public var clusterMedianRadiusM: Double = 35.0

    public init() {}
}

// MARK: - Live runtime

/// On-device scheduling. Has no counterpart in the Python baseline, which is a
/// batch processor: these are the knobs that make the same algorithm affordable
/// at 50 Hz on a phone.
public struct RuntimeConfig: Codable, Sendable, Equatable {
    /// Parking tracker propagation step. It is a cheap 6-state filter, so it
    /// runs close to the raw IMU rate and sees the whole parking manoeuvre.
    public var parkingStepSeconds: Double = 0.04          // 25 Hz
    /// Road filter propagation step: IMU is accumulated into batches this long.
    public var roadStepSeconds: Double = 0.2              // 5 Hz
    /// How often a UI snapshot is published.
    public var uiUpdateSeconds: Double = 0.5              // 2 Hz
    /// How often the motion-only weight update runs: the heading-against-road
    /// and speed-limit likelihoods, plus the resampling check.
    ///
    /// This is evidence, and applying it more often than it actually arrives
    /// over-counts it: running it at the propagation rate collapses the weights
    /// and triggers needless resampling. The batch baseline applies it once per
    /// output step, and so does this.
    public var outputSeconds: Double = 1.0

    /// How often the road belief is turned into confidence polygons. Building
    /// them is the expensive part, and the map cannot show more than this.
    public var polygonUpdateSeconds: Double = 2.0
    /// How often live results are appended to disk.
    public var persistSeconds: Double = 5.0
    /// Cap on retained road-position history, to bound memory on a long trip.
    public var maxRouteSamples: Int = 20000
    /// Cap on retained parking trajectory points (the tracker only ever needs
    /// the rolling window, but the diagnostics keep a little more).
    public var maxParkingTrajectorySamples: Int = 4000

    /// Particle counts per GPS state. The road filter is only expensive when it
    /// is actually uncertain; while GPS is healthy a small cloud tracks a
    /// single edge perfectly well.
    public var particlesTrusted: Int = 800
    public var particlesSuspect: Int = 2000
    public var particlesLost: Int = 5000
    public var particlesRecovering: Int = 3000

    /// Set false in replay/parity mode so the particle count matches the Python
    /// baseline's fixed `pf.n_particles`.
    public var adaptiveParticleCounts: Bool = true

    /// How much IMU the *uncalibrated* heading fallback collects before it
    /// commits to a world-frame alignment.
    ///
    /// With a mount calibration the vehicle forward axis is known exactly and
    /// the alignment is instant. Without one it has to be inferred from the
    /// direction of the strongest accelerations, and a third of a second of
    /// data gives an answer tens of degrees wrong. The batch baseline uses a
    /// 30 s window; matching it costs a delayed start to the road tracker on an
    /// uncalibrated trip, which is the honest trade.
    public var headingFallbackWindowSeconds: Double = 30.0

    public init() {}
}
