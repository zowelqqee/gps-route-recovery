import Foundation

/// One resampled IMU step handed to a tracker. Mirrors `ImuControl`.
public struct IMUControl: Sendable, Equatable {
    public var t: Double
    public var dt: Double
    /// Mean world-frame acceleration over the bin, m/s^2.
    ///
    /// Before the heading offset is resolved this is still in CMDeviceMotion's
    /// arbitrary reference frame; `MotionProcessor.aligned(_:)` rotates it into
    /// local East/North.
    public var aWorld: SIMD3<Double>
    /// Mean world-frame yaw rate over the bin, rad/s, positive counter-clockwise.
    public var yawRate: Double
    /// The IMU is quiet: near-zero acceleration and near-zero yaw rate. A
    /// necessary but *not* sufficient condition for a stop, because steady
    /// cruising on a straight road looks exactly the same. The consumer must
    /// also check its own speed estimate before applying a ZUPT.
    public var isQuiet: Bool
    /// The source samples were further apart than `maxGapSeconds`. Integrating
    /// across such a step invents position, so trackers must refuse it.
    public var gapExceeded: Bool
    /// A likely handset/mount impact. The vehicle filters must preserve their
    /// current motion rather than integrating this as a car manoeuvre.
    public var isShock: Bool
    public var peakAccelMS2: Double
    public var peakGyroRadS: Double

    public init(
        t: Double, dt: Double, aWorld: SIMD3<Double>, yawRate: Double,
        isQuiet: Bool = false, gapExceeded: Bool = false,
        isShock: Bool = false, peakAccelMS2: Double = 0, peakGyroRadS: Double = 0
    ) {
        self.t = t
        self.dt = dt
        self.aWorld = aWorld
        self.yawRate = yawRate
        self.isQuiet = isQuiet
        self.gapExceeded = gapExceeded
        self.isShock = isShock
        self.peakAccelMS2 = peakAccelMS2
        self.peakGyroRadS = peakGyroRadS
    }

    /// Longitudinal projection: `a_par = a_E cos(psi) + a_N sin(psi)`.
    public func longitudinalAcceleration(heading: Double) -> Double {
        aWorld.x * cos(heading) + aWorld.y * sin(heading)
    }
}

/// Rotation helpers for CoreMotion attitude quaternions.
public enum Quaternion {
    /// Rotation matrix `R_WD` from a `(w, x, y, z)` quaternion, so that
    /// `a_W = R_WD(q) a_D` takes a device-frame vector into the reference frame
    /// CMDeviceMotion was started with.
    public static func matrix(_ q: SIMD4<Double>) -> (SIMD3<Double>, SIMD3<Double>, SIMD3<Double>) {
        let norm = (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w).squareRoot()
        guard norm > 1e-12 else {
            return (SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, 1))
        }
        let w = q.x / norm, x = q.y / norm, y = q.z / norm, z = q.w / norm
        return (
            SIMD3(1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            SIMD3(2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            SIMD3(2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))
        )
    }

    /// `R_WD(q) * vector`.
    public static func rotate(_ vector: SIMD3<Double>, by q: SIMD4<Double>) -> SIMD3<Double> {
        let (r0, r1, r2) = matrix(q)
        return SIMD3(
            (r0 * vector).sum(),
            (r1 * vector).sum(),
            (r2 * vector).sum()
        )
    }

    /// `R_WD(q)^T * vector`: reference frame back into the device frame.
    public static func rotateInverse(_ vector: SIMD3<Double>, by q: SIMD4<Double>) -> SIMD3<Double> {
        let (r0, r1, r2) = matrix(q)
        return SIMD3(
            r0.x * vector.x + r1.x * vector.y + r2.x * vector.z,
            r0.y * vector.x + r1.y * vector.y + r2.y * vector.z,
            r0.z * vector.x + r1.z * vector.y + r2.z * vector.z
        )
    }
}

/// Turns 50 Hz `CMDeviceMotion` frames into fixed-rate control steps.
///
/// A port of `geotrace.motion_model.build_imu_stream`, restructured to run
/// incrementally: the offline version bins the whole recording at once, which a
/// live pipeline cannot do.
///
/// One deliberate difference is documented and measured rather than hidden. The
/// Python quiet-IMU detector is non-causal: it marks a run of quiet samples only
/// once it can see that the run lasted `zuptWindowSeconds`. Live, the future is
/// not available, so quietness here is declared once the condition *has held*
/// for that long. The effect is that a ZUPT engages up to one window later than
/// in the batch baseline; both converge at a real stop.
public final class MotionProcessor {
    private let config: MotionConfig

    /// Yaw rotation that maps CMDeviceMotion's arbitrary reference frame onto
    /// local East/North. Nil until the initial vehicle heading is known.
    public private(set) var headingOffset: Double?

    private var binStart: Double?
    private var binEnd: Double = 0
    private var accelSum = SIMD3<Double>(repeating: 0)
    private var yawSum: Double = 0
    private var binCount: Int = 0
    private var binQuietCount: Int = 0
    private var binMaxGap: Double = 0
    private var binHasShock = false
    private var binPeakAccelMS2: Double = 0
    private var binPeakGyroRadS: Double = 0
    private var lastSampleTime: Double?
    private var shockUntil: Double = -.infinity

    /// Start of the current uninterrupted run of quiet samples.
    private var quietRunStart: Double?

    /// Samples kept for the fallback heading estimate, when no calibration
    /// forward axis is available. Bounded: only the first 30 s can contribute.
    private var headingCandidates: [(magnitude: Double, vector: SIMD3<Double>)] = []
    private var firstSampleTime: Double?

    public private(set) var processedSamples: Int = 0

    public init(config: MotionConfig) {
        self.config = config
    }

    /// Feed one raw motion frame. Returns a control when a bin closes.
    ///
    /// Out-of-order and duplicate timestamps are dropped rather than integrated:
    /// a negative `dt` would run the filters backwards.
    public func ingest(_ sample: TrackerMotion) -> IMUControl? {
        guard sample.monotonicTime.isFinite else { return nil }
        if let last = lastSampleTime, sample.monotonicTime <= last { return nil }

        processedSamples += 1
        if firstSampleTime == nil { firstSampleTime = sample.monotonicTime }

        let rotation = Quaternion.matrix(sample.quaternion)
        let accelDevice = sample.userAccelerationMS2
        let aWorld = SIMD3(
            (rotation.0 * accelDevice).sum(),
            (rotation.1 * accelDevice).sum(),
            (rotation.2 * accelDevice).sum()
        )
        let rateWorld = SIMD3(
            (rotation.0 * sample.rotationRate).sum(),
            (rotation.1 * sample.rotationRate).sum(),
            (rotation.2 * sample.rotationRate).sum()
        )
        let yaw = rateWorld.z

        // Quietness is rotation-invariant, so it can be decided before the
        // heading offset is known.
        let horizontalMagnitude = (aWorld.x * aWorld.x + aWorld.y * aWorld.y).squareRoot()
        let gyroMagnitude = (
            sample.rotationRate.x * sample.rotationRate.x
                + sample.rotationRate.y * sample.rotationRate.y
                + sample.rotationRate.z * sample.rotationRate.z
        ).squareRoot()
        let accelMagnitude = (
            aWorld.x * aWorld.x + aWorld.y * aWorld.y + aWorld.z * aWorld.z
        ).squareRoot()
        if accelMagnitude >= config.shockAccelMS2 || gyroMagnitude >= config.shockGyroRadS {
            shockUntil = Swift.max(shockUntil, sample.monotonicTime + config.shockHoldSeconds)
        }
        let shockActive = sample.monotonicTime <= shockUntil
        let instantaneouslyQuiet = horizontalMagnitude < config.zuptAccelMS2
            && gyroMagnitude < config.zuptGyroRadS
        if instantaneouslyQuiet {
            if quietRunStart == nil { quietRunStart = sample.monotonicTime }
        } else {
            quietRunStart = nil
        }
        let quietLongEnough = quietRunStart.map {
            sample.monotonicTime - $0 >= config.zuptWindowSeconds
        } ?? false

        if headingOffset == nil,
           let first = firstSampleTime,
           sample.monotonicTime - first <= 30.0,
           headingCandidates.count < 4000 {
            headingCandidates.append((horizontalMagnitude, aWorld))
        }

        var emitted: IMUControl?
        if binStart == nil {
            binStart = sample.monotonicTime
            binEnd = sample.monotonicTime + config.filterDT
        } else if sample.monotonicTime >= binEnd {
            emitted = closeBin()
            binStart = binEnd
            binEnd += config.filterDT
            // A very long hole can skip many bins; snap forward rather than
            // emitting hundreds of empty ones.
            if sample.monotonicTime >= binEnd {
                binStart = sample.monotonicTime
                binEnd = sample.monotonicTime + config.filterDT
            }
        }

        if let last = lastSampleTime {
            binMaxGap = max(binMaxGap, sample.monotonicTime - last)
        }
        accelSum += aWorld
        yawSum += yaw
        binCount += 1
        if quietLongEnough { binQuietCount += 1 }
        if shockActive {
            binHasShock = true
            binPeakAccelMS2 = Swift.max(binPeakAccelMS2, accelMagnitude)
            binPeakGyroRadS = Swift.max(binPeakGyroRadS, gyroMagnitude)
        }
        lastSampleTime = sample.monotonicTime
        return emitted
    }

    /// Close the pending bin, e.g. when the trip ends.
    public func flush() -> IMUControl? {
        guard binCount > 0 else { return nil }
        return closeBin()
    }

    private func closeBin() -> IMUControl? {
        guard binCount > 0 else { return nil }
        let control = IMUControl(
            t: binEnd,
            dt: config.filterDT,
            aWorld: accelSum / Double(binCount),
            yawRate: yawSum / Double(binCount),
            isQuiet: binQuietCount == binCount,
            gapExceeded: binMaxGap > config.maxGapSeconds,
            isShock: binHasShock,
            peakAccelMS2: binPeakAccelMS2,
            peakGyroRadS: binPeakGyroRadS
        )
        accelSum = SIMD3(repeating: 0)
        yawSum = 0
        binCount = 0
        binQuietCount = 0
        binMaxGap = 0
        binHasShock = false
        binPeakAccelMS2 = 0
        binPeakGyroRadS = 0
        return control
    }

    // MARK: - Heading alignment

    /// Resolve the yaw offset between CMDeviceMotion's reference frame and
    /// local East/North, given the true initial vehicle heading.
    ///
    /// Preference order matches `_estimate_reference_offset`:
    /// 1. the mount calibration's vehicle forward axis, when the straight-line
    ///    step was performed - this is the only exact route;
    /// 2. otherwise the mean direction of the strongest horizontal
    ///    accelerations in the first 30 s, which on a straight-line start all
    ///    point forward.
    @discardableResult
    public func resolveHeadingOffset(
        referenceHeading: Double, calibration: TrackerCalibration?
    ) -> Double {
        if let calibration, let forward = calibration.forwardAxisDevice {
            let forwardWorld = Quaternion.rotate(forward, by: calibration.referenceQuaternion)
            let horizontal = (forwardWorld.x * forwardWorld.x
                + forwardWorld.y * forwardWorld.y).squareRoot()
            if horizontal > 1e-6 {
                let measured = atan2(forwardWorld.y, forwardWorld.x)
                let offset = Angles.wrap(referenceHeading - measured)
                headingOffset = offset
                headingCandidates.removeAll(keepingCapacity: false)
                return offset
            }
        }

        guard !headingCandidates.isEmpty else {
            headingOffset = 0
            return 0
        }
        let magnitudes = headingCandidates.map(\.magnitude)
        guard let peak = magnitudes.max(), peak >= 0.2 else {
            headingOffset = 0
            headingCandidates.removeAll(keepingCapacity: false)
            return 0
        }
        // Mean direction of the strongest decile, as in the baseline. The
        // quantile has to be numpy's default linear interpolation, not a floor
        // index: a slightly different threshold selects a slightly different
        // set of samples, which rotates the whole world frame by a fraction of
        // a degree and shows up later as diverging gate decisions.
        let sorted = magnitudes.sorted()
        let threshold = Swift.max(Self.linearQuantile(sorted, 0.9), 0.2)
        var sum = SIMD3<Double>(repeating: 0)
        var used = 0
        for candidate in headingCandidates where candidate.magnitude >= threshold {
            sum += candidate.vector
            used += 1
        }
        headingCandidates.removeAll(keepingCapacity: false)
        guard used > 0, (sum.x * sum.x + sum.y * sum.y).squareRoot() > 1e-6 else {
            headingOffset = 0
            return 0
        }
        let measured = atan2(sum.y, sum.x)
        let offset = Angles.wrap(referenceHeading - measured)
        headingOffset = offset
        return offset
    }

    /// numpy's default quantile: linear interpolation between order statistics.
    static func linearQuantile(_ sorted: [Double], _ q: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        if sorted.count == 1 { return sorted[0] }
        let position = Double(sorted.count - 1) * q
        let low = Int(position.rounded(.down))
        let high = Swift.min(low + 1, sorted.count - 1)
        let fraction = position - Double(low)
        return sorted[low] + fraction * (sorted[high] - sorted[low])
    }

    /// Rotate a control's world acceleration into local East/North.
    public func aligned(_ control: IMUControl) -> IMUControl {
        guard let offset = headingOffset, offset != 0 else { return control }
        var rotated = control
        let c = cos(offset), s = sin(offset)
        rotated.aWorld = SIMD3(
            c * control.aWorld.x - s * control.aWorld.y,
            s * control.aWorld.x + c * control.aWorld.y,
            control.aWorld.z
        )
        return rotated
    }
}

/// Initial sensor biases from the stationary period before the drive.
///
/// A port of `estimate_initial_biases`. The accelerometer bias must be handled
/// as a *vector*: the longitudinal bias the motion model needs is its
/// projection onto the vehicle heading. Taking the magnitude instead injects a
/// positive bias whatever the true sign, and 0.2 m/s^2 of phantom acceleration
/// integrates to roughly 200 m over a 45 s outage.
public enum BiasEstimator {
    public static func estimate(
        controls: [IMUControl], heading: Double, config: MotionConfig, maxSeconds: Double = 12.0
    ) -> (accel: Double, gyro: Double) {
        guard let first = controls.first else { return (0, 0) }
        let quiet = controls.filter {
            $0.t - first.t <= maxSeconds && $0.isQuiet && !$0.gapExceeded
        }
        guard quiet.count >= 5 else { return (0, 0) }
        var sum = SIMD2<Double>(repeating: 0)
        var yawSum = 0.0
        for control in quiet {
            sum += SIMD2(control.aWorld.x, control.aWorld.y)
            yawSum += control.yawRate
        }
        let mean = sum / Double(quiet.count)
        var accelBias = mean.x * cos(heading) + mean.y * sin(heading)
        var gyroBias = yawSum / Double(quiet.count)
        // Refuse absurd values: a real MEMS bias is small, anything larger
        // means the "stationary" period was not stationary.
        if abs(accelBias) > 1.0 { accelBias = 0 }
        if abs(gyroBias) > 0.2 { gyroBias = 0 }
        return (accelBias, gyroBias)
    }
}
