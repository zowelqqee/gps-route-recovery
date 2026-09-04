import CoreLocation
import CoreMotion
import Foundation
import simd

/// Captures how the phone is mounted, before the trip starts.
///
/// The driver is asked to:
///   1. fix the phone in its holder,
///   2. leave it there for the whole trip,
///   3. stand still for a few seconds,
///   4. start moving in a straight line.
///
/// Step 3 gives the stationary period the processor uses to estimate sensor
/// bias. Step 4 gives the vehicle's forward axis in the device frame, which is
/// what lets the gyro-integrated heading be tied to a real compass direction
/// without trusting the magnetometer.
public final class CalibrationManager: ObservableObject {
    public enum Phase: Equatable {
        case idle
        case holdStill(remaining: TimeInterval)
        case driveStraight
        case done
        case failed(String)
    }

    /// How long the driver must stand still.
    public static let stillDuration: TimeInterval = 6.0
    /// Speed above which the calibration drive counts as under way.
    public static let straightLineSpeed: CLLocationSpeed = 4.0

    @Published public private(set) var phase: Phase = .idle
    @Published public private(set) var calibration: MountCalibration?

    private var stillStart: TimeInterval?
    private var referenceQuaternion: [Double]?
    private var gravitySamples: [SIMD3<Double>] = []
    private var straightSamples: [(heading: Double, quaternion: simd_quatd)] = []

    public init() {}

    public func begin() {
        phase = .holdStill(remaining: Self.stillDuration)
        stillStart = nil
        referenceQuaternion = nil
        gravitySamples.removeAll()
        straightSamples.removeAll()
        calibration = nil
    }

    public func reset() {
        phase = .idle
        calibration = nil
    }

    /// Feed a motion frame. Returns true while calibration is still running.
    @discardableResult
    public func ingest(motion: CMDeviceMotion, now: TimeInterval) -> Bool {
        switch phase {
        case .holdStill:
            let rotation = simd_length(SIMD3(
                motion.rotationRate.x, motion.rotationRate.y, motion.rotationRate.z))
            let acceleration = simd_length(SIMD3(
                motion.userAcceleration.x, motion.userAcceleration.y, motion.userAcceleration.z))
            // Anything more than a nudge restarts the countdown: a calibration
            // taken while the driver is still adjusting the holder is worthless.
            guard rotation < 0.08, acceleration < 0.05 else {
                stillStart = nil
                gravitySamples.removeAll()
                phase = .holdStill(remaining: Self.stillDuration)
                return true
            }
            if stillStart == nil { stillStart = now }
            gravitySamples.append(SIMD3(motion.gravity.x, motion.gravity.y, motion.gravity.z))
            let elapsed = now - (stillStart ?? now)
            if elapsed >= Self.stillDuration {
                let q = motion.attitude.quaternion
                referenceQuaternion = [q.w, q.x, q.y, q.z]
                phase = .driveStraight
            } else {
                phase = .holdStill(remaining: Self.stillDuration - elapsed)
            }
            return true

        case .driveStraight:
            return true

        default:
            return false
        }
    }

    /// Feed a fix during the straight-line drive.
    ///
    /// Returns true while calibration is still running. A fix that arrives
    /// during the hold-still phase is simply not relevant yet - it must not be
    /// read as "calibration finished", or the whole calibration is skipped the
    /// moment the first GPS fix lands.
    @discardableResult
    public func ingest(location: LocationSample, motion: CMDeviceMotion?) -> Bool {
        switch phase {
        case .holdStill:
            return true
        case .driveStraight:
            break
        case .idle, .done, .failed:
            return false
        }
        guard let motion,
              let speed = location.speed, speed >= Self.straightLineSpeed,
              let course = location.course, course >= 0
        else { return true }

        let q = motion.attitude.quaternion
        straightSamples.append((
            heading: Self.headingRadians(fromCourse: course),
            quaternion: simd_quatd(ix: q.x, iy: q.y, iz: q.z, r: q.w)
        ))

        // A couple of seconds of consistent straight-line driving is plenty.
        guard straightSamples.count >= 20 else { return true }
        finish()
        return false
    }

    /// Finish with whatever has been gathered. Called when the driver skips the
    /// straight-line step, or when enough samples have arrived.
    public func finish() {
        guard let referenceQuaternion else {
            phase = .failed("The phone never held still long enough to calibrate.")
            return
        }
        let gravity = gravitySamples.isEmpty
            ? SIMD3<Double>(0, 0, -1)
            : gravitySamples.reduce(SIMD3<Double>.zero, +) / Double(gravitySamples.count)

        var forwardAxis: [Double]?
        var initialHeading: Double?
        var source = "unknown"

        if !straightSamples.isEmpty {
            // Average the vehicle heading over the straight-line drive, then
            // express that direction back in the device frame.
            let meanVector = straightSamples.reduce(SIMD2<Double>.zero) { partial, sample in
                partial + SIMD2(cos(sample.heading), sin(sample.heading))
            }
            let heading = atan2(meanVector.y, meanVector.x)
            let worldForward = SIMD3<Double>(cos(heading), sin(heading), 0)
            let attitude = straightSamples[straightSamples.count / 2].quaternion
            // Device <- world is the inverse of the recorded device -> world.
            let deviceForward = attitude.inverse.act(worldForward)
            forwardAxis = [deviceForward.x, deviceForward.y, deviceForward.z]
            initialHeading = Self.courseDegrees(fromHeading: heading)
            source = "gps_course"
        }

        calibration = MountCalibration(
            referenceQuaternion: referenceQuaternion,
            gravityDevice: [gravity.x, gravity.y, gravity.z],
            forwardAxisDevice: forwardAxis,
            initialHeadingDeg: initialHeading,
            headingSource: source,
            stillDurationS: Self.stillDuration,
            capturedAt: Date()
        )
        phase = .done
    }

    /// CoreLocation course (degrees clockwise from north) -> psi (radians CCW
    /// from east), the convention the processor's state vector uses.
    public static func headingRadians(fromCourse course: CLLocationDirection) -> Double {
        let radians = (90.0 - course) * .pi / 180.0
        return atan2(sin(radians), cos(radians))
    }

    public static func courseDegrees(fromHeading heading: Double) -> Double {
        let degrees = 90.0 - heading * 180.0 / .pi
        return degrees.truncatingRemainder(dividingBy: 360.0) < 0
            ? degrees.truncatingRemainder(dividingBy: 360.0) + 360.0
            : degrees.truncatingRemainder(dividingBy: 360.0)
    }
}
