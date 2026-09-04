import Foundation

/// The individual admissibility tests applied to a GPS fix.
///
/// A direct port of the free functions in `geotrace.gps_quality`. A single
/// `horizontalAccuracy` is not evidence: a receiver that has just teleported the
/// car four kilometres across the Neva will happily report 12 m, so every fix is
/// tested against several independent things.
public enum GPSGate {
    /// `d_max = v*dt + 0.5*a_max*dt^2 + m`; the fix is an outlier when
    /// `d > d_max`.
    public static func physical(
        distanceM: Double, dt: Double, previousSpeed: Double,
        config: GPSQualityConfig, maxAccel: Double
    ) -> (passed: Bool, maxDistance: Double) {
        let clamped = Swift.max(dt, 0.0)
        let maxDistance = previousSpeed * clamped
            + 0.5 * maxAccel * clamped * clamped
            + config.physicalMarginM
        return (distanceM <= maxDistance, maxDistance)
    }

    /// `D^2 = r^T S^-1 r`, with `S = H P H^T + R`, for the 2x2 position block.
    ///
    /// A singular `S` fails the test rather than throwing: a degenerate
    /// covariance means the filter has no opinion and must not be allowed to
    /// bless an arbitrary fix.
    public static func mahalanobis(
        residual: Point, covariance: Matrix2, threshold: Double
    ) -> (passed: Bool, distanceSquared: Double) {
        let determinant = covariance.a * covariance.d - covariance.b * covariance.c
        guard abs(determinant) > 1e-12 else { return (false, .infinity) }
        // S^-1 r for a 2x2.
        let ix = (covariance.d * residual.x - covariance.b * residual.y) / determinant
        let iy = (-covariance.c * residual.x + covariance.a * residual.y) / determinant
        let d2 = residual.x * ix + residual.y * iy
        guard d2.isFinite, d2 >= 0 else { return (false, .infinity) }
        return (d2 <= threshold, d2)
    }

    /// Turn the reported accuracy into a usable sigma with a sane floor.
    /// Receivers routinely under-report.
    public static func measurementSigma(
        _ sample: TrackerLocation, config: GPSQualityConfig
    ) -> Double {
        guard let accuracy = sample.horizontalAccuracy, accuracy >= 0 else {
            return config.maxHorizontalAccuracyM
        }
        return Swift.max(config.minAccuracySigmaM, config.accuracySigmaScale * accuracy)
    }
}

/// Minimal symmetric 2x2, enough for the position block of a covariance.
public struct Matrix2: Sendable, Equatable {
    public var a: Double, b: Double, c: Double, d: Double

    public init(a: Double, b: Double, c: Double, d: Double) {
        self.a = a; self.b = b; self.c = c; self.d = d
    }

    public static func diagonal(_ value: Double) -> Matrix2 {
        Matrix2(a: value, b: 0, c: 0, d: value)
    }

    public static func + (lhs: Matrix2, rhs: Matrix2) -> Matrix2 {
        Matrix2(a: lhs.a + rhs.a, b: lhs.b + rhs.b, c: lhs.c + rhs.c, d: lhs.d + rhs.d)
    }
}
