import Foundation

/// Angle helpers, shared by every tracker.
///
/// Conventions, fixed here once and relied on everywhere else:
///
/// * `heading` (`psi`, `theta`) is **radians, counter-clockwise from local
///   East**, i.e. the mathematical convention that matches the `(E, N)` state
///   layout. `atan2(north, east)`.
/// * `course` is **degrees clockwise from true north**, which is what
///   `CLLocation.course` reports.
/// * Yaw rate is positive counter-clockwise (turning left), matching the sign
///   of the world-frame Z component of the gyro.
/// * Nothing here ever touches magnetic north. Vehicle heading comes from the
///   gyro, corrected by a trusted GPS *course*; the magnetometer is recorded
///   but is not a heading source inside a car.
public enum Angles {
    /// Wrap into `(-pi, pi]`. The half-open convention matters: `wrap(pi) == pi`
    /// and `wrap(-pi) == pi`, matching `geotrace.coordinates.wrap_angle`.
    public static func wrap(_ angle: Double) -> Double {
        var wrapped = fmod(angle + .pi, 2 * .pi) - .pi
        if wrapped <= -.pi { wrapped += 2 * .pi }
        return wrapped
    }

    public static func wrapDegrees(_ angle: Double) -> Double {
        var wrapped = fmod(angle + 180.0, 360.0) - 180.0
        if wrapped <= -180.0 { wrapped += 360.0 }
        return wrapped
    }

    /// CLLocation course (deg clockwise from north) -> heading (rad CCW from east).
    public static func headingFromCourse(_ courseDegrees: Double) -> Double {
        wrap((90.0 - courseDegrees) * .pi / 180.0)
    }

    /// Inverse of `headingFromCourse`, returned in `[0, 360)`.
    public static func courseFromHeading(_ heading: Double) -> Double {
        let degrees = fmod(90.0 - heading * 180.0 / .pi, 360.0)
        return degrees < 0 ? degrees + 360.0 : degrees
    }
}

/// WGS84 ellipsoid constants.
public enum WGS84 {
    public static let a = 6_378_137.0
    public static let f = 1.0 / 298.257223563
    public static let b = a * (1.0 - f)
}

/// Geodesic problems on the WGS84 ellipsoid, by Vincenty's formulae.
///
/// Needed because the Python baseline projects with
/// `+proj=aeqd +datum=WGS84`, and PROJ's ellipsoidal azimuthal-equidistant is
/// exactly "geodesic distance along the geodesic azimuth from the origin".
/// Verified against pyproj: the two agree to about 2e-9 m over tens of
/// kilometres, which is six orders of magnitude below anything that matters
/// here.
public enum Geodesic {
    public struct InverseResult {
        public let distance: Double
        /// Forward azimuth at the first point, radians clockwise from north.
        public let azimuth: Double
    }

    /// Distance and forward azimuth from point 1 to point 2.
    public static func inverse(
        lat1: Double, lon1: Double, lat2: Double, lon2: Double
    ) -> InverseResult {
        let a = WGS84.a, b = WGS84.b, f = WGS84.f
        let phi1 = lat1 * .pi / 180, phi2 = lat2 * .pi / 180
        let L = Angles.wrap((lon2 - lon1) * .pi / 180)

        let U1 = atan((1 - f) * tan(phi1))
        let U2 = atan((1 - f) * tan(phi2))
        let sinU1 = sin(U1), cosU1 = cos(U1)
        let sinU2 = sin(U2), cosU2 = cos(U2)

        var lambda = L
        var sinSigma = 0.0, cosSigma = 1.0, sigma = 0.0
        var cos2Alpha = 1.0, cos2SigmaM = 1.0
        var sinLambda = 0.0, cosLambda = 1.0

        for _ in 0..<200 {
            sinLambda = sin(lambda); cosLambda = cos(lambda)
            let t1 = cosU2 * sinLambda
            let t2 = cosU1 * sinU2 - sinU1 * cosU2 * cosLambda
            sinSigma = (t1 * t1 + t2 * t2).squareRoot()
            if sinSigma == 0 {
                // Coincident points.
                return InverseResult(distance: 0, azimuth: 0)
            }
            cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda
            sigma = atan2(sinSigma, cosSigma)
            let sinAlpha = cosU1 * cosU2 * sinLambda / sinSigma
            cos2Alpha = 1 - sinAlpha * sinAlpha
            cos2SigmaM = cos2Alpha == 0 ? 0 : cosSigma - 2 * sinU1 * sinU2 / cos2Alpha
            let C = f / 16 * cos2Alpha * (4 + f * (4 - 3 * cos2Alpha))
            let previous = lambda
            lambda = L + (1 - C) * f * sinAlpha
                * (sigma + C * sinSigma
                    * (cos2SigmaM + C * cosSigma * (-1 + 2 * cos2SigmaM * cos2SigmaM)))
            if abs(lambda - previous) < 1e-14 { break }
        }

        let uSq = cos2Alpha * (a * a - b * b) / (b * b)
        let A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
        let B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))
        let deltaSigma = B * sinSigma * (cos2SigmaM + B / 4
            * (cosSigma * (-1 + 2 * cos2SigmaM * cos2SigmaM)
                - B / 6 * cos2SigmaM * (-3 + 4 * sinSigma * sinSigma)
                    * (-3 + 4 * cos2SigmaM * cos2SigmaM)))
        let distance = b * A * (sigma - deltaSigma)
        let azimuth = atan2(
            cosU2 * sinLambda, cosU1 * sinU2 - sinU1 * cosU2 * cosLambda
        )
        return InverseResult(distance: distance, azimuth: azimuth)
    }

    /// Point reached by travelling `distance` along `azimuth` from (lat1, lon1).
    public static func direct(
        lat1: Double, lon1: Double, azimuth: Double, distance: Double
    ) -> (latitude: Double, longitude: Double) {
        let a = WGS84.a, b = WGS84.b, f = WGS84.f
        if distance == 0 { return (lat1, lon1) }
        let phi1 = lat1 * .pi / 180
        let sinAlpha1 = sin(azimuth), cosAlpha1 = cos(azimuth)
        let tanU1 = (1 - f) * tan(phi1)
        let cosU1 = 1 / (1 + tanU1 * tanU1).squareRoot()
        let sinU1 = tanU1 * cosU1
        let sigma1 = atan2(tanU1, cosAlpha1)
        let sinAlpha = cosU1 * sinAlpha1
        let cos2Alpha = 1 - sinAlpha * sinAlpha
        let uSq = cos2Alpha * (a * a - b * b) / (b * b)
        let A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
        let B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))

        var sigma = distance / (b * A)
        var sinSigma = 0.0, cosSigma = 0.0, cos2SigmaM = 0.0
        for _ in 0..<200 {
            cos2SigmaM = cos(2 * sigma1 + sigma)
            sinSigma = sin(sigma); cosSigma = cos(sigma)
            let deltaSigma = B * sinSigma * (cos2SigmaM + B / 4
                * (cosSigma * (-1 + 2 * cos2SigmaM * cos2SigmaM)
                    - B / 6 * cos2SigmaM * (-3 + 4 * sinSigma * sinSigma)
                        * (-3 + 4 * cos2SigmaM * cos2SigmaM)))
            let previous = sigma
            sigma = distance / (b * A) + deltaSigma
            if abs(sigma - previous) < 1e-14 { break }
        }

        let tmp = sinU1 * sinSigma - cosU1 * cosSigma * cosAlpha1
        let phi2 = atan2(
            sinU1 * cosSigma + cosU1 * sinSigma * cosAlpha1,
            (1 - f) * (sinAlpha * sinAlpha + tmp * tmp).squareRoot()
        )
        let lambda = atan2(
            sinSigma * sinAlpha1, cosU1 * cosSigma - sinU1 * sinSigma * cosAlpha1
        )
        let C = f / 16 * cos2Alpha * (4 + f * (4 - 3 * cos2Alpha))
        let L = lambda - (1 - C) * f * sinAlpha
            * (sigma + C * sinSigma
                * (cos2SigmaM + C * cosSigma * (-1 + 2 * cos2SigmaM * cos2SigmaM)))
        return (phi2 * 180 / .pi, lon1 + L * 180 / .pi)
    }
}

/// Metric working frame for one trip.
///
/// Azimuthal equidistant, centred on the trip origin, identical to the Python
/// baseline's `geotrace.coordinates.LocalFrame`:
///
///     E = s * sin(azimuth),  N = s * cos(azimuth)
///
/// with `s` the geodesic distance and `azimuth` the forward geodesic azimuth
/// from the origin. X is East, Y is North, both in metres.
public struct LocalFrame: Sendable, Equatable {
    public let latitude0: Double
    public let longitude0: Double

    public init(latitude: Double, longitude: Double) {
        precondition(abs(latitude) <= 90 && abs(longitude) <= 180, "origin out of range")
        self.latitude0 = latitude
        self.longitude0 = longitude
    }

    public func toLocal(latitude: Double, longitude: Double) -> Point {
        let result = Geodesic.inverse(
            lat1: latitude0, lon1: longitude0, lat2: latitude, lon2: longitude
        )
        return Point(
            x: result.distance * sin(result.azimuth),
            y: result.distance * cos(result.azimuth)
        )
    }

    public func toGeo(_ point: Point) -> Coordinate {
        let distance = (point.x * point.x + point.y * point.y).squareRoot()
        if distance == 0 { return Coordinate(latitude: latitude0, longitude: longitude0) }
        let azimuth = atan2(point.x, point.y)
        let result = Geodesic.direct(
            lat1: latitude0, lon1: longitude0, azimuth: azimuth, distance: distance
        )
        return Coordinate(latitude: result.latitude, longitude: result.longitude)
    }

    public func toLocal(_ coordinate: Coordinate) -> Point {
        toLocal(latitude: coordinate.latitude, longitude: coordinate.longitude)
    }
}

/// A point in the local metric frame: metres East, metres North.
public struct Point: Sendable, Equatable, Codable {
    public var x: Double
    public var y: Double

    public init(x: Double, y: Double) {
        self.x = x
        self.y = y
    }

    public static let zero = Point(x: 0, y: 0)

    public func distance(to other: Point) -> Double {
        let dx = x - other.x, dy = y - other.y
        return (dx * dx + dy * dy).squareRoot()
    }

    public var magnitude: Double { (x * x + y * y).squareRoot() }

    public static func - (lhs: Point, rhs: Point) -> Point {
        Point(x: lhs.x - rhs.x, y: lhs.y - rhs.y)
    }

    public static func + (lhs: Point, rhs: Point) -> Point {
        Point(x: lhs.x + rhs.x, y: lhs.y + rhs.y)
    }

    public static func * (lhs: Point, rhs: Double) -> Point {
        Point(x: lhs.x * rhs, y: lhs.y * rhs)
    }
}

/// WGS84 latitude/longitude, degrees.
public struct Coordinate: Sendable, Equatable, Codable {
    public var latitude: Double
    public var longitude: Double

    public init(latitude: Double, longitude: Double) {
        self.latitude = latitude
        self.longitude = longitude
    }
}
