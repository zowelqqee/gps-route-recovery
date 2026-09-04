import Foundation
@testable import LiveTracking

/// Shared fixtures: small hand-drawn graphs and a synthetic sensor generator, so
/// the expected answer of every test can be worked out on paper.
enum Fixtures {
    static let origin = Coordinate(latitude: 59.9311, longitude: 30.3609)

    /// A straight road east, then a fork:
    ///
    ///                 branch A (north-east)
    ///                /
    ///     start --- junction
    ///                \
    ///                 branch B (south-east)
    static func fork() -> RoadNetwork {
        RoadNetwork.synthetic(
            segments: [
                .init(name: "Stem", points: [Point(x: -200, y: 0), Point(x: 500, y: 0)]),
                .init(name: "Branch A", points: [
                    Point(x: 500, y: 0), Point(x: 1000, y: 500), Point(x: 1400, y: 900),
                ]),
                .init(name: "Branch B", points: [
                    Point(x: 500, y: 0), Point(x: 1000, y: -500), Point(x: 1400, y: -900),
                ]),
            ],
            origin: origin
        )
    }

    /// A stem feeding a one-way street that may only be driven eastwards, plus a
    /// two-way side road.
    static func oneway() -> RoadNetwork {
        RoadNetwork.synthetic(
            segments: [
                .init(name: "Stem", points: [Point(x: 0, y: 0), Point(x: 300, y: 0)]),
                .init(name: "One way east", points: [Point(x: 300, y: 0), Point(x: 900, y: 0)],
                      oneway: true),
                .init(name: "Side", points: [Point(x: 300, y: 0), Point(x: 300, y: 400)]),
            ],
            origin: origin
        )
    }

    /// A single straight road running east, for the parking scenarios: the car
    /// has to be able to leave it.
    static func straight() -> RoadNetwork {
        RoadNetwork.synthetic(
            segments: [
                .init(name: "Prospekt", points: [
                    Point(x: -500, y: 0), Point(x: 0, y: 0), Point(x: 800, y: 0),
                ]),
            ],
            origin: origin
        )
    }

    static func config(particles: Int = 400) -> LiveTrackingConfig {
        var config = LiveTrackingConfig()
        config.particleFilter.particleCount = particles
        config.runtime.adaptiveParticleCounts = false
        return config
    }

    /// A calibration whose straight-line step was performed, so the pipeline can
    /// align the world frame immediately.
    static func calibration(headingDegrees: Double = 0) -> TrackerCalibration {
        TrackerCalibration(
            referenceQuaternion: SIMD4(1, 0, 0, 0),
            gravityDevice: SIMD3(0, 0, -1),
            forwardAxisDevice: SIMD3(
                cos(Angles.headingFromCourse(headingDegrees)),
                sin(Angles.headingFromCourse(headingDegrees)),
                0
            ),
            initialHeadingDegrees: headingDegrees,
            headingSource: "gps_course"
        )
    }

    static func location(
        t: Double, point: Point, frame: LocalFrame,
        accuracy: Double = 8, speed: Double? = 10, course: Double? = 90
    ) -> TrackerLocation {
        let coordinate = frame.toGeo(point)
        return TrackerLocation(
            monotonicTime: t,
            latitude: coordinate.latitude,
            longitude: coordinate.longitude,
            horizontalAccuracy: accuracy,
            speed: speed ?? -1,
            speedAccuracy: 1,
            course: course ?? -1,
            courseAccuracy: course == nil ? -1 : 5
        )
    }

    /// Motion frames for a straight, level drive with a given longitudinal
    /// acceleration and yaw rate, with the phone mounted flat.
    static func motion(
        t: Double, accelMS2: Double, yawRate: Double = 0, heading: Double = 0
    ) -> TrackerMotion {
        // Device frame == world frame here, so the world acceleration is just
        // the longitudinal one rotated into the heading.
        let world = SIMD3(accelMS2 * cos(heading), accelMS2 * sin(heading), 0)
        return TrackerMotion(
            monotonicTime: t,
            userAccelerationG: world / gravityMetresPerSecondSquared,
            rotationRate: SIMD3(0, 0, yawRate),
            gravity: SIMD3(0, 0, -1),
            quaternion: SIMD4(1, 0, 0, 0)
        )
    }

    static func controls(
        from: Double, to: Double, dt: Double, accel: Double, yawRate: Double = 0,
        heading: Double = 0, quiet: Bool = false
    ) -> [IMUControl] {
        var out: [IMUControl] = []
        var t = from
        while t <= to + 1e-9 {
            out.append(
                IMUControl(
                    t: t, dt: dt,
                    aWorld: SIMD3(accel * cos(heading), accel * sin(heading), 0),
                    yawRate: yawRate, isQuiet: quiet
                )
            )
            t += dt
        }
        return out
    }
}
