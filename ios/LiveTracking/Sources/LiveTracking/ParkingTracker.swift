import Foundation

/// Independent free-space tracker for the last manoeuvre before parking.
///
/// A port of `geotrace.parking_tracker.ParkingTracker`. Its state is
///
///     X = [x, y, u, theta, b_a, b_omega]
///
/// in the local ENU frame, where `u` is a **signed** speed. Nothing clamps it
/// with `max(0, u)`: a car reversing into a space is moving backwards, and a
/// tracker that cannot represent that will place it several metres past where it
/// actually stopped.
///
/// Two things separate it from the road tracker and are deliberate:
/// it may leave the OSM graph entirely (courtyards, car parks, forecourts), and
/// it never writes anything back into the road belief.
///
/// The evaluation is a pure function of the rolling window, so the live runtime
/// can simply re-run it as the window slides. That also means it is exactly the
/// batch algorithm the Python baseline runs, with no incremental approximation
/// to drift apart from it.
public struct ParkingTracker {
    private let config: LiveTrackingConfig
    private let frame: LocalFrame

    public init(config: LiveTrackingConfig, frame: LocalFrame) {
        self.config = config
        self.frame = frame
    }

    /// Six-state vector `[x, y, u, theta, b_a, b_omega]`.
    public struct State {
        public var x = 0.0, y = 0.0
        public var u = 0.0
        public var theta = 0.0
        public var accelBias = 0.0
        public var gyroBias = 0.0
        public var positionVariance = 25.0

        public init() {}

        var point: Point { Point(x: x, y: y) }
    }

    /// Test seams: the transition and the measurement update are the two places
    /// signed velocity is decided, and they are worth asserting on directly
    /// rather than only through a whole manoeuvre.
    public func testPredict(_ state: inout State, control: IMUControl) {
        predict(&state, control: control)
    }

    public func testGPSUpdate(_ state: inout State, fix: TrackerLocation, point: Point) {
        gpsUpdate(&state, fix: fix, point: point)
    }

    public struct Inputs {
        public var controls: [IMUControl]
        public var fixes: [TrackerLocation]
        public var endedAt: Double
        public var roadPrior: Point
        public var roadHeading: Double
        public var roadSpeed: Double
        public var calibrationPresent: Bool
        public var roadDistance: ((Point) -> Double)?

        public init(
            controls: [IMUControl], fixes: [TrackerLocation], endedAt: Double,
            roadPrior: Point, roadHeading: Double, roadSpeed: Double,
            calibrationPresent: Bool, roadDistance: ((Point) -> Double)? = nil
        ) {
            self.controls = controls
            self.fixes = fixes
            self.endedAt = endedAt
            self.roadPrior = roadPrior
            self.roadHeading = roadHeading
            self.roadSpeed = roadSpeed
            self.calibrationPresent = calibrationPresent
            self.roadDistance = roadDistance
        }
    }

    public func evaluate(_ inputs: Inputs) -> ParkingTrackingResult {
        let cfg = config.parkingTracker
        let start = inputs.endedAt - cfg.windowSeconds
        let controls = inputs.controls.filter { $0.t >= start && $0.t <= inputs.endedAt }
        let fixes = inputs.fixes
            .filter { $0.monotonicTime >= start && $0.monotonicTime <= inputs.endedAt && $0.isUsable }
            .sorted { $0.monotonicTime < $1.monotonicTime }

        var state = State()
        state.x = inputs.roadPrior.x
        state.y = inputs.roadPrior.y
        state.u = inputs.roadSpeed
        state.theta = inputs.roadHeading

        var trajectory: [Point] = []
        var accepted: [(fix: TrackerLocation, point: Point)] = []
        var rejected = 0
        var cursor = 0
        var hasReverse = false

        for control in controls {
            predict(&state, control: control)
            if state.u < cfg.reverseSpeedThresholdMS { hasReverse = true }
            while cursor < fixes.count && fixes[cursor].monotonicTime <= control.t {
                let fix = fixes[cursor]
                cursor += 1
                let point = frame.toLocal(latitude: fix.latitude, longitude: fix.longitude)
                guard reachable(
                    fix: fix, point: point, accepted: accepted,
                    roadPrior: inputs.roadPrior, start: start
                ) else {
                    rejected += 1
                    continue
                }
                gpsUpdate(&state, fix: fix, point: point)
                accepted.append((fix, point))
            }
            trajectory.append(state.point)
        }

        // No prediction after `endedAt`: the end of recording is a terminal
        // stop constraint, never an excuse to extrapolate further.
        state.u = 0

        var radius: Double
        var confidence: Double
        var status: ParkingStatus
        var reason: String
        var clusterCount = 0

        if let cluster = terminalCluster(accepted: accepted, endedAt: inputs.endedAt) {
            let weightSum = cluster.weights.reduce(0, +)
            var centre = Point.zero
            for (point, weight) in zip(cluster.points, cluster.weights) {
                centre = Point(x: centre.x + point.x * weight, y: centre.y + point.y * weight)
            }
            centre = Point(x: centre.x / weightSum, y: centre.y / weightSum)
            var scatterSum = 0.0
            for (point, weight) in zip(cluster.points, cluster.weights) {
                let dx = point.x - centre.x, dy = point.y - centre.y
                scatterSum += (dx * dx + dy * dy) * weight
            }
            let scatter = (scatterSum / weightSum).squareRoot()
            radius = Swift.max(4.0, scatter * 2.0, (1.0 / weightSum).squareRoot() * 2.0)
            confidence = Swift.min(
                0.95,
                0.55 + 0.08 * Double(cluster.points.count)
                    + (inputs.calibrationPresent ? 0.08 : 0.0)
            )
            status = confidence >= 0.78 ? .confident : .probable
            reason = "robust low-speed terminal GPS cluster"
            clusterCount = cluster.points.count
            state.x = centre.x
            state.y = centre.y
        } else if !accepted.isEmpty {
            radius = cfg.uncertainRadiusM
            confidence = 0.32
            status = .uncertain
            reason = "no mutually consistent terminal GPS cluster"
        } else {
            radius = cfg.uncertainRadiusM
            confidence = 0.12
            status = .insufficientData
            reason = "no usable GPS in terminal window"
        }

        if !inputs.calibrationPresent {
            confidence *= 0.65
            if status == .confident { status = .probable }
            reason += "; mount calibration unavailable"
        }

        var maxRoadDistance: Double?
        if let roadDistance = inputs.roadDistance, !trajectory.isEmpty {
            maxRoadDistance = trajectory.map(roadDistance).max()
        }

        // With genuinely nothing to go on there is no honest position to
        // report. Drawing the road endpoint here instead would be a lie: the
        // road tracker is pinned to the carriageway and cannot represent a car
        // parked in a courtyard.
        let hasPosition = status != .insufficientData
        let position = hasPosition ? frame.toGeo(state.point) : nil

        let lastFixGap = accepted.last.map { inputs.endedAt - $0.fix.monotonicTime }

        return ParkingTrackingResult(
            status: status,
            position: position,
            positionLocal: hasPosition ? state.point : nil,
            confidence: confidence,
            polygonRadiusM: radius,
            polygon: hasPosition ? circle(around: state.point, radius: radius) : [],
            terminalClusterFixes: clusterCount,
            rejectedFixes: rejected,
            hasReverseMotion: hasReverse,
            reason: reason,
            maxDistanceFromRoadM: maxRoadDistance,
            trajectory: trajectory.map { frame.toGeo($0) },
            lastFixSecondsBeforeEnd: lastFixGap
        )
    }

    // MARK: - Transition

    /// The signed-velocity transition, exactly as the baseline runs it:
    ///
    ///     theta <- wrap(theta + (omega - b_omega) dt)
    ///     a      = a_long(theta) - b_a
    ///     ds     = u dt + 0.5 a dt^2
    ///     x     <- x + ds cos(theta)
    ///     y     <- y + ds sin(theta)
    ///     u     <- u + a dt          (signed: reverse is a valid state)
    private func predict(_ state: inout State, control: IMUControl) {
        guard control.dt > 0, !control.gapExceeded else { return }
        let theta = Angles.wrap(state.theta + (control.yawRate - state.gyroBias) * control.dt)
        let a = control.longitudinalAcceleration(heading: theta) - state.accelBias
        let ds = state.u * control.dt + 0.5 * a * control.dt * control.dt
        state.x += ds * cos(theta)
        state.y += ds * sin(theta)
        state.u += a * control.dt
        state.theta = theta
        let growth = config.parkingTracker.processPositionSigmaM * control.dt
        state.positionVariance += growth * growth
    }

    // MARK: - Measurement

    private func reachable(
        fix: TrackerLocation, point: Point,
        accepted: [(fix: TrackerLocation, point: Point)],
        roadPrior: Point, start: Double
    ) -> Bool {
        let cfg = config.parkingTracker
        guard let accuracy = fix.horizontalAccuracy, accuracy > 0, accuracy <= cfg.maxAccuracyM
        else { return false }
        if let previous = accepted.last {
            let dt = Swift.max(0, fix.monotonicTime - previous.fix.monotonicTime)
            let maximum = config.motion.maxSpeedMS * dt + cfg.physicalMarginM
            return point.distance(to: previous.point) <= maximum
        }
        let dt = Swift.max(0, fix.monotonicTime - start)
        let maximum = config.motion.maxSpeedMS * dt + cfg.physicalMarginM
        return point.distance(to: roadPrior) <= maximum
    }

    private func gpsUpdate(_ state: inout State, fix: TrackerLocation, point: Point) {
        let sigma = Swift.max(fix.horizontalAccuracy ?? 8.0, config.gps.minAccuracySigmaM)
        let gain = Swift.min(Swift.max(30.0 / (30.0 + sigma * sigma), 0.08), 0.65)
        let previous = state.point
        state.x += gain * (point.x - state.x)
        state.y += gain * (point.y - state.y)

        if fix.hasValidCourse, fix.hasValidSpeed,
           (fix.speed ?? 0) >= config.parkingTracker.lowSpeedMS {
            state.theta = Angles.headingFromCourse(fix.course ?? 0)
        }
        if fix.hasValidSpeed {
            // CLLocation.speed is an unsigned magnitude. The sign has to come
            // from somewhere else: the direction the tracker actually moved,
            // compared with where it believes it is pointing. Without this a
            // reversing car is integrated forwards.
            let displacement = point - previous
            let direction = Point(x: cos(state.theta), y: sin(state.theta))
            let dot = displacement.x * direction.x + displacement.y * direction.y
            state.u = (dot < 0 ? -1.0 : 1.0) * (fix.speed ?? 0)
        }
    }

    // MARK: - Terminal cluster

    private struct Cluster {
        var points: [Point]
        var weights: [Double]
    }

    /// The set of slow, mutually consistent fixes right at the end of the trip.
    ///
    /// This is what turns "somewhere along the last manoeuvre" into a position:
    /// several low-speed fixes agreeing with each other is strong evidence, and
    /// their scatter is an honest radius.
    private func terminalCluster(
        accepted: [(fix: TrackerLocation, point: Point)], endedAt: Double
    ) -> Cluster? {
        let cfg = config.parkingTracker
        let tail = accepted.filter {
            $0.fix.monotonicTime >= endedAt - cfg.terminalClusterWindowSeconds
                && $0.fix.hasValidSpeed
                && ($0.fix.speed ?? 0) <= cfg.lowSpeedMS
        }
        guard tail.count >= cfg.minClusterFixes else { return nil }

        let medianX = median(tail.map(\.point.x))
        let medianY = median(tail.map(\.point.y))
        let centre = Point(x: medianX, y: medianY)
        let kept = tail.filter { $0.point.distance(to: centre) <= cfg.clusterMedianRadiusM }
        guard kept.count >= cfg.minClusterFixes else { return nil }

        return Cluster(
            points: kept.map(\.point),
            weights: kept.map { 1.0 / pow(Swift.max($0.fix.horizontalAccuracy ?? 5.0, 5.0), 2) }
        )
    }

    private func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        // Component-wise median, matching numpy's even-length averaging.
        return sorted.count % 2 == 1
            ? sorted[middle]
            : 0.5 * (sorted[middle - 1] + sorted[middle])
    }

    private func circle(around centre: Point, radius: Double, segments: Int = 48) -> [Coordinate] {
        (0...segments).map { step in
            let angle = 2 * Double.pi * Double(step) / Double(segments)
            return frame.toGeo(
                Point(x: centre.x + radius * cos(angle), y: centre.y + radius * sin(angle))
            )
        }
    }
}
