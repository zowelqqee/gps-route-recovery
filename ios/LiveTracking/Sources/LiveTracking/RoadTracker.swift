import Foundation

/// OSM-constrained route and road-corridor tracker.
///
/// A port of `geotrace.road_tracker.RoadTracker` (which is
/// `geotrace.particle_filter.RoadParticleFilter` under a name that stops
/// free-space terminal logic from creeping back in).
///
/// The road tracker deliberately has **no** parking state: no free movement
/// through a courtyard, no lateral offset, no terminal GPS cluster and no final
/// parked position. Those belong to `ParkingTracker`, and nothing flows back
/// from it into the road belief.
public final class RoadTracker {
    public let network: RoadNetwork
    private let config: LiveTrackingConfig
    private var rng: ParityRNG

    public internal(set) var cloud = ParticleCloud()

    /// Overwrite every particle's speed. Test seam: it lets a scenario put the
    /// cloud in an exactly known state instead of driving it there with noise.
    public func setAllParticleSpeeds(_ speed: Double) {
        for index in cloud.speed.indices { cloud.speed[index] = speed }
    }
    public private(set) var isInitialized = false

    /// Best particle's GPS likelihood at the last update; drives divergence
    /// detection.
    public private(set) var lastGPSLikelihood: Double = 1.0

    public private(set) var resampleCount = 0
    public private(set) var reinitializations = 0
    public private(set) var deadEndParticles = 0
    public private(set) var junctionTransitions = 0
    public private(set) var skippedGaps = 0
    public private(set) var injectedParticles = 0

    private var successorCache: [[Int32]]
    private var successorCacheUTurn: [[Int32]]
    private var routePrior: [Double]
    private var edgeLengths: [Double]
    private var edgeSpeedLimits: [Double]

    public init(network: RoadNetwork, config: LiveTrackingConfig, seed: UInt64? = nil) {
        self.network = network
        self.config = config
        self.rng = ParityRNG(seed: seed ?? config.seed)
        // Successor lists are hot: a junction crossing happens for a handful of
        // particles on most steps, and recomputing the filtered list each time
        // dominated the profile.
        self.successorCache = (0..<network.edgeCount).map {
            network.forwardSuccessors(edge: $0, allowUTurn: false)
        }
        self.successorCacheUTurn = (0..<network.edgeCount).map {
            network.forwardSuccessors(edge: $0, allowUTurn: true)
        }
        self.routePrior = network.edges.map { RoadClass.prior($0.roadClass) }
        self.edgeLengths = network.edges.map(\.length)
        self.edgeSpeedLimits = network.edges.map(\.speedLimitMS)
    }

    public var particleCount: Int { cloud.count }
    public var effectiveSampleSize: Double { ParticleMath.effectiveSampleSize(cloud.weight) }

    // MARK: - Seeding

    /// Seed the cloud around the last trusted fix.
    ///
    /// Particles are spread over the several nearest drivable edges, not just
    /// the closest one: at 15 m accuracy the nearest edge is often the wrong
    /// side of a dual carriageway.
    @discardableResult
    public func initialize(
        at point: Point, heading: Double, speed: Double,
        accelBias: Double = 0, gyroBias: Double = 0, positionSigma: Double? = nil,
        particleCount: Int? = nil
    ) -> Bool {
        let n = particleCount ?? config.particleFilter.particleCount
        let radius = positionSigma ?? config.particleFilter.initRadiusM
        let candidates = network.nearestEdges(
            to: point,
            k: config.particleFilter.initCandidateEdges,
            radius: Swift.max(radius * 3, 120)
        )
        guard !candidates.isEmpty else { return false }

        // Prefer edges that both lie close to the fix and point the right way.
        var scores: [Double] = []
        var projections: [(s: Double, bearing: Double)] = []
        for index in candidates {
            let projection = network.project(point, onto: index)
            let bearing = network.bearing(edge: index, s: projection.s)
            let alignment = exp(
                -pow(Angles.wrap(bearing - heading), 2)
                    / (2 * pow(config.particleFilter.sigmaHeadingRad, 2))
            )
            let proximity = exp(-pow(projection.offset, 2) / (2 * pow(Swift.max(radius, 5), 2)))
            scores.append(Swift.max(alignment * proximity, 1e-6))
            projections.append((projection.s, bearing))
        }
        let total = scores.reduce(0, +)
        let normalized = scores.map { $0 / total }
        let counts = rng.multinomial(count: n, weights: normalized)

        cloud.resize(to: n)
        var cursor = 0
        for (position, index) in candidates.enumerated() {
            let count = counts[position]
            if count == 0 { continue }
            let sNoise = rng.normals(count: count, mean: 0, sigma: Swift.max(radius * 0.5, 3))
            let headingNoise = rng.normals(
                count: count, mean: 0, sigma: config.particleFilter.sigmaHeadingRad * 0.5
            )
            let sOn = projections[position].s
            let bearing = projections[position].bearing
            for offset in 0..<count {
                let slot = cursor + offset
                cloud.edge[slot] = Int32(index)
                cloud.s[slot] = Swift.min(Swift.max(sOn + sNoise[offset], 0), edgeLengths[index])
                cloud.heading[slot] = Angles.wrap(bearing + headingNoise[offset])
            }
            cursor += count
        }

        let speedNoise = rng.normals(
            count: n, mean: 0, sigma: Swift.max(0.5, speed * 0.15)
        )
        // A wide spread on purpose: the accelerometer bias is what decides how
        // far the cloud travels during an outage, and it is only observable
        // through the GPS speed likelihood while GPS is still there. Seeding it
        // too tightly means there is no particle carrying the right value.
        let accelNoise = rng.normals(count: n, mean: 0, sigma: 0.12)
        let gyroNoise = rng.normals(count: n, mean: 0, sigma: 0.008)
        for index in 0..<n {
            cloud.speed[index] = Swift.min(
                Swift.max(speed + speedNoise[index], 0), config.motion.maxSpeedMS
            )
            cloud.accelBias[index] = accelBias + accelNoise[index]
            cloud.gyroBias[index] = gyroBias + gyroNoise[index]
        }
        isInitialized = true
        return true
    }

    /// Full re-seed after divergence, keeping the learned sensor biases.
    public func reinitialize(at point: Point, heading: Double, speed: Double, sigma: Double) {
        let meanAccelBias = weightedMean(cloud.accelBias)
        let meanGyroBias = weightedMean(cloud.gyroBias)
        if initialize(
            at: point, heading: heading, speed: speed,
            accelBias: meanAccelBias, gyroBias: meanGyroBias,
            positionSigma: sigma, particleCount: cloud.count
        ) {
            reinitializations += 1
        }
    }

    private func weightedMean(_ values: [Double]) -> Double {
        var sum = 0.0, weightSum = 0.0
        for index in values.indices {
            sum += values[index] * cloud.weight[index]
            weightSum += cloud.weight[index]
        }
        return weightSum > 0 ? sum / weightSum : 0
    }

    // MARK: - Prediction

    /// One motion step for every particle.
    public func predict(aWorld: SIMD3<Double>, yawRate: Double, dt: Double) {
        guard isInitialized, dt > 0 else { return }
        let n = cloud.count
        guard n > 0 else { return }

        if dt > config.motion.maxGapSeconds {
            // Do not integrate across a hole in the IMU stream. Diffuse the
            // cloud along the roads instead, so the belief widens honestly.
            skippedGaps += 1
            let noise = rng.normals(
                count: n, mean: 0, sigma: config.motion.maxSpeedMS * dt * 0.5
            )
            for index in 0..<n { cloud.s[index] += noise[index] }
            resolveEdges()
            return
        }

        let pf = config.particleFilter
        let scale = (dt / Swift.max(config.motion.filterDT, 1e-6)).squareRoot()
        let epsPsi = rng.normals(count: n, mean: 0, sigma: pf.sigmaPsiRad * scale)
        let epsS = rng.normals(count: n, mean: 0, sigma: pf.sigmaS * scale)
        let epsV = rng.normals(count: n, mean: 0, sigma: pf.sigmaV * scale)
        let biasA = rng.normals(
            count: n, mean: 0, sigma: config.motion.accelBiasRandomWalk * dt.squareRoot()
        )
        let biasW = rng.normals(
            count: n, mean: 0, sigma: config.motion.gyroBiasRandomWalk * dt.squareRoot()
        )

        let maxAccel = config.motion.maxAccelMS2
        let maxSpeed = config.motion.maxSpeedMS
        for index in 0..<n {
            // Longitudinal projection, per particle:
            // a_par = aE cos(psi) + aN sin(psi)
            let psi = cloud.heading[index]
            let aLong = aWorld.x * cos(psi) + aWorld.y * sin(psi)
            let aHat = Swift.min(Swift.max(aLong - cloud.accelBias[index], -maxAccel), maxAccel)
            let wHat = yawRate - cloud.gyroBias[index]

            cloud.heading[index] = Angles.wrap(psi + wHat * dt + epsPsi[index])
            let ds = cloud.speed[index] * dt + 0.5 * aHat * dt * dt + epsS[index]
            cloud.speed[index] = Swift.min(
                Swift.max(cloud.speed[index] + aHat * dt + epsV[index], 0), maxSpeed
            )
            cloud.s[index] += Swift.max(ds, 0)
            cloud.accelBias[index] += biasA[index]
            cloud.gyroBias[index] += biasW[index]
        }
        resolveEdges()
    }

    /// Move particles that ran off the end of their edge through junctions.
    ///
    /// A particle may cross several short edges in one step, so this iterates
    /// until nothing overflows, with a hard cap so a pathological graph cannot
    /// hang the run.
    private func resolveEdges() {
        for index in cloud.s.indices { cloud.s[index] = Swift.max(cloud.s[index], 0) }
        for _ in 0..<12 {
            var overflowing: [Int] = []
            for index in cloud.s.indices
            where cloud.s[index] > edgeLengths[Int(cloud.edge[index])] {
                overflowing.append(index)
            }
            if overflowing.isEmpty { return }
            crossJunctions(overflowing)
        }
        // Anything still overflowing is clamped to the end of its edge.
        for index in cloud.s.indices {
            cloud.s[index] = Swift.min(cloud.s[index], edgeLengths[Int(cloud.edge[index])])
        }
    }

    private func crossJunctions(_ indices: [Int]) {
        let pf = config.particleFilter
        let sigmaTurnSquared = 2.0 * pf.sigmaTurnRad * pf.sigmaTurnRad
        let gain = pf.headingSnapGain
        for index in indices {
            let current = Int(cloud.edge[index])
            let remainder = cloud.s[index] - edgeLengths[current]
            let allowUTurn = pf.allowUTurn && cloud.speed[index] <= pf.uturnMaxSpeedMS
            let options = allowUTurn ? successorCacheUTurn[current] : successorCache[current]

            if options.isEmpty {
                // Dead end: the car cannot be here. Park the particle at the end
                // of the edge and mark it so the weight update kills it.
                cloud.s[index] = edgeLengths[current]
                cloud.speed[index] = 0
                cloud.mapLikelihood[index] = pf.deadEndWeight
                deadEndParticles += 1
                continue
            }

            let psi = cloud.heading[index]
            var choice: Int
            if options.count == 1 {
                choice = Int(options[0])
            } else {
                // P(e' | p) ~ exp(-wrap(theta_e' - psi)^2 / 2 sigma_turn^2) * P_route(e')
                var probabilities = [Double](repeating: 0, count: options.count)
                var maximum = -Double.greatestFiniteMagnitude
                for (position, option) in options.enumerated() {
                    let turn = Angles.wrap(network.startBearing(edge: Int(option)) - psi)
                    let logit = -(turn * turn) / sigmaTurnSquared
                    probabilities[position] = logit
                    maximum = Swift.max(maximum, logit)
                }
                var total = 0.0
                for position in probabilities.indices {
                    probabilities[position] = exp(probabilities[position] - maximum)
                        * routePrior[Int(options[position])]
                    total += probabilities[position]
                }
                if total <= 0 || !total.isFinite {
                    choice = Int(options[rng.integer(upperBound: options.count)])
                } else {
                    choice = Int(options[rng.categorical(weights: probabilities)])
                }
                junctionTransitions += 1
            }

            cloud.edge[index] = Int32(choice)
            cloud.s[index] = Swift.min(remainder, edgeLengths[choice])
            // Nudge the heading onto the new road; the gyro still drives the rest.
            let newBearing = network.startBearing(edge: choice)
            cloud.heading[index] = Angles.wrap(psi + gain * Angles.wrap(newBearing - psi))
        }
    }

    // MARK: - Weights

    public func positions() -> [Point] {
        (0..<cloud.count).map { network.position(edge: Int(cloud.edge[$0]), s: cloud.s[$0]) }
    }

    /// `w~_i = w_i * L_GPS * L_psi * L_v * L_map`, then normalise.
    public func updateWeights(
        gps: Point? = nil, gpsSigma: Double? = nil,
        gpsCourseRad: Double? = nil, gpsSpeed: Double? = nil
    ) {
        guard isInitialized, cloud.count > 0 else { return }
        let pf = config.particleFilter
        let n = cloud.count
        var likelihood = cloud.mapLikelihood

        for index in 0..<n {
            // L_psi: particle heading against the bearing of the road it sits on.
            let roadBearing = network.bearing(edge: Int(cloud.edge[index]), s: cloud.s[index])
            let headingError = Angles.wrap(cloud.heading[index] - roadBearing)
            likelihood[index] *= exp(
                -(headingError * headingError) / (2 * pf.sigmaHeadingRad * pf.sigmaHeadingRad)
            )
            // L_v: a particle must not be doing 90 km/h in a courtyard.
            let limit = edgeSpeedLimits[Int(cloud.edge[index])] * 1.4
            let excess = Swift.max(0, cloud.speed[index] - limit)
            likelihood[index] *= exp(
                -(excess * excess) / (2 * pf.sigmaSpeedMS * pf.sigmaSpeedMS)
            )
        }

        lastGPSLikelihood = 1.0
        if let gps {
            let reported = gpsSigma ?? pf.sigmaGPSM
            let sigma = (reported * reported + pf.sigmaGPSM * pf.sigmaGPSM).squareRoot()
            var best = 0.0
            let positions = positions()
            for index in 0..<n {
                let delta = positions[index] - gps
                let d2 = delta.x * delta.x + delta.y * delta.y
                let term = exp(-d2 / (2 * sigma * sigma))
                best = Swift.max(best, term)
                likelihood[index] *= term
            }
            lastGPSLikelihood = best

            if let gpsCourseRad, (gpsSpeed ?? 0) >= config.gps.minSpeedForCourseMS {
                let sigmaCourse = pf.sigmaHeadingRad * 1.5
                for index in 0..<n {
                    let error = Angles.wrap(cloud.heading[index] - gpsCourseRad)
                    likelihood[index] *= exp(-(error * error) / (2 * sigmaCourse * sigmaCourse))
                }
            }
            if let gpsSpeed, gpsSpeed >= 0 {
                for index in 0..<n {
                    let error = cloud.speed[index] - gpsSpeed
                    likelihood[index] *= exp(
                        -(error * error) / (2 * pf.sigmaGPSSpeedMS * pf.sigmaGPSSpeedMS)
                    )
                }
            }
        }

        for index in 0..<n { cloud.weight[index] *= likelihood[index] }
        ParticleMath.normalize(&cloud.weight)
        for index in 0..<n { cloud.mapLikelihood[index] = 1 }
    }

    /// The vehicle is provably standing still.
    ///
    /// Particles that believe they are still rolling are down-weighted and then
    /// stopped. This is what pins down the accelerometer bias without GPS: at a
    /// red light the true speed is known exactly, so any speed a particle has
    /// accumulated is integrated bias, and resampling keeps the particles whose
    /// bias explained it.
    public func zeroVelocityUpdate(maxSpeedMS: Double) {
        guard isInitialized else { return }
        let sigma = config.particleFilter.zuptSpeedSigmaMS
        for index in 0..<cloud.count {
            let speed = cloud.speed[index]
            cloud.weight[index] *= exp(-(speed * speed) / (2 * sigma * sigma))
        }
        ParticleMath.normalize(&cloud.weight)
        // Only particles that already agree the car is nearly stopped are
        // stopped; one convinced it is doing 40 km/h is a different hypothesis,
        // not a bias error, and is left for the weights to judge.
        for index in 0..<cloud.count where cloud.speed[index] <= maxSpeedMS {
            cloud.speed[index] = 0
        }
    }

    /// True when no particle can explain the current trusted fix.
    public func hasDiverged() -> Bool {
        lastGPSLikelihood < config.particleFilter.divergenceLikelihood
    }

    /// Resample when `N_eff` drops below the configured fraction of N.
    @discardableResult
    public func maybeResample() -> Bool {
        guard isInitialized, cloud.count > 0 else { return false }
        let n = cloud.count
        guard ParticleMath.effectiveSampleSize(cloud.weight) < config.particleFilter.resampleThreshold * Double(n)
        else { return false }
        let picks = ParticleMath.systematicResample(weights: cloud.weight, count: n, rng: &rng)
        cloud = cloud.gathered(picks)
        resampleCount += 1
        return true
    }

    /// Replace the worst particles with fresh ones drawn around a trusted fix.
    ///
    /// Standard rejuvenation. It is what lets the filter climb back out of a
    /// branch it wrongly committed to during an outage, and it costs nothing
    /// while the filter is already tracking correctly, because fresh particles
    /// near the fix are exactly where the good ones already are.
    public func inject(from point: Point, heading: Double, speed: Double, sigma: Double) {
        guard isInitialized, cloud.count > 0 else { return }
        let n = cloud.count
        let count = Swift.max(1, Int(config.particleFilter.injectFraction * Double(n)))
        // Sorted by (weight, index) so the tie-break is defined: right after a
        // resample every weight is identical, and an unspecified order there
        // would make the two implementations diverge for no reason.
        let victims = Array(
            cloud.weight.enumerated()
                .sorted { $0.element != $1.element ? $0.element < $1.element : $0.offset < $1.offset }
                .prefix(count).map(\.offset)
        )
        let candidates = network.nearestEdges(
            to: point,
            k: config.particleFilter.initCandidateEdges,
            radius: Swift.max(sigma * 4, 120)
        )
        guard !candidates.isEmpty else { return }

        let meanAccelBias = weightedMean(cloud.accelBias)
        let meanGyroBias = weightedMean(cloud.gyroBias)
        var picks: [Int] = []
        picks.reserveCapacity(count)
        for _ in 0..<count { picks.append(rng.integer(upperBound: candidates.count)) }

        var sortedWeights = cloud.weight.sorted()
        let median = sortedWeights.isEmpty ? 0 : sortedWeights[sortedWeights.count / 2]
        sortedWeights.removeAll(keepingCapacity: false)

        for (slot, pick) in zip(victims, picks) {
            let index = candidates[pick]
            let projection = network.project(point, onto: index)
            cloud.edge[slot] = Int32(index)
            cloud.s[slot] = Swift.min(
                Swift.max(projection.s + rng.normal(sigma: Swift.max(sigma, 5)), 0),
                edgeLengths[index]
            )
            cloud.heading[slot] = Angles.wrap(
                network.bearing(edge: index, s: projection.s)
                    + rng.normal(sigma: config.particleFilter.sigmaHeadingRad * 0.5)
            )
            cloud.speed[slot] = Swift.max(0, speed + rng.normal(sigma: 1.0))
            cloud.accelBias[slot] = meanAccelBias + rng.normal(sigma: 0.05)
            cloud.gyroBias[slot] = meanGyroBias + rng.normal(sigma: 0.005)
            cloud.weight[slot] = median
        }
        ParticleMath.normalize(&cloud.weight)
        injectedParticles += count
    }

    /// Change the particle budget without discarding the belief.
    ///
    /// The cloud is systematically resampled to the new size, so the
    /// distribution is preserved: cutting from 5000 to 800 while GPS is healthy
    /// keeps the same posterior, it just represents it with fewer samples.
    public func resize(to newCount: Int) {
        guard isInitialized, newCount > 0, newCount != cloud.count else { return }
        let picks = ParticleMath.systematicResample(
            weights: cloud.weight, count: newCount, rng: &rng
        )
        cloud = cloud.gathered(picks)
    }
}
