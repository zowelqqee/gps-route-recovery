import XCTest
@testable import LiveTracking

/// Particle mechanics: resampling, reinjection, one-way streets and junctions.
final class RoadTrackerTests: XCTestCase {

    // MARK: - Weight algebra

    func testNormalisationSumsToOneAndPreservesRatios() {
        var weights = [1.0, 3.0, 6.0]
        ParticleMath.normalize(&weights)
        XCTAssertEqual(weights.reduce(0, +), 1, accuracy: 1e-12)
        XCTAssertEqual(weights[1] / weights[0], 3, accuracy: 1e-12)
    }

    /// A fully collapsed set means "the filter knows nothing", which uniform
    /// says correctly; a division by zero does not.
    func testATotalCollapseFallsBackToUniform() {
        var weights = [Double](repeating: 0, count: 5)
        ParticleMath.normalize(&weights)
        XCTAssertEqual(weights, [Double](repeating: 0.2, count: 5))
    }

    func testNaNAndNegativeWeightsAreDiscarded() {
        var weights = [1.0, .nan, -2.0, 3.0]
        ParticleMath.normalize(&weights)
        XCTAssertEqual(weights.reduce(0, +), 1, accuracy: 1e-12)
        XCTAssertEqual(weights[1], 0)
        XCTAssertEqual(weights[2], 0)
    }

    func testEffectiveSampleSize() {
        XCTAssertEqual(ParticleMath.effectiveSampleSize([Double](repeating: 0.01, count: 100)),
                       100, accuracy: 1e-9)
        var collapsed = [Double](repeating: 0, count: 100)
        collapsed[0] = 1
        XCTAssertEqual(ParticleMath.effectiveSampleSize(collapsed), 1, accuracy: 1e-9)
    }

    // MARK: - Resampling

    func testSystematicResampleReturnsTheRequestedCount() {
        var rng = ParityRNG(seed: 1)
        let weights = (0..<500).map { _ in 1.0 / 500 }
        let picks = ParticleMath.systematicResample(weights: weights, count: 500, rng: &rng)
        XCTAssertEqual(picks.count, 500)
        XCTAssertTrue(picks.allSatisfy { $0 >= 0 && $0 < 500 })
    }

    func testSystematicResampleFollowsTheWeights() {
        var rng = ParityRNG(seed: 7)
        var weights = [Double](repeating: 0.002, count: 500)
        for index in 0..<100 { weights[index] = 0.007 }
        ParticleMath.normalize(&weights)
        // The first 100 carry 100 * 0.007 = 0.7 of a total of 1.5, so 46.7%.
        let expectedShare = (100 * 0.007) / (100 * 0.007 + 400 * 0.002)
        let picks = ParticleMath.systematicResample(weights: weights, count: 1000, rng: &rng)
        let heavy = picks.filter { $0 < 100 }.count
        XCTAssertEqual(Double(heavy), expectedShare * 1000, accuracy: 5,
                       "systematic resampling should hit the expected count almost exactly")
    }

    func testSystematicResampleIsLowVariance() {
        var rng = ParityRNG(seed: 3)
        let weights = [Double](repeating: 1.0 / 1000, count: 1000)
        let picks = ParticleMath.systematicResample(weights: weights, count: 1000, rng: &rng)
        var counts = [Int](repeating: 0, count: 1000)
        for pick in picks { counts[pick] += 1 }
        XCTAssertLessThanOrEqual(counts.max() ?? 0, 2,
                                 "uniform weights must give an almost exact 1:1 copy")
    }

    func testSystematicResampleKeepsOnlyTheSurvivor() {
        var rng = ParityRNG(seed: 1)
        var weights = [Double](repeating: 0, count: 50)
        weights[17] = 1
        let picks = ParticleMath.systematicResample(weights: weights, count: 50, rng: &rng)
        XCTAssertTrue(picks.allSatisfy { $0 == 17 })
    }

    func testResamplingTriggersWhenNeffCollapses() {
        let tracker = RoadTracker(network: Fixtures.straight(), config: Fixtures.config(particles: 500))
        XCTAssertTrue(tracker.initialize(at: Point(x: 0, y: 0), heading: 0, speed: 10))
        var weights = [Double](repeating: 1e-9, count: 500)
        weights[0] = 1000
        ParticleMath.normalize(&weights)
        tracker.cloud.weight = weights
        XCTAssertTrue(tracker.maybeResample())
        XCTAssertEqual(tracker.cloud.weight[0], 1.0 / 500, accuracy: 1e-12)
    }

    func testResamplingDoesNotTriggerOnAHealthyCloud() {
        let tracker = RoadTracker(network: Fixtures.straight(), config: Fixtures.config(particles: 500))
        XCTAssertTrue(tracker.initialize(at: Point(x: 0, y: 0), heading: 0, speed: 10))
        XCTAssertFalse(tracker.maybeResample())
    }

    /// Changing the particle budget must preserve the posterior, not reset it.
    func testResizingPreservesTheDistribution() {
        let network = Fixtures.fork()
        let tracker = RoadTracker(network: network, config: Fixtures.config(particles: 2000))
        XCTAssertTrue(tracker.initialize(at: Point(x: 400, y: 0), heading: 0, speed: 12))
        for _ in 0..<80 { tracker.predict(aWorld: SIMD3(0, 0, 0), yawRate: 0, dt: 0.1) }
        tracker.updateWeights()

        let before = tracker.positions()
        let beforeMeanX = zip(before, tracker.cloud.weight).reduce(0.0) { $0 + $1.0.x * $1.1 }
        tracker.resize(to: 500)
        XCTAssertEqual(tracker.particleCount, 500)
        let after = tracker.positions()
        let afterMeanX = zip(after, tracker.cloud.weight).reduce(0.0) { $0 + $1.0.x * $1.1 }
        XCTAssertEqual(afterMeanX, beforeMeanX, accuracy: 12,
                       "resizing must resample, not reset")
    }

    // MARK: - Reinjection

    func testInjectionReplacesTheWorstParticlesNearTheFix() {
        let network = Fixtures.straight()
        let tracker = RoadTracker(network: network, config: Fixtures.config(particles: 500))
        XCTAssertTrue(tracker.initialize(at: Point(x: 0, y: 0), heading: 0, speed: 10))
        let target = Point(x: 400, y: 0)
        let before = tracker.positions().filter { $0.distance(to: target) < 60 }.count
        for _ in 0..<20 {
            tracker.inject(from: target, heading: 0, speed: 10, sigma: 8)
        }
        let after = tracker.positions().filter { $0.distance(to: target) < 60 }.count
        XCTAssertGreaterThan(after, before, "rejuvenation must seed particles at the fix")
    }

    func testInjectionKeepsTheParticleCount() {
        let tracker = RoadTracker(network: Fixtures.straight(), config: Fixtures.config(particles: 300))
        XCTAssertTrue(tracker.initialize(at: Point(x: 0, y: 0), heading: 0, speed: 10))
        tracker.inject(from: Point(x: 100, y: 0), heading: 0, speed: 10, sigma: 8)
        XCTAssertEqual(tracker.particleCount, 300)
        XCTAssertEqual(tracker.cloud.weight.reduce(0, +), 1, accuracy: 1e-9)
    }

    func testDivergenceIsDetected() {
        let tracker = RoadTracker(network: Fixtures.straight(), config: Fixtures.config(particles: 300))
        XCTAssertTrue(tracker.initialize(at: Point(x: 0, y: 0), heading: 0, speed: 10))
        tracker.updateWeights(gps: Point(x: 5000, y: 5000), gpsSigma: 8)
        XCTAssertTrue(tracker.hasDiverged())
    }

    func testReinitialisationKeepsTheLearnedBiases() {
        let tracker = RoadTracker(network: Fixtures.straight(), config: Fixtures.config(particles: 300))
        XCTAssertTrue(tracker.initialize(at: Point(x: 0, y: 0), heading: 0, speed: 10,
                                         accelBias: 0.2))
        let before = zip(tracker.cloud.accelBias, tracker.cloud.weight).reduce(0.0) { $0 + $1.0 * $1.1 }
        tracker.reinitialize(at: Point(x: 600, y: 0), heading: 0, speed: 10, sigma: 8)
        let after = zip(tracker.cloud.accelBias, tracker.cloud.weight).reduce(0.0) { $0 + $1.0 * $1.1 }
        XCTAssertEqual(after, before, accuracy: 0.06)
        XCTAssertGreaterThan(tracker.positions().map(\.x).reduce(0, +) / 300, 400)
    }

    // MARK: - Topology

    /// Driving up a one-way street has to be structurally impossible, not merely
    /// penalised.
    func testNoParticleEntersAOneWayStreetBackwards() {
        let network = Fixtures.oneway()
        let tracker = RoadTracker(network: network, config: Fixtures.config(particles: 400))
        XCTAssertTrue(tracker.initialize(at: Point(x: 880, y: 0), heading: 0, speed: 8))
        for _ in 0..<100 { tracker.predict(aWorld: SIMD3(0, 0, 0), yawRate: 0, dt: 0.1) }
        // Every particle must still be on an edge that exists; the far end of
        // the one-way street is a dead end, so nothing may travel back down it.
        for index in 0..<tracker.cloud.count {
            let edge = network.edges[Int(tracker.cloud.edge[index])]
            let name = network.name(edge: edge.index)
            if name == "One way east" {
                XCTAssertLessThan(
                    network.position(edge: edge.index, s: tracker.cloud.s[index]).x, 901
                )
            }
        }
        XCTAssertGreaterThan(tracker.deadEndParticles, 0, "the far end really is a dead end")
    }

    func testAOneWayStreetHasNoReverseEdge() {
        let network = Fixtures.oneway()
        let forward = network.edges.first { network.name(edge: $0.index) == "One way east" }!
        XCTAssertTrue(network.edges.allSatisfy {
            !($0.startNode == forward.endNode && $0.endNode == forward.startNode)
        })
    }

    /// Crossing a fork must split the belief, not pick a road arbitrarily.
    func testParticlesCrossTheJunctionOntoBothBranches() {
        let network = Fixtures.fork()
        let tracker = RoadTracker(network: network, config: Fixtures.config(particles: 2000))
        XCTAssertTrue(tracker.initialize(at: Point(x: 450, y: 0), heading: 0, speed: 12))
        for _ in 0..<120 { tracker.predict(aWorld: SIMD3(0, 0, 0), yawRate: 0, dt: 0.1) }

        var onA = 0, onB = 0
        for index in 0..<tracker.cloud.count {
            switch network.name(edge: Int(tracker.cloud.edge[index])) {
            case "Branch A": onA += 1
            case "Branch B": onB += 1
            default: break
            }
        }
        XCTAssertGreaterThan(onA, 0)
        XCTAssertGreaterThan(onB, 0)
        let share = Double(onA) / Double(onA + onB)
        XCTAssertGreaterThan(share, 0.2, "neither branch may be discarded without evidence")
        XCTAssertLessThan(share, 0.8)
    }

    /// The core claim of the whole approach: with GPS gone, the turn the gyro
    /// recorded is what decides which road the car took.
    func testTheGyroDecidesWhichBranchGetsTheWeight() {
        let network = Fixtures.fork()
        let tracker = RoadTracker(network: network, config: Fixtures.config(particles: 3000))
        XCTAssertTrue(tracker.initialize(at: Point(x: 450, y: 0), heading: 0, speed: 12))
        // Branch A leaves at +45 degrees; turn at a rate that reaches it.
        let yaw = (45.0 * .pi / 180.0) / 3.0
        for _ in 0..<120 { tracker.predict(aWorld: SIMD3(0, 0, 0), yawRate: yaw, dt: 0.1) }
        tracker.updateWeights()

        var weightA = 0.0, weightB = 0.0
        for index in 0..<tracker.cloud.count {
            switch network.name(edge: Int(tracker.cloud.edge[index])) {
            case "Branch A": weightA += tracker.cloud.weight[index]
            case "Branch B": weightB += tracker.cloud.weight[index]
            default: break
            }
        }
        XCTAssertGreaterThan(weightA, weightB)
        XCTAssertGreaterThan(weightA, 0.6)
    }

    func testTurningTheOtherWayFlipsTheAnswer() {
        let network = Fixtures.fork()
        let tracker = RoadTracker(network: network, config: Fixtures.config(particles: 3000))
        XCTAssertTrue(tracker.initialize(at: Point(x: 450, y: 0), heading: 0, speed: 12))
        let yaw = -(45.0 * .pi / 180.0) / 3.0
        for _ in 0..<120 { tracker.predict(aWorld: SIMD3(0, 0, 0), yawRate: yaw, dt: 0.1) }
        tracker.updateWeights()

        var weightA = 0.0, weightB = 0.0
        for index in 0..<tracker.cloud.count {
            switch network.name(edge: Int(tracker.cloud.edge[index])) {
            case "Branch A": weightA += tracker.cloud.weight[index]
            case "Branch B": weightB += tracker.cloud.weight[index]
            default: break
            }
        }
        XCTAssertGreaterThan(weightB, weightA,
                             "the filter must be reading the gyro, not favouring one road")
    }

    /// The road tracker is constrained to the graph by construction; that is
    /// what separates it from the parking tracker.
    func testEveryParticleStaysOnTheGraph() {
        let network = Fixtures.fork()
        let tracker = RoadTracker(network: network, config: Fixtures.config(particles: 800))
        XCTAssertTrue(tracker.initialize(at: Point(x: 0, y: 0), heading: 0, speed: 12))
        for step in 0..<300 {
            tracker.predict(
                aWorld: SIMD3(0, 0, 0),
                yawRate: step > 150 ? 0.3 : 0, dt: 0.1
            )
        }
        for point in tracker.positions() {
            XCTAssertLessThan(network.distanceToRoad(point), 0.5,
                              "a road particle is always on a road")
        }
    }

    func testRefusesToIntegrateAcrossALargeGap() {
        let tracker = RoadTracker(network: Fixtures.straight(), config: Fixtures.config(particles: 200))
        XCTAssertTrue(tracker.initialize(at: Point(x: 0, y: 0), heading: 0, speed: 10))
        tracker.predict(aWorld: SIMD3(0, 0, 0), yawRate: 0, dt: 5.0)
        XCTAssertEqual(tracker.skippedGaps, 1)
    }

    // MARK: - Reproducibility

    private func run(seed: UInt64) -> [Point] {
        var config = Fixtures.config(particles: 300)
        config.seed = seed
        let tracker = RoadTracker(network: Fixtures.fork(), config: config, seed: seed)
        _ = tracker.initialize(at: Point(x: 0, y: 0), heading: 0, speed: 10)
        for _ in 0..<80 {
            tracker.predict(aWorld: SIMD3(0.2, 0, 0), yawRate: 0.02, dt: 0.1)
            tracker.updateWeights()
            tracker.maybeResample()
        }
        return tracker.positions()
    }

    func testTheSameSeedGivesTheSameRun() {
        XCTAssertEqual(run(seed: 42), run(seed: 42))
    }

    func testADifferentSeedGivesADifferentRun() {
        XCTAssertNotEqual(run(seed: 42), run(seed: 43))
    }
}
