import XCTest
@testable import LiveTracking

/// The cross-language generator has to produce the identical stream in both
/// languages, or the "same seed, same thresholds, same road graph" parity claim
/// means nothing.
final class RNGTests: XCTestCase {
    func testUInt32StreamMatchesPython() {
        var rng = ParityRNG(seed: 42)
        for expected in RNGFixtures.uint32Seed42 {
            XCTAssertEqual(rng.nextUInt32(), expected)
        }
    }

    func testDoubleStreamMatchesPython() {
        var rng = ParityRNG(seed: 42)
        for expected in RNGFixtures.doublesSeed42 {
            XCTAssertEqual(rng.nextDouble(), expected, accuracy: 0)
        }
    }

    func testNormalsMatchPython() {
        var rng = ParityRNG(seed: 42)
        let values = rng.normals(count: RNGFixtures.normalsSeed42.count)
        for (actual, expected) in zip(values, RNGFixtures.normalsSeed42) {
            XCTAssertEqual(actual, expected, accuracy: 1e-12)
        }
    }

    func testScaledNormalsMatchPython() {
        var rng = ParityRNG(seed: 42)
        let values = rng.normals(
            count: RNGFixtures.scaledNormalsSeed42.count, mean: 2.5, sigma: 0.25
        )
        for (actual, expected) in zip(values, RNGFixtures.scaledNormalsSeed42) {
            XCTAssertEqual(actual, expected, accuracy: 1e-12)
        }
    }

    func testIntegersMatchPython() {
        var rng = ParityRNG(seed: 7)
        for expected in RNGFixtures.integersSeed7 {
            XCTAssertEqual(rng.integer(upperBound: 11), expected)
        }
    }

    func testCategoricalMatchesPython() {
        var rng = ParityRNG(seed: 7)
        let weights = [0.1, 0.2, 0.3, 0.4]
        for expected in RNGFixtures.choicesSeed7 {
            XCTAssertEqual(rng.categorical(weights: weights), expected)
        }
    }

    func testMultinomialMatchesPython() {
        var rng = ParityRNG(seed: 7)
        let counts = rng.multinomial(count: 50, weights: [0.5, 0.2, 0.2, 0.1])
        XCTAssertEqual(counts, RNGFixtures.multinomialSeed7)
        XCTAssertEqual(counts.reduce(0, +), 50)
    }

    /// The consumption order is the contract, not just the marginals: an odd
    /// normal count still burns a whole Box-Muller pair, and the next call has
    /// to resume at the same point in the stream.
    func testMixedConsumptionOrderMatchesPython() {
        var rng = ParityRNG(seed: 99)
        var values: [Double] = []
        values.append(contentsOf: rng.normals(count: 3))
        values.append(rng.nextDouble())
        values.append(Double(rng.integer(upperBound: 5)))
        values.append(rng.normal())
        values.append(contentsOf: rng.normals(count: 2, mean: 0, sigma: 2.0))
        XCTAssertEqual(values.count, RNGFixtures.mixedSeed99.count)
        for (actual, expected) in zip(values, RNGFixtures.mixedSeed99) {
            XCTAssertEqual(actual, expected, accuracy: 1e-12)
        }
    }

    func testSameSeedGivesTheSameStream() {
        var a = ParityRNG(seed: 1234)
        var b = ParityRNG(seed: 1234)
        for _ in 0..<200 { XCTAssertEqual(a.nextUInt32(), b.nextUInt32()) }
    }

    func testDifferentSeedsDiverge() {
        var a = ParityRNG(seed: 1)
        var b = ParityRNG(seed: 2)
        var same = 0
        for _ in 0..<200 where a.nextUInt32() == b.nextUInt32() { same += 1 }
        XCTAssertLessThan(same, 5)
    }

    func testNormalsAreRoughlyStandard() {
        var rng = ParityRNG(seed: 5)
        let values = rng.normals(count: 20_000)
        let mean = values.reduce(0, +) / Double(values.count)
        let variance = values.map { ($0 - mean) * ($0 - mean) }.reduce(0, +)
            / Double(values.count)
        XCTAssertEqual(mean, 0, accuracy: 0.03)
        XCTAssertEqual(variance, 1, accuracy: 0.05)
    }

    func testIntegersCoverTheRangeUniformly() {
        var rng = ParityRNG(seed: 3)
        var counts = [Int](repeating: 0, count: 6)
        for _ in 0..<60_000 { counts[rng.integer(upperBound: 6)] += 1 }
        for count in counts { XCTAssertEqual(Double(count), 10_000, accuracy: 500) }
    }
}
