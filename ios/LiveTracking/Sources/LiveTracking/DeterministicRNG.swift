import Foundation

/// A pseudo-random generator specified precisely enough to be reimplemented
/// bit-for-bit in another language.
///
/// The Python baseline normally uses `numpy.random.default_rng`, whose PCG64
/// bit stream and whose ziggurat normal sampler cannot be reproduced here
/// without porting a great deal of numpy. Instead both sides can be switched to
/// this generator (`--rng parity` in the Python CLI, the default in Swift), so
/// the particle filter consumes an identical stream of random numbers and the
/// two implementations can be compared far more tightly than "the endpoints are
/// within a few metres".
///
/// The algorithm is PCG-XSH-RR 64/32, exactly as published by O'Neill. Every
/// derived distribution below is defined by its consumption order, because that
/// is the part that has to match, not just the marginal distribution.
public struct ParityRNG: RandomNumberGenerator, Sendable {
    private var state: UInt64
    private let increment: UInt64

    public init(seed: UInt64, sequence: UInt64 = 0xda3e39cb94b95bdb) {
        self.state = 0
        self.increment = (sequence &<< 1) | 1
        _ = nextUInt32()
        self.state = self.state &+ seed
        _ = nextUInt32()
    }

    /// One step of the PCG output function.
    public mutating func nextUInt32() -> UInt32 {
        let old = state
        state = old &* 6_364_136_223_846_793_005 &+ increment
        let xorshifted = UInt32(truncatingIfNeeded: ((old &>> 18) ^ old) &>> 27)
        let rot = UInt32(truncatingIfNeeded: old &>> 59)
        return (xorshifted &>> rot) | (xorshifted &<< ((0 &- rot) & 31))
    }

    public mutating func next() -> UInt64 {
        let high = UInt64(nextUInt32())
        let low = UInt64(nextUInt32())
        return (high &<< 32) | low
    }

    /// Uniform in [0, 1). 53 significant bits, the same construction numpy uses.
    public mutating func nextDouble() -> Double {
        Double(next() &>> 11) * (1.0 / 9_007_199_254_740_992.0)  // 2^-53
    }

    /// `count` standard normals via Box-Muller, generated in pairs.
    ///
    /// The pairing is part of the contract: an odd count still consumes a whole
    /// pair's worth of uniforms and discards the second value, so that the next
    /// call starts at the same point in the stream in both languages.
    public mutating func normals(count: Int) -> [Double] {
        guard count > 0 else { return [] }
        var out = [Double](repeating: 0, count: count)
        var index = 0
        while index < count {
            let u1 = Swift.max(nextDouble(), 1e-300)
            let u2 = nextDouble()
            let radius = (-2.0 * Foundation.log(u1)).squareRoot()
            let theta = 2.0 * Double.pi * u2
            out[index] = radius * Foundation.cos(theta)
            if index + 1 < count {
                out[index + 1] = radius * Foundation.sin(theta)
            }
            index += 2
        }
        return out
    }

    public mutating func normals(count: Int, mean: Double, sigma: Double) -> [Double] {
        normals(count: count).map { mean + sigma * $0 }
    }

    public mutating func normal(mean: Double = 0, sigma: Double = 1) -> Double {
        normals(count: 1, mean: mean, sigma: sigma)[0]
    }

    /// Uniform integer in `0..<upperBound`, by rejection so the result is exact.
    public mutating func integer(upperBound: Int) -> Int {
        precondition(upperBound > 0)
        let bound = UInt32(truncatingIfNeeded: upperBound)
        let threshold = (0 &- bound) % bound
        while true {
            let value = nextUInt32()
            if value >= threshold { return Int(value % bound) }
        }
    }

    /// Categorical draw from unnormalised weights, by inverse CDF.
    public mutating func categorical(weights: [Double]) -> Int {
        let total = weights.reduce(0, +)
        guard total > 0, total.isFinite else { return integer(upperBound: weights.count) }
        let target = nextDouble() * total
        var running = 0.0
        for (index, weight) in weights.enumerated() {
            running += weight
            if target < running { return index }
        }
        return weights.count - 1
    }

    /// Counts of `count` independent categorical draws.
    ///
    /// Deliberately *not* numpy's conditional-binomial multinomial: drawing one
    /// category at a time is trivial to mirror exactly in Python, and at the
    /// sizes used here (a few thousand particles, once per initialisation) the
    /// cost is irrelevant.
    public mutating func multinomial(count: Int, weights: [Double]) -> [Int] {
        var counts = [Int](repeating: 0, count: weights.count)
        guard count > 0, !weights.isEmpty else { return counts }
        for _ in 0..<count {
            counts[categorical(weights: weights)] += 1
        }
        return counts
    }
}
