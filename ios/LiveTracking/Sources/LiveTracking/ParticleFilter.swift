import Foundation

/// Weight algebra and resampling, shared by the road tracker.
///
/// Ports of the free functions in `geotrace.particle_filter`.
public enum ParticleMath {
    /// Normalise to sum 1. A fully collapsed set falls back to uniform, which is
    /// the correct statement of "the filter knows nothing" rather than a crash.
    public static func normalize(_ weights: inout [Double]) {
        var total = 0.0
        for index in weights.indices {
            let value = weights[index]
            if value.isFinite && value > 0 { total += value } else { weights[index] = 0 }
        }
        if total <= 0 || !total.isFinite {
            let uniform = 1.0 / Double(max(1, weights.count))
            for index in weights.indices { weights[index] = uniform }
            return
        }
        for index in weights.indices { weights[index] /= total }
    }

    /// `N_eff = 1 / sum(w_i^2)`.
    public static func effectiveSampleSize(_ weights: [Double]) -> Double {
        var sum = 0.0
        for weight in weights { sum += weight * weight }
        return sum <= 0 ? .infinity : 1.0 / sum
    }

    /// Low-variance systematic resampling: one uniform draw, then `count`
    /// equally spaced pointers into the cumulative weight.
    ///
    /// `count` may differ from `weights.count`, which is how the live runtime
    /// changes the particle budget without discarding the distribution it has
    /// already built up.
    public static func systematicResample(
        weights: [Double], count: Int, rng: inout ParityRNG
    ) -> [Int] {
        let n = weights.count
        guard n > 0, count > 0 else { return [] }
        let start = rng.nextDouble()
        var picks = [Int](repeating: 0, count: count)
        var cumulative = 0.0
        var source = 0
        for index in 0..<count {
            let target = (start + Double(index)) / Double(count)
            while source < n - 1 && cumulative + weights[source] < target {
                cumulative += weights[source]
                source += 1
            }
            picks[index] = source
        }
        return picks
    }
}

/// The particle cloud, as a structure of arrays.
///
/// One particle is a hypothesis about *where on the road network* the car is:
/// `(edge, distance along edge, speed, heading, accel bias, gyro bias, weight)`.
/// Because a particle lives on an edge rather than in free space, the belief
/// after a junction is a set of distinct branches, not a blob covering the
/// buildings in between.
public struct ParticleCloud {
    public var edge: [Int32] = []
    public var s: [Double] = []
    public var speed: [Double] = []
    public var heading: [Double] = []
    public var accelBias: [Double] = []
    public var gyroBias: [Double] = []
    public var weight: [Double] = []
    /// `L_map`: carries the dead-end penalty from prediction into the update.
    public var mapLikelihood: [Double] = []

    public var count: Int { edge.count }

    public mutating func resize(to n: Int) {
        edge = [Int32](repeating: 0, count: n)
        s = [Double](repeating: 0, count: n)
        speed = [Double](repeating: 0, count: n)
        heading = [Double](repeating: 0, count: n)
        accelBias = [Double](repeating: 0, count: n)
        gyroBias = [Double](repeating: 0, count: n)
        weight = [Double](repeating: 1.0 / Double(max(1, n)), count: n)
        mapLikelihood = [Double](repeating: 1, count: n)
    }

    /// Gather by index, used by both resampling and particle-count changes.
    public func gathered(_ picks: [Int]) -> ParticleCloud {
        var out = ParticleCloud()
        out.edge = picks.map { edge[$0] }
        out.s = picks.map { s[$0] }
        out.speed = picks.map { speed[$0] }
        out.heading = picks.map { heading[$0] }
        out.accelBias = picks.map { accelBias[$0] }
        out.gyroBias = picks.map { gyroBias[$0] }
        out.mapLikelihood = picks.map { mapLikelihood[$0] }
        out.weight = [Double](repeating: 1.0 / Double(max(1, picks.count)), count: picks.count)
        return out
    }
}
