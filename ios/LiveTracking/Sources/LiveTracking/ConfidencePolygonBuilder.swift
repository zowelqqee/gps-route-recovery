import Foundation

/// Probabilistic position polygons.
///
/// A port of `geotrace.polygons`. Take the smallest set of particles carrying at
/// least `gamma` of the probability mass:
///
///     sum_{i=1..K} w_(i) >= gamma        (weights sorted descending)
///
/// group those particles by *connected branch of the road graph*, and build one
/// corridor per branch as a buffered ribbon around the occupied road segments,
/// with `r = r_min + k * sigma_perp`.
///
/// Deliberately **not** a convex hull. A hull over a post-junction belief would
/// swallow the courtyards, the buildings and the empty space between two
/// independent roads, and would claim the car might be inside a block of flats.
/// Two branches that have diverged produce two separate polygons with separate
/// probabilities; while the cloud still straddles the junction they are
/// genuinely one connected Y-shaped region and are reported as such.
public struct ConfidencePolygonBuilder {
    private let network: RoadNetwork
    private let config: PolygonConfig

    public init(network: RoadNetwork, config: PolygonConfig) {
        self.network = network
        self.config = config
    }

    /// Smallest index set whose weights sum to at least `gamma`, ordered by
    /// descending weight.
    public static func selectHighProbabilitySet(
        weights: [Double], gamma: Double
    ) -> [Int] {
        guard !weights.isEmpty else { return [] }
        let order = weights.enumerated().sorted { $0.element > $1.element }.map(\.offset)
        let total = weights.reduce(0, +)
        let target = gamma * (total > 0 ? total : 1)
        var running = 0.0
        var selected: [Int] = []
        for index in order {
            selected.append(index)
            running += weights[index]
            if running >= target { break }
        }
        return selected
    }

    /// Lateral uncertainty of a particle that sits on the road centreline.
    ///
    /// A particle has no lateral spread by construction, but the car does: it is
    /// in one of the lanes, and the longer GPS has been unavailable the less
    /// certain even the lane-level along-track position is.
    public func crossTrackSigma(secondsSinceTrusted: Double) -> Double {
        config.crossTrackSigmaBaseM
            + config.crossTrackSigmaPerSecond * Swift.max(0, secondsSinceTrusted)
    }

    public struct Output {
        public var components: [ConfidenceComponent]
        /// Branch-aware point estimate: the weighted mean *within the
        /// highest-probability branch*, snapped to the carriageway. Averaging
        /// across branches would put the car in the block between them.
        public var estimate: Point?
        public var estimateEdge: Int?
        public var selectedCount: Int
    }

    public func build(
        cloud: ParticleCloud, secondsSinceTrusted: Double
    ) -> Output {
        let selected = Self.selectHighProbabilitySet(
            weights: cloud.weight, gamma: config.confidence
        )
        guard !selected.isEmpty else {
            return Output(components: [], estimate: nil, estimateEdge: nil, selectedCount: 0)
        }

        let sigma = crossTrackSigma(secondsSinceTrusted: secondsSinceTrusted)
        let radius = Swift.min(config.minRadiusM + config.kSigma * sigma, config.maxRadiusM)

        // Occupied span per edge: the particles on it define a sub-segment.
        var spans: [Int: (lo: Double, hi: Double, weight: Double, members: [Int])] = [:]
        var totalSelectedWeight = 0.0
        for index in selected {
            let edge = Int(cloud.edge[index])
            let s = cloud.s[index]
            let weight = cloud.weight[index]
            totalSelectedWeight += weight
            if var existing = spans[edge] {
                existing.lo = Swift.min(existing.lo, s)
                existing.hi = Swift.max(existing.hi, s)
                existing.weight += weight
                existing.members.append(index)
                spans[edge] = existing
            } else {
                spans[edge] = (s, s, weight, [index])
            }
        }
        if totalSelectedWeight <= 0 { totalSelectedWeight = 1 }

        // Connect occupied edges that actually touch: consecutive in the graph
        // *and* both occupied near the shared node. That is what makes two
        // branches that have diverged come out as separate components, while a
        // belief still sitting on the junction stays a single connected region.
        let occupied = Array(spans.keys)
        var parent = Array(0..<occupied.count)
        var position: [Int: Int] = [:]
        for (slot, edge) in occupied.enumerated() { position[edge] = slot }

        func find(_ x: Int) -> Int {
            var root = x
            while parent[root] != root { root = parent[root] }
            var current = x
            while parent[current] != current {
                let next = parent[current]
                parent[current] = root
                current = next
            }
            return root
        }
        func union(_ a: Int, _ b: Int) {
            let ra = find(a), rb = find(b)
            if ra != rb { parent[ra] = rb }
        }

        let touch = radius
        for edge in occupied {
            guard let span = spans[edge] else { continue }
            let length = network.edges[edge].length
            // Occupied up to the far node?
            if span.hi >= length - touch {
                for successor in network.forwardSuccessors(edge: edge, allowUTurn: true) {
                    let other = Int(successor)
                    if let otherSpan = spans[other], otherSpan.lo <= touch,
                       let a = position[edge], let b = position[other] {
                        union(a, b)
                    }
                }
            }
        }

        var groups: [Int: [Int]] = [:]
        for (slot, edge) in occupied.enumerated() {
            groups[find(slot), default: []].append(edge)
        }

        var components: [ConfidenceComponent] = []
        for (_, edgesInGroup) in groups {
            var probability = 0.0
            var names = Set<String>()
            var rings: [[Coordinate]] = []
            var area = 0.0
            for edge in edgesInGroup {
                guard let span = spans[edge] else { continue }
                probability += span.weight
                if let name = network.name(edge: edge) { names.insert(name) }
                let length = network.edges[edge].length
                let lo = Swift.max(0, span.lo - radius * 0.5)
                let hi = Swift.min(length, Swift.max(span.hi + radius * 0.5, lo + 0.5))
                let centreline = polyline(edge: edge, from: lo, to: hi)
                guard centreline.count >= 2 else { continue }
                let ring = Corridor.buffer(polyline: centreline, radius: radius)
                guard ring.count >= 4 else { continue }
                area += abs(Corridor.signedArea(ring))
                rings.append(ring.map { network.frame.toGeo($0) })
            }
            guard !rings.isEmpty else { continue }
            if probability / totalSelectedWeight < config.minComponentProbability { continue }
            components.append(
                ConfidenceComponent(
                    componentID: "branch-00",
                    probability: probability,
                    areaM2: area,
                    rings: rings,
                    streetNames: Array(names.sorted().prefix(6))
                )
            )
        }

        components.sort { $0.probability > $1.probability }
        for index in components.indices {
            components[index].componentID = String(format: "branch-%02d", index + 1)
        }

        // Branch-aware estimate, over the highest-probability group.
        var estimate: Point?
        var estimateEdge: Int?
        if let best = groups.max(by: { lhs, rhs in
            let lw = lhs.value.reduce(0.0) { $0 + (spans[$1]?.weight ?? 0) }
            let rw = rhs.value.reduce(0.0) { $0 + (spans[$1]?.weight ?? 0) }
            return lw < rw
        }) {
            // Weighted mean along the single dominant edge of the branch, so the
            // estimate stays on the carriageway even when the branch curves.
            if let dominant = best.value.max(by: { (spans[$0]?.weight ?? 0) < (spans[$1]?.weight ?? 0) }),
               let span = spans[dominant] {
                var weighted = 0.0, weightSum = 0.0
                for member in span.members {
                    weighted += cloud.s[member] * cloud.weight[member]
                    weightSum += cloud.weight[member]
                }
                let s = weightSum > 0 ? weighted / weightSum : span.lo
                estimate = network.position(edge: dominant, s: s)
                estimateEdge = dominant
            }
        }

        return Output(
            components: components, estimate: estimate,
            estimateEdge: estimateEdge, selectedCount: selected.count
        )
    }

    /// The sub-polyline of an edge between two distances along it.
    private func polyline(edge index: Int, from lo: Double, to hi: Double) -> [Point] {
        var out: [Point] = [network.position(edge: index, s: lo)]
        let edge = network.edges[index]
        for point in (edge.pointOffset + 1)..<(edge.pointOffset + edge.pointCount - 1) {
            let s = network.cumulative[point]
            if s > lo && s < hi { out.append(network.points[point]) }
        }
        out.append(network.position(edge: index, s: hi))
        return out
    }
}

/// Buffered-polyline geometry.
///
/// Replaces Shapely's `buffer` for the one shape that is actually needed here: a
/// constant-radius corridor around an open polyline. Offsetting each vertex
/// along the bisector of its adjacent segments and closing with semicircular
/// caps produces the same ribbon, without pulling a geometry library onto the
/// phone.
public enum Corridor {
    public static func buffer(
        polyline: [Point], radius: Double, capSegments: Int = 8
    ) -> [Point] {
        guard polyline.count >= 2, radius > 0 else { return [] }
        var cleaned: [Point] = [polyline[0]]
        for point in polyline.dropFirst() where point.distance(to: cleaned[cleaned.count - 1]) > 1e-6 {
            cleaned.append(point)
        }
        guard cleaned.count >= 2 else { return [] }

        let left = offsetSide(cleaned, radius: radius)
        let right = offsetSide(cleaned.reversed(), radius: radius)

        var ring: [Point] = []
        ring.append(contentsOf: left)
        ring.append(contentsOf: cap(
            at: cleaned[cleaned.count - 1],
            from: left[left.count - 1],
            to: right[0],
            radius: radius,
            segments: capSegments
        ))
        ring.append(contentsOf: right)
        ring.append(contentsOf: cap(
            at: cleaned[0],
            from: right[right.count - 1],
            to: left[0],
            radius: radius,
            segments: capSegments
        ))
        if let first = ring.first { ring.append(first) }
        return ring
    }

    /// Offset the polyline to its left by `radius`, mitring at the joints and
    /// falling back to a bevel where the miter would spike.
    private static func offsetSide<C: Collection>(
        _ points: C, radius: Double
    ) -> [Point] where C.Element == Point {
        let list = Array(points)
        guard list.count >= 2 else { return [] }
        var out: [Point] = []
        for index in list.indices {
            let normalIn = index > 0 ? normal(from: list[index - 1], to: list[index]) : nil
            let normalOut = index < list.count - 1 ? normal(from: list[index], to: list[index + 1]) : nil
            switch (normalIn, normalOut) {
            case let (nil, .some(n)):
                out.append(Point(x: list[index].x + n.x * radius, y: list[index].y + n.y * radius))
            case let (.some(n), nil):
                out.append(Point(x: list[index].x + n.x * radius, y: list[index].y + n.y * radius))
            case let (.some(a), .some(b)):
                var bisector = Point(x: a.x + b.x, y: a.y + b.y)
                let length = bisector.magnitude
                if length < 1e-9 {
                    // Reversal: bevel rather than spike to infinity.
                    out.append(Point(x: list[index].x + a.x * radius, y: list[index].y + a.y * radius))
                    out.append(Point(x: list[index].x + b.x * radius, y: list[index].y + b.y * radius))
                    continue
                }
                bisector = Point(x: bisector.x / length, y: bisector.y / length)
                let cosHalf = bisector.x * a.x + bisector.y * a.y
                let scale = cosHalf > 0.2 ? radius / cosHalf : radius / 0.2
                out.append(
                    Point(x: list[index].x + bisector.x * scale, y: list[index].y + bisector.y * scale)
                )
            case (nil, nil):
                break
            }
        }
        return out
    }

    private static func normal(from a: Point, to b: Point) -> Point {
        let dx = b.x - a.x, dy = b.y - a.y
        let length = (dx * dx + dy * dy).squareRoot()
        guard length > 1e-12 else { return Point(x: 0, y: 0) }
        // Left-hand normal.
        return Point(x: -dy / length, y: dx / length)
    }

    private static func cap(
        at centre: Point, from start: Point, to end: Point, radius: Double, segments: Int
    ) -> [Point] {
        let startAngle = atan2(start.y - centre.y, start.x - centre.x)
        let endAngle = atan2(end.y - centre.y, end.x - centre.x)
        var sweep = endAngle - startAngle
        while sweep <= 0 { sweep += 2 * .pi }
        guard segments > 0 else { return [] }
        return (1..<segments).map { step in
            let angle = startAngle + sweep * Double(step) / Double(segments)
            return Point(x: centre.x + radius * cos(angle), y: centre.y + radius * sin(angle))
        }
    }

    /// Shoelace area of a closed ring, in square metres.
    public static func signedArea(_ ring: [Point]) -> Double {
        guard ring.count >= 3 else { return 0 }
        var sum = 0.0
        for index in 0..<(ring.count - 1) {
            sum += ring[index].x * ring[index + 1].y - ring[index + 1].x * ring[index].y
        }
        return 0.5 * sum
    }

    /// Point-in-ring test by ray casting; used by coverage metrics.
    public static func contains(ring: [Point], point: Point) -> Bool {
        guard ring.count >= 3 else { return false }
        var inside = false
        var j = ring.count - 1
        for i in 0..<ring.count {
            let a = ring[i], b = ring[j]
            if (a.y > point.y) != (b.y > point.y) {
                let t = (point.y - a.y) / (b.y - a.y)
                if point.x < a.x + t * (b.x - a.x) { inside.toggle() }
            }
            j = i
        }
        return inside
    }
}
