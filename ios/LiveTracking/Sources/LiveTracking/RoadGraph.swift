import Foundation

/// Route prior over road classes.
///
/// Without turn restrictions or traffic data the only honest prior is that a
/// driver is somewhat more likely to stay on a larger road. Mirrors
/// `geotrace.particle_filter.ROUTE_PRIOR`; the index order must match
/// `graph_export.ROAD_CLASSES`.
public enum RoadClass {
    public static let names = [
        "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
        "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
        "residential", "living_street", "service", "road", "unknown",
    ]

    public static let routePrior: [Double] = [
        1.6, 1.2, 1.5, 1.2, 1.4, 1.1,
        1.25, 1.05, 1.1, 1.0, 1.0,
        1.0, 0.8, 0.6, 1.0, 1.0,
    ]

    public static func name(_ index: Int) -> String {
        index >= 0 && index < names.count ? names[index] : "unknown"
    }

    public static func prior(_ index: Int) -> Double {
        index >= 0 && index < routePrior.count ? routePrior[index] : 1.0
    }
}

/// One directed road segment, pre-projected into the trip's local frame.
///
/// Mirrors `geotrace.road_graph.Edge`.
public struct RoadEdge: Sendable {
    public var index: Int
    public var startNode: Int
    public var endNode: Int
    /// Offset into `RoadNetwork.points`.
    public var pointOffset: Int
    public var pointCount: Int
    public var length: Double
    public var speedLimitMS: Double
    public var isOneway: Bool
    public var roadClass: Int
    /// Index of the edge that is this one driven backwards, or -1. Used to
    /// forbid U-turns.
    public var reverseEdge: Int
    public var nameID: Int32
}

/// Local-frame view of a drivable road graph.
///
/// A port of `geotrace.road_graph.RoadNetwork`, answering the three questions
/// the particle filter asks millions of times: where is a given
/// `(edge, distance along)` in metres, which edges leave a node, and which edges
/// are near a point.
public final class RoadNetwork: @unchecked Sendable {
    public let frame: LocalFrame
    public private(set) var edges: [RoadEdge] = []
    /// Flattened polyline points of every edge, in local metres.
    public private(set) var points: [Point] = []
    /// Cumulative distance along each edge's polyline, parallel to `points`.
    public private(set) var cumulative: [Double] = []
    /// Bearing of each polyline segment, radians CCW from East. There is one
    /// fewer of these per edge than there are points.
    public private(set) var bearings: [Double] = []
    private var bearingOffsets: [Int] = []
    /// CSR adjacency: outgoing edges per node.
    private var adjacencyOffsets: [Int] = []
    private var adjacencyEdges: [Int32] = []
    private var edgeNames: [String] = []

    /// Uniform grid over the local frame for nearest-edge queries. A linear scan
    /// over 41 000 edges on every GPS update is not affordable on a phone.
    private var gridCellSize: Double = 200
    private var gridMinX: Double = 0
    private var gridMinY: Double = 0
    private var gridCols: Int = 1
    private var gridRows: Int = 1
    private var cellOffsets: [Int] = []
    private var cellEdges: [Int32] = []

    public private(set) var bounds: (minX: Double, minY: Double, maxX: Double, maxY: Double) =
        (0, 0, 0, 0)
    public private(set) var geographicBounds: (minLat: Double, minLon: Double, maxLat: Double, maxLon: Double) =
        (0, 0, 0, 0)

    public var edgeCount: Int { edges.count }

    init(
        frame: LocalFrame,
        edges: [RoadEdge],
        points: [Point],
        cumulative: [Double],
        bearings: [Double],
        bearingOffsets: [Int],
        adjacencyOffsets: [Int],
        adjacencyEdges: [Int32],
        edgeNames: [String],
        geographicBounds: (minLat: Double, minLon: Double, maxLat: Double, maxLon: Double),
        gridCellSize: Double
    ) {
        self.frame = frame
        self.edges = edges
        self.points = points
        self.cumulative = cumulative
        self.bearings = bearings
        self.bearingOffsets = bearingOffsets
        self.adjacencyOffsets = adjacencyOffsets
        self.adjacencyEdges = adjacencyEdges
        self.edgeNames = edgeNames
        self.geographicBounds = geographicBounds
        self.gridCellSize = gridCellSize
        buildGrid()
    }

    // MARK: - Geometry

    /// Point at distance `s` along `edge`, clamped to the edge.
    public func position(edge index: Int, s: Double) -> Point {
        let edge = edges[index]
        let clamped = Swift.min(Swift.max(s, 0), edge.length)
        let segment = segmentIndex(edge: edge, s: clamped)
        let start = cumulative[segment]
        let end = cumulative[segment + 1]
        let span = end - start
        let fraction = span > 1e-9 ? (clamped - start) / span : 0
        let p0 = points[segment], p1 = points[segment + 1]
        return Point(x: p0.x + fraction * (p1.x - p0.x), y: p0.y + fraction * (p1.y - p0.y))
    }

    /// Heading of the road at distance `s` along `edge`.
    public func bearing(edge index: Int, s: Double) -> Double {
        let edge = edges[index]
        let clamped = Swift.min(Swift.max(s, 0), edge.length)
        let segment = segmentIndex(edge: edge, s: clamped)
        let local = segment - edge.pointOffset
        let bearingIndex = bearingOffsets[index] + Swift.min(local, Swift.max(0, edge.pointCount - 2))
        return bearings[Swift.min(bearingIndex, bearings.count - 1)]
    }

    public func startBearing(edge index: Int) -> Double {
        bearings[bearingOffsets[index]]
    }

    private func segmentIndex(edge: RoadEdge, s: Double) -> Int {
        // Binary search within this edge's own cumulative slice.
        var low = edge.pointOffset
        var high = edge.pointOffset + edge.pointCount - 1
        while low + 1 < high {
            let mid = (low + high) / 2
            if cumulative[mid] <= s { low = mid } else { high = mid }
        }
        return Swift.min(low, edge.pointOffset + edge.pointCount - 2)
    }

    public func name(edge index: Int) -> String? {
        let id = edges[index].nameID
        guard id >= 0, Int(id) < edgeNames.count else { return nil }
        let name = edgeNames[Int(id)]
        return name.isEmpty ? nil : name
    }

    // MARK: - Topology

    /// Edges leaving the far node of `edge`.
    ///
    /// Direction of travel is already encoded in the directed graph, so an edge
    /// simply does not exist against a one-way street. The reverse twin of the
    /// current edge is excluded unless a U-turn is explicitly allowed.
    public func successors(edge index: Int, allowUTurn: Bool = false) -> ArraySlice<Int32> {
        let node = Int(edges[index].endNode)
        guard node >= 0, node + 1 < adjacencyOffsets.count else { return [][...] }
        let slice = adjacencyEdges[adjacencyOffsets[node]..<adjacencyOffsets[node + 1]]
        return slice
    }

    /// Successors with the U-turn twin removed, materialised.
    public func forwardSuccessors(edge index: Int, allowUTurn: Bool) -> [Int32] {
        let all = successors(edge: index, allowUTurn: allowUTurn)
        if allowUTurn { return Array(all) }
        let edge = edges[index]
        return all.filter { candidate in
            let other = edges[Int(candidate)]
            return !(other.startNode == edge.endNode && other.endNode == edge.startNode)
        }
    }

    // MARK: - Spatial queries

    private func buildGrid() {
        guard !points.isEmpty else {
            cellOffsets = [0]
            return
        }
        var minX = Double.greatestFiniteMagnitude, minY = Double.greatestFiniteMagnitude
        var maxX = -Double.greatestFiniteMagnitude, maxY = -Double.greatestFiniteMagnitude
        for point in points {
            minX = Swift.min(minX, point.x); maxX = Swift.max(maxX, point.x)
            minY = Swift.min(minY, point.y); maxY = Swift.max(maxY, point.y)
        }
        bounds = (minX, minY, maxX, maxY)
        gridMinX = minX
        gridMinY = minY
        gridCols = Swift.max(1, Int(((maxX - minX) / gridCellSize).rounded(.up)) + 1)
        gridRows = Swift.max(1, Int(((maxY - minY) / gridCellSize).rounded(.up)) + 1)

        var buckets = [[Int32]](repeating: [], count: gridCols * gridRows)
        for edge in edges {
            var seen = Set<Int>()
            for i in edge.pointOffset..<(edge.pointOffset + edge.pointCount) {
                var cells: Set<Int> = [cellIndex(points[i])]
                if i + 1 < edge.pointOffset + edge.pointCount {
                    // Register the segment midpoint too, so a long straight edge
                    // is present in the cells it merely passes through.
                    let mid = Point(
                        x: 0.5 * (points[i].x + points[i + 1].x),
                        y: 0.5 * (points[i].y + points[i + 1].y)
                    )
                    cells.insert(cellIndex(mid))
                }
                for cell in cells where !seen.contains(cell) {
                    seen.insert(cell)
                    buckets[cell].append(Int32(edge.index))
                }
            }
        }
        cellOffsets = [0]
        cellOffsets.reserveCapacity(buckets.count + 1)
        cellEdges.reserveCapacity(buckets.reduce(0) { $0 + $1.count })
        for bucket in buckets {
            cellEdges.append(contentsOf: bucket)
            cellOffsets.append(cellEdges.count)
        }
    }

    private func cellIndex(_ point: Point) -> Int {
        let col = Swift.min(gridCols - 1, Swift.max(0, Int((point.x - gridMinX) / gridCellSize)))
        let row = Swift.min(gridRows - 1, Swift.max(0, Int((point.y - gridMinY) / gridCellSize)))
        return row * gridCols + col
    }

    /// Candidate edges within `radius` of `point`, from the grid.
    private func candidates(near point: Point, radius: Double) -> [Int32] {
        guard !cellEdges.isEmpty else { return [] }
        let span = Swift.max(1, Int((radius / gridCellSize).rounded(.up)))
        let col = Swift.min(gridCols - 1, Swift.max(0, Int((point.x - gridMinX) / gridCellSize)))
        let row = Swift.min(gridRows - 1, Swift.max(0, Int((point.y - gridMinY) / gridCellSize)))
        var seen = Set<Int32>()
        var result: [Int32] = []
        for r in Swift.max(0, row - span)...Swift.min(gridRows - 1, row + span) {
            for c in Swift.max(0, col - span)...Swift.min(gridCols - 1, col + span) {
                let cell = r * gridCols + c
                for edge in cellEdges[cellOffsets[cell]..<cellOffsets[cell + 1]] where !seen.contains(edge) {
                    seen.insert(edge)
                    result.append(edge)
                }
            }
        }
        return result
    }

    /// Squared distance from a point to an edge's polyline, plus the distance
    /// along the edge of the closest point.
    public func project(_ point: Point, onto index: Int) -> (s: Double, offset: Double) {
        let edge = edges[index]
        var bestOffsetSquared = Double.greatestFiniteMagnitude
        var bestS = 0.0
        for i in edge.pointOffset..<(edge.pointOffset + edge.pointCount - 1) {
            let p0 = points[i], p1 = points[i + 1]
            let dx = p1.x - p0.x, dy = p1.y - p0.y
            let lengthSquared = dx * dx + dy * dy
            var t = 0.0
            if lengthSquared > 1e-12 {
                t = ((point.x - p0.x) * dx + (point.y - p0.y) * dy) / lengthSquared
                t = Swift.min(Swift.max(t, 0), 1)
            }
            let cx = p0.x + t * dx, cy = p0.y + t * dy
            let ex = point.x - cx, ey = point.y - cy
            let distanceSquared = ex * ex + ey * ey
            if distanceSquared < bestOffsetSquared {
                bestOffsetSquared = distanceSquared
                bestS = cumulative[i] - cumulative[edge.pointOffset]
                    + (lengthSquared > 1e-12 ? t * lengthSquared.squareRoot() : 0)
            }
        }
        return (bestS, bestOffsetSquared.squareRoot())
    }

    /// Indices of the `k` nearest edges to a local-frame point.
    public func nearestEdges(to point: Point, k: Int = 5, radius: Double = 250) -> [Int] {
        var searchRadius = radius
        for _ in 0..<4 {
            let found = candidates(near: point, radius: searchRadius)
            if !found.isEmpty {
                let scored = found.map { (index: Int($0), offset: project(point, onto: Int($0)).offset) }
                    .sorted { $0.offset < $1.offset }
                return scored.prefix(k).map(\.index)
            }
            searchRadius *= 3
        }
        return []
    }

    /// Metres from a point to the closest drivable edge.
    public func distanceToRoad(_ point: Point) -> Double {
        var searchRadius = 250.0
        for _ in 0..<5 {
            let found = candidates(near: point, radius: searchRadius)
            if !found.isEmpty {
                return found.map { project(point, onto: Int($0)).offset }.min() ?? .infinity
            }
            searchRadius *= 3
        }
        return .infinity
    }

    /// Best `(edge, s, offset)` for a point, or nil if nothing is near.
    public func snap(_ point: Point, k: Int = 5) -> (edge: Int, s: Double, offset: Double)? {
        var best: (edge: Int, s: Double, offset: Double)?
        for index in nearestEdges(to: point, k: k) {
            let projection = project(point, onto: index)
            if best == nil || projection.offset < best!.offset {
                best = (index, projection.s, projection.offset)
            }
        }
        return best
    }

    /// True when a WGS84 coordinate falls inside the graph's own bounding box.
    public func covers(latitude: Double, longitude: Double) -> Bool {
        latitude >= geographicBounds.minLat && latitude <= geographicBounds.maxLat
            && longitude >= geographicBounds.minLon && longitude <= geographicBounds.maxLon
    }
}

// MARK: - Synthetic graphs

extension RoadNetwork {
    /// Build a network directly from local-metre polylines.
    ///
    /// The counterpart of `geotrace.road_graph.build_graph_from_segments`. Used
    /// by the tests, where the expected answer has to be workable out on paper,
    /// and available as a fallback when no binary graph is present.
    ///
    /// A segment is created in both directions unless `oneway` is set, which is
    /// how a one-way street becomes structurally impossible to drive up rather
    /// than merely penalised.
    public struct SyntheticSegment {
        public var name: String
        public var points: [Point]
        public var oneway: Bool
        public var roadClass: Int
        public var speedLimitMS: Double

        public init(
            name: String, points: [Point], oneway: Bool = false,
            roadClass: Int = 11, speedLimitMS: Double = 11.1
        ) {
            self.name = name
            self.points = points
            self.oneway = oneway
            self.roadClass = roadClass
            self.speedLimitMS = speedLimitMS
        }
    }

    public static func synthetic(
        segments: [SyntheticSegment],
        origin: Coordinate = Coordinate(latitude: 59.9311, longitude: 30.3609)
    ) -> RoadNetwork {
        let frame = LocalFrame(latitude: origin.latitude, longitude: origin.longitude)
        var nodeIndex: [String: Int] = [:]
        func node(for point: Point) -> Int {
            let key = "\(Int((point.x * 100).rounded())):\(Int((point.y * 100).rounded()))"
            if let existing = nodeIndex[key] { return existing }
            let created = nodeIndex.count
            nodeIndex[key] = created
            return created
        }

        var edges: [RoadEdge] = []
        var points: [Point] = []
        var cumulative: [Double] = []
        var bearings: [Double] = []
        var bearingOffsets: [Int] = []
        var names: [String] = []

        func addEdge(_ polyline: [Point], _ segment: SyntheticSegment) {
            guard polyline.count >= 2 else { return }
            let offset = points.count
            var length = 0.0
            cumulative.append(0)
            points.append(polyline[0])
            for index in 1..<polyline.count {
                length += polyline[index].distance(to: polyline[index - 1])
                cumulative.append(length)
                let dx = polyline[index].x - polyline[index - 1].x
                let dy = polyline[index].y - polyline[index - 1].y
                bearings.append(atan2(dy, dx))
                points.append(polyline[index])
            }
            bearingOffsets.append(bearings.count - (polyline.count - 1))
            names.append(segment.name)
            edges.append(
                RoadEdge(
                    index: edges.count,
                    startNode: node(for: polyline[0]),
                    endNode: node(for: polyline[polyline.count - 1]),
                    pointOffset: offset,
                    pointCount: polyline.count,
                    length: length,
                    speedLimitMS: segment.speedLimitMS,
                    isOneway: segment.oneway,
                    roadClass: segment.roadClass,
                    reverseEdge: -1,
                    nameID: Int32(names.count - 1)
                )
            )
        }

        for segment in segments {
            for index in 0..<(segment.points.count - 1) {
                let pair = [segment.points[index], segment.points[index + 1]]
                addEdge(pair, segment)
                if !segment.oneway { addEdge(pair.reversed(), segment) }
            }
        }

        var byPair: [String: Int] = [:]
        for edge in edges { byPair["\(edge.startNode)>\(edge.endNode)"] = edge.index }
        for index in edges.indices {
            let key = "\(edges[index].endNode)>\(edges[index].startNode)"
            edges[index].reverseEdge = byPair[key] ?? -1
        }

        var buckets = [[Int32]](repeating: [], count: nodeIndex.count)
        for edge in edges { buckets[edge.startNode].append(Int32(edge.index)) }
        var adjacencyOffsets: [Int] = [0]
        var adjacencyEdges: [Int32] = []
        for bucket in buckets {
            adjacencyEdges.append(contentsOf: bucket)
            adjacencyOffsets.append(adjacencyEdges.count)
        }

        return RoadNetwork(
            frame: frame,
            edges: edges,
            points: points,
            cumulative: cumulative,
            bearings: bearings,
            bearingOffsets: bearingOffsets,
            adjacencyOffsets: adjacencyOffsets,
            adjacencyEdges: adjacencyEdges,
            edgeNames: names,
            geographicBounds: (
                origin.latitude - 0.5, origin.longitude - 0.5,
                origin.latitude + 0.5, origin.longitude + 0.5
            ),
            gridCellSize: 100
        )
    }
}
