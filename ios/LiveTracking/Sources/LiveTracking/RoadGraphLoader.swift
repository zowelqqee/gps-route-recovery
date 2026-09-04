import Foundation

/// Reads the compact binary road graph produced by
/// `processor/src/geotrace/graph_export.py`.
///
/// The file stores WGS84 coordinates; the loader projects them into the trip's
/// own local frame. That is deliberate: starting both implementations from
/// exactly the same latitudes and longitudes is what makes the Swift and Python
/// geometry comparable at all.
///
/// Nothing here parses XML. A 25 MB GraphML costs seconds of CPU and hundreds of
/// megabytes of peak memory, which is not something to do at the start of every
/// trip on a phone.
public enum RoadGraphLoader {
    public enum LoadError: LocalizedError {
        case unreadable(String)
        case badMagic
        case unsupportedVersion(UInt32)
        case truncated
        case emptyAfterClipping(Double)

        public var errorDescription: String? {
            switch self {
            case .unreadable(let path):
                return "Road graph could not be read: \(path)"
            case .badMagic:
                return "That file is not a GeoTrace road graph."
            case .unsupportedVersion(let version):
                return "Road graph format version \(version) is not supported by this build."
            case .truncated:
                return "The road graph file is truncated."
            case .emptyAfterClipping(let radius):
                return "No road within \(Int(radius)) m of the trip origin. "
                    + "The graph does not cover this area."
            }
        }
    }

    struct Header {
        var nodeCount: Int
        var edgeCount: Int
        var pointCount: Int
        var gridCols: Int
        var gridRows: Int
        var minLat: Double
        var minLon: Double
        var maxLat: Double
        var maxLon: Double
    }

    static let magic = Array("GTRGRAPH".utf8)
    /// `HEADER_STRUCT.size` in graph_export.py: 8s + 8 UInt32 + 6 Double.
    static let headerSize = 88

    /// Read the header only, to answer coverage questions before committing to
    /// projecting anything.
    public static func readBounds(at url: URL) throws
        -> (minLat: Double, minLon: Double, maxLat: Double, maxLon: Double, edges: Int)
    {
        guard let handle = try? FileHandle(forReadingFrom: url),
              let data = try handle.read(upToCount: headerSize), data.count == headerSize
        else { throw LoadError.unreadable(url.path) }
        try? handle.close()
        let header = try parseHeader(data)
        return (header.minLat, header.minLon, header.maxLat, header.maxLon, header.edgeCount)
    }

    private static func parseHeader(_ data: Data) throws -> Header {
        guard data.count >= headerSize else { throw LoadError.truncated }
        return try data.withUnsafeBytes { raw -> Header in
            for (index, byte) in magic.enumerated() where raw[index] != byte {
                throw LoadError.badMagic
            }
            let version = raw.loadUnaligned(fromByteOffset: 8, as: UInt32.self)
            guard version == 1 else { throw LoadError.unsupportedVersion(version) }
            return Header(
                nodeCount: Int(raw.loadUnaligned(fromByteOffset: 16, as: UInt32.self)),
                edgeCount: Int(raw.loadUnaligned(fromByteOffset: 20, as: UInt32.self)),
                pointCount: Int(raw.loadUnaligned(fromByteOffset: 24, as: UInt32.self)),
                gridCols: Int(raw.loadUnaligned(fromByteOffset: 28, as: UInt32.self)),
                gridRows: Int(raw.loadUnaligned(fromByteOffset: 32, as: UInt32.self)),
                minLat: raw.loadUnaligned(fromByteOffset: 40, as: Double.self),
                minLon: raw.loadUnaligned(fromByteOffset: 48, as: Double.self),
                maxLat: raw.loadUnaligned(fromByteOffset: 56, as: Double.self),
                maxLon: raw.loadUnaligned(fromByteOffset: 64, as: Double.self)
            )
        }
    }

    /// Load and project a graph into a frame anchored at `origin`.
    ///
    /// Only edges whose polyline comes within `radiusM` of the origin are kept,
    /// which is what makes a city-wide graph affordable: a 12 km disc around a
    /// trip is a fraction of Saint Petersburg.
    public static func load(
        contentsOf url: URL, origin: Coordinate, radiusM: Double = 12_000
    ) throws -> RoadNetwork {
        guard let data = try? Data(contentsOf: url, options: [.mappedIfSafe]) else {
            throw LoadError.unreadable(url.path)
        }
        return try load(data: data, origin: origin, radiusM: radiusM)
    }

    public static func load(
        data: Data, origin: Coordinate, radiusM: Double
    ) throws -> RoadNetwork {
        let header = try parseHeader(data)
        let frame = LocalFrame(latitude: origin.latitude, longitude: origin.longitude)

        // Section offsets, matching graph_export.py's layout.
        var cursor = headerSize
        let nodeLatOffset = cursor; cursor += header.nodeCount * 8
        let nodeLonOffset = cursor; cursor += header.nodeCount * 8
        let edgeOffset = cursor; cursor += header.edgeCount * 32
        cursor = (cursor + 7) & ~7
        let pointLatOffset = cursor; cursor += header.pointCount * 8
        let pointLonOffset = cursor; cursor += header.pointCount * 8
        _ = nodeLatOffset; _ = nodeLonOffset
        let adjacencyOffsetsOffset = cursor; cursor += (header.nodeCount + 1) * 4
        let adjacencyEdgesOffset = cursor; cursor += header.edgeCount * 4
        guard data.count >= cursor else { throw LoadError.truncated }

        // A cheap lat/lon pre-filter, so the expensive geodesic projection only
        // runs on edges that can possibly be in range.
        let latPadding = radiusM / 110_000.0
        let lonPadding = radiusM / (111_320.0 * Swift.max(0.2, cos(origin.latitude * .pi / 180)))
        let minLat = origin.latitude - latPadding, maxLat = origin.latitude + latPadding
        let minLon = origin.longitude - lonPadding, maxLon = origin.longitude + lonPadding

        var edges: [RoadEdge] = []
        var points: [Point] = []
        var cumulative: [Double] = []
        var bearings: [Double] = []
        var bearingOffsets: [Int] = []
        var keptIndexByOriginal = [Int32](repeating: -1, count: header.edgeCount)
        var nodeRemap: [Int32: Int] = [:]
        var originalEdges: [(u: Int, v: Int, offset: Int, count: Int, speed: Double,
                             oneway: Bool, roadClass: Int, reverse: Int)] = []

        data.withUnsafeBytes { raw in
            let pointLat = raw.baseAddress!.advanced(by: pointLatOffset)
                .assumingMemoryBound(to: Double.self)
            let pointLon = raw.baseAddress!.advanced(by: pointLonOffset)
                .assumingMemoryBound(to: Double.self)

            for index in 0..<header.edgeCount {
                let base = edgeOffset + index * 32
                let u = Int(raw.loadUnaligned(fromByteOffset: base, as: UInt32.self))
                let v = Int(raw.loadUnaligned(fromByteOffset: base + 4, as: UInt32.self))
                let pointStart = Int(raw.loadUnaligned(fromByteOffset: base + 8, as: UInt32.self))
                let pointCount = Int(raw.loadUnaligned(fromByteOffset: base + 12, as: UInt32.self))
                let speed = Double(raw.loadUnaligned(fromByteOffset: base + 20, as: Float.self))
                let reverse = Int(raw.loadUnaligned(fromByteOffset: base + 24, as: Int32.self))
                let oneway = raw.loadUnaligned(fromByteOffset: base + 28, as: UInt8.self) != 0
                let roadClass = Int(raw.loadUnaligned(fromByteOffset: base + 29, as: UInt8.self))

                var inRange = false
                for i in pointStart..<(pointStart + pointCount) {
                    let lat = pointLat[i], lon = pointLon[i]
                    if lat >= minLat && lat <= maxLat && lon >= minLon && lon <= maxLon {
                        inRange = true
                        break
                    }
                }
                guard inRange else { continue }

                keptIndexByOriginal[index] = Int32(originalEdges.count)
                originalEdges.append(
                    (u, v, pointStart, pointCount, Double(speed), oneway, roadClass, reverse)
                )
            }

            // Project only the surviving edges.
            points.reserveCapacity(originalEdges.reduce(0) { $0 + $1.count })
            for (keptIndex, record) in originalEdges.enumerated() {
                let offset = points.count
                var runningLength = 0.0
                cumulative.append(0)
                for i in record.offset..<(record.offset + record.count) {
                    let projected = frame.toLocal(
                        latitude: pointLat[i], longitude: pointLon[i]
                    )
                    if i > record.offset {
                        let previous = points[points.count - 1]
                        runningLength += projected.distance(to: previous)
                        cumulative.append(runningLength)
                        let dx = projected.x - previous.x, dy = projected.y - previous.y
                        bearings.append(atan2(dy, dx))
                    }
                    points.append(projected)
                }
                bearingOffsets.append(bearings.count - (record.count - 1))
                if !nodeRemap.keys.contains(Int32(record.u)) {
                    nodeRemap[Int32(record.u)] = nodeRemap.count
                }
                if !nodeRemap.keys.contains(Int32(record.v)) {
                    nodeRemap[Int32(record.v)] = nodeRemap.count
                }
                edges.append(
                    RoadEdge(
                        index: keptIndex,
                        startNode: nodeRemap[Int32(record.u)]!,
                        endNode: nodeRemap[Int32(record.v)]!,
                        pointOffset: offset,
                        pointCount: record.count,
                        length: runningLength,
                        speedLimitMS: record.speed,
                        isOneway: record.oneway,
                        roadClass: record.roadClass,
                        reverseEdge: -1,
                        nameID: -1
                    )
                )
            }
        }

        guard !edges.isEmpty else { throw LoadError.emptyAfterClipping(radiusM) }

        // Remap the reverse-twin pointers into the clipped index space.
        for index in edges.indices {
            let original = originalEdges[index].reverse
            if original >= 0, original < keptIndexByOriginal.count {
                let kept = keptIndexByOriginal[original]
                edges[index].reverseEdge = kept >= 0 ? Int(kept) : -1
            }
        }

        // Rebuild CSR adjacency over the clipped node numbering.
        let nodeCount = nodeRemap.count
        var buckets = [[Int32]](repeating: [], count: nodeCount)
        for edge in edges { buckets[edge.startNode].append(Int32(edge.index)) }
        var adjacencyOffsets: [Int] = [0]
        var adjacencyEdges: [Int32] = []
        for bucket in buckets {
            adjacencyEdges.append(contentsOf: bucket)
            adjacencyOffsets.append(adjacencyEdges.count)
        }
        _ = adjacencyOffsetsOffset
        _ = adjacencyEdgesOffset

        return RoadNetwork(
            frame: frame,
            edges: edges,
            points: points,
            cumulative: cumulative,
            bearings: bearings,
            bearingOffsets: bearingOffsets,
            adjacencyOffsets: adjacencyOffsets,
            adjacencyEdges: adjacencyEdges,
            edgeNames: [],
            geographicBounds: (header.minLat, header.minLon, header.maxLat, header.maxLon),
            gridCellSize: 200
        )
    }
}
