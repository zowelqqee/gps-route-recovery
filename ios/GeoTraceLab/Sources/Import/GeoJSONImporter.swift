import CoreLocation
import Foundation

/// Reads the processor's results back into the app.
///
/// The phone records; the reconstruction happens in Python. Bringing
/// `reconstructed-route.geojson` and `uncertainty-polygons.geojson` back lets
/// the same map show the original, the corrupted and the recovered route side
/// by side.
///
private func sameCoordinate(
    _ lhs: CLLocationCoordinate2D, _ rhs: CLLocationCoordinate2D
) -> Bool {
    lhs.latitude == rhs.latitude && lhs.longitude == rhs.longitude
}

private func coordinates(
    _ lhs: [CLLocationCoordinate2D], equal rhs: [CLLocationCoordinate2D]
) -> Bool {
    lhs.count == rhs.count && zip(lhs, rhs).allSatisfy(sameCoordinate)
}

private func coordinateLists(
    _ lhs: [[CLLocationCoordinate2D]], equal rhs: [[CLLocationCoordinate2D]]
) -> Bool {
    lhs.count == rhs.count && zip(lhs, rhs).allSatisfy { coordinates($0, equal: $1) }
}

/// Only the subset of GeoJSON the processor emits is handled - LineString,
/// Polygon, MultiPolygon and Point - which is deliberate: a general GeoJSON
/// parser is not what this needs.
public enum GeoJSONImporter {
    public struct Layer: Identifiable, Equatable {
        public let id = UUID()
        public var name: String
        public var lines: [[CLLocationCoordinate2D]]
        public var polygons: [PolygonFeature]
        public var points: [PointFeature]

        public var isEmpty: Bool { lines.isEmpty && polygons.isEmpty && points.isEmpty }

        // CLLocationCoordinate2D is a C struct and is not Equatable, so the
        // synthesised conformance is unavailable and equality is spelled out.
        public static func == (lhs: Layer, rhs: Layer) -> Bool {
            lhs.name == rhs.name
                && lhs.polygons == rhs.polygons
                && lhs.points == rhs.points
                && coordinateLists(lhs.lines, equal: rhs.lines)
        }
    }

    public struct PolygonFeature: Equatable {
        public var componentId: String?
        public var probability: Double?
        public var ring: [CLLocationCoordinate2D]
        public var holes: [[CLLocationCoordinate2D]]

        public static func == (lhs: PolygonFeature, rhs: PolygonFeature) -> Bool {
            lhs.componentId == rhs.componentId
                && lhs.probability == rhs.probability
                && coordinates(lhs.ring, equal: rhs.ring)
                && coordinateLists(lhs.holes, equal: rhs.holes)
        }
    }

    public struct PointFeature: Equatable {
        public var coordinate: CLLocationCoordinate2D
        public var label: String?

        public static func == (lhs: PointFeature, rhs: PointFeature) -> Bool {
            lhs.label == rhs.label && sameCoordinate(lhs.coordinate, rhs.coordinate)
        }
    }

    public enum ImportError: LocalizedError {
        case unreadable
        case notGeoJSON

        public var errorDescription: String? {
            switch self {
            case .unreadable: return "The file could not be read."
            case .notGeoJSON: return "That file is not GeoJSON the app understands."
            }
        }
    }

    public static func load(from url: URL, name: String? = nil) throws -> Layer {
        let needsScope = url.startAccessingSecurityScopedResource()
        defer { if needsScope { url.stopAccessingSecurityScopedResource() } }

        guard let data = try? Data(contentsOf: url) else { throw ImportError.unreadable }
        return try parse(data: data, name: name ?? url.deletingPathExtension().lastPathComponent)
    }

    public static func parse(data: Data, name: String) throws -> Layer {
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ImportError.notGeoJSON
        }
        var layer = Layer(name: name, lines: [], polygons: [], points: [])

        let features: [[String: Any]]
        if let collection = root["features"] as? [[String: Any]] {
            features = collection
        } else if root["geometry"] != nil {
            features = [root]
        } else {
            throw ImportError.notGeoJSON
        }

        for feature in features {
            guard let geometry = feature["geometry"] as? [String: Any],
                  let type = geometry["type"] as? String
            else { continue }
            let properties = feature["properties"] as? [String: Any] ?? [:]

            switch type {
            case "LineString":
                if let coordinates = geometry["coordinates"] as? [[Double]] {
                    let line = coordinates.compactMap(coordinate(from:))
                    if line.count > 1 { layer.lines.append(line) }
                }
            case "MultiLineString":
                if let parts = geometry["coordinates"] as? [[[Double]]] {
                    for part in parts {
                        let line = part.compactMap(coordinate(from:))
                        if line.count > 1 { layer.lines.append(line) }
                    }
                }
            case "Polygon":
                if let rings = geometry["coordinates"] as? [[[Double]]],
                   let polygon = polygon(from: rings, properties: properties) {
                    layer.polygons.append(polygon)
                }
            case "MultiPolygon":
                if let parts = geometry["coordinates"] as? [[[[Double]]]] {
                    for rings in parts {
                        if let polygon = polygon(from: rings, properties: properties) {
                            layer.polygons.append(polygon)
                        }
                    }
                }
            case "Point":
                if let pair = geometry["coordinates"] as? [Double],
                   let point = coordinate(from: pair) {
                    layer.points.append(
                        PointFeature(coordinate: point, label: properties["name"] as? String))
                }
            default:
                continue
            }
        }
        return layer
    }

    /// GeoJSON positions are [longitude, latitude].
    private static func coordinate(from pair: [Double]) -> CLLocationCoordinate2D? {
        guard pair.count >= 2 else { return nil }
        let coordinate = CLLocationCoordinate2D(latitude: pair[1], longitude: pair[0])
        return CLLocationCoordinate2DIsValid(coordinate) ? coordinate : nil
    }

    private static func polygon(
        from rings: [[[Double]]], properties: [String: Any]
    ) -> PolygonFeature? {
        guard let outer = rings.first else { return nil }
        let ring = outer.compactMap(coordinate(from:))
        guard ring.count > 2 else { return nil }
        let holes = rings.dropFirst().map { $0.compactMap(coordinate(from:)) }.filter { $0.count > 2 }
        return PolygonFeature(
            componentId: properties["component_id"] as? String,
            probability: properties["probability"] as? Double,
            ring: ring,
            holes: Array(holes)
        )
    }
}
