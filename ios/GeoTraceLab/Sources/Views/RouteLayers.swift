import CoreLocation
import Foundation
import SwiftUI

/// Which route overlays the map is currently drawing.
///
/// The original GPS is recorded on the phone. The corrupted and reconstructed
/// routes, and the probability polygons, come back from the Python processor as
/// GeoJSON and are imported.
public struct RouteLayers: Equatable {
    public var showOriginal = true
    public var showCorrupted = true
    public var showReconstructed = true
    public var showPolygons = true

    public init() {}
}

public struct ImportedResults: Equatable {
    public var corrupted: GeoJSONImporter.Layer?
    public var reconstructed: GeoJSONImporter.Layer?
    public var polygons: GeoJSONImporter.Layer?

    public init() {}

    public var isEmpty: Bool {
        corrupted == nil && reconstructed == nil && polygons == nil
    }

    /// Route the imported file into the right slot from its filename.
    ///
    /// Falls back on the geometry: a file with polygons is an uncertainty layer
    /// whatever it is called.
    public mutating func absorb(_ layer: GeoJSONImporter.Layer) {
        let name = layer.name.lowercased()
        if name.contains("polygon") || name.contains("uncertainty") {
            polygons = layer
        } else if name.contains("corrupt") || name.contains("broken") {
            corrupted = layer
        } else if name.contains("reconstruct") || name.contains("route") || name.contains("baseline") {
            reconstructed = layer
        } else if !layer.polygons.isEmpty {
            polygons = layer
        } else {
            reconstructed = layer
        }
    }
}

public enum LayerStyle {
    public static let original = Color(red: 0.10, green: 0.66, blue: 0.31)
    public static let corrupted = Color(red: 0.84, green: 0.19, blue: 0.15)
    public static let reconstructed = Color(red: 0.13, green: 0.40, blue: 0.67)
    public static let polygon = Color(red: 0.48, green: 0.20, blue: 0.58)
}
