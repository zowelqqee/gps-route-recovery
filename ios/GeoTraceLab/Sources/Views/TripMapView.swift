import CoreLocation
import LiveTracking
import MapKit
import SwiftUI

/// The MapKit view.
///
/// SwiftUI's iOS 17 `Map` is used directly: `MapPolyline` and `MapPolygon`
/// cover everything needed here, so there is no `UIViewRepresentable` bridge to
/// maintain. `MKPolyline` is still what backs `MapPolyline` underneath.
struct TripMapView: View {
    let route: [CLLocationCoordinate2D]
    let layers: RouteLayers
    let results: ImportedResults
    /// On-device causal estimate while a recording is active. Imported GeoJSON
    /// remains available for reviewing an offline reconstruction afterwards.
    let live: LiveSnapshot?
    @Binding var position: MapCameraPosition

    var body: some View {
        Map(position: $position) {
            UserAnnotation()

            // Corridors first, so the route lines stay legible over them.
            if layers.showPolygons {
                if let live, !live.roadComponents.isEmpty {
                    ForEach(Array(liveRoadRings.enumerated()), id: \.offset) { _, ring in
                        MapPolygon(coordinates: ring)
                            .foregroundStyle(LayerStyle.polygon.opacity(0.16))
                            .stroke(LayerStyle.polygon.opacity(0.55), lineWidth: 1)
                    }
                } else if let polygons = results.polygons {
                    ForEach(Array(polygons.polygons.enumerated()), id: \.offset) { _, polygon in
                        MapPolygon(coordinates: polygon.ring)
                            .foregroundStyle(LayerStyle.polygon.opacity(0.16))
                            .stroke(LayerStyle.polygon.opacity(0.55), lineWidth: 1)
                    }
                }
            }

            if layers.showOriginal, route.count > 1 {
                MapPolyline(coordinates: route)
                    .stroke(LayerStyle.original, style: .init(lineWidth: 5, lineCap: .round, lineJoin: .round))
            }

            if layers.showCorrupted, let corrupted = results.corrupted {
                ForEach(Array(corrupted.lines.enumerated()), id: \.offset) { _, line in
                    MapPolyline(coordinates: line)
                        .stroke(
                            LayerStyle.corrupted,
                            style: .init(lineWidth: 4, lineCap: .round, dash: [8, 6])
                        )
                }
            }

            if layers.showReconstructed {
                if let live, live.roadRoute.count > 1 {
                    MapPolyline(coordinates: live.roadRoute.map(clLocation))
                        .stroke(
                            LayerStyle.reconstructed,
                            style: .init(lineWidth: 5, lineCap: .round, lineJoin: .round)
                        )
                } else if let reconstructed = results.reconstructed {
                    ForEach(Array(reconstructed.lines.enumerated()), id: \.offset) { _, line in
                        MapPolyline(coordinates: line)
                            .stroke(
                                LayerStyle.reconstructed,
                                style: .init(lineWidth: 5, lineCap: .round, lineJoin: .round)
                            )
                    }
                }
            }
        }
        .mapControls {
            MapUserLocationButton()
            MapCompass()
            MapScaleView()
        }
    }

    private var liveRoadRings: [[CLLocationCoordinate2D]] {
        (live?.roadComponents ?? []).flatMap { component in
            component.rings.map { $0.map(clLocation) }
        }
    }

    private func clLocation(_ coordinate: Coordinate) -> CLLocationCoordinate2D {
        CLLocationCoordinate2D(latitude: coordinate.latitude, longitude: coordinate.longitude)
    }
}

/// Legend and layer switches.
struct LayerToggleView: View {
    @Binding var layers: RouteLayers
    let results: ImportedResults
    let hasRoute: Bool
    let live: LiveSnapshot?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            toggle(
                "Recorded GPS", color: LayerStyle.original,
                isOn: $layers.showOriginal, enabled: hasRoute
            )
            toggle(
                "Corrupted GPS", color: LayerStyle.corrupted,
                isOn: $layers.showCorrupted, enabled: results.corrupted != nil
            )
            toggle(
                "Reconstructed route", color: LayerStyle.reconstructed,
                isOn: $layers.showReconstructed,
                enabled: (live?.roadRoute.count ?? 0) > 1 || results.reconstructed != nil
            )
            toggle(
                "95% polygons", color: LayerStyle.polygon,
                isOn: $layers.showPolygons,
                enabled: !(live?.roadComponents.isEmpty ?? true) || results.polygons != nil
            )
            if results.isEmpty && (live?.roadRoute.isEmpty ?? true) {
                Text("Start a calibrated trip or import processor results.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .padding(.top, 2)
            }
        }
        .padding(10)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
    }

    private func toggle(
        _ title: String, color: Color, isOn: Binding<Bool>, enabled: Bool
    ) -> some View {
        Toggle(isOn: isOn) {
            HStack(spacing: 7) {
                Capsule()
                    .fill(color)
                    .frame(width: 18, height: 4)
                Text(title)
                    .font(.caption)
            }
        }
        .toggleStyle(.switch)
        .controlSize(.mini)
        .disabled(!enabled)
        .opacity(enabled ? 1 : 0.45)
    }
}
