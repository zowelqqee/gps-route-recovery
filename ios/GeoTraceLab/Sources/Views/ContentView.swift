import CoreLocation
import MapKit
import SwiftUI
import UniformTypeIdentifiers

/// The main screen: map, live status, and the recording controls.
struct ContentView: View {
    @EnvironmentObject private var recorder: TripRecorder

    @State private var layers = RouteLayers()
    @State private var results = ImportedResults()
    @State private var camera: MapCameraPosition = .userLocation(
        fallback: .region(MKCoordinateRegion(
            // Palace Square, so the map opens on Saint Petersburg before the
            // first fix arrives.
            center: CLLocationCoordinate2D(latitude: 59.9390, longitude: 30.3158),
            span: MKCoordinateSpan(latitudeDelta: 0.08, longitudeDelta: 0.08)
        ))
    )

    @State private var showCalibration = false
    @State private var showTrips = false
    @State private var showCamera = false
    @State private var showImporter = false
    @State private var shareURL: URL?
    @State private var reviewing: (record: PhotoRecord, url: URL)?
    @State private var message: String?

    var body: some View {
        ZStack(alignment: .top) {
            TripMapView(route: recorder.route, layers: layers, results: results, position: $camera)
                .ignoresSafeArea(edges: .top)

            VStack(alignment: .leading, spacing: 10) {
                statusCard
                LayerToggleView(
                    layers: $layers, results: results, hasRoute: recorder.route.count > 1)
            }
            .padding(.horizontal, 14)
            .padding(.top, 8)

            VStack {
                Spacer()
                controls
                    .padding(.horizontal, 14)
                    .padding(.bottom, 10)
            }
        }
        .sheet(isPresented: $showCalibration) { CalibrationView() }
        .sheet(isPresented: $showTrips) { TripListView() }
        .sheet(isPresented: $showCamera) {
            CameraPicker { image, exif in
                Task { await savePhoto(image, exif: exif) }
            }
            .ignoresSafeArea()
        }
        .sheet(item: Binding(
            get: { shareURL.map(IdentifiedURL.init) },
            set: { shareURL = $0?.url }
        )) { item in
            ShareSheet(items: [item.url])
        }
        .sheet(item: Binding(
            get: { reviewing.map { ReviewItem(record: $0.record, url: $0.url) } },
            set: { if $0 == nil { reviewing = nil } }
        )) { item in
            PhotoReviewView(record: item.record, imageURL: item.url)
        }
        .fileImporter(
            isPresented: $showImporter,
            allowedContentTypes: [.json, UTType(filenameExtension: "geojson") ?? .json],
            allowsMultipleSelection: true
        ) { result in
            importResults(result)
        }
        .alert(
            "GeoTraceLab",
            isPresented: Binding(get: { message != nil }, set: { if !$0 { message = nil } })
        ) {
            Button("OK", role: .cancel) { message = nil }
        } message: {
            Text(message ?? "")
        }
        .task { recorder.requestPermissions() }
        .onChange(of: recorder.errorMessage) { _, new in
            if let new { message = new }
        }
    }

    // MARK: - Status

    private var statusCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Circle()
                    .fill(stateColor)
                    .frame(width: 9, height: 9)
                Text(stateText)
                    .font(.subheadline.weight(.semibold))
                Spacer()
                if recorder.state == .recording {
                    Text(formatted(recorder.elapsed))
                        .font(.subheadline.monospacedDigit().weight(.semibold))
                }
            }

            HStack(spacing: 16) {
                metric("GPS", accuracyText)
                metric("Fixes", "\(recorder.locationCount)")
                metric("Motion", "\(recorder.motionCount)")
                if recorder.photoCount > 0 { metric("Photos", "\(recorder.photoCount)") }
            }

            if recorder.needsLocationPermission {
                Label(
                    "Location permission is required to record a trip.",
                    systemImage: "exclamationmark.triangle.fill"
                )
                .font(.caption)
                .foregroundStyle(.orange)
            }
            if !recorder.motion.isAvailable {
                Label(
                    "No motion sensors on this device: GPS only, no reconstruction.",
                    systemImage: "exclamationmark.triangle"
                )
                .font(.caption)
                .foregroundStyle(.orange)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    private func metric(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(value).font(.footnote.monospacedDigit().weight(.medium))
            Text(label).font(.caption2).foregroundStyle(.secondary)
        }
    }

    private var accuracyText: String {
        guard let accuracy = recorder.horizontalAccuracy, accuracy >= 0 else { return "–" }
        return String(format: "±%.0f m", accuracy)
    }

    private var stateText: String {
        switch recorder.state {
        case .idle: return "Ready"
        case .calibrating: return "Calibrating"
        case .recording: return "Recording"
        case .stopped: return recorder.isStopped ? "Stopped" : "Finished"
        }
    }

    private var stateColor: Color {
        switch recorder.state {
        case .idle: return .secondary
        case .calibrating: return .orange
        case .recording: return .red
        case .stopped: return .green
        }
    }

    // MARK: - Controls

    private var controls: some View {
        VStack(spacing: 10) {
            HStack(spacing: 10) {
                if recorder.state == .recording {
                    Button(role: .destructive) {
                        recorder.stopRecording()
                    } label: {
                        Label("Finish trip", systemImage: "stop.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                } else {
                    Button {
                        recorder.beginCalibration()
                        showCalibration = true
                    } label: {
                        Label("Start trip", systemImage: "record.circle")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(recorder.needsLocationPermission)
                }
            }

            // Deliberately unavailable while the car is moving: the driver
            // must not be invited to take a photograph mid-trip.
            HStack(spacing: 8) {
                actionButton("Photo", systemImage: "camera.fill", enabled: canTakePhoto) {
                    showCamera = true
                }
                actionButton(
                    "Export", systemImage: "square.and.arrow.up",
                    enabled: recorder.trip != nil && recorder.state != .recording
                ) {
                    exportCurrentTrip()
                }
                actionButton("Results", systemImage: "square.and.arrow.down", enabled: true) {
                    showImporter = true
                }
                actionButton("Trips", systemImage: "list.bullet", enabled: true) {
                    showTrips = true
                }
            }

            if recorder.state == .recording && !recorder.isStopped {
                Text("The camera unlocks once the car has been stopped for a few seconds.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
    }

    /// A compact icon-over-caption button; four of these fit one row on the
    /// narrowest supported phone without the captions wrapping.
    private func actionButton(
        _ title: String, systemImage: String, enabled: Bool, action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            VStack(spacing: 3) {
                Image(systemName: systemImage)
                    .font(.system(size: 17))
                Text(title)
                    .font(.caption2)
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 6)
        }
        .buttonStyle(.bordered)
        .disabled(!enabled)
    }

    /// A photo may only be taken after the trip has finished, or during a
    /// confirmed standstill.
    private var canTakePhoto: Bool {
        guard recorder.trip != nil else { return false }
        return recorder.state == .stopped || (recorder.state == .recording && recorder.isStopped)
    }

    // MARK: - Actions

    private func savePhoto(_ image: UIImage, exif: CLLocationCoordinate2D?) async {
        guard let trip = recorder.trip else { return }
        do {
            let result = try await PhotoSaver.save(
                image: image,
                into: trip,
                basename: recorder.nextPhotoName(),
                monotonicTime: recorder.location.latest?.monotonicTime,
                assumedCoordinate: recorder.lastKnownCoordinate,
                assumedAccuracy: recorder.lastKnownAccuracy,
                deviceHeadingDeg: recorder.location.latest?.course,
                exifCoordinate: exif
            )
            recorder.notePhotoSaved()
            reviewing = (result.record, result.imageURL)
        } catch {
            message = "Could not save the photo: \(error.localizedDescription)"
        }
    }

    private func exportCurrentTrip() {
        guard let trip = recorder.trip else { return }
        do {
            shareURL = try TripExporter.makeArchive(for: trip)
        } catch {
            message = error.localizedDescription
        }
    }

    private func importResults(_ result: Result<[URL], Error>) {
        switch result {
        case .failure(let error):
            message = error.localizedDescription
        case .success(let urls):
            var imported = 0
            for url in urls {
                do {
                    results.absorb(try GeoJSONImporter.load(from: url))
                    imported += 1
                } catch {
                    message = "\(url.lastPathComponent): \(error.localizedDescription)"
                }
            }
            if imported > 0 {
                message = "Imported \(imported) result layer\(imported == 1 ? "" : "s")."
            }
        }
    }

    private func formatted(_ interval: TimeInterval) -> String {
        let total = Int(interval.rounded())
        return String(format: "%02d:%02d", total / 60, total % 60)
    }
}

struct ReviewItem: Identifiable {
    let record: PhotoRecord
    let url: URL
    var id: String { record.imagePath }
}
