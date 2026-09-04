import SwiftUI

/// Saved trips, with export and delete.
struct TripListView: View {
    @EnvironmentObject private var recorder: TripRecorder
    @Environment(\.dismiss) private var dismiss

    @State private var shareURL: URL?
    @State private var exportError: String?
    @State private var pendingDelete: TripDirectory?

    var body: some View {
        NavigationStack {
            Group {
                if recorder.savedTrips.isEmpty {
                    ContentUnavailableView(
                        "No trips yet",
                        systemImage: "car",
                        description: Text("Record a trip and it will appear here.")
                    )
                } else {
                    List {
                        ForEach(recorder.savedTrips) { trip in
                            row(for: trip)
                        }
                    }
                }
            }
            .navigationTitle("Saved trips")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
            .sheet(item: Binding(
                get: { shareURL.map(IdentifiedURL.init) },
                set: { shareURL = $0?.url }
            )) { item in
                ShareSheet(items: [item.url])
            }
            .alert(
                "Export failed",
                isPresented: Binding(get: { exportError != nil }, set: { if !$0 { exportError = nil } })
            ) {
                Button("OK", role: .cancel) { exportError = nil }
            } message: {
                Text(exportError ?? "")
            }
            .confirmationDialog(
                "Delete this trip?",
                isPresented: Binding(
                    get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } }),
                titleVisibility: .visible
            ) {
                Button("Delete", role: .destructive) {
                    if let trip = pendingDelete { recorder.delete(trip) }
                    pendingDelete = nil
                }
                Button("Cancel", role: .cancel) { pendingDelete = nil }
            } message: {
                Text("The recording and its photos are removed from this device. This cannot be undone.")
            }
            .onAppear { recorder.refreshSavedTrips() }
        }
    }

    private func row(for trip: TripDirectory) -> some View {
        let metadata = recorder.metadata(for: trip)
        let photos = recorder.photos(for: trip)
        return VStack(alignment: .leading, spacing: 6) {
            Text(trip.id)
                .font(.subheadline.monospaced())
                .lineLimit(1)
                .truncationMode(.middle)

            HStack(spacing: 12) {
                stat("location", "\(metadata?.locationSampleCount ?? 0) fixes")
                stat("gyroscope.fill", "\(metadata?.motionSampleCount ?? 0) motion")
                if !photos.isEmpty { stat("camera.fill", "\(photos.count)") }
            }
            .font(.caption)
            .foregroundStyle(.secondary)

            if let started = metadata?.startedAt {
                Text(started.formatted(date: .abbreviated, time: .shortened))
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            if metadata?.calibration == nil {
                Label("no mount calibration", systemImage: "exclamationmark.triangle")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }
        }
        .padding(.vertical, 3)
        .swipeActions(edge: .trailing) {
            Button(role: .destructive) { pendingDelete = trip } label: {
                Label("Delete", systemImage: "trash")
            }
            Button { export(trip) } label: {
                Label("Export", systemImage: "square.and.arrow.up")
            }
            .tint(.blue)
        }
    }

    private func stat(_ systemImage: String, _ text: String) -> some View {
        Label(text, systemImage: systemImage)
    }

    private func export(_ trip: TripDirectory) {
        do {
            shareURL = try TripExporter.makeArchive(for: trip)
        } catch {
            exportError = error.localizedDescription
        }
    }
}

/// `sheet(item:)` needs an `Identifiable`; a bare URL is not one.
struct IdentifiedURL: Identifiable {
    let url: URL
    var id: String { url.absoluteString }
}
