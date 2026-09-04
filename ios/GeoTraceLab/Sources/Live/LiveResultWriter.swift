import Foundation
import LiveTracking
import os

/// Streams live tracking output into the trip directory.
///
/// The `.geotrip` package gains these files alongside the existing
/// `metadata.json` and `samples.jsonl`, so an export carries both the raw
/// recording *and* what the phone concluded from it:
///
///     live-road-result.jsonl      road position per output step
///     live-parking-result.jsonl   parking estimate as the window slid
///     live-diagnostics.jsonl      every gate decision, with its reasons
///     live-final-result.json      the finished result the driver was shown
///     road-graph-metadata.json    which graph was used, and its extent
///     fault-manifest.json         only when a debug fault was armed
///
/// Older trips simply do not have these files, and the loader treats them as
/// optional, so nothing already recorded stops working.
///
/// Appends are periodic rather than at the end: an app that is killed mid-trip
/// should cost the last few seconds, not the whole drive.
final class LiveResultWriter {
    private let trip: TripDirectory
    private let queue = DispatchQueue(label: "com.geotrace.liveresults", qos: .utility)
    private let logger = Logger(subsystem: "com.geotrace.GeoTraceLab", category: "LiveResults")
    private var handles: [String: FileHandle] = [:]
    private let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return encoder
    }()

    init(trip: TripDirectory) {
        self.trip = trip
    }

    private func handle(_ name: String) -> FileHandle? {
        if let existing = handles[name] { return existing }
        let url = trip.url.appendingPathComponent(name)
        if !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: nil)
        }
        guard let handle = try? FileHandle(forWritingTo: url) else { return nil }
        try? handle.seekToEnd()
        handles[name] = handle
        return handle
    }

    private func appendLines<T: Encodable>(_ values: [T], to name: String) {
        guard !values.isEmpty else { return }
        var blob = Data()
        for value in values {
            guard let data = try? encoder.encode(value) else { continue }
            blob.append(data)
            blob.append(0x0A)
        }
        guard !blob.isEmpty, let handle = handle(name) else { return }
        do { try handle.write(contentsOf: blob) } catch {
            logger.error("live result write failed: \(error.localizedDescription, privacy: .public)")
        }
    }

    func append(_ batch: LiveRecordBatch) {
        queue.sync {
            appendLines(batch.roadPositions, to: "live-road-result.jsonl")
            appendLines(batch.parkingResults, to: "live-parking-result.jsonl")
            appendLines(batch.gateDecisions, to: "live-diagnostics.jsonl")
            appendLines(batch.lateFixes, to: "live-late-fixes.jsonl")
        }
    }

    func writeFinal(_ result: LiveTrackingResult) {
        queue.sync {
            let pretty = JSONEncoder()
            pretty.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            guard let data = try? pretty.encode(result) else { return }
            try? data.write(
                to: trip.url.appendingPathComponent("live-final-result.json"), options: .atomic
            )
        }
    }

    /// Record which graph produced the answer, so a later run can be compared
    /// against the same map rather than a newer one.
    func writeGraphMetadata() {
        queue.sync {
            guard let source = Bundle.main.url(
                forResource: "spb-graph-metadata", withExtension: "json"
            ), let data = try? Data(contentsOf: source) else { return }
            try? data.write(
                to: trip.url.appendingPathComponent("road-graph-metadata.json"), options: .atomic
            )
        }
    }

    func writeFaultManifest(_ manifest: [String: String]) {
        queue.sync {
            guard let data = try? JSONSerialization.data(
                withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys]
            ) else { return }
            try? data.write(
                to: trip.url.appendingPathComponent("fault-manifest.json"), options: .atomic
            )
        }
    }

    func close() {
        queue.sync {
            for handle in handles.values { try? handle.close() }
            handles.removeAll()
        }
    }
}
