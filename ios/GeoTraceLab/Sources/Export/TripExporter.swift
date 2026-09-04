import Foundation

/// Packages a trip directory into a single `.zip` for the share sheet.
///
/// Uses `NSFileCoordinator`'s `.forUploading` option, which is the system's own
/// directory-to-zip path. That keeps the dependency list empty - no
/// ZIPFoundation - and produces an archive the Python processor opens directly:
///
///     geotrace reconstruct --trip trip-<UUID>.zip ...
public enum TripExporter {
    public enum ExportError: LocalizedError {
        case coordinationFailed(String)
        case nothingToExport

        public var errorDescription: String? {
            switch self {
            case .coordinationFailed(let message):
                return "Could not package the trip: \(message)"
            case .nothingToExport:
                return "This trip has no recorded samples yet."
            }
        }
    }

    /// Zip `trip` into a temporary file and return its URL.
    ///
    /// The caller keeps the URL alive for as long as the share sheet is up;
    /// everything lives under a unique temporary directory so repeated exports
    /// never collide.
    public static func makeArchive(
        for trip: TripDirectory,
        fileManager: FileManager = .default
    ) throws -> URL {
        let samples = trip.samplesURL
        let size = (try? fileManager.attributesOfItem(atPath: samples.path)[.size] as? Int) ?? 0
        guard fileManager.fileExists(atPath: samples.path), (size ?? 0) > 0 else {
            throw ExportError.nothingToExport
        }

        let staging = fileManager.temporaryDirectory
            .appendingPathComponent("export-\(UUID().uuidString)", isDirectory: true)
        try fileManager.createDirectory(at: staging, withIntermediateDirectories: true)
        let destination = staging.appendingPathComponent("\(trip.id).zip")

        var coordinationError: NSError?
        var thrown: Error?
        NSFileCoordinator().coordinate(
            readingItemAt: trip.url, options: [.forUploading], error: &coordinationError
        ) { zippedURL in
            do {
                try fileManager.copyItem(at: zippedURL, to: destination)
            } catch {
                thrown = error
            }
        }
        if let coordinationError {
            throw ExportError.coordinationFailed(coordinationError.localizedDescription)
        }
        if let thrown {
            throw ExportError.coordinationFailed(thrown.localizedDescription)
        }
        return destination
    }

    /// Human-readable size of a trip, for the trip list.
    public static func formattedSize(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useKB, .useMB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }
}
