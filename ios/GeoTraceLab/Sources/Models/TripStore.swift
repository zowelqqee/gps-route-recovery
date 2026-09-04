import Foundation

/// On-disk layout of one recorded trip.
///
///     trip-<UUID>/
///       metadata.json
///       samples.jsonl
///       photos/photo-001.jpg + photo-001.json
///       results/
///
/// This is exactly what the Python processor reads, with no manual editing.
public struct TripDirectory: Identifiable, Hashable, Sendable {
    public let id: String
    public let url: URL

    public init(id: String, url: URL) {
        self.id = id
        self.url = url
    }

    public var metadataURL: URL { url.appendingPathComponent("metadata.json") }
    public var samplesURL: URL { url.appendingPathComponent("samples.jsonl") }
    public var photosURL: URL { url.appendingPathComponent("photos", isDirectory: true) }
    public var resultsURL: URL { url.appendingPathComponent("results", isDirectory: true) }
}

/// Creates, lists and deletes trip directories in the app's Documents folder.
///
/// Documents is used (with `UIFileSharingEnabled`) so a trip can also be pulled
/// off the device with Finder or the Files app when AirDrop is inconvenient.
public final class TripStore {
    public static let shared = TripStore()

    private let fileManager: FileManager
    public let root: URL

    public init(fileManager: FileManager = .default, root: URL? = nil) {
        self.fileManager = fileManager
        if let root {
            self.root = root
        } else {
            let documents = fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
            self.root = documents.appendingPathComponent("Trips", isDirectory: true)
        }
        try? fileManager.createDirectory(at: self.root, withIntermediateDirectories: true)
    }

    @discardableResult
    public func createTrip(id: UUID = UUID()) throws -> TripDirectory {
        let name = "trip-\(id.uuidString.lowercased())"
        let url = root.appendingPathComponent(name, isDirectory: true)
        try fileManager.createDirectory(at: url, withIntermediateDirectories: true)
        let trip = TripDirectory(id: name, url: url)
        try fileManager.createDirectory(at: trip.photosURL, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: trip.resultsURL, withIntermediateDirectories: true)
        // Create the sample file immediately so appending never has to branch.
        if !fileManager.fileExists(atPath: trip.samplesURL.path) {
            fileManager.createFile(atPath: trip.samplesURL.path, contents: nil)
        }
        return trip
    }

    /// All trips, newest first.
    public func listTrips() -> [TripDirectory] {
        let contents = (try? fileManager.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        )) ?? []

        return contents
            .filter { $0.hasDirectoryPath && $0.lastPathComponent.hasPrefix("trip-") }
            .map { TripDirectory(id: $0.lastPathComponent, url: $0) }
            .sorted { modificationDate($0.url) > modificationDate($1.url) }
    }

    public func delete(_ trip: TripDirectory) throws {
        try fileManager.removeItem(at: trip.url)
    }

    public func writeMetadata(_ metadata: TripMetadata, to trip: TripDirectory) throws {
        let data = try TripJSON.encoder(prettyPrinted: true).encode(metadata)
        try data.write(to: trip.metadataURL, options: .atomic)
    }

    public func readMetadata(from trip: TripDirectory) -> TripMetadata? {
        guard let data = try? Data(contentsOf: trip.metadataURL) else { return nil }
        return try? TripJSON.decoder().decode(TripMetadata.self, from: data)
    }

    public func readPhotos(from trip: TripDirectory) -> [PhotoRecord] {
        let sidecars = (try? fileManager.contentsOfDirectory(
            at: trip.photosURL, includingPropertiesForKeys: nil
        )) ?? []
        return sidecars
            .filter { $0.pathExtension.lowercased() == "json" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            .compactMap { url in
                guard let data = try? Data(contentsOf: url) else { return nil }
                return try? TripJSON.decoder().decode(PhotoRecord.self, from: data)
            }
    }

    /// Next free `photo-NNN` basename in a trip.
    public func nextPhotoName(in trip: TripDirectory) -> String {
        let existing = (try? fileManager.contentsOfDirectory(
            at: trip.photosURL, includingPropertiesForKeys: nil
        )) ?? []
        let numbers = existing.compactMap { url -> Int? in
            let name = url.deletingPathExtension().lastPathComponent
            guard name.hasPrefix("photo-") else { return nil }
            return Int(name.dropFirst("photo-".count))
        }
        return String(format: "photo-%03d", (numbers.max() ?? 0) + 1)
    }

    public func sizeOnDisk(of trip: TripDirectory) -> Int64 {
        guard let enumerator = fileManager.enumerator(
            at: trip.url, includingPropertiesForKeys: [.fileSizeKey]
        ) else { return 0 }
        var total: Int64 = 0
        for case let url as URL in enumerator {
            let size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            total += Int64(size)
        }
        return total
    }

    private func modificationDate(_ url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate
            ?? .distantPast
    }
}
