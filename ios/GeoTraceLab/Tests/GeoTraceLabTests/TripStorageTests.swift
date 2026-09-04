import CoreLocation
import XCTest

@testable import GeoTraceLab

/// Directory layout, JSONL writing and export packaging.
final class TripStorageTests: XCTestCase {
    private var root: URL!
    private var store: TripStore!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("geotrace-tests-\(UUID().uuidString)", isDirectory: true)
        store = TripStore(root: root)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    func testCreatingATripMakesTheDocumentedLayout() throws {
        let trip = try store.createTrip()
        XCTAssertTrue(trip.id.hasPrefix("trip-"))
        XCTAssertTrue(FileManager.default.fileExists(atPath: trip.url.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: trip.samplesURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: trip.photosURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: trip.resultsURL.path))
    }

    func testEachTripGetsItsOwnDirectory() throws {
        let first = try store.createTrip()
        let second = try store.createTrip()
        XCTAssertNotEqual(first.id, second.id)
        XCTAssertEqual(store.listTrips().count, 2)
    }

    func testTripsAreListedNewestFirst() throws {
        _ = try store.createTrip()
        Thread.sleep(forTimeInterval: 1.05)
        let newest = try store.createTrip()
        XCTAssertEqual(store.listTrips().first?.id, newest.id)
    }

    func testDeletingATripRemovesIt() throws {
        let trip = try store.createTrip()
        try store.delete(trip)
        XCTAssertTrue(store.listTrips().isEmpty)
    }

    func testMetadataRoundTrips() throws {
        let trip = try store.createTrip()
        let metadata = TripMetadata(
            tripId: trip.id, startedAt: Date(timeIntervalSince1970: 1_788_000_000),
            deviceModel: "iPhone", locationSampleCount: 12, motionSampleCount: 600
        )
        try store.writeMetadata(metadata, to: trip)
        let loaded = try XCTUnwrap(store.readMetadata(from: trip))
        XCTAssertEqual(loaded.tripId, trip.id)
        XCTAssertEqual(loaded.motionSampleCount, 600)
    }

    func testCompletedCalibrationPersistsThroughMetadataStorage() throws {
        let trip = try store.createTrip()
        let calibration = MountCalibration(
            referenceQuaternion: [0.9, 0.1, 0.2, 0.3],
            gravityDevice: [0, -0.2, -0.97],
            forwardAxisDevice: [1, 0, 0],
            initialHeadingDeg: 274,
            headingSource: "gps_course",
            stillDurationS: 6
        )
        try store.writeMetadata(TripMetadata(tripId: trip.id, calibration: calibration), to: trip)
        let loaded = try XCTUnwrap(store.readMetadata(from: trip))
        XCTAssertEqual(loaded.calibration, calibration)
    }

    // MARK: - SampleWriter

    func testSamplesAreWrittenOnePerLine() throws {
        let trip = try store.createTrip()
        let writer = try SampleWriter(url: trip.samplesURL)
        writer.append(location: makeLocation(monotonic: 1))
        writer.append(motion: makeMotion(monotonic: 1.02))
        writer.append(location: makeLocation(monotonic: 2))
        writer.close()

        let lines = try lines(of: trip.samplesURL)
        XCTAssertEqual(lines.count, 3)
        for line in lines {
            XCTAssertNoThrow(
                try JSONSerialization.jsonObject(with: Data(line.utf8)),
                "every line must be standalone JSON: \(line)"
            )
        }
    }

    func testBothStreamsShareOneFile() throws {
        let trip = try store.createTrip()
        let writer = try SampleWriter(url: trip.samplesURL)
        writer.append(location: makeLocation(monotonic: 1))
        writer.append(motion: makeMotion(monotonic: 1.02))
        writer.close()

        let types = try lines(of: trip.samplesURL).compactMap { line -> String? in
            let object = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any]
            return object?["type"] as? String
        }
        XCTAssertEqual(types, ["location", "motion"])
    }

    func testWriterCountsBothStreams() throws {
        let trip = try store.createTrip()
        let writer = try SampleWriter(url: trip.samplesURL)
        for index in 0..<5 { writer.append(location: makeLocation(monotonic: Double(index))) }
        for index in 0..<50 { writer.append(motion: makeMotion(monotonic: Double(index) / 50)) }
        let counts = writer.counts()
        writer.close()
        XCTAssertEqual(counts.locations, 5)
        XCTAssertEqual(counts.motions, 50)
    }

    func testWriterSurvivesAFullFiftyHertzMinute() throws {
        // 50 Hz for a minute is 3000 frames; buffering must keep up without loss.
        let trip = try store.createTrip()
        let writer = try SampleWriter(url: trip.samplesURL)
        for index in 0..<3000 { writer.append(motion: makeMotion(monotonic: Double(index) / 50)) }
        writer.close()
        XCTAssertEqual(try lines(of: trip.samplesURL).count, 3000)
    }

    func testAppendingReopensWithoutTruncating() throws {
        let trip = try store.createTrip()
        let first = try SampleWriter(url: trip.samplesURL)
        first.append(location: makeLocation(monotonic: 1))
        first.close()

        let second = try SampleWriter(url: trip.samplesURL)
        second.append(location: makeLocation(monotonic: 2))
        second.close()

        XCTAssertEqual(try lines(of: trip.samplesURL).count, 2)
    }

    // MARK: - Photos

    func testPhotoNamesIncrement() throws {
        let trip = try store.createTrip()
        XCTAssertEqual(store.nextPhotoName(in: trip), "photo-001")
        try Data("{}".utf8).write(
            to: trip.photosURL.appendingPathComponent("photo-001.json"))
        XCTAssertEqual(store.nextPhotoName(in: trip), "photo-002")
    }

    func testPhotoSidecarsAreRead() throws {
        let trip = try store.createTrip()
        let record = PhotoRecord(
            imagePath: "photo-001.jpg",
            ocr: [OCRLine(text: "Садовая улица", confidence: 0.9)]
        )
        let data = try TripJSON.encoder(prettyPrinted: true).encode(record)
        try data.write(to: trip.photosURL.appendingPathComponent("photo-001.json"))

        let photos = store.readPhotos(from: trip)
        XCTAssertEqual(photos.count, 1)
        XCTAssertEqual(photos.first?.ocr.first?.text, "Садовая улица")
    }

    // MARK: - Export

    func testExportProducesAZipContainingTheTrip() throws {
        let trip = try store.createTrip()
        let writer = try SampleWriter(url: trip.samplesURL)
        writer.append(location: makeLocation(monotonic: 1))
        writer.close()
        try store.writeMetadata(TripMetadata(tripId: trip.id), to: trip)

        let archive = try TripExporter.makeArchive(for: trip)
        addTeardownBlock { try? FileManager.default.removeItem(at: archive.deletingLastPathComponent()) }

        XCTAssertEqual(archive.pathExtension, "zip")
        let size = try FileManager.default.attributesOfItem(atPath: archive.path)[.size] as? Int
        XCTAssertGreaterThan(size ?? 0, 0)

        // A zip starts with the local-file-header magic "PK\u{03}\u{04}".
        let handle = try FileHandle(forReadingFrom: archive)
        defer { try? handle.close() }
        let magic = try XCTUnwrap(handle.read(upToCount: 4))
        XCTAssertEqual([UInt8](magic), [0x50, 0x4B, 0x03, 0x04])
    }

    func testExportingAnEmptyTripIsRefusedWithAClearMessage() throws {
        let trip = try store.createTrip()
        XCTAssertThrowsError(try TripExporter.makeArchive(for: trip)) { error in
            XCTAssertTrue(
                error.localizedDescription.lowercased().contains("no recorded samples"),
                "unhelpful message: \(error.localizedDescription)"
            )
        }
    }

    // MARK: - Helpers

    private func makeLocation(monotonic: TimeInterval) -> LocationSample {
        LocationSample(
            wallTime: Date(timeIntervalSince1970: 1_788_000_000 + monotonic),
            monotonicTime: monotonic, latitude: 59.93, longitude: 30.33,
            horizontalAccuracy: 9, speed: 11, course: 90
        )
    }

    private func makeMotion(monotonic: TimeInterval) -> MotionSample {
        MotionSample(
            monotonicTime: monotonic, userAccelerationG: [0.01, 0, 0],
            rotationRate: [0, 0, 0.01], gravity: [0, 0, -1], quaternion: [1, 0, 0, 0]
        )
    }

    private func lines(of url: URL) throws -> [String] {
        try String(contentsOf: url, encoding: .utf8)
            .split(separator: "\n", omittingEmptySubsequences: true)
            .map(String.init)
    }
}
