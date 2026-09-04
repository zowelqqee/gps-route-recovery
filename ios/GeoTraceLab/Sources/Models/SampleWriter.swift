import Foundation
import os

/// Append-only writer for `samples.jsonl`.
///
/// Both sensor streams land in one file, one JSON object per line, so the
/// processor sees a single merged timeline. Writes are batched on a serial
/// queue: at 50 Hz a `write(2)` per sample would burn battery and stall the
/// motion callback.
///
/// The file is flushed whenever the buffer fills, so a crash loses at most one
/// buffer rather than the whole trip - which is also why the Python loader
/// tolerates a truncated final line.
public final class SampleWriter {
    private let handle: FileHandle
    private let queue = DispatchQueue(label: "com.geotrace.samplewriter", qos: .utility)
    private let encoder = TripJSON.encoder()
    private var buffer = Data()
    private let flushThreshold: Int
    private let logger = Logger(subsystem: "com.geotrace.GeoTraceLab", category: "SampleWriter")

    private var locationCount = 0
    private var motionCount = 0

    public init(url: URL, flushThreshold: Int = 64 * 1024) throws {
        if !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: nil)
        }
        self.handle = try FileHandle(forWritingTo: url)
        self.flushThreshold = flushThreshold
        try handle.seekToEnd()
    }

    public func append(location: LocationSample) {
        queue.async { [self] in
            encode(location)
            locationCount += 1
        }
    }

    public func append(motion: MotionSample) {
        queue.async { [self] in
            encode(motion)
            motionCount += 1
        }
    }

    private func encode<T: Encodable>(_ value: T) {
        do {
            var line = try encoder.encode(value)
            line.append(0x0A)  // newline
            buffer.append(line)
            if buffer.count >= flushThreshold { flushBuffer() }
        } catch {
            logger.error("failed to encode sample: \(error.localizedDescription, privacy: .public)")
        }
    }

    private func flushBuffer() {
        guard !buffer.isEmpty else { return }
        do {
            try handle.write(contentsOf: buffer)
            buffer.removeAll(keepingCapacity: true)
        } catch {
            logger.error("failed to write samples: \(error.localizedDescription, privacy: .public)")
        }
    }

    /// Counts written so far. Blocks until the queue has caught up.
    public func counts() -> (locations: Int, motions: Int) {
        queue.sync { (locationCount, motionCount) }
    }

    public func flush() {
        queue.sync { flushBuffer() }
    }

    public func close() {
        queue.sync {
            flushBuffer()
            try? handle.close()
        }
    }

    deinit {
        // Best effort; `close()` should already have been called.
        try? handle.close()
    }
}
