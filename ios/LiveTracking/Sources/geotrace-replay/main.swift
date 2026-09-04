import Foundation
import LiveTracking

/// Swift replay harness.
///
/// Runs a recorded `.geotrip` through exactly the code the phone runs, on a Mac,
/// so the result can be compared against the Python baseline. Reference GPS is
/// never read here: replay sees only what the device recorded.
///
///     geotrace-replay --trip DIR --graph FILE.geograph --output DIR [--seed N]
///                     [--particles N] [--radius M] [--adaptive]

struct Options {
    var trip: URL?
    var graph: URL?
    var output: URL?
    var seed: UInt64 = 42
    var particles: Int?
    var radius: Double = 12_000
    var adaptive = false
    /// Road-filter step. Defaults to the IMU bin size so replay advances the
    /// particle filter exactly as the batch baseline does; the live runtime
    /// batches more coarsely for cost.
    var roadStep: Double?
    var dumpControls: URL?
}

func parseOptions() -> Options {
    var options = Options()
    var index = 1
    let arguments = CommandLine.arguments
    while index < arguments.count {
        let flag = arguments[index]
        func value() -> String? {
            index += 1
            return index < arguments.count ? arguments[index] : nil
        }
        switch flag {
        case "--trip": options.trip = value().map { URL(fileURLWithPath: $0) }
        case "--graph": options.graph = value().map { URL(fileURLWithPath: $0) }
        case "--output": options.output = value().map { URL(fileURLWithPath: $0) }
        case "--seed": options.seed = UInt64(value() ?? "42") ?? 42
        case "--particles": options.particles = Int(value() ?? "")
        case "--radius": options.radius = Double(value() ?? "") ?? 12_000
        case "--adaptive": options.adaptive = true
        case "--road-step": options.roadStep = Double(value() ?? "")
        case "--dump-controls": options.dumpControls = value().map { URL(fileURLWithPath: $0) }
        case "-h", "--help":
            print("""
            geotrace-replay --trip DIR --graph FILE.geograph --output DIR
                            [--seed N] [--particles N] [--radius M] [--adaptive]
                            [--road-step SECONDS]

            Replays a recorded trip through the on-device Swift trackers.
            Without --adaptive the particle count is fixed, matching the Python
            baseline; with it, the live adaptive budget is used instead.
            """)
            exit(0)
        default:
            FileHandle.standardError.write(Data("unknown option: \(flag)\n".utf8))
            exit(2)
        }
        index += 1
    }
    return options
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("error: \(message)\n".utf8))
    exit(2)
}

let options = parseOptions()
guard let tripURL = options.trip else { fail("--trip is required") }
guard let outputURL = options.output else { fail("--output is required") }

var config = LiveTrackingConfig()
config.seed = options.seed
if let particles = options.particles { config.particleFilter.particleCount = particles }
config.runtime.adaptiveParticleCounts = options.adaptive
// Replay defaults to stepping the road filter at the IMU bin rate, which is
// what the Python baseline does. The live runtime batches at 5 Hz instead,
// because a phone cannot afford a 5000-particle update 10 times a second.
config.runtime.roadStepSeconds = options.roadStep ?? config.motion.filterDT
config.runtime.polygonUpdateSeconds = 1.0
config.runtime.outputSeconds = 1.0

let trip: TripReplay.Trip
do {
    trip = try TripReplay.load(directory: tripURL)
} catch {
    fail(error.localizedDescription)
}

print("trip: \(trip.locations.count) fixes, \(trip.motions.count) motion samples, "
    + String(format: "%.1f s", trip.endedAt - trip.startedAt))
print("calibration: \(trip.calibration.map { $0.definesVehicleFrame ? "present" : "incomplete" } ?? "absent")")
if trip.malformedLines > 0 { print("malformed lines skipped: \(trip.malformedLines)") }

let pipeline = LiveTrackingPipeline(
    config: config,
    graphURL: options.graph,
    graphRadiusM: options.radius,
    calibration: trip.calibration
)

if options.dumpControls != nil { await pipeline.enableControlTrace() }
let wallStart = Date()
let result = await TripReplay.run(trip: trip, pipeline: pipeline)
let wallElapsed = -wallStart.timeIntervalSinceNow
let records = await pipeline.drainRecords()
if let dumpURL = options.dumpControls {
    let trace = await pipeline.drainControlTrace()
    var text = "t,dt,aE,aN,yaw,quiet,gap\n"
    for control in trace {
        text += String(
            format: "%.4f,%.4f,%.9f,%.9f,%.9f,%d,%d\n",
            control.t, control.dt, control.aWorld.x, control.aWorld.y, control.yawRate,
            control.isQuiet ? 1 : 0, control.gapExceeded ? 1 : 0
        )
    }
    try? text.write(to: dumpURL, atomically: true, encoding: .utf8)
    print("wrote control trace to \(dumpURL.path)")
}

try? FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)
let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]

func write<T: Encodable>(_ value: T, to name: String) {
    guard let data = try? encoder.encode(value) else { return }
    try? data.write(to: outputURL.appendingPathComponent(name), options: .atomic)
}

func writeLines<T: Encodable>(_ values: [T], to name: String) {
    let lineEncoder = JSONEncoder()
    lineEncoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    var blob = Data()
    for value in values {
        guard let data = try? lineEncoder.encode(value) else { continue }
        blob.append(data)
        blob.append(0x0A)
    }
    try? blob.write(to: outputURL.appendingPathComponent(name), options: .atomic)
}

write(result, to: "live-final-result.json")
writeLines(records.roadPositions, to: "live-road-result.jsonl")
writeLines(records.parkingResults, to: "live-parking-result.jsonl")
writeLines(records.gateDecisions, to: "live-diagnostics.jsonl")

// A stable digest of the geometry, recorded before any reference track is
// brought in, so it is provable that comparing against ground truth did not
// change what the algorithm produced.
struct Fingerprint: Encodable {
    var roadEndpoint: Coordinate?
    var parkingPosition: Coordinate?
    var parkingStatus: String
    var parkingConfidence: Double
    var polygonRadiusM: Double
    var terminalClusterFixes: Int
    var branchCount: Int
    var acceptedFixes: Int
    var rejectedFixes: Int
    var digest: String
}

func digest(_ parts: [String]) -> String {
    // FNV-1a: enough to detect an accidental change, and dependency-free.
    var hash: UInt64 = 0xcbf29ce484222325
    for part in parts {
        for byte in part.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x100000001b3
        }
    }
    return String(format: "%016llx", hash)
}

let fingerprint = Fingerprint(
    roadEndpoint: result.roadResult.endpoint,
    parkingPosition: result.parkingResult.position,
    parkingStatus: result.parkingResult.status.rawValue,
    parkingConfidence: result.parkingResult.confidence,
    polygonRadiusM: result.parkingResult.polygonRadiusM,
    terminalClusterFixes: result.parkingResult.terminalClusterFixes,
    branchCount: result.roadResult.branchCount,
    acceptedFixes: result.diagnostics.acceptedFixes,
    rejectedFixes: result.diagnostics.rejectedFixes,
    digest: digest([
        String(format: "%.6f", result.roadResult.endpoint?.latitude ?? 0),
        String(format: "%.6f", result.roadResult.endpoint?.longitude ?? 0),
        String(format: "%.6f", result.parkingResult.position?.latitude ?? 0),
        String(format: "%.6f", result.parkingResult.position?.longitude ?? 0),
        result.parkingResult.status.rawValue,
        String(result.parkingResult.terminalClusterFixes),
    ])
)
write(fingerprint, to: "swift-fingerprint.json")

print("")
print("GPS: \(result.diagnostics.acceptedFixes) accepted / \(result.diagnostics.rejectedFixes) rejected")
for window in result.diagnostics.outageWindows {
    print(String(format: "outage: %.1fs -> %.1fs (%@)",
                 window.startSeconds, window.endSeconds, window.state.rawValue))
}
if let endpoint = result.roadResult.endpoint {
    print(String(format: "road endpoint:    %.6f, %.6f  (%d branches, %d particles)",
                 endpoint.latitude, endpoint.longitude,
                 result.roadResult.branchCount, result.roadResult.particleCount))
} else {
    print("road endpoint:    none (tracker never initialised)")
}
if let position = result.parkingResult.position {
    print(String(format: "parking position: %.6f, %.6f  %@  confidence %.2f  radius %.1f m",
                 position.latitude, position.longitude,
                 result.parkingResult.status.rawValue,
                 result.parkingResult.confidence, result.parkingResult.polygonRadiusM))
} else {
    print("parking position: none (\(result.parkingResult.status.rawValue))")
}
print("reverse motion detected: \(result.parkingResult.hasReverseMotion)")
print("terminal cluster fixes:  \(result.parkingResult.terminalClusterFixes)")
print(String(format: "road step:    mean %.2f ms, p95 %.2f ms",
             result.diagnostics.roadStepMeanMS, result.diagnostics.roadStepP95MS))
print(String(format: "parking step: mean %.2f ms, p95 %.2f ms",
             result.diagnostics.parkingStepMeanMS, result.diagnostics.parkingStepP95MS))
print(String(format: "replay wall time: %.1f s for %.0f s of trip", wallElapsed, trip.endedAt - trip.startedAt))
print(String(format: "heading offset: %.4f deg", await pipeline.headingOffsetDegrees()))
print("fingerprint: \(fingerprint.digest)")
print("wrote \(outputURL.path)")
