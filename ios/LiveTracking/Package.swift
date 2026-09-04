// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "LiveTracking",
    platforms: [.iOS(.v17), .macOS(.v13)],
    products: [
        .library(name: "LiveTracking", targets: ["LiveTracking"]),
        .executable(name: "geotrace-replay", targets: ["geotrace-replay"]),
    ],
    targets: [
        // Pure Foundation. No CoreLocation, no CoreMotion, no UIKit: the same
        // code has to run inside the app on the phone and inside the replay
        // harness on a Mac, and it must be comparable against Python.
        .target(name: "LiveTracking"),
        .executableTarget(name: "geotrace-replay", dependencies: ["LiveTracking"]),
        .testTarget(name: "LiveTrackingTests", dependencies: ["LiveTracking"]),
    ]
)
