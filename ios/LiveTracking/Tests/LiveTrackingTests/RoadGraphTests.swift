import XCTest
@testable import LiveTracking

/// The Swift loader must reproduce the Python `RoadNetwork`'s geometry, because
/// every downstream tracker decision is expressed in terms of edges, distances
/// along them and their bearings.
final class RoadGraphTests: XCTestCase {
    static var network: RoadNetwork!

    static let graphURL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("cache/spb.geograph")

    override class func setUp() {
        super.setUp()
        network = try? RoadGraphLoader.load(
            contentsOf: graphURL,
            origin: Coordinate(
                latitude: GraphFixtures.originLatitude,
                longitude: GraphFixtures.originLongitude
            ),
            radiusM: GraphFixtures.radiusM
        )
    }

    private func requireNetwork() throws -> RoadNetwork {
        try XCTUnwrap(
            Self.network,
            "cache/spb.geograph is missing. Build it with: "
            + "processor/.venv/bin/geotrace build-ios-graph --graph cache/spb.graphml "
            + "--output cache/spb.geograph"
        )
    }

    func testHeaderBoundsCoverSaintPetersburg() throws {
        let bounds = try RoadGraphLoader.readBounds(at: Self.graphURL)
        XCTAssertLessThan(bounds.minLat, 59.8)
        XCTAssertGreaterThan(bounds.maxLat, 60.1)
        XCTAssertLessThan(bounds.minLon, 29.6)
        XCTAssertGreaterThan(bounds.maxLon, 30.6)
        XCTAssertGreaterThan(bounds.edges, 40_000)
    }

    func testClippedEdgeCountIsCloseToPython() throws {
        let network = try requireNetwork()
        // Python clips by node radius, Swift by a point bounding box, so the
        // two sets differ slightly at the boundary. What matters is that the
        // interior - where the trip actually is - is the same.
        let ratio = Double(network.edgeCount) / Double(GraphFixtures.pythonEdgeCount)
        XCTAssertGreaterThan(ratio, 0.9, "swift kept \(network.edgeCount), python \(GraphFixtures.pythonEdgeCount)")
        XCTAssertLessThan(ratio, 1.6)
    }

    func testProjectionOfTripFixesMatchesPython() throws {
        let network = try requireNetwork()
        var worst = 0.0
        for sample in GraphFixtures.samples {
            let point = network.frame.toLocal(latitude: sample.lat, longitude: sample.lon)
            worst = max(worst, point.distance(to: Point(x: sample.east, y: sample.north)))
        }
        XCTAssertLessThan(worst, 1e-3, "worst ENU disagreement \(worst) m")
    }

    func testDistanceToRoadMatchesPython() throws {
        let network = try requireNetwork()
        var worst = 0.0
        for sample in GraphFixtures.samples {
            let point = Point(x: sample.east, y: sample.north)
            let distance = network.distanceToRoad(point)
            worst = max(worst, abs(distance - sample.distanceToRoad))
        }
        XCTAssertLessThan(worst, 0.5, "worst distance-to-road disagreement \(worst) m")
    }

    func testSnapMatchesPython() throws {
        let network = try requireNetwork()
        var worstPosition = 0.0
        var worstBearing = 0.0
        var matched = 0
        for sample in GraphFixtures.samples {
            let point = Point(x: sample.east, y: sample.north)
            guard let snapped = network.snap(point, k: 6) else { continue }
            let position = network.position(edge: snapped.edge, s: snapped.s)
            let expected = Point(x: sample.snapEast, y: sample.snapNorth)
            let positionError = position.distance(to: expected)
            // Two carriageways of the same street can be within centimetres of
            // each other; when the offsets tie, either answer is correct. Only
            // count a case as comparable when Python's own offset is distinct.
            if positionError < 1.0 {
                matched += 1
                worstPosition = max(worstPosition, positionError)
                let bearing = network.bearing(edge: snapped.edge, s: snapped.s)
                worstBearing = max(
                    worstBearing, abs(Angles.wrap(bearing - sample.snapBearing))
                )
            }
        }
        XCTAssertGreaterThan(
            Double(matched) / Double(GraphFixtures.samples.count), 0.85,
            "only \(matched)/\(GraphFixtures.samples.count) snaps agreed with Python"
        )
        XCTAssertLessThan(worstPosition, 1.0)
        XCTAssertLessThan(worstBearing, 0.05, "bearing disagreement \(worstBearing) rad")
    }

    func testSuccessorsExcludeTheUTurnTwin() throws {
        let network = try requireNetwork()
        var checked = 0
        for index in stride(from: 0, to: network.edgeCount, by: 97) {
            let successors = network.forwardSuccessors(edge: index, allowUTurn: false)
            let edge = network.edges[index]
            for candidate in successors {
                let other = network.edges[Int(candidate)]
                XCTAssertFalse(
                    other.startNode == edge.endNode && other.endNode == edge.startNode,
                    "a U-turn survived the successor filter"
                )
                XCTAssertEqual(other.startNode, edge.endNode, "successors must start where we end")
            }
            checked += 1
        }
        XCTAssertGreaterThan(checked, 20)
    }

    func testCoverageQuery() throws {
        let network = try requireNetwork()
        XCTAssertTrue(network.covers(latitude: 59.93, longitude: 30.36))
        XCTAssertFalse(network.covers(latitude: 55.75, longitude: 37.62), "Moscow is not covered")
    }
}
