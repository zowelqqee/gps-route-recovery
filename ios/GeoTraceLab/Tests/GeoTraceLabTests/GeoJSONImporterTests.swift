import CoreLocation
import XCTest

@testable import GeoTraceLab

/// Reading the processor's results back into the app.
final class GeoJSONImporterTests: XCTestCase {

    func testALineStringBecomesARoute() throws {
        let json = """
        {"type":"FeatureCollection","features":[
          {"type":"Feature","properties":{"name":"road_particle_filter"},
           "geometry":{"type":"LineString","coordinates":[[30.33,59.93],[30.34,59.94]]}}]}
        """
        let layer = try GeoJSONImporter.parse(data: Data(json.utf8), name: "reconstructed-route")
        XCTAssertEqual(layer.lines.count, 1)
        XCTAssertEqual(layer.lines[0].count, 2)
        // GeoJSON is [longitude, latitude]; getting this backwards puts Saint
        // Petersburg in the Indian Ocean.
        XCTAssertEqual(layer.lines[0][0].latitude, 59.93, accuracy: 1e-9)
        XCTAssertEqual(layer.lines[0][0].longitude, 30.33, accuracy: 1e-9)
    }

    func testAPolygonKeepsItsProbability() throws {
        let json = """
        {"type":"FeatureCollection","features":[
          {"type":"Feature","properties":{"component_id":"branch-01","probability":0.71},
           "geometry":{"type":"Polygon","coordinates":[[[30.33,59.93],[30.34,59.93],[30.34,59.94],[30.33,59.93]]]}}]}
        """
        let layer = try GeoJSONImporter.parse(data: Data(json.utf8), name: "uncertainty-polygons")
        XCTAssertEqual(layer.polygons.count, 1)
        XCTAssertEqual(layer.polygons[0].componentId, "branch-01")
        XCTAssertEqual(layer.polygons[0].probability ?? 0, 0.71, accuracy: 1e-9)
    }

    func testAMultiPolygonBecomesSeveralBranches() throws {
        let json = """
        {"type":"FeatureCollection","features":[
          {"type":"Feature","properties":{"component_id":"branch-01"},
           "geometry":{"type":"MultiPolygon","coordinates":[
             [[[30.33,59.93],[30.34,59.93],[30.34,59.94],[30.33,59.93]]],
             [[[30.35,59.95],[30.36,59.95],[30.36,59.96],[30.35,59.95]]]]}}]}
        """
        let layer = try GeoJSONImporter.parse(data: Data(json.utf8), name: "polygons")
        XCTAssertEqual(layer.polygons.count, 2, "each branch must stay a separate shape")
    }

    func testPolygonHolesArePreserved() throws {
        let json = """
        {"type":"FeatureCollection","features":[
          {"type":"Feature","properties":{},
           "geometry":{"type":"Polygon","coordinates":[
             [[30.30,59.90],[30.40,59.90],[30.40,60.00],[30.30,59.90]],
             [[30.33,59.93],[30.34,59.93],[30.34,59.94],[30.33,59.93]]]}}]}
        """
        let layer = try GeoJSONImporter.parse(data: Data(json.utf8), name: "polygons")
        XCTAssertEqual(layer.polygons.first?.holes.count, 1)
    }

    func testInvalidCoordinatesAreDropped() throws {
        let json = """
        {"type":"FeatureCollection","features":[
          {"type":"Feature","properties":{},
           "geometry":{"type":"LineString","coordinates":[[30.33,59.93],[999,999],[30.34,59.94]]}}]}
        """
        let layer = try GeoJSONImporter.parse(data: Data(json.utf8), name: "route")
        XCTAssertEqual(layer.lines.first?.count, 2)
    }

    func testDegenerateGeometryIsIgnored() throws {
        let json = """
        {"type":"FeatureCollection","features":[
          {"type":"Feature","properties":{},"geometry":{"type":"LineString","coordinates":[[30.33,59.93]]}},
          {"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[30.33,59.93]]]}}]}
        """
        let layer = try GeoJSONImporter.parse(data: Data(json.utf8), name: "route")
        XCTAssertTrue(layer.isEmpty)
    }

    func testUnknownGeometryTypesAreSkippedNotFatal() throws {
        let json = """
        {"type":"FeatureCollection","features":[
          {"type":"Feature","properties":{},"geometry":{"type":"GeometryCollection","geometries":[]}},
          {"type":"Feature","properties":{},
           "geometry":{"type":"LineString","coordinates":[[30.33,59.93],[30.34,59.94]]}}]}
        """
        let layer = try GeoJSONImporter.parse(data: Data(json.utf8), name: "route")
        XCTAssertEqual(layer.lines.count, 1)
    }

    func testNonGeoJSONIsRejectedWithAClearError() {
        XCTAssertThrowsError(
            try GeoJSONImporter.parse(data: Data("{\"hello\":1}".utf8), name: "x")
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("GeoJSON"))
        }
    }

    // MARK: - Routing files into layers

    func testResultFilesAreRoutedToTheRightLayerByName() {
        var results = ImportedResults()
        results.absorb(.init(name: "uncertainty-polygons", lines: [], polygons: [], points: []))
        results.absorb(.init(name: "corrupted-gps", lines: [], polygons: [], points: []))
        results.absorb(.init(name: "reconstructed-route", lines: [], polygons: [], points: []))
        XCTAssertEqual(results.polygons?.name, "uncertainty-polygons")
        XCTAssertEqual(results.corrupted?.name, "corrupted-gps")
        XCTAssertEqual(results.reconstructed?.name, "reconstructed-route")
        XCTAssertFalse(results.isEmpty)
    }

    func testAnOddlyNamedFileIsRoutedByItsGeometry() {
        var results = ImportedResults()
        let polygon = GeoJSONImporter.PolygonFeature(
            componentId: nil, probability: nil,
            ring: [
                CLLocationCoordinate2D(latitude: 59.93, longitude: 30.33),
                CLLocationCoordinate2D(latitude: 59.94, longitude: 30.34),
                CLLocationCoordinate2D(latitude: 59.95, longitude: 30.33),
            ],
            holes: []
        )
        results.absorb(.init(name: "run-output", lines: [], polygons: [polygon], points: []))
        XCTAssertNotNil(results.polygons)
        XCTAssertNil(results.reconstructed)
    }

    func testAnEmptyResultSetIsEmpty() {
        XCTAssertTrue(ImportedResults().isEmpty)
    }
}
