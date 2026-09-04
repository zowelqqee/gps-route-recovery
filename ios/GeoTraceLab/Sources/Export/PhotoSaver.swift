import CoreLocation
import Foundation
import ImageIO
import UIKit

/// Writes a stop photo and its OCR sidecar into a trip.
public enum PhotoSaver {
    public struct Result {
        public let imageURL: URL
        public let record: PhotoRecord
    }

    public enum SaveError: LocalizedError {
        case encodingFailed

        public var errorDescription: String? {
            "The photo could not be encoded as JPEG."
        }
    }

    /// Save `image`, run OCR, and write `photo-NNN.jpg` + `photo-NNN.json`.
    ///
    /// `assumedCoordinate` is the last position the app believed in. It is
    /// recorded as a starting point for later work, explicitly not as a
    /// verified address.
    public static func save(
        image: UIImage,
        into trip: TripDirectory,
        basename: String,
        monotonicTime: TimeInterval?,
        assumedCoordinate: CLLocationCoordinate2D?,
        assumedAccuracy: Double?,
        deviceHeadingDeg: Double?,
        exifCoordinate: CLLocationCoordinate2D? = nil,
        store: TripStore = .shared
    ) async throws -> Result {
        try FileManager.default.createDirectory(
            at: trip.photosURL, withIntermediateDirectories: true)

        guard let jpeg = image.jpegData(compressionQuality: 0.85) else {
            throw SaveError.encodingFailed
        }
        let imageURL = trip.photosURL.appendingPathComponent("\(basename).jpg")
        try jpeg.write(to: imageURL, options: .atomic)

        // A failed OCR pass must not lose the photograph.
        let lines = (try? await TextRecognizer.recognize(in: image)) ?? []

        let record = PhotoRecord(
            imagePath: "\(basename).jpg",
            capturedAt: Date(),
            monotonicTime: monotonicTime,
            ocr: lines,
            cameraOrientation: image.imageOrientation.label,
            deviceHeadingDeg: deviceHeadingDeg,
            assumedLatitude: assumedCoordinate?.latitude,
            assumedLongitude: assumedCoordinate?.longitude,
            assumedAccuracy: assumedAccuracy,
            exifLatitude: exifCoordinate?.latitude,
            exifLongitude: exifCoordinate?.longitude,
            note: "Assumed position from the last GPS fix. Not a verified address."
        )
        let sidecarURL = trip.photosURL.appendingPathComponent("\(basename).json")
        let data = try TripJSON.encoder(prettyPrinted: true).encode(record)
        try data.write(to: sidecarURL, options: .atomic)
        return Result(imageURL: imageURL, record: record)
    }

    /// Coordinates embedded in a picked image's EXIF, if it has any.
    public static func exifCoordinate(from info: [UIImagePickerController.InfoKey: Any])
        -> CLLocationCoordinate2D?
    {
        if let asset = info[.phAsset] as? NSObject,
           let location = asset.value(forKey: "location") as? CLLocation {
            return location.coordinate
        }
        guard let url = info[.imageURL] as? URL,
              let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
                as? [CFString: Any],
              let gps = properties[kCGImagePropertyGPSDictionary] as? [CFString: Any],
              let latitude = gps[kCGImagePropertyGPSLatitude] as? Double,
              let longitude = gps[kCGImagePropertyGPSLongitude] as? Double
        else { return nil }

        let latitudeRef = gps[kCGImagePropertyGPSLatitudeRef] as? String ?? "N"
        let longitudeRef = gps[kCGImagePropertyGPSLongitudeRef] as? String ?? "E"
        return CLLocationCoordinate2D(
            latitude: latitudeRef == "S" ? -latitude : latitude,
            longitude: longitudeRef == "W" ? -longitude : longitude
        )
    }
}
