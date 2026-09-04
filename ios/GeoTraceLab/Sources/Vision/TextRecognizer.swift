import Foundation
import UIKit
import Vision

/// Vision OCR for stop photos.
///
/// What this produces is text and per-line confidence, nothing more. It is not
/// a geocoder and it does not identify an address: without a reference image
/// database a single photograph cannot place the car, and the app never claims
/// otherwise.
public enum TextRecognizer {
    public enum RecognizerError: LocalizedError {
        case badImage
        case visionFailed(String)

        public var errorDescription: String? {
            switch self {
            case .badImage:
                return "The photo could not be read."
            case .visionFailed(let message):
                return "Text recognition failed: \(message)"
            }
        }
    }

    /// Languages worth trying on a Saint Petersburg street sign.
    public static let languages = ["ru-RU", "en-US"]

    public static func recognize(in image: UIImage) async throws -> [OCRLine] {
        guard let cgImage = image.cgImage else { throw RecognizerError.badImage }
        return try await recognize(in: cgImage, orientation: image.imageOrientation.cgOrientation)
    }

    public static func recognize(
        in cgImage: CGImage,
        orientation: CGImagePropertyOrientation = .up
    ) async throws -> [OCRLine] {
        try await withCheckedThrowingContinuation { continuation in
            let request = VNRecognizeTextRequest { request, error in
                if let error {
                    continuation.resume(
                        throwing: RecognizerError.visionFailed(error.localizedDescription))
                    return
                }
                let observations = request.results as? [VNRecognizedTextObservation] ?? []
                let lines: [OCRLine] = observations.compactMap { observation in
                    guard let candidate = observation.topCandidates(1).first else { return nil }
                    let text = candidate.string.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !text.isEmpty else { return nil }
                    let box = observation.boundingBox
                    return OCRLine(
                        text: text,
                        confidence: Double(candidate.confidence),
                        boundingBox: [
                            Double(box.origin.x), Double(box.origin.y),
                            Double(box.size.width), Double(box.size.height),
                        ]
                    )
                }
                continuation.resume(returning: lines)
            }
            request.recognitionLevel = .accurate
            request.usesLanguageCorrection = true
            request.recognitionLanguages = languages

            let handler = VNImageRequestHandler(cgImage: cgImage, orientation: orientation, options: [:])
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    try handler.perform([request])
                } catch {
                    continuation.resume(
                        throwing: RecognizerError.visionFailed(error.localizedDescription))
                }
            }
        }
    }
}

extension UIImage.Orientation {
    var cgOrientation: CGImagePropertyOrientation {
        switch self {
        case .up: return .up
        case .down: return .down
        case .left: return .left
        case .right: return .right
        case .upMirrored: return .upMirrored
        case .downMirrored: return .downMirrored
        case .leftMirrored: return .leftMirrored
        case .rightMirrored: return .rightMirrored
        @unknown default: return .up
        }
    }

    var label: String {
        switch self {
        case .up, .upMirrored, .down, .downMirrored: return "portrait"
        case .left, .leftMirrored: return "landscapeLeft"
        case .right, .rightMirrored: return "landscapeRight"
        @unknown default: return "unknown"
        }
    }
}
