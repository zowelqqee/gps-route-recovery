import CoreLocation
import SwiftUI
import UIKit

/// Camera (or photo library on a device without one, e.g. the Simulator).
///
/// The system picker is used rather than a custom AVFoundation session: the
/// task is one still image after the car has stopped, and the system UI already
/// handles orientation, permissions and hardware differences.
struct CameraPicker: UIViewControllerRepresentable {
    @Environment(\.dismiss) private var dismiss
    let onPicked: (UIImage, CLLocationCoordinate2D?) -> Void

    /// True on hardware with a camera; false in the Simulator, where the
    /// library is offered instead so the flow can still be exercised.
    static var cameraAvailable: Bool {
        UIImagePickerController.isSourceTypeAvailable(.camera)
    }

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        picker.sourceType = Self.cameraAvailable ? .camera : .photoLibrary
        if Self.cameraAvailable {
            picker.cameraCaptureMode = .photo
        }
        return picker
    }

    func updateUIViewController(_ controller: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(onPicked: onPicked, dismiss: { dismiss() })
    }

    final class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        private let onPicked: (UIImage, CLLocationCoordinate2D?) -> Void
        private let dismiss: () -> Void

        init(
            onPicked: @escaping (UIImage, CLLocationCoordinate2D?) -> Void,
            dismiss: @escaping () -> Void
        ) {
            self.onPicked = onPicked
            self.dismiss = dismiss
        }

        func imagePickerController(
            _ picker: UIImagePickerController,
            didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]
        ) {
            let image = (info[.editedImage] as? UIImage) ?? (info[.originalImage] as? UIImage)
            if let image {
                onPicked(image, PhotoSaver.exifCoordinate(from: info))
            }
            dismiss()
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            dismiss()
        }
    }
}
