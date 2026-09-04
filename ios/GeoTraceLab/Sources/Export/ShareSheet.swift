import SwiftUI
import UIKit

/// Thin `UIActivityViewController` wrapper so a trip archive can be AirDropped,
/// saved to Files, or sent to a Mac for processing.
public struct ShareSheet: UIViewControllerRepresentable {
    public let items: [Any]

    public init(items: [Any]) {
        self.items = items
    }

    public func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    public func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}
