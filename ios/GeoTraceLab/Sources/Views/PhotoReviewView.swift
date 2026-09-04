import SwiftUI

/// What OCR actually found on a stop photo.
struct PhotoReviewView: View {
    let record: PhotoRecord
    let imageURL: URL?
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    if let imageURL, let image = UIImage(contentsOfFile: imageURL.path) {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFit()
                            .clipShape(RoundedRectangle(cornerRadius: 10))
                    }

                    Text("Recognised text")
                        .font(.headline)

                    if record.ocr.isEmpty {
                        Text("Vision found no readable text in this photo.")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(record.ocr) { line in
                            HStack(alignment: .firstTextBaseline) {
                                Text(line.text)
                                Spacer()
                                Text(String(format: "%.2f", line.confidence))
                                    .font(.caption.monospacedDigit())
                                    .foregroundStyle(.secondary)
                            }
                            .padding(.vertical, 3)
                            Divider()
                        }
                    }

                    if let latitude = record.assumedLatitude, let longitude = record.assumedLongitude {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Assumed position")
                                .font(.headline)
                            Text(String(format: "%.5f, %.5f", latitude, longitude))
                                .font(.subheadline.monospaced())
                        }
                        .padding(.top, 6)
                    }

                    Text(
                        "This is the last position the app believed in plus whatever text "
                        + "Vision could read. It is not a verified address: identifying a "
                        + "place from one photograph needs a database of geo-referenced "
                        + "images, which this app does not have."
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.top, 8)
                }
                .padding(20)
            }
            .navigationTitle("Stop photo")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}
