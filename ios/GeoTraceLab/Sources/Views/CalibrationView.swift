import SwiftUI

/// The short instruction sheet shown before the first recording.
struct CalibrationView: View {
    @EnvironmentObject private var recorder: TripRecorder
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 22) {
                Text("Before you drive")
                    .font(.title2.bold())

                VStack(alignment: .leading, spacing: 16) {
                    step(1, "Fix the phone in its holder",
                         "The maths assumes the phone is rigidly mounted.")
                    step(2, "Do not move it during the trip",
                         "Repositioning it mid-trip invalidates the mount calibration.")
                    step(3, "Stand still for a few seconds",
                         "This is what lets the sensor bias be measured. Bias is the "
                         + "single biggest source of drift once GPS goes.")
                    step(4, "Then drive straight for a short stretch",
                         "This ties the gyro heading to a real compass direction "
                         + "without relying on the magnetometer, which a car body distorts.")
                }

                statusBox

                Spacer()

                VStack(spacing: 10) {
                    Text("Recording will start automatically after the green confirmation.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)

                    Button("Cancel", role: .cancel) {
                        recorder.cancelCalibration()
                        dismiss()
                    }
                }
            }
            .padding(22)
            .navigationBarTitleDisplayMode(.inline)
            .interactiveDismissDisabled()
            .onChange(of: recorder.state) { _, newValue in
                if newValue == .recording { dismiss() }
            }
        }
    }

    @ViewBuilder
    private var statusBox: some View {
        switch recorder.calibration.phase {
        case .holdStill(let remaining):
            label(
                "Hold still: \(Int(remaining.rounded())) s",
                detail: "Any movement restarts the countdown.",
                systemImage: "hand.raised.fill", tint: .orange
            )
        case .driveStraight:
            label(
                "Now drive straight ahead",
                detail: "Recording starts automatically once the heading settles.",
                systemImage: "arrow.up.forward", tint: .blue
            )
        case .done:
            label("Calibrated", detail: "Ready to record.",
                  systemImage: "checkmark.circle.fill", tint: .green)
        case .failed(let message):
            label("Calibration incomplete", detail: message,
                  systemImage: "exclamationmark.triangle.fill", tint: .red)
        case .idle:
            label(
                "Waiting for sensors",
                detail: recorder.motion.isAvailable
                    ? "Starting up."
                    : "This device has no motion sensors, so only GPS will be recorded.",
                systemImage: "sensor.tag.radiowaves.forward", tint: .secondary
            )
        }
    }

    private func label(
        _ title: String, detail: String, systemImage: String, tint: Color
    ) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: systemImage)
                .foregroundStyle(tint)
                .font(.title3)
            VStack(alignment: .leading, spacing: 3) {
                Text(title).font(.subheadline.weight(.semibold))
                Text(detail).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding(14)
        .background(tint.opacity(0.10), in: RoundedRectangle(cornerRadius: 10))
    }

    private func step(_ number: Int, _ title: String, _ detail: String) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Text("\(number)")
                .font(.footnote.bold())
                .frame(width: 22, height: 22)
                .background(Color.accentColor.opacity(0.15), in: Circle())
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.subheadline.weight(.medium))
                Text(detail).font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}
