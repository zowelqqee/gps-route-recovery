import SwiftUI

@main
struct GeoTraceLabApp: App {
    @StateObject private var recorder = TripRecorder()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(recorder)
        }
    }
}
