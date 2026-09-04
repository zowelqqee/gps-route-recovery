import CoreLocation
import Foundation
import os

/// CoreLocation front end, configured for in-car navigation.
///
///     desiredAccuracy = kCLLocationAccuracyBestForNavigation
///     activityType    = .automotiveNavigation
///     distanceFilter  = kCLDistanceFilterNone
///
/// Fixes CoreLocation itself marks invalid (negative accuracy) are dropped here
/// and counted, so the recorded file never contains them.
public final class LocationRecorder: NSObject, ObservableObject {
    public enum AuthorizationState: Equatable {
        case unknown, denied, whenInUse, always
    }

    @Published public private(set) var authorization: AuthorizationState = .unknown
    @Published public private(set) var latest: LocationSample?
    @Published public private(set) var acceptedCount = 0
    @Published public private(set) var rejectedCount = 0
    @Published public private(set) var isRecording = false

    /// Called for every accepted fix, on the main thread.
    public var onSample: ((LocationSample) -> Void)?

    private let manager: CLLocationManager
    private let timebase: Timebase
    private let logger = Logger(subsystem: "com.geotrace.GeoTraceLab", category: "Location")

    public init(manager: CLLocationManager = CLLocationManager(), timebase: Timebase = Timebase()) {
        self.manager = manager
        self.timebase = timebase
        super.init()
        manager.delegate = self
        manager.desiredAccuracy = kCLLocationAccuracyBestForNavigation
        manager.activityType = .automotiveNavigation
        manager.distanceFilter = kCLDistanceFilterNone
        // Keep recording with the screen locked; a trip must not be lost
        // because the driver put the phone to sleep.
        manager.pausesLocationUpdatesAutomatically = false
        updateAuthorization(manager.authorizationStatus)
    }

    public func requestAuthorization() {
        if authorization == .whenInUse {
            manager.requestAlwaysAuthorization()
        } else {
            manager.requestWhenInUseAuthorization()
        }
    }

    /// The app can record with When In Use permission while visible, but an
    /// uninterrupted locked-screen drive requires Always permission.
    public func requestAlwaysAuthorization() {
        manager.requestAlwaysAuthorization()
    }

    public func start() {
        guard !isRecording else { return }
        acceptedCount = 0
        rejectedCount = 0
        latest = nil
        if authorization == .whenInUse || authorization == .always {
            manager.allowsBackgroundLocationUpdates = true
            manager.showsBackgroundLocationIndicator = authorization != .always
        }
        manager.startUpdatingLocation()
        manager.startUpdatingHeading()
        isRecording = true
    }

    public func stop() {
        guard isRecording else { return }
        manager.stopUpdatingLocation()
        manager.stopUpdatingHeading()
        manager.allowsBackgroundLocationUpdates = false
        isRecording = false
    }

    private func updateAuthorization(_ status: CLAuthorizationStatus) {
        switch status {
        case .authorizedAlways: authorization = .always
        case .authorizedWhenInUse: authorization = .whenInUse
        case .denied, .restricted: authorization = .denied
        default: authorization = .unknown
        }
    }
}

extension LocationRecorder: CLLocationManagerDelegate {
    public func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        for location in locations {
            let sample = LocationSample(location: location, timebase: timebase)
            guard sample.isUsable else {
                rejectedCount += 1
                continue
            }
            acceptedCount += 1
            latest = sample
            onSample?(sample)
        }
    }

    public func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        updateAuthorization(manager.authorizationStatus)
    }

    public func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        logger.error("location failure: \(error.localizedDescription, privacy: .public)")
    }
}
