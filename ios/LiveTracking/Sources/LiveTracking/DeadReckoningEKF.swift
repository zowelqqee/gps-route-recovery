import Foundation

/// Fixed-size 6x6 matrix, enough for the dead-reckoning EKF.
///
/// Written out rather than pulled from Accelerate so the numerics are identical
/// on every platform the package builds for, and so the operation order matches
/// the Python baseline's numpy expressions.
public struct Matrix6 {
    public var m: [Double]  // row-major, 36 entries

    public init(repeating value: Double = 0) {
        m = [Double](repeating: value, count: 36)
    }

    public static func identity() -> Matrix6 {
        var out = Matrix6()
        for i in 0..<6 { out[i, i] = 1 }
        return out
    }

    public static func diagonal(_ values: [Double]) -> Matrix6 {
        var out = Matrix6()
        for i in 0..<Swift.min(6, values.count) { out[i, i] = values[i] }
        return out
    }

    public subscript(row: Int, column: Int) -> Double {
        get { m[row * 6 + column] }
        set { m[row * 6 + column] = newValue }
    }

    public func multiplied(by other: Matrix6) -> Matrix6 {
        var out = Matrix6()
        for i in 0..<6 {
            for k in 0..<6 {
                let a = self[i, k]
                if a == 0 { continue }
                for j in 0..<6 { out[i, j] += a * other[k, j] }
            }
        }
        return out
    }

    public func transposed() -> Matrix6 {
        var out = Matrix6()
        for i in 0..<6 { for j in 0..<6 { out[j, i] = self[i, j] } }
        return out
    }

    public static func + (lhs: Matrix6, rhs: Matrix6) -> Matrix6 {
        var out = lhs
        for i in 0..<36 { out.m[i] += rhs.m[i] }
        return out
    }

    public mutating func symmetrise() {
        for i in 0..<6 {
            for j in (i + 1)..<6 {
                let value = 0.5 * (self[i, j] + self[j, i])
                self[i, j] = value
                self[j, i] = value
            }
        }
    }
}

/// EKF over `[E, N, v, psi, b_a, b_omega]`.
///
/// A port of `geotrace.ekf.ExtendedKalmanFilter`. In the live runtime it plays
/// the role it plays in the baseline: it is the source of the predicted
/// position, covariance, speed and heading that the GPS gates test each fix
/// against, and it is the honest `ekf_dead_reckoning` reference for how far pure
/// inertial dead reckoning drifts.
///
/// Its speed is clamped at zero, which is correct for this forward-motion road
/// model. Signed velocity - the thing a reversing car needs - lives in
/// `ParkingTracker`, which is a separate estimator for exactly that reason.
public final class DeadReckoningEKF {
    public private(set) var state: [Double]
    public private(set) var P: Matrix6
    public private(set) var skippedGaps = 0
    private let config: MotionConfig

    static let idxE = 0, idxN = 1, idxV = 2, idxPsi = 3, idxBa = 4, idxBw = 5

    public init(config: MotionConfig, initialState: [Double]) {
        self.config = config
        self.state = initialState
        self.P = Matrix6.diagonal([25.0, 25.0, 4.0, 0.35, 0.25, 0.01])
    }

    public var position: Point { Point(x: state[Self.idxE], y: state[Self.idxN]) }
    public var speed: Double { state[Self.idxV] }
    public var heading: Double { state[Self.idxPsi] }
    public var biases: (accel: Double, gyro: Double) {
        (state[Self.idxBa], state[Self.idxBw])
    }

    /// Position block of the covariance, for the Mahalanobis gate.
    public var positionCovariance: Matrix2 {
        Matrix2(a: P[0, 0], b: P[0, 1], c: P[1, 0], d: P[1, 1])
    }

    public func predict(aWorld: SIMD3<Double>, yawRate: Double, dt: Double) {
        guard dt > 0 else { return }
        if dt > config.maxGapSeconds {
            // Do not integrate across a hole; inflate the covariance instead so
            // the filter honestly reports that it lost track.
            skippedGaps += 1
            let growth = pow(config.maxSpeedMS * dt, 2)
            P[0, 0] += growth
            P[1, 1] += growth
            P[3, 3] += pow(.pi / 2, 2)
            return
        }

        let psi = state[Self.idxPsi]
        let aLong = aWorld.x * cos(psi) + aWorld.y * sin(psi)
        let F = transitionJacobian(aLong: aLong, yawRate: yawRate, dt: dt)
        let G = noiseJacobian(dt: dt)
        let Q = processNoise(dt: dt)
        state = propagate(state, aLong: aLong, yawRate: yawRate, dt: dt)

        var next = F.multiplied(by: P).multiplied(by: F.transposed())
        // G Q G^T with Q diagonal 4x4 and G 6x4, expanded inline.
        for i in 0..<6 {
            for j in 0..<6 {
                var sum = 0.0
                for k in 0..<4 { sum += G[i][k] * Q[k] * G[j][k] }
                next[i, j] += sum
            }
        }
        next.symmetrise()
        P = next
    }

    /// Keep the constant-velocity state through a likely phone/mount impact,
    /// while explicitly widening what the filter admits it no longer knows.
    public func inflateForMountDisturbance(
        positionNoiseMSqrt: Double, headingNoiseRadSqrt: Double, dt: Double
    ) {
        guard dt > 0 else { return }
        P[Self.idxE, Self.idxE] += positionNoiseMSqrt * positionNoiseMSqrt * dt
        P[Self.idxN, Self.idxN] += positionNoiseMSqrt * positionNoiseMSqrt * dt
        P[Self.idxPsi, Self.idxPsi] += headingNoiseRadSqrt * headingNoiseRadSqrt * dt
        P.symmetrise()
    }

    /// The transition from the specification:
    ///
    ///     psi_bar   = psi + 0.5 w_hat dt
    ///     E        += v dt cos(psi_bar) + 0.5 a_hat dt^2 cos(psi_bar)
    ///     N        += v dt sin(psi_bar) + 0.5 a_hat dt^2 sin(psi_bar)
    ///     v         = max(0, v + a_hat dt)
    ///     psi       = wrap(psi + w_hat dt)
    func propagate(_ x: [Double], aLong: Double, yawRate: Double, dt: Double) -> [Double] {
        let aHat = Swift.min(
            Swift.max(aLong - x[Self.idxBa], -config.maxAccelMS2), config.maxAccelMS2
        )
        let wHat = yawRate - x[Self.idxBw]
        let psiBar = x[Self.idxPsi] + 0.5 * wHat * dt
        let step = x[Self.idxV] * dt + 0.5 * aHat * dt * dt
        var out = x
        out[Self.idxE] = x[Self.idxE] + step * cos(psiBar)
        out[Self.idxN] = x[Self.idxN] + step * sin(psiBar)
        out[Self.idxV] = Swift.min(Swift.max(0, x[Self.idxV] + aHat * dt), config.maxSpeedMS)
        out[Self.idxPsi] = Angles.wrap(x[Self.idxPsi] + wHat * dt)
        return out
    }

    /// Analytic dF/dX of `propagate`, checked against finite differences in the
    /// tests so the two can never drift apart.
    func transitionJacobian(aLong: Double, yawRate: Double, dt: Double) -> Matrix6 {
        let v = state[Self.idxV], psi = state[Self.idxPsi]
        let aRaw = aLong - state[Self.idxBa]
        let saturated = abs(aRaw) >= config.maxAccelMS2
        let aHat = Swift.min(Swift.max(aRaw, -config.maxAccelMS2), config.maxAccelMS2)
        let wHat = yawRate - state[Self.idxBw]
        let psiBar = psi + 0.5 * wHat * dt
        let cosBar = cos(psiBar), sinBar = sin(psiBar)
        let step = v * dt + 0.5 * aHat * dt * dt
        let dStepDBa = saturated ? 0.0 : -0.5 * dt * dt
        let dPsiBarDBw = -0.5 * dt

        var F = Matrix6.identity()
        F[0, Self.idxV] = dt * cosBar
        F[0, Self.idxPsi] = -step * sinBar
        F[0, Self.idxBa] = dStepDBa * cosBar
        F[0, Self.idxBw] = -step * sinBar * dPsiBarDBw
        F[1, Self.idxV] = dt * sinBar
        F[1, Self.idxPsi] = step * cosBar
        F[1, Self.idxBa] = dStepDBa * sinBar
        F[1, Self.idxBw] = step * cosBar * dPsiBarDBw

        let vNext = v + aHat * dt
        if vNext <= 0 || vNext >= config.maxSpeedMS {
            // max(0, .) / clip is flat here, so the row is zero apart from itself.
            F[Self.idxV, Self.idxV] = 0
            F[Self.idxV, Self.idxBa] = 0
        } else {
            F[Self.idxV, Self.idxV] = 1
            F[Self.idxV, Self.idxBa] = saturated ? 0 : -dt
        }
        F[Self.idxPsi, Self.idxBw] = -dt
        return F
    }

    /// dF/du for `[accel noise, gyro noise, b_a random walk, b_w random walk]`.
    func noiseJacobian(dt: Double) -> [[Double]] {
        let psi = state[Self.idxPsi]
        let cosP = cos(psi), sinP = sin(psi)
        var G = [[Double]](repeating: [Double](repeating: 0, count: 4), count: 6)
        G[0][0] = 0.5 * dt * dt * cosP
        G[1][0] = 0.5 * dt * dt * sinP
        G[2][0] = dt
        G[0][1] = -0.5 * state[Self.idxV] * dt * dt * sinP
        G[1][1] = 0.5 * state[Self.idxV] * dt * dt * cosP
        G[3][1] = dt
        G[4][2] = 1
        G[5][3] = 1
        return G
    }

    func processNoise(dt: Double) -> [Double] {
        [
            config.accelNoise * config.accelNoise,
            config.gyroNoise * config.gyroNoise,
            config.accelBiasRandomWalk * config.accelBiasRandomWalk * dt,
            config.gyroBiasRandomWalk * config.gyroBiasRandomWalk * dt,
        ]
    }

    // MARK: - Updates

    public func updatePosition(_ measurement: Point, sigma: Double) {
        let residual = [measurement.x - state[0], measurement.y - state[1]]
        let r = sigma * sigma
        // S = H P H^T + R for H selecting (E, N).
        let s00 = P[0, 0] + r, s01 = P[0, 1]
        let s10 = P[1, 0], s11 = P[1, 1] + r
        let determinant = s00 * s11 - s01 * s10
        guard abs(determinant) > 1e-12 else { return }
        let i00 = s11 / determinant, i01 = -s01 / determinant
        let i10 = -s10 / determinant, i11 = s00 / determinant

        // K = P H^T S^-1, a 6x2.
        var K = [[Double]](repeating: [0, 0], count: 6)
        for row in 0..<6 {
            let p0 = P[row, 0], p1 = P[row, 1]
            K[row][0] = p0 * i00 + p1 * i10
            K[row][1] = p0 * i01 + p1 * i11
        }
        for row in 0..<6 {
            state[row] += K[row][0] * residual[0] + K[row][1] * residual[1]
        }
        state[Self.idxPsi] = Angles.wrap(state[Self.idxPsi])
        state[Self.idxV] = Swift.max(0, state[Self.idxV])

        // Joseph form: stays positive definite even with a rough gain.
        var IKH = Matrix6.identity()
        for row in 0..<6 {
            IKH[row, 0] -= K[row][0]
            IKH[row, 1] -= K[row][1]
        }
        var next = IKH.multiplied(by: P).multiplied(by: IKH.transposed())
        for i in 0..<6 {
            for j in 0..<6 {
                next[i, j] += (K[i][0] * K[j][0] + K[i][1] * K[j][1]) * r
            }
        }
        next.symmetrise()
        P = next
    }

    public func updateSpeed(_ speed: Double, sigma: Double) {
        scalarUpdate(row: Self.idxV, residual: speed - state[Self.idxV], sigma: sigma)
    }

    public func updateHeading(_ heading: Double, sigma: Double) {
        scalarUpdate(
            row: Self.idxPsi,
            residual: Angles.wrap(heading - state[Self.idxPsi]),
            sigma: sigma
        )
    }

    /// ZUPT: while the car is provably still, `v` is 0 and the residual speed is
    /// accumulated accelerometer bias.
    public func zeroVelocityUpdate(sigma: Double = 0.05) {
        scalarUpdate(row: Self.idxV, residual: -state[Self.idxV], sigma: sigma)
    }

    private func scalarUpdate(row measurementRow: Int, residual: Double, sigma: Double) {
        let r = sigma * sigma
        let s = P[measurementRow, measurementRow] + r
        guard abs(s) > 1e-12 else { return }
        var K = [Double](repeating: 0, count: 6)
        for row in 0..<6 { K[row] = P[row, measurementRow] / s }
        for row in 0..<6 { state[row] += K[row] * residual }
        state[Self.idxPsi] = Angles.wrap(state[Self.idxPsi])
        state[Self.idxV] = Swift.max(0, state[Self.idxV])

        var IKH = Matrix6.identity()
        for row in 0..<6 { IKH[row, measurementRow] -= K[row] }
        var next = IKH.multiplied(by: P).multiplied(by: IKH.transposed())
        for i in 0..<6 { for j in 0..<6 { next[i, j] += K[i] * K[j] * r } }
        next.symmetrise()
        P = next
    }

    /// Hard reset onto a fix that trust has just been restored to.
    ///
    /// After a long outage the dead-reckoned state is not a prior worth blending
    /// with: it is stale by hundreds of metres and its covariance understates
    /// that. The bias estimates are the one thing worth keeping - they were
    /// learned before the outage and are still valid.
    public func reanchor(to point: Point, sigma: Double, heading: Double?, speed: Double?) {
        state[Self.idxE] = point.x
        state[Self.idxN] = point.y
        if let heading { state[Self.idxPsi] = Angles.wrap(heading) }
        if let speed { state[Self.idxV] = Swift.max(0, speed) }
        let variance = pow(Swift.max(sigma, 1.0), 2)
        for i in 0..<6 {
            P[0, i] = 0; P[i, 0] = 0
            P[1, i] = 0; P[i, 1] = 0
        }
        P[0, 0] = variance
        P[1, 1] = variance
        if heading != nil { P[3, 3] = pow(20.0 * .pi / 180.0, 2) }
        P.symmetrise()
    }
}
