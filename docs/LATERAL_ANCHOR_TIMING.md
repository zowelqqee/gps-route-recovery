# Lateral-anchor timestamp and sequencing investigation

Date: 2026-09-11  
Primary trip: `rf-07-26` (`runs/review-final/2026-07-26/trip`)

## Verdict

**No delayed-measurement bug was found. Keep the production sequencing and
defaults unchanged.** The lateral observation is a rolling, uniformly weighted,
centered 1 s measurement stamped at its window centre. It is not a completed
turn event reported at the end of a turn. Across all 535 accepted anchors on
`rf-07-26`, the call timestamp trails the exact raw-window centre of mass by a
median 0.006 s, p90 0.006 s, p95 0.006 s, and maximum 0.0106 s. That is raw/grid
alignment below one 50 Hz sensor sample, not actionable out-of-sequence latency.

A forced 0.5 s rewind is therefore a negative sensitivity test, not a timestamp
correction. It materially worsens `rf-07-26` odometry and is unstable across
control trips. The same-step trapezoid control is effectively neutral. The
Phase-34 residual was primarily a predict/posterior-semantics diagnostic, not
evidence that the lateral observation belongs to an earlier turn timestamp.

No topology, road graph, branch selection, route commit, spectral estimator,
display-position correction, saturation/isotonic logic, gamma, gate, threshold,
or hidden-GPS behavior was changed.

## 1. Established delayed-measurement patterns

| System | Delayed-measurement strategy | State/history stored | Replay/update method | Relevance here |
|---|---|---|---|---|
| [Autoware `ekf_localizer`](https://github.com/autowarefoundation/autoware_ai_perception/blob/master/ekf_localizer/src/ekf_localizer.cpp) | Computes `delay_step = round((now - header.stamp + additional_delay) / ekf_dt)` and calls `updateWithDelay`; observations older than the configured augmented horizon are rejected. | An extended/augmented vector of current and past EKF states (`extend_state_step`). | A measurement matrix selects the delayed block; the augmented-state Kalman update propagates correlation to the current block. | Closest to this repository's small linear state/cross-covariance diagnostic. It depends on an honest sensor timestamp. |
| [`robot_localization`](https://github.com/cra-ros-pkg/robot_localization/blob/rolling-devel/doc/state_estimation_nodes.rst) | With `smooth_lagged_data`, detects an observation older than the last filter time and reverts to the last state before it. | Bounded [`filter_state_history_` and `measurement_history_`](https://github.com/cra-ros-pkg/robot_localization/blob/rolling-devel/src/ros_filter.cpp), plus stored controls on measurements. | Restores state and measurement queue, inserts the late observation in time order, then calls `processMeasurement` for all later measurements/controls again. | Preferred lightweight design if Pacman later gains a genuinely lagged, nonlinear observation source. It handles state-dependent gates and side effects by replaying them. |
| [GTSAM fixed-lag smoother](https://borglab.github.io/gtsam/fixedlagsmootherexample/) | Adds factors to timestamped states inside a sliding time window; states older than the lag are marginalized. | Recent state variables, factors, initial values, and a key-to-timestamp map. | Batch re-optimization or incremental iSAM2 update; no hand-authored distance patch. | Architectural reference for timestamped observations and bounded history, but too heavy for this 5-state Python filter. |
| [GTSAM IMU preintegration](https://borglab.github.io/gtsam/imufactor/) | Accumulates high-rate IMU samples between timestamped navigation states. | Preintegrated delta rotation, velocity, position, covariance, and bias linearization. | An IMU factor connects pose/velocity/bias states at the endpoints; re-optimization incorporates later factors consistently. | Reinforces that propagation inputs belong to explicit time intervals and absolute observations to explicit state timestamps. |
| [KumarRobotics/glider](https://github.com/KumarRobotics/glider) | GNSS/IMU factor graph; configurable fixed-lag smoothing (`optimizer.smooth`, `optimizer.lag_time`) or iSAM2. | Timestamped navigation states/factors and IMU preintegration through GTSAM. | Sliding-window factor-graph optimization. | A real GNSS/IMU example of the same bounded-history pattern, not code to import here. |

The common rule is simple: first establish the physical observation timestamp.
Only when it precedes the current filter time should a retained state be updated
and later propagation/measurements be replayed (or their joint covariance be
conditioned in an equivalent linear-Gaussian formulation).

## 2. Exact Pacman lateral timestamp semantics

The complete lifecycle is:

1. `tracker._lateral_channel` sorts raw motion records by `monotonic_time`
   (median spacing 0.020 s on `rf-07-26`). It rotates device-frame user
   acceleration and gyro by the recorded attitude. `omega` is the world-vertical
   gyro component. `a_lat` is the world acceleration projected on the horizontal
   left-normal of the calibrated forward axis.
2. `build_imu_samples` sends **both** raw channels through `_smooth_to_steps`
   using `lateral_smooth_s = 1.0`. On this trip that is a 50-sample uniform
   boxcar with symmetric padding (25 raw samples before, 24 after), then
   `searchsorted` resampling onto the 0.1 s tracker grid.
3. The filter does not wait for a turn detector, peak, event end, or window
   maturity. On every non-stationary, non-shock tracker step it calls
   `lateral_anchor(a_lat, yaw_rate_smooth, ..., t=sample.t)`.
4. `lateral_anchor` subtracts the current gyro bias and forms
   `v_lat = a_lat / omega`. It rejects only ill-conditioned/wrong-sign/
   implausible/high-sigma ratios. The accepted anchor's sigma comes from ratio
   error propagation plus the model term. There is no separate confidence
   timestamp.

Thus the physical observation is a **uniform average over the rolling window**,
conventionally located at that window's centre of mass. It is not a point
observation at the peak of the turn. The `|omega|`-weighted content centroid was
also computed as a diagnostic: its absolute offset has median 0.0377 s, p95
0.1983 s, and max 0.3450 s. That describes where turning energy happens inside
the symmetric window; it is not processing delay and does not change the
timestamp of the equally weighted ratio measurement.

The smoother is noncausal because the reconstruction is batch/offline: a live
implementation would have to wait roughly half a window before publishing the
centre-stamped value. That wall-clock availability latency does not make the
offline observation a window-end measurement.

### `rf-07-26` delay distribution

| quantity | median | p90 | p95 | max |
|---|---:|---:|---:|---:|
| received time − physical window centre (s) | 0.0060 | 0.0060 | 0.0060 | 0.0106 |
| absolute value (s) | 0.0060 | 0.0060 | 0.0060 | 0.0106 |
| absolute received time − `|omega|` content centroid (s), diagnostic only | 0.0377 | 0.1588 | 0.1983 | 0.3450 |

The full requested 535-row table contains `anchor_received_time`,
`physical_measurement_time`, `delay_seconds`, `v_lat`, `sigma`, inverse-variance
confidence, window start/end, yaw-content centroid, peak time, and the complete
pre/post filter state trace:
[`lateral_anchors_rf-07-26.csv`](plots/phase35/lateral_anchors_rf-07-26.csv).

## 3. Current `GlobalSpeedTracker` state semantics

For a normal moving step the runtime order is:

```text
D, v at previous posterior
  -> predict: D += v*dt + 0.5*a*dt^2; v += a*dt
  -> lateral velocity update (when valid)
  -> spectral velocity update (when present)
  -> report posterior D_route and v_route
```

`D_before_lateral_update == D_after_predict`. However, `lateral_anchor` uses a
normal Kalman update with `H = [0, 1, 0, 0, 0]`; because `P[D,v]` is nonzero,
the Kalman gain has a distance component. Therefore the lateral update changes
both posterior `v` **and correlated posterior `D`**. On `rf-07-26`, all 535
accepted updates changed `D`; median absolute change was 0.240 m, p95 1.976 m,
max 3.717 m, signed sum +31.238 m. This is a correlated EKF state correction,
not a display-position jump. The display branch was not involved.

Consequently, comparing a distance increment with only the final reported
posterior velocity is not an exact reconstruction of the step: the reported
sample is after lateral and spectral corrections, and `D` may also have been
conditioned through cross-covariance.

## 4. Delayed-path diagnostic

The existing feature flag remains **OFF by default**. Its retained-state update
uses a bounded state/cross-covariance history, conceptually like Autoware's
augmented delayed-state update. One diagnostic defect was fixed: a delayed
lateral residual must be `v_lat - v(t_measurement)`, not
`v_lat - v(t_received)`. A unit test covers that direction.

Because the measured physical lag is below one raw sample, the decision-rule
precondition for a production rewind/replay implementation is false. No new
production replay mechanism was enabled or recommended. Enabling the diagnostic
with zero delay reproduces the baseline speed-distance trace exactly. A forced
0.5 s lag was retained only as a deliberately wrong sensitivity control.

## 5. `rf-07-26` A/B/C

`B` at the physically correct timestamp is identical to `A` because the anchor
is centre-stamped at the current step. `B*` below is the artificial 0.5 s rewind
negative control. `C` is the requested same-step trapezoid control and is not a
production candidate.

| run | median abs error (m) | p95 (m) | max (m) | endpoint signed (m) | `D_route/D_true` | speed MAE (m/s) | lateral updates | decisions | `real_wrong` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A production | 265.4 | 624.7 | 731.6 | -731.6 | 0.834 | 2.63 | 535 | 13 | 0 |
| B correct timestamp (zero delay) | identical to A | identical | identical | identical | identical | identical | 535 | 13 | 0 |
| B* forced 0.5 s rewind | 389.4 | 785.4 | 902.2 | -902.2 | 0.795 | 2.78 | 535 | 13 | 0 |
| C same-step trapezoid | 265.6 | 624.6 | 731.7 | -731.7 | 0.834 | 2.63 | 535 | 13 | 0 |

Chosen topology is identical for A/B*/C on the primary trip. A/B* commit timing
is not identical (maximum shift 17.6 s); A/C differs by at most 0.1 s. This is
expected when a diagnostic deliberately changes `D_route`, but it means B* does
not satisfy an “identical commit timing” acceptance check. Production A remains
the stated 13 decisions and `real_wrong = 0`.

### Requested intervals

Error is signed `D_route - D_true`; change is end minus start. Delay is the real
window-centre delay, not the artificial B* assumption.

| elapsed (s) | A start | A end | A change | B* start | B* end | B* change | C start | C end | C change | lateral n | mean delay (s) | max delay (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 27–115 | 21.4 | -236.7 | -258.1 | 4.6 | -226.7 | -231.3 | 21.6 | -236.9 | -258.5 | 140 | 0.00365 | 0.0106 |
| 212–304 | -215.2 | -571.4 | -356.2 | -378.1 | -732.0 | -353.9 | -215.2 | -571.4 | -356.2 | 7 | 0.00314 | 0.0060 |
| 316–390 | -557.5 | -620.7 | -63.2 | -718.1 | -783.4 | -65.3 | -557.5 | -620.7 | -63.2 | 17 | -0.0040 | -0.0040 |
| 436–455 | -623.6 | -725.6 | -102.0 | -787.3 | -896.0 | -108.7 | -623.1 | -725.7 | -102.6 | 123 | 0.0060 | 0.0060 |

The key 212–304 s interval contains only seven accepted lateral updates and a
3 ms mean timestamp offset, so late turn detection cannot explain its 356 m
error growth.

## 6. Cross-trip controls

The forced 0.5 s rewind is not consistently beneficial. The table reports A →
B*; the correct-timestamp B is identical to A by construction.

| trip | median abs (m) | p95 (m) | max (m) | endpoint (m) | speed MAE (m/s) | topology/decision effect |
|---|---:|---:|---:|---:|---:|---|
| `rf-07-26` | 265.4 → 389.4 | 624.7 → 785.4 | 731.6 → 902.2 | -731.6 → -902.2 | 2.63 → 2.78 | same chosen topology; timing shifted |
| `07-22` | 103.2 → 66.0 | 136.8 → 255.6 | 146.0 → 273.4 | -142.1 → +195.7 | 1.99 → 2.12 | regress: 26/6 wrong → 29/9 wrong |
| `07-23` | 29.8 → 13.9 | 58.0 → 51.3 | 74.5 → 56.9 | +50.4 → +56.9 | 0.55 → 0.54 | no decisions in either run |
| `07-22-s0` | 222.7 → 391.4 | 278.5 → 428.3 | 431.7 → 452.0 | -431.7 → +314.1 | 1.06 → 1.07 | regress: 35/15 wrong → 45/24 wrong |
| `07-26-s1` | 1478.4 → 1552.1 | 1890.0 → 1911.7 | 1933.5 → 1958.2 | -1771.9 → -1685.4 | 2.65 → 2.67 | chosen sequence/timing changed; wrong count stayed 53 |
| `07-26-s3` | 948.3 → 819.8 | 1982.4 → 2263.0 | 2063.3 → 2351.2 | -1776.4 → -1844.2 | 3.08 → 3.11 | decision count changed 96 → 70; both topologies already failed |

The trapezoid control was also run on the first five listed datasets. Its
largest change on the primary trip was 0.2 m in median error and 0.1 m in max;
on `07-22`, `07-23`, `07-22-s0`, and `07-26-s1` it likewise left odometry and
speed metrics essentially unchanged (at most about 1.4 m in the reported
distance-error summaries). This does not justify a production distance patch.

`07-24` and `07-25` review-final could not be evaluated with the locally
available graphs; they were skipped rather than silently using an invalid map.
The Phase-31 segments above provide the available fair-set controls, but several
have already-bad baseline topology, so they are sensitivity checks rather than
clean localization acceptance trips.

## 7. Test status and artifacts

- New focused test: delayed lateral innovation is evaluated against the
  retained measurement-time velocity.
- Full processor suite: **653 passed in 105.85 s**.
- Full per-anchor state/timestamp trace and summary are generated by
  `tools/phase35_lateral_timestamps.py` without GPS truth in the filter.
- Production defaults remain unchanged (`lateral_delayed_correction_enabled =
  False`).
- See `plots/phase35/lateral_timing_summary_rf-07-26.json` for machine-readable
  delay/state-update summary.

Final interpretation: current predict → lateral correct → spectral correct
sequencing is appropriate for the timestamp the lateral channel actually
carries. The remaining `D_route` error is real estimator error, but the evidence
does not attribute it to late lateral-anchor timing.
