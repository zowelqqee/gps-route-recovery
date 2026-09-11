# GPS-free physical interval estimator takeover report

## Outcome

Methods 5/6/7/8 are implemented and guarded, but the central real-data result
is negative: neither review trip contains a map-distance interval that passes
the independence gate. The correct production action is therefore **no global
map correction and no learned accelerometer scale**. Forcing the inherited
candidate updates would have made the route belief self-confirming.

The production baseline remains the strongest generally defensible live route
configuration. A 30 s fixed-lag output improves retrospective local speed and
distance errors, but does not improve the live Pacman route belief and adds
latency and memory. Disabling spectral helps route survival on 07-26 but harms
07-22 and makes local speed error much worse; there is still no GPS-free signal
that tells the tracker when the spectral model is saturated.

## Inherited worktree

At takeover the repository was on `main` at `67d13ad`, with a deliberately
dirty tree: 29 tracked modified entries and 19 untracked status entries. The
entire Pacman package and Pacman tests were untracked, alongside earlier review
tools, live logs, documentation, generated reports, and unrelated legacy
pipeline edits. Nothing was reset or discarded.

The tracked-only starting/current diff stat is:

```
29 files changed, 38108 insertions(+), 38382 deletions(-)
```

This number excludes all untracked Pacman source/tests and is therefore not a
measure of this takeover's implementation size.

Starting verification found one failure:

- focused Pacman suite: 119 passed, 1 failed;
- full suite: 533 passed, 1 failed;
- failure: a Mahony characterization test incorrectly expected sustained
  longitudinal acceleration not to be absorbed as pitch.

## What was kept, repaired, and rejected

Kept from the interrupted work:

- the five-state speed-filter direction and frozen interval-distance state;
- manager-side junction alignment measurements;
- the experimental Mahony implementation behind a disabled flag;
- existing ZUPT, lateral-speed, spectral-deployment-uncertainty, route-manager,
  and oracle-ceiling work.

Repaired:

- global route mass is retained through gating; selected subsets are never
  renormalized into fictitious consensus;
- duplicate descendants on the same evidence edge are aggregated before they
  count as support;
- corroboration requires majority mass, at least two meaningful evidence
  groups, minimum per-route mass, and effective route count;
- route anchors must match a physical IMU turn, wait for the event to settle,
  and yield at most one candidate update per physical event;
- turn intervals use `D_B - D_A = L_map`, not terminal speed equal to interval
  average;
- common global drift and differential per-route `offset_bias` are separated;
- `k_a` uses the correct prediction Jacobians and can receive information only
  from an accepted absolute-distance interval;
- delayed measurements use retained state/cross-covariance history and a
  bounded fixed-lag update;
- history is not allocated when intervals and lag are disabled;
- diagnostics now carry event type/times, evidence IDs, mass/spread, INS/map
  length, discrepancy, state before/after values, lag, modified-state count,
  acceptance, and rejection reason.

Rejected/disabled:

- Mahony remains off. Its apparently good parked/moving offset was caused by
  interpreting sustained vehicle acceleration as pitch and deleting signal.
- A stop is not treated as a map landmark merely because a previous turn
  changed route offset. The stop position is still derived from `D`, so such a
  constraint would be circular. ZUPT remains active, while stop-to-stop map
  distance is rejected until an independent stop landmark exists.
- NHC was not added: the current state has no lateral or vertical velocity, so
  it cannot provide a meaningful forward-speed observation.
- No dynamics prior was promoted; it cannot create absolute speed and risks
  suppressing real manoeuvres.
- No censored spectral likelihood was added because the model output itself
  does not identify saturation without hidden GPS.

## Implemented mathematics

The global filter state is

```
x = [D, v, b_a, D_anchor, k_a]
a_corrected = k_a (a_measured - b_a)
```

Prediction includes

```
d a_corrected / d b_a = -k_a
d a_corrected / d k_a = a_measured - b_a
```

At the first independently supported turn, `D_anchor` is attached to the
retained state at the physical event time. At the next supported turn, the
linear measurement is

```
z = L_map
h(x) = D_event - D_anchor
H = [1, 0, 0, -1, 0]
```

For route event `i`, the manager supplies common discrepancy
`r_i = D_i - M_i`. Thus the GPS-free interval length is

```
M_B - M_A = (D_B - D_A) - (r_B - r_A)
```

After a global correction, every hypothesis `offset_bias` is shifted by the
same `D` delta. This preserves every route's map position while leaving
route-specific residual mismatch in the route state.

The fixed-lag path stores filtered state, covariance, and live-state cross
covariance at 10 Hz. A delayed scalar interval is applied to the current state
through that cross covariance and to retained outputs over the configured
horizon. Tests cover 0, 5, 10, 20, and 30 s. This is bounded online smoothing,
not full-trip offline smoothing.

## Anti-circularity evidence

Deterministic tests cover all required cases:

- independent `+38/+41/+39 m` routes accept a common drift near `+39 m`;
- a lone `+40 m` route rejects;
- `+40/-20/+5 m` high spread rejects;
- reconverged duplicate descendants count as one evidence group and reject;
- a 92% leader plus 8% sibling fails the effective-route gate.

On real data, accepted map intervals were `0` on both trips. Typical rejected
turn candidates either did not match an IMU turn, had zero/one meaningful
route group, or carried only 19--39% agreeing global mass versus the required
60%. Stop candidates are explicitly rejected as not independently
map-localized. Consequently every accepted-interval before/after diagnostic is
empty rather than fabricated.

## Accelerometer-scale observability

Synthetic stop-bounded triangular motion proves that an accepted integrated
distance measurement moves `k_a` in the correct direction without converting
distance into terminal speed. Separate tests prove spectral, lateral, ZUPT,
and envelope updates cannot change either the mean or marginal variance of
`k_a`.

Real GPS-free inference accepted no interval, so both trips finish at the
generic prior:

```
07-22: k_a = 1.000, sigma = 0.258
07-26: k_a = 1.000, sigma = 0.258
```

Only after inference, hidden-GPS diagnostics suggest the scale needed to undo
the measured attenuation would be about `1/0.790 = 1.266` and
`1/0.655 = 1.527`, respectively. Those values were never estimator inputs.
The failure to move toward them establishes that `k_a` is not observable from
the available independently supported events. It is not enabled in the chosen
production configuration.

## Ablation definitions

- `base`: previous production configuration.
- `5`: turn-to-turn common-mode distance feedback.
- `6`: stop-to-stop path (ZUPT active; map distance requires an independent
  stop landmark and therefore rejects on these data).
- `7`: acceleration-scale state.
- `8`: 30 s fixed-lag output.
- combined rows enable the named methods.
- `best-no-spectral`: 5+6+7+8 with spectral disabled.
- `best-with-spectral`: 5+6+7+8 with spectral retained.

`e10/e30/e60` are mean absolute local integrated-distance errors in metres;
`max30/max60` are worst local errors. Coverage columns are fractions.

<!-- FINAL_ABLATION_TABLES -->

## Interpretation and selected configuration

Methods 5 and 6 are metric-identical to base because the safety gates accept
no absolute interval. Method 7 cannot infer scale and is not a production win.
Method 8 lowers delayed-output local error, especially on 07-22, but does not
change live route survival and does not repair final distance. The all-methods
configuration therefore does not solve the 07-26 high-speed shortfall.

Removing spectral is not a general solution. It improves 07-26 final distance
and route survival, but substantially worsens 07-22 distance/survival and makes
07-26 speed MAE/RMSE and 30/60 s errors worse. With no online saturation
indicator, choosing per trip would itself require unavailable truth.

Selected live production configuration: unchanged baseline. Optional 30 s lag
is defensible only for delayed telemetry/history consumers, not for improving
the live Pacman decision.

## Oracle and leakage regression

Oracle-distance Pacman after the changes:

| Trip | Truth survival | Top1 | Top3 | Top5 | First loss | Permanent loss |
|---|---:|---:|---:|---:|---:|---:|
| 07-22 | 0.9962 | 0.0750 | 0.0863 | 0.0911 | 747.7 s | none |
| 07-26 | 1.0000 | 0.0768 | 0.1166 | 0.1336 | none | none |

Reference GPS enters only benchmark metrics and explicit oracle modes. The
production input builder cannot construct an oracle series; interval event
selection, route gate, stop gate, and scale update read no withheld GPS.
Leakage-focused tests pass.

## Runtime, memory, and latency

The disabled-path history bug was removed. Patched base runtime is 44.2 s on
07-22 and 84.9 s on 07-26, with identical metrics to the inherited baseline.
Enabling lag/interval history adds work proportional to retained history.

At 10 Hz, 30 s retains about 300 states. Pairwise 5x5 cross-covariance arrays
occupy about 9 MB of numeric storage plus Python object overhead; memory is
bounded by lag and cannot grow with trip duration. A 30 s smoothed output is
delayed by 30 s, while the current live filter remains available immediately.
A physical turn candidate additionally waits 6.5 s for event settlement before
selection.

## Verification

- final focused interval/speed tests: 34 passed;
- final leakage/attitude/interval tests: 32 passed;
- final full suite: 549 passed;
- oracle-distance route ceiling preserved on both trips.

## Files and functions changed in this takeover

- `processor/src/geotrace/pacman_tracker/intervals.py`:
  `IntervalConfig`, `DriftObservation`, `common_drift`.
- `processor/src/geotrace/pacman_tracker/manager.py`:
  `_resolve_alignment_anchors` and pending common-drift evidence.
- `processor/src/geotrace/pacman_tracker/speed.py`:
  `SpeedConfig`, `SpeedSample`, `_HistoryState`, `predict`, `open_interval`,
  `apply_interval`, history/smoothing helpers, constrained `_update`.
- `processor/src/geotrace/pacman_tracker/tracker.py`:
  physical turn detection, `run`, `_collect_drift`, `_apply_turn_interval`,
  `_collect_stop`, diagnostics.
- `processor/tests/pacman/test_attitude.py`: failed Mahony expectation changed
  into an explicit rejection/disabled-default characterization.
- `processor/tests/pacman/test_intervals.py`: deterministic anti-circularity,
  integrated-distance, scale-observability, non-map scale-isolation, and lag
  horizon tests.
- `tools/pacman_physical_ablation.py`: reproducible real-trip ablation runner.
- `docs/PHYSICAL_INTERVAL_RESULTS.md`: this evidence report.

No commit was created, and unrelated inherited changes were preserved.
