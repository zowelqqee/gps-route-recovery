# Phase 36 — EKF measurement quality and authority

Date: 2026-09-11  
Trip: `rf-07-26`  
Scope: baseline `GlobalSpeedTracker`; no estimator or display changes

## Executive finding

The two large failures have different mechanisms.

- **27–115 s:** the filter is already under-propagating distance by 219.3 m.
  Biased-low lateral anchors directly remove another 30.8 m. Two false ZUPT
  runs during ~14 m/s cruising directly remove 10.2 m and 20.6 m/s of summed
  velocity corrections; their downstream effect is much larger than their
  direct distance correction. Spectral updates are not the cause here: their
  net direct distance effect is +3.7 m.
- **212–304 s:** lateral is essentially innocent (+1.0 m direct correction).
  Propagation loses 215.2 m and spectral updates directly remove another
  142.1 m. During the 47.6 high-speed seconds (`v_true >= 18 m/s`), mean truth
  is 20.34 m/s while mean spectral output is 13.79 m/s, a -6.56 m/s bias.

So the next technical targets are now specific: lateral bias/uncertainty and
false ZUPT handling for the first episode; saturation-aware spectral authority
for the main episode. Removing any source wholesale is not justified.

GPS truth is used only after the filter replay to score measurements and
distance error. It never enters prediction or any update.

## Exact distance attribution

For every 0.1 s step, the diagnostic records the exact `predict()` distance
increment and the exact `D_after - D_before` of each measurement method. The
identity closes to below `6e-13 m` in every interval:

```text
error change
  = prediction distance - true distance
  + lateral D correction
  + spectral D correction
  + ZUPT D correction
```

| interval (s) | true distance | ordinary prediction | prediction − truth | lateral ΔD | spectral ΔD | ZUPT ΔD | total error change |
|---|---:|---:|---:|---:|---:|---:|---:|
| 27–115 | 1035.8 | 816.5 | **-219.3** | **-30.8** | +3.7 | **-11.8** | **-258.2** |
| 212–304 | 1456.3 | 1241.2 | **-215.2** | +1.0 | **-142.1** | 0.0 | **-356.2** |
| 316–390 | 1018.6 | 946.5 | -72.1 | -8.0 | +16.7 | 0.0 | -63.4 |
| 436–455 | 287.1 | 199.4 | **-87.7** | **-29.1** | +9.3 | 0.0 | **-107.6** |

“Prediction − truth” is an exact bookkeeping term, not an independent causal
source: its starting `v` and bias state already contain the downstream effects
of previous lateral/spectral/ZUPT updates.

## 27–115 s: lateral is biased and nominal sigma is under-dispersed

There are 140 accepted lateral updates.

| quantity | value |
|---|---:|
| mean `v_lat` | 10.97 m/s |
| mean `v_true` at anchors | 13.78 m/s |
| mean bias | **-2.82 m/s** |
| median bias | **-3.45 m/s** |
| MAE / RMSE | 5.06 / 6.17 m/s |
| mean nominal sigma | 3.26 m/s |
| mean effective per-update sigma after correlation inflation | 10.31 m/s |
| truth within nominal 1σ / 2σ | **27.1% / 65.7%** |
| mean innovation (`z - v_pred`) | -1.14 m/s |
| negative-innovation fraction | 78.6% |
| mean `K_D` / `K_v` | 0.307 / 0.0537 |
| summed direct `ΔD` / `Δv` | **-30.82 m / -7.92 m/s** |

The nominal physical measurement model is too optimistic for this episode:
Gaussian calibration would suggest roughly 68% within 1σ and 95% within 2σ,
not 27% and 66%. The correlation-time inflation makes each individual update
much weaker than the nominal sigma implies, but it does not remove the
systematic negative bias. The nonzero `P[D,v]` converts that repeated bias into
a real -30.8 m posterior distance correction, and the lower posterior speed
then reduces later prediction distance.

The same failure is even stronger at 436–455 s: 123 lateral anchors average
9.20 m/s against 16.03 m/s truth (mean bias -6.83 m/s); only 14.6% lie within
nominal 1σ, and the direct distance correction is -29.1 m. Removing lateral in
a diagnostic replay changes that interval's error growth from -107.6 m to
-35.6 m.

## 27–115 s: two false ZUPT runs

The interval contains 109 ZUPT update ticks, but only the final 37 belong to a
real stop. The other 72 ticks form two smooth-cruise false positives:

| elapsed run (s) | ticks | mean truth speed | detector run length | direct ΔD | summed Δv |
|---|---:|---:|---:|---:|---:|
| 36.799–38.099 | 14 | 14.00 m/s | 1.4 s | -0.60 m | -3.08 m/s |
| 50.099–55.799 | 58 | 13.82 m/s | 5.8 s | -9.56 m | -17.55 m/s |
| 111.399–114.999 | 37 | 0.005 m/s | 3.7 s | -1.64 m | -7.28 m/s |

The speed-only ZUPT sub-update deliberately has `K_D = 0`. The observed ZUPT
distance movement comes from the following accelerometer-bias sub-update:
across the interval its mean `K_D` is -0.239 and it contributes the full
-11.8 m. Thus ZUPT affects future distance in three ways: it lowers `v`, it
changes the bias state, and that bias measurement directly conditions `D`
through covariance.

The reachability credibility gate softens the false stop target (mean applied
pseudo-target 3.79 m/s instead of raw `z=0`) but does not make a 5.8 s false
stationary run harmless. A no-ZUPT diagnostic improves this interval's error
change from -258.2 m to -133.3 m. It is not a production proposal: removing
valid stops worsens whole-trip speed MAE from 2.63 to 5.02 m/s and max distance
error from 731.6 to 1020.6 m.

## 212–304 s: spectral saturation dominates measurement authority

There are 920 spectral updates and only seven lateral updates.

| measurement | n | mean z | mean truth | mean bias | mean innovation | negative innovation | mean `K_D` / `K_v` | summed direct `ΔD` / `Δv` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lateral | 7 | 15.93 | 16.01 | -0.07 | +1.01 | 71.4% | 0.221 / 0.0326 | +1.01 m / +0.09 m/s |
| spectral | 920 | 11.96 | 15.83 | **-3.87** | **-1.54** | **77.3%** | 0.0977 / 0.0131 | **-142.09 m / -18.65 m/s** |

The interval-wide mean includes slower sections. Restricting to
`v_true >= 18 m/s` leaves 476 ticks (47.6 s): mean truth 20.34 m/s, mean
spectral 13.79 m/s, mean bias **-6.56 m/s**. Those high-speed ticks alone
directly remove 52.0 m through `K_D`, in addition to depressing velocity used
by later propagation.

The spectral sigma is not numerically small: mean nominal sigma is 7.76 m/s
and temporal-correlation inflation makes the per-update effective sigma
38.8 m/s. The problem is repeated one-sided model bias at the saturation
plateau. Even a small mean `K_v` of 0.0131 and `K_D` of 0.0977 becomes strong
authority over 920 correlated updates.

Removing spectral entirely is catastrophic (whole-trip speed MAE 10.78 m/s,
p95 distance error 1385.8 m). The evidence calls for saturation-aware,
directional authority—not removal or a globally larger sigma.

## Source-removal sensitivity controls

These counterfactuals use identical IMU inputs and gates but omit one update
source. They are nonlinear and not additive; they diagnose causal sensitivity,
not production candidates.

| run | median abs D error | p95 | max | endpoint | speed MAE | error change 27–115 | error change 212–304 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 265.3 | 624.8 | 731.6 | -731.6 | 2.63 | -258.2 | -356.2 |
| no lateral | 209.2 | 609.9 | 661.5 | -654.9 | 2.73 | -214.7 | -362.8 |
| no spectral | 282.4 | 1385.8 | 1910.4 | -1137.8 | 10.78 | -292.0 | +1251.5 |
| no ZUPT | 367.5 | 591.3 | 1020.6 | -598.5 | 5.02 | -133.3 | -342.5 |
| prediction only | 333.4 | 1474.8 | 1504.6 | +109.1 | 9.43 | -1478.2 | -104.5 |

## Recommendation

1. Keep Phase-35 timestamp behavior unchanged.
2. Treat the first episode as two concrete diagnostics:
   - characterize `v_lat` bias versus `|omega|`, turn direction, acceleration
     offset, and window dynamics; recalibrate the nominal lateral model only
     from cross-trip evidence;
   - investigate why the stationary variance detector accepts the two cruising
     runs, preserving real-stop behavior and the existing reachability concept.
3. Return to the established spectral saturation branch for 212–304 s. Test a
   one-sided/downward-authority treatment specifically when the source is on
   its high-speed plateau; do not globally disable spectral or tune to GPS
   truth on `rf-07-26`.
4. Before any production change, repeat measurement calibration and ablations
   on `rf-07-22` and the fair Phase-31/32 set, then re-check topology and route
   commit behavior.

Artifacts:

- `tools/phase36_measurement_authority.py`
- `docs/plots/phase36/measurement_updates_rf-07-26.csv`
- `docs/plots/phase36/step_attribution_rf-07-26.csv`
- `docs/plots/phase36/measurement_authority_summary_rf-07-26.json`

