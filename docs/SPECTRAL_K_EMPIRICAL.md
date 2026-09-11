# Empirical spectral correction coefficient `k_required = v_true / v_spectral`

**Diagnostic / forensic only. Hidden GPS used here only, for this analysis.
Nothing in production changed.** Reproduces the pipeline's own spectral speed
model (`SpectralSpeedModel`, fit on each trip's visible GPS warm-up window,
predicted over the whole trip) and compares it, in robust time windows, to the
withheld GPS speed on trips **07-22** and **07-26**.

Script: `tools/k_spectral_forensics.py`
Artefacts: `docs/plots/`
- `k_windows.csv` — every windowed sample (3 s / 5 s / 10 s, non-overlapping)
- `k_vs_speed_bins.csv` — binned k distribution (07-22 / 07-26 / pooled)
- `k_feature_correlations.csv`, `k_trip_effect.csv` — forensic Spearman table
- `k_cross_trip.csv` — overlapping-bin transfer
- `k_fit_candidates.csv` — descriptive fits + cross-trip errors
- `k_vs_true_speed_scatter.png`, `k_vs_true_speed_binned.png`,
  `vspec_vs_vtrue.png`, `k_vs_vspec.png`, `k_fits.png`

## Method notes

- **`v_spectral`** = `model.predict_many(features)` clipped 0–33 m/s — exactly
  the series `build_imu_samples` feeds the tracker (`spectral_scale_enabled`
  is `False`, so this is the raw model output).
- **`v_true`** = withheld-GPS Doppler speed (10 Hz, clean, max gap 0.12 s),
  plus the visible-window GPS speed for `t < ~145 s`.
- Everything resampled to a uniform 0.5 s grid; windows tiled **non-overlapping**
  (so the spread statistics are not inflated by autocorrelation).
- Primary coefficient: **`k_required = mean(v_true) / mean(v_spectral)`** over
  the window. `k_median_ratio = median(v_true/v_spectral)` also stored.
- **STEADY** = `max(v_true) − min(v_true)` over the window `< 2 m/s` (with a
  `< 1 m/s` sensitivity variant); windows with `v_spec_mean < 2` or either
  speed mean `< 1` are flagged and excluded from the k analysis.
- Main k(v) analysis set: **5 s, outage only, STEADY (<2 m/s), moving,
  reliable `v_spectral`** → n = 137 (07-22: 48, 07-26: 89). 3 s (n=330) and
  10 s (n=33) give the same medians (1.12, IQR ≈ 0.93–1.41) — robust to window
  length.
- In-sample sanity: steady 5 s windows *inside* the visible warm-up have median
  `k = 1.00` (n=25), as they must.

---

## Answers

### 1. How does `k(v_true)` look by eye?
Monotone rising, from ~0.8 at a crawl to ~1.6–1.9 near 21 m/s, with a knee near
each trip's warm-up top speed. Pooled binned medians:

| v_true bin | 0–3 | 3–6 | 6–9 | 9–12 | 12–14 | 14–16 | 16–18 | 18–20 | 20–22 |
|---|---|---|---|---|---|---|---|---|---|
| **pooled median k** | 0.78 | 0.98 | 0.96 | 1.22 | 1.10 | 1.12 | 1.23 | 1.49 | 1.61 |
| 07-22 median k | 0.78 | 0.98 | 1.00 | 1.36 | 1.55 | 1.86 | — | — | — |
| 07-26 median k | — | — | 0.74 | 0.88 | 0.94 | 1.11 | 1.23 | 1.49 | 1.61 |

The dip in the *pooled* row at 12–16 m/s is an artefact of mixing the two
trips (07-22 ends, 07-26's low-k points dominate); **each trip alone is
monotone**.

### 2. Is the dependence monotone?
Yes — within each trip, strictly monotone-increasing in the binned median. The
raw scatter is noisy but the trend is unambiguous (`vspec_vs_vtrue.png` shows
`v_spectral` flat-lining while `v_true` keeps rising).

### 3. Where does saturation start?
**It is trip-specific, and it sits at roughly the fastest speed in that trip's
GPS warm-up:**
- **07-22**: `v_spectral` tracks `y = x` to ~6 m/s, then plateaus at a ceiling
  of **~8 m/s**. `k` leaves 1.0 at **~8 m/s true** and is already 1.55 by
  13 m/s. (07-22 warm-up: mean 3 m/s, max ~8.)
- **07-26**: tracks `y = x` to ~13 m/s, plateaus at **~13–14 m/s**. `k` leaves
  1.0 at **~14 m/s true**, reaches ~1.6 by 21 m/s. (07-26 warm-up: mean 11 m/s,
  max ~17.)

So there is no universal saturation speed — the ridge model cannot extrapolate
past the speeds it was calibrated on, and each trip calibrates on a different
range.

### 4. How big is the scatter of `k` at fixed true speed?
Large, and it must not be hidden. Per-bin (steady 5 s windows):

| set / bin | median k | p10–p90 | k_std |
|---|---|---|---|
| pooled 12–14 m/s | 1.10 | **0.85 – 1.63** | 0.32 |
| pooled 16–18 m/s | 1.23 | 1.07 – 1.56 | 0.20 |
| 07-26 14–16 m/s | 1.11 | 0.92 – 1.34 | 0.17 |
| 07-26 18–20 m/s | 1.49 | 1.34 – 1.75 | 0.23 |
| 07-22 9–12 m/s | 1.36 | 1.12 – 1.46 | 0.18 |

Within one trip σ(k) ≈ 0.15–0.25 per bin; pooled it widens to ~0.3 because the
trip offset adds on top. **At 18 m/s the honest statement is "median k ≈ 1.4,
p10–p90 ≈ 1.15–1.75", not "k(18) = 1.4".**

### 5. Does the dependence match between 07-22 and 07-26?
**No.** In every overlapping bin 07-22 needs a markedly *higher* `k` than 07-26
(`k_cross_trip.csv`):

| bin | median k 07-22 | median k 07-26 | diff | 95 % CI on diff |
|---|---|---|---|---|
| 6–9 | 1.00 | 0.74 | −0.26 | [−0.66, +0.07] |
| 9–12 | 1.36 | 0.88 | −0.49 | [−0.65, +0.01] |
| 12–14 | 1.55 | 0.94 | **−0.61** | **[−0.83, −0.38]** |

The 12–14 m/s CI excludes zero. The curves are offset by ~0.4–0.6 — **not a
shared physical calibration curve.**

### 6. Is there a simple `k(v)` that actually describes the data?
Only weakly, and **only within a trip**. Pooled descriptive fits:

| fit | pooled RMSE | pooled MAE | fit 07-22 → test 07-26 RMSE | fit 07-26 → test 07-22 RMSE |
|---|---|---|---|---|
| piecewise linear | 0.237 | 0.184 | 0.90 | 0.42 |
| linear after threshold (`v0≈12`, `a≈0.070`) | 0.252 | 0.192 | 0.74 | 0.34 |
| quadratic after threshold | 0.246 | 0.190 | 0.94 | 0.36 |
| isotonic (diagnostic) | 0.215 | 0.160 | 0.67 | 0.46 |

A pooled fit carries **RMSE ≈ 0.22–0.25 in `k`** (≈ ±3–4 m/s at cruise) with
residuals ±0.4 at every speed — mediocre. **Cross-trip it collapses**: a curve
fitted on 07-22 over-predicts 07-26 by ~0.7 RMSE (it would push `D/D_true` well
past 1). No simple formula transfers.

### 7. Best diagnostic-only fit?
`isotonic` has the lowest pooled error but is just interpolating the bin
medians. The most defensible *parametric* description is
**`k(v) = 1` for `v ≤ v0`, `k(v) = 1 + a·(v − v0)` above**, with
`v0 ≈ 12 m/s`, `a ≈ 0.07 /(m/s)` pooled — but `v0` is really per-trip
(≈ 6–8 m/s for 07-22, ≈ 13–14 for 07-26), i.e. it *is* the warm-up ceiling, not
a constant. `k_fits.png` overlays all four; no fit is imposed on the binned
plot.

### 8. How well does it transfer cross-trip?
Poorly (table in #6). Fit→test RMSE is 2–4× the within-trip fit RMSE. The
`v0` knot alone differs by ~6 m/s between trips.

### 9. Can speed be treated as the main latent variable for `k`?
**Partly.** `Spearman(k, v_true) = 0.64` — by far the strongest single
predictor (`k_feature_correlations.csv`). But it is not the *only* one: after
regressing `k` on `v_true` (local median), a **significant residual trip
effect remains** — median residual +0.05 (07-22) vs −0.02 (07-26),
Mann–Whitney p = 0.0017 — and the per-trip curves are offset by ~0.5. Speed is
necessary but not sufficient.

### 10. What else does `k` depend on?
`Spearman(k, feature)` on steady 5 s windows:

| feature | ρ with k | note |
|---|---|---|
| v_true_mean | **+0.64** | dominant |
| vib_low (0.5–2 Hz accel log-power) | −0.32 | collinear with speed & trip |
| gyro_mag_mean | −0.30 | collinear with speed |
| gyro_power | −0.29 | collinear with speed |
| vib_total | −0.21 | " |
| v_spec_mean | **+0.01** | the *observable* carries no info about its own error |
| accel magnitude / \|Δv\| | −0.13 | n.s. |
| lateral accel | −0.02 | n.s. |
| time | −0.12 | n.s. |
| temperature | — | not recorded |

The only material second factor is **trip identity**, and its physical meaning
is **the speed range of that trip's GPS warm-up**: 07-22 was calibrated on
0–8 m/s driving, so its model saturates at 8 m/s and needs `k > 1` from
moderate speed; 07-26 was calibrated to ~17 m/s and stays near `k = 1` until
14 m/s. `v_spectral` itself (the online-observable) has ρ = 0.01 with `k` —
**it cannot tell you its own correction**, which is why `k_vs_vspec.png` shows
`k` from 0.6 to 2.1 all stacked at `v_spectral ≈ 12–14`.

---

## Follow-up: can a linear regression do it?

`tools/k_linreg_forensics.py` → `docs/plots/k_linreg_models.csv`, `k_linreg.png`.

**The current spectral speed model already *is* a linear regression** — ridge on
42 log band-power features (`SpectralSpeedModel.fit`). The question is whether a
*different* linear model (bigger, multivariate, or targeting `k` instead of
`v`) transfers. Every variant was scored pooled in-sample **and**
leave-one-trip-out (the only generalisation that matters):

| model | target | pooled R² (in-sample) | cross-trip R² | cross-trip bias | cross-trip RMSE |
|---|---|---|---|---|---|
| `k ~ v_true` | k | 0.32 | −1.0 / −2.1 | −0.41 / +0.52 | 0.44 / 0.58 |
| `k ~ v_spectral` (observable) | k | **0.00** | −2.2 / −2.4 | +0.42 / +0.42 | 0.57 / 0.60 |
| `k ~ v_spectral + 5 vib/gyro` (observable) | k | 0.09 | −2.4 / −4.3 | +0.15 / +0.60 | 0.59 / 0.76 |
| `k ~ 42 log band powers` (observable, ridge) | k | 0.55 | **−1.4 / −1.2** | −0.14 / +0.29 | 0.49 / 0.49 |
| `v_true ~ a + b·v_spectral` (affine rescale) | v | 0.60 | −0.85 / −0.61 | **+4.4 / +2.3 m/s** | 5.2 / 4.9 m/s |
| `v_true ~ 42 powers`, windowed, LOTO | v | 0.93 | +0.43 / **−0.98** | +1.7 / **−4.2 m/s** | 2.9 / 5.5 m/s |
| `v_true ~ 42 powers`, in-trip refit (ceiling, needs outage GPS) | v | — | 0.88 (07-26) | **−1.5 m/s at ≥15 m/s** | 2.5 (2.9 hi) |

(cross-trip pairs are *fit 07-26 → test 07-22* / *fit 07-22 → test 07-26*.)

Reading:

- **`k` is not a linear (or any) function of the observables.** `k ~ v_spectral`
  has R² = 0.00; adding vibration/gyro features raises pooled R² to 0.09 and
  makes cross-trip *worse* (it fits one trip's noise). `k ~ 42 powers` looks
  good in-sample (R² 0.55) and collapses to R² ≈ −1.3 leave-one-trip-out — the
  textbook overfit, and exactly what `SPECTRAL_CALIBRATION_FORENSICS.md` Phase 1
  predicted.
- **Affine recalibration `v_true ≈ a + b·v_spectral`** (the simplest
  "linear regression" fix, equivalent to a speed-dependent `k`) fitted on one
  trip mis-predicts the other by **+2–4 m/s of bias**: the two trips need
  different slopes because their spectral models saturate at different speeds
  (`k_linreg.png` left panel — the blue and red fit lines cross).
- **Refitting the full band-power regression on a whole trip** and testing on
  the other still fails at the top end: 07-22 → 07-26 leaves a **−4 to −6 m/s
  bias in the 15–22 m/s cruise** (`k_linreg.png` right panel), because 07-22
  contains no fast driving to learn from.
- **Even the non-deployable ceiling** — a linear regression refit on the trip's
  *own* withheld outage GPS — still under-reads the ≥15 m/s cruise by ~1.5 m/s
  (07-26). The features stop separating speeds up there (Phase B5/B6:
  16 m/s and 25 m/s driving are not statistically distinguishable in the
  spectrum), so no linear combination of them can either.

**Verdict: a linear regression is worth *nothing new* here.** The one that is
deployable (`v_true ~ a + b·v_spectral`, refit online) needs a high-speed
absolute-speed anchor to set the slope, which is the exact thing these trips
don't provide during the outage — the same wall as every prior phase. The one
that looks good (`k ~ 42 powers`) is fitting noise. The bend-anchor `k_high`
(Phase 18) remains the only formulation that moves `D/D_true` on 07-26 without a
cross-trip blow-up.

---

## Bottom line

`k_required` is a **monotone-rising function of true speed with a trip-specific
onset** (the warm-up top speed) and **±0.2–0.3 of irreducible scatter** at any
fixed speed. It is **not transferable between 07-22 and 07-26** (offset ~0.5 in
the overlap band) and **not a function of any online-observable quantity** —
`v_spectral`, vibration and gyro features add nothing once you condition on
speed, and none of them is available without already knowing `v_true`. This is
the same identifiability wall reached in `SPECTRAL_CALIBRATION_FORENSICS.md`,
now shown directly from the data. No production change; no fitted `k(v)` is
proposed for deployment here.
