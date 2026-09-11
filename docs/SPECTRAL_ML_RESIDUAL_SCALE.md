# Phase 31 — large-scale cross-trip ML validation

**Diagnostic / offline only. Hidden GPS is used for the target and evaluation.
Nothing in production changed; no model is wired.**

Phase 30 (`SPECTRAL_ML_RESIDUAL.md`) reached PARTIAL on 5 trips: a CatBoost
residual model transferred weakly cross-trip and halved the 15–20 m/s speed
error, but the offline replay benefit was fragile and the sample was tiny.
Phase 31 answers the same questions at scale.

Scripts: `tools/phase31_dataset.py`, `phase31_models.py`, `phase31_honest.py`,
`phase31_scaling_unc.py`, `phase31_replay.py`, `phase31_plots.py`.
Artefacts: `docs/plots/phase31/` (`inventory.csv`, `windows_{5s,2s}.csv`,
`grid_*.csv`, `task*.csv`, `models_out.txt`, `honest_out.txt`, `replay_out.txt`,
`phase31_summary.png`).

---

## TASK 1 — inventory

The vehicle-logger day files (`live_logs/{gps,imu}_logs/`) cover **13 days,
2026-07-22 … 2026-08-03**. Split into logger sessions and imported with the
standard protocol (120 s GPS warm-up, then every fix withheld; `--max-duration
1800` to bound compute), the moving sessions are:

| status | count | notes |
|---|---|---|
| trip dirs discovered | 29 | 24 fresh session imports + 5 `runs/review-final` |
| **usable distinct sessions** | **23** | ≈ 5.4 h of moving withheld-GPS driving (Phase 30 had ~0.5 h) |
| — with meaningful high speed | 11 | ≥ 6 % of outage time above 15 m/s |
| excluded — spectral model unfit | 1 | `08-01-s0` (warm-up 111 windows, 27 moving) |
| excluded — review-final duplicates | 4 | `rf-07-2{2,4,5,6}` are 600 s prefixes of the 1800 s imports; `rf-07-23` barely moves |
| flagged **pathological** (kept, but a "primary" set excludes them) | 2 | `07-24-s6` — pure highway, median 30 m/s / max 42, the spectral model extrapolates ~2× its warm-up range; `07-26-s0` — mount coherence 0.38, 28 GPS jumps, `v_spectral` model broken high (`f_vspec>15` = 0.999) |

Full per-trip table (duration, outage/moving seconds, speed-regime fractions,
saturation fraction, ZUPT episodes, turn/bend counts, spectral coverage, sensor
availability) is `inventory.csv`. No trip has a magnetometer (as in
`MAGNETOMETER_FORENSICS.md`).

---

## TASK 2–4 — dataset and grouped validation

Same observable-only feature set and BASELINE speed-EKF replay as Phase 30. 23
trips → 6 894 non-overlapping 5 s windows / 17 269 2 s windows (2 247 / 9 778
steady-moving-reliable). Primary split: **leave-one-trip-out** over the 23
sessions; leakage-controlled check: **leave-one-day-out** (10 day groups).

### Residual prediction, trip-LOTO (2 s dense, n = 9 778, `task34_2s.csv`)

| model | resid MAE | R² | csMAE | csMAE 15–25 | per-trip Δ median | worst trip | helped |
|---|---|---|---|---|---|---|---|
| predict 0 | 5.70 | −0.02 | 5.70 | 7.47 | — | — | — |
| LOTO trip-mean | 6.02 | −0.13 | 6.02 | 7.15 | +0.19 | +2.4 | 5/23 |
| Ridge | 5.92 | +0.05 | 5.92 | 6.14 | +0.60 | +3.1 | 6/23 |
| GradientBoosting | 6.49 | −0.67 | 6.49 | 8.58 | −0.43 | **+17.9** | 16/23 |
| **CatBoost** | **5.59** | **−0.01** | 5.59 | 6.61 | −0.37 | **+5.98** | 16/23 |

**The pooled R² has collapsed from Phase 30's +0.3 to ≈ 0.** CatBoost still has
the lowest MAE and helps the median trip, but two things are now visible that
five trips hid:

* **Two trips poison the pool.** `07-24-s6` (2× extrapolation) and `07-26-s0`
  (bad mount) contribute residuals of +15…+40 m/s that no warm-up-calibrated
  model can predict, and their presence drags every other trip's prediction
  positive. Held out, they are the +6…+18 worst-trip regressions.
* **Day-grouping tames the tail.** LODO CatBoost: per-day Δ median −0.37, worst
  **+0.77** (vs +5.98 trip-split) — i.e. the big trip-split regressions are
  concentrated in the pathological sessions, not a same-day-leakage artefact.

### The clean picture — 21 in-regime trips (`honest_out.txt`)

Excluding the two pathological trips, CatBoost (with the three per-trip-constant
features dropped — see TASK 9) **improves every speed regime and 20 of 21
trips**, worst per-trip regression **+0.04 m/s**:

| true speed | n | MAE `v_spectral` | MAE `v_ml` | improvement |
|---|---|---|---|---|
| 0–5 | 1937 | 2.38 | 1.90 | +0.47 (+20 %) |
| 5–10 | 2628 | 2.58 | 2.20 | +0.38 (+15 %) |
| 10–15 | 2536 | 3.18 | 2.12 | +1.05 (+33 %) |
| **15–20** | 1015 | 5.08 | 3.44 | **+1.64 (+32 %)** |
| 20–25 | 166 | 10.56 | 8.85 | +1.71 (+16 %) |
| 25+ | 41 | 18.32 | 16.17 | +2.16 (+12 %) |

This is a **cleaner and safer window-level result than Phase 30** — the
low-speed harm Phase 30 avoided by using the residual target is now genuinely
absent once the extrapolation trips are removed, and the mid-speed win holds
across 20 trips with essentially no regressions.

---

## TASK 5 / 6 — regime and identifiability at scale

**Identifiability got *stronger* with data.** Windows with `12 ≤ v_spectral ≤
15` (n = 1 364, 18 trips), split by hidden true speed:

| true speed | n | `v_prior − v_spectral` | `v_prior` |
|---|---|---|---|
| 12–15 | 960 | **−2.70** | 10.6 |
| 15–18 | 290 | **−0.24** | 13.3 |
| 18–21 | 88 | **+0.35** | 13.7 |
| 21+ | 26 | +0.15 | 13.3 |

LOTO logistic classifier for "true ≥ 17 m/s given `v_spectral` 12–15": **median
AUC 0.80** (p25 0.77, p75 0.82, 8 folds) — up from Phase 30's 0.60. The
information to tell a saturated-and-slow window from a saturated-and-fast one
**genuinely exists**, carried by `v_prior − v_spectral` (the IMU-integrated
prior), and it is not domain noise: it strengthens as trips are added.

---

## TASK 7 / 9 — feature stability and trip-domain leakage

**Trip identity is classifiable from the features at 90.7 % accuracy** (chance
4.3 %, `task9_leakage.json`). The three most trip-predictive features are
`feat_spec_gap`, `feat_sigma_spec`, `feat_dep_sigma_spec` — the spectral
model's own fit-quality scalars, **one constant value per trip**. They are also
CatBoost's top-4 residual-model features by importance. Removing them:

* on the 23-trip pool: csMAE 15–25 worsens 6.6 → 7.9 (the model was leaning on
  the fingerprint to cope with the pathological trips);
* on the clean 21-trip set: no loss — the honest feature set gives the TASK 5
  table above.

So on the messy pool a real part of the "cross-trip signal" is domain
memorisation; on the in-regime set the correction survives without any
per-trip-constant feature. The features that are **stable across trips** (sign
of `corr(feature, residual)` consistent on ≥ 87 % of trips, `task7_stability.csv`):
`feat_vekf_minus_vspec` (+, 100 %), `feat_vspec_std_10s` (−, 96 %),
`feat_pvv` (+, 87 %), `feat_vspec_std_20s` (−, 87 %). `v_spectral` level:
corr −0.01, 52 % — still carries nothing about its own error.

---

## TASK 8 — train-size scaling (`task8_scaling_clean.csv`)

7 held-out clean trips, CatBoost trained on 2 → 14 of the remaining:

| train trips | 2 | 3 | 5 | 8 | 11 | 14 |
|---|---|---|---|---|---|---|
| held-out csMAE | 3.79 | 2.94 | 2.98 | 2.76 | 2.68 | 2.64 |
| held-out csMAE 15–25 | 4.66 | 3.83 | 4.37 | 4.08 | 3.97 | 3.73 |
| (baseline `v_spectral`: 3.54 all, 4.55 for 15–25) |

**The curve flattens by ≈ 8 trips.** Going 8 → 14 trips buys ~0.35 m/s in the
key regime; the first 3 trips buy most of it. We are near the
observability/representation ceiling for this feature set — **more trips will
not materially move this**, which answers the main open question from Phase 30.

---

## TASK 10–12 — saturation gate and offline replay (`task12_replay.csv`)

Eight covered held-out trips, four regimes: **A** baseline, **B** ML always on
(σ = 2), **C** binary saturation gate, **D** soft gate × adaptive σ (authority
scaled by observable saturation strength and CatBoost/GBR agreement). The gate
uses only rolling `v_spectral` statistics.

* **The gate correctly suppresses on no-saturation trips.** `07-22-s0`,
  `07-26-s3`, `07-25-s1` (sat ≈ 0): C and D stay within ~1–2 % of baseline D/D_true,
  ≤ +1 wrong decision. **B (always on) overshoots them badly** (D/D_true → 1.10–1.33).
  So the safety requirement "trips that don't need it are left alone" is met by
  the gate but not by unconditional ML.
* **The benefit on the trips the gate *does* fire on is inconsistent:**

  | held-out trip | sat | baseline D/D_true, median \|D_err\| | gated (C) | verdict |
  |---|---|---|---|---|
  | 07-26-s1 | 0.51 | 0.887, 1478 m | 0.918, **1086 m** | helps (−390 m lag), +3 wrong |
  | 07-24-s7 | 0.32 | 0.97, 104 m | 1.02, 91 m; **real_wrong 45 → 0** | helps |
  | 07-25-s1 | 0.02 | 0.919, 211 m | 0.907, **196 m** (soft D) | mild help |
  | 07-31-s1 | 0.16 | 0.946, **74 m** | 0.892, 472 m | **hurts** (+400 m lag) |
  | 07-28-s1 | 0.41 | **1.09**, 263 m | 1.15, 564 m | **hurts** — baseline already over-reads D; adding speed makes it worse, +1 wrong |

  The gate fires on "spectral looks saturated", but that does **not** imply "the
  odometer is undershooting and needs more speed". `07-28-s1` and `07-31-s1`
  saturate *and* have D/D_true ≥ 0.95 already — the correction there is pure
  harm. This is the same wall as every prior phase: **D error is unobservable
  online**, so you cannot know whether the correction is needed.

* **The soft/adaptive σ (D) is milder than C** but does not fix the direction
  problem — it still regresses `07-31-s1` (74 → 193 m) and `07-28-s1`.

## TASK 11 — uncertainty (`task11_uncertainty.json`)

**There is no usable data-driven authority signal.** CatBoost-vs-GBR
disagreement correlates with actual prediction error at **ρ = 0.09**. Binned:
mean \|err\| 2.35 → 2.69 → 3.10 as disagreement rises to 2 — a weak trend
swamped by noise. The empirical `σ_ml` regression is dominated by its intercept
(≈ 2.3 m/s) with small, partly wrong-signed slopes. A conservative fixed
`σ_ml ≈ 3 m/s` is the honest choice, and at that authority (between the native
6 and the "helpful" 1–2) the replay effect on D is weak.

## TASK 14 — comparison with simple corrections (`honest_out.txt`)

On the clean 21-trip set, LOTO:

| correction | csMAE 15–25 | csMAE < 10 m/s | worst trip |
|---|---|---|---|
| isotonic `residual(v_prior − v_spectral)` | 4.9 | 2.8 | +0.4 |
| isotonic `residual(v_prior)` | 5.1 | 2.8 | +0.7 |
| **CatBoost (honest features)** | **4.2** | **2.1** | +0.04 |

CatBoost beats the monotone 1-D map, but by ~0.7 m/s in the key regime — a
single isotonic curve on `v_prior − v_spectral` captures most of the effect and
is far simpler to reason about and gate.

---

## TASK 15 — Verdict: **PARTIAL** (leaning pessimistic on deployment)

1. **How many usable trips?** 23 distinct logger sessions (≈ 5.4 h moving
   withheld-GPS), from 13 days.
2. **How many high-speed?** 11 with ≥ 6 % of outage time above 15 m/s; 2 of
   those are pathological (highway 2× extrapolation / broken mount).
3. **CatBoost / GBR LOTO MAE?** CatBoost residual MAE 5.6 (2 s) / 6.0 (5 s),
   **pooled R² ≈ 0** (collapsed from Phase 30's +0.3). GBR similar MAE but a
   much heavier tail (worst held-out trip +17.9 m/s). On the clean 21-trip set,
   CatBoost residual MAE 1.9–3.4 by regime, R² is still ~0 pooled but the
   corrected speed beats `v_spectral` in every regime.
4. **High-speed MAE improvement?** Clean set, 15–20 m/s: **5.08 → 3.44 m/s
   (+32 %, +1.6 m/s)**; 20–25: +1.7 m/s; > 25: +2.2 m/s but absolute MAE stays
   ~16 m/s (useless). Full pool: the 15–20 gain shrinks to +0.6 m/s and low
   speed *regresses* −0.1…−0.5 m/s.
5. **Does more data help?** **Barely.** The scaling curve flattens by ~8 trips;
   8 → 14 trips buys ~0.35 m/s at 15–25. We are at the observability ceiling
   for this representation.
6. **Gate precision/recall?** The binary saturation gate has good *negative*
   behaviour — it leaves the 3 no-saturation held-out trips within ~2 % of
   baseline. But its *positive* firing is only ~50 % useful: of the 5 held-out
   trips it fires on, it clearly helps 2, is mild on 1, and clearly hurts 2.
   It fires on "spectral saturated", which is not the same event as "odometer
   undershooting".
7. **Replay median / p25 / p75 effect?** Gated (C), median \|D_err\| change
   across 8 held-out trips: **median +6 m** (≈ neutral), p25 −70 m, p75 +200 m,
   best −390 m, **worst +400 m**. D/D_true: median +0.01, range −0.05 … +0.06.
8. **Worst regression?** Offline replay: `07-31-s1` median \|D_err\| 74 → 472 m
   and `07-28-s1` 263 → 564 m (both saturate but their D was already fine).
   Window-level on the full pool: held-out `07-24-s6` +6 m/s, `07-26-s0`
   +11–18 m/s (both pathological, correctly excluded from the primary set).
9. **Safe adaptive σ?** No. Model disagreement is uninformative (ρ = 0.09);
   there is no observable that says "trust this correction now".
10. **Production candidate: NO.**
    * The window-level correction transfers well on in-regime trips (20/21,
      +1.6 m/s at 15–20, worst +0.04) — this part is real and reproducible.
    * But it does not translate into a reliable localisation gain: the offline
      replay is a coin-flip on the trips the gate fires on, because whether the
      correction *helps D* depends on the hidden D error, not on the observable
      saturation. Two of eight held-out trips regress by 300–400 m of lag.
    * More data will not fix this (flat scaling curve), and a simple isotonic
      `residual(v_prior − v_spectral)` captures most of the window-level effect
      anyway.
    * The honest description: **the spectral residual is partly predictable
      cross-trip, but the correction cannot be safely applied because its
      benefit is conditional on an unobservable quantity.** Same wall as
      `SPECTRAL_CALIBRATION_FORENSICS.md`, now confirmed at 23 trips.

No production change. All flags remain `False`. If anything from this line of
work is ever wired, it should be the isotonic `residual(v_prior − v_spectral)`
map behind the binary saturation gate, applied at a conservative `σ_ml ≈ 3`,
and only after a trip-level acceptance test that the current 5-trip + 23-trip
evidence does not yet support.
