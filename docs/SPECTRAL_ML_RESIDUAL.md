# Phase 30 — learned spectral residual correction

**Diagnostic / offline only. Hidden GPS is used here for the target and the
evaluation. Nothing in production changed; no model is wired.**

The question: can a gradient-boosted model recover the missing absolute speed
during a GPS outage from *outage-observable* features, where every linear /
isotonic fit in `SPECTRAL_K_EMPIRICAL.md` and `SPECTRAL_CALIBRATION_FORENSICS.md`
failed cross-trip?

The new ingredient that makes this worth re-asking:

* **five trips**, not two — `runs/review-final/2026-07-2{2,3,4,5,6}/trip`, each a
  120 s GPS warm-up then a ~455 s withheld-GPS outage. Leave-one-trip-out is a
  real 5-fold test instead of a single 07-22 ↔ 07-26 swap.
* **residual target** `y = v_true − v_spectral` (then `v_ml = v_spectral + ŷ`),
  not `k = v_true / v_spectral`.
* a **compact interpretable feature set** (≈50), including the EKF velocity
  prior, which the earlier `k ~ 42 band powers` regression never had.

Scripts: `tools/phase30_dataset.py`, `phase30_models.py`, `phase30_pertrip.py`,
`phase30_offline_replay.py`, `phase30_plots.py`.
Artefacts: `docs/plots/phase30/` (`windows_5s.csv`, `windows_2s.csv`,
`grid_<trip>.csv`, `task45_crosstrip.csv`, `task7_ablation.csv`,
`task6_importance.csv`, `task10_offline_replay.csv`, `phase30_summary.png`).

---

## Method

* **Baseline speed EKF replay** — `predict + ZUPT + lateral + spectral_update`,
  every calibration extra OFF (`spectral_scale`, `bend_scale`, `spectral_censor`
  all `False` — the production default). Per step it logs the fused speed
  `v_ekf`, the pre-spectral prior `v_prior`, `√P[v,v]`, `b_a`, `D`, and the raw
  spectral measurement `v_spectral` and its σ.
* **`v_spectral`** = the pipeline's own `SpectralSpeedModel`, fit on each trip's
  visible warm-up only (`_fit_spectral`), predicted over the outage — exactly
  what `build_imu_samples` feeds the tracker.
* **`v_true`** = withheld-GPS Doppler speed (+ visible-window speed before the
  outage).
* Everything resampled to a 0.5 s grid; windows tiled **non-overlapping** for
  the headline numbers (5 s, n=128 steady-moving; 2 s, n=630), overlapping
  hop-0.5 s only for the offline replay feed.
* Target `residual = mean(v_true) − mean(v_spectral)` over the window. Analysis
  set = outage, moving, `v_spectral ≥ 2`, `max−min v_true < 2 m/s` (steady).
* **All `feat_*` columns are outage-observable.** Hidden GPS touches only
  `v_true_*`, `residual`, `k`.

Feature blocks: `v_spectral` · EKF prior (`v_prior`, `v_ekf`, their gaps to
`v_spectral`, `P[v,v]`, σ_spec, the model's own `deployment_sigma`/
`generalisation_gap`) · IMU history (∫(a_long−b_a) over 2/5/10/20 s, `v_prior`
slopes, a_long rms/mean, positive-accel & braking fractions) · spectral
representation (6 accel + gyro log band powers, `band_tilt` = high−low,
`v_spectral` slope/std/max over 10/20 s, fraction of the rolling 30 s above
13 / 15 m/s, a saturation indicator) · context (gyro mag/std, |yaw| mean/max,
accel magnitude, stationary fraction, time since outage start).

---

## Results

### 1. Cross-trip residual prediction (leave-one-trip-out)

Non-overlapping windows, LOTO over the 5 trips (`task45_crosstrip.csv`):

| model | 5 s steady (n=128) | | | 2 s dense (n=630) | | |
|---|---|---|---|---|---|---|
| | resid MAE | RMSE | R² | resid MAE | RMSE | R² |
| predict 0 | 2.77 | 3.57 | −0.28 | 2.38 | 3.16 | −0.06 |
| LOTO trip-mean constant | 2.42 | 3.29 | −0.09 | 2.39 | 3.16 | −0.06 |
| Ridge | 4.14 | 5.28 | −1.80 | 3.01 | 4.09 | −0.78 |
| **CatBoost** | **2.07** | **2.81** | **+0.21** | **1.82** | **2.55** | **+0.31** |
| GradientBoosting | 2.13 | 2.84 | +0.19 | 1.85 | 2.57 | +0.30 |
| MLP (32,16) | 2.33 | 3.49 | −0.22 | 3.40 | 4.86 | −1.51 |

CatBoost is the only model that beats *both* trivial baselines cross-trip. It
does regress toward the mean hard (panel b of `phase30_summary.png`: it predicts
≈ +1…+2 m/s for almost every window, actual residual −4…+8) — R² ≈ 0.2–0.3 is
real but modest.

### 2. Corrected speed, by true-speed regime (LOTO, 2 s dense)

| true speed | n | MAE `v_spectral` | MAE `v_ekf` | MAE `v_ml` (CatBoost) |
|---|---|---|---|---|
| 0–5 | 139 | 1.90 | 2.46 | **1.64** |
| 5–10 | 132 | 1.91 | 2.57 | **1.62** |
| 10–15 | 195 | 1.99 | 3.35 | 1.88 |
| **15–20** | 149 | 3.21 | 4.03 | **1.66** |
| 20+ | 15 | 7.61 | 6.56 | 6.29 |

The **15–20 m/s saturation regime is halved** (3.21 → 1.66) and low speed is not
hurt (the reason `residual` was chosen over `k`). Above 20 m/s the model cannot
follow — it was trained on four trips with little >20 m/s driving and the
prediction floor there is ≈ the true-speed spread.

### 3. Per-held-out-trip (2 s dense, corrected-speed MAE)

| held-out trip | v_true range | resid mean | MAE `v_spectral` | MAE `v_ml` | 15+ `v_spectral` → `v_ml` |
|---|---|---|---|---|---|
| 07-22 | 1–15 | +1.4 | 2.02 | **1.75** | — |
| 07-23 | 1–7 | −1.8 | 2.06 | **1.83** | — |
| 07-24 | 2–19 | +1.4 | 2.45 | **1.60** | 3.83 → **1.40** |
| 07-25 | 1–20 | −0.8 | 2.50 | **1.99** | 2.39 → **1.40** |
| 07-26 | 2–23 | +1.5 | 2.60 | **1.98** | 4.14 → **3.13** |

CatBoost beats raw `v_spectral` on **all five** held-out trips at the window
level, and the 15+ m/s win reproduces on the three trips that reach that speed
(not just 07-26).

### 4. Feature ablation (LOTO, CatBoost, residual, `task7_ablation.csv`)

| stage | features | 5 s MAE | 2 s MAE |
|---|---|---|---|
| `v_spectral` only | 1 | 2.45 | 2.45 |
| + EKF prior block | 10 | 2.20 | 2.09 |
| + IMU history | 27 | 2.12 | 2.06 |
| + spectral band powers | 46 | 2.04 | 1.82 |
| + temporal / context | 52 | 2.07 | 1.82 |

`v_spectral` alone carries essentially nothing about its own error (matches
`SPECTRAL_K_EMPIRICAL.md` §10, ρ = 0.01). The gain is built from **the EKF
prior** (biggest single jump) and **the raw band-power representation** —
CatBoost pulls signal out of the 42 powers that ridge cannot (ridge *degrades*
as those features are added). Adding context on top is flat.

### 5. Identifiability — the core question (`task6_importance.csv`)

Windows with `12 ≤ v_spectral ≤ 15` (n=172), split by hidden true speed:

| true speed | n | `v_spectral` | `v_prior` | `v_prior − v_spectral` | ∫a over 20 s |
|---|---|---|---|---|---|
| 12–15 | 90 | 13.3 | 11.0 | **−2.22** | 2.5 |
| 15–18 | 59 | 13.5 | 13.4 | **−0.18** | 2.8 |
| 18+ | 23 | 13.5 | 14.7 | **+1.20** | 2.3 |

When `v_spectral` is pinned in its plateau, **the EKF/IMU prior still moves
monotonically with true speed** — `v_prior − v_spectral` goes −2.2 → −0.2 → +1.2
across the three groups. That is the information the earlier `k`-vs-`v_spectral`
analysis was missing: it exists, but it is *weak* — a LOTO logistic classifier
for "true ≥ 16" inside the band scores mean AUC 0.60 (0.51–0.66). CatBoost's top
features are `v_spectral` slope/std over 5–10 s, `vib_high`, `v_prior − v_spectral`,
`v_ekf` — spectral *dynamics* plus the prior, never `v_spectral` level alone.

### 6. Leakage audit (`phase30_models.py` TASK 9/12)

| split | 5 s CatBoost MAE / R² | 2 s CatBoost MAE / R² |
|---|---|---|
| random K-fold (leakage) | 1.77 / +0.40 | 1.42 / +0.54 |
| contiguous time blocks | 1.85 / +0.31 | 1.74 / +0.38 |
| **leave-one-trip-out** | **2.07 / +0.21** | **1.82 / +0.31** |

LOTO is only ≈ 0.3–0.4 m/s worse than the random-window ceiling, and both beat
the baselines. This is *not* the `k ~ 42 powers` collapse (R² −1.3 LOTO in
`SPECTRAL_K_EMPIRICAL.md`) — the residual target + regularised trees + the
5-trip pool hold up.

### 7. Offline EKF replay (`task10_offline_replay.csv`) — TASK 10

Train CatBoost on the other four trips, predict a rolling residual, feed
`v_ml = v_spectral + ŷ` as the spectral measurement into the full single-path
tracker.

* **R_spec unchanged (as the task specifies first):** negligible everywhere.
  The spectral update gain is `K_v ≈ 0.01` (Phase 28) — a corrected reading at
  native σ ≈ 6 m/s barely moves the fused speed.
* **With measurement authority (σ_spec forced to 1–2 m/s):**

  | 07-26 | baseline | v_ml σ=2 | v_ml σ=1 |
  |---|---|---|---|
  | D/D_true | 0.834 | 0.853 | **0.862** |
  | median \|D_err\| m | 265 | 211 | **173** |
  | max \|D_err\| m | 732 | 650 | **608** |
  | speed bias (moving) | −2.51 | −2.13 | **−1.90** |
  | real_wrong decisions | 0 | 0 | **0** (topology intact) |

  On 07-26 (the one covered trip that saturates) the learned correction
  materially reduces the along-track lag and the speed bias with no topology
  regression.

  **But the same authority regresses 07-22** — which does *not* need a spectral
  correction (its shortfall is the AHRS/accel channel, per
  `ACCEL_SCALE_FORENSICS.md`): survival 0.47 → 0.36 → 0.03 as σ tightens, and
  its fragile t+497 shallow fork is perturbed. The model, trained mostly on
  positive-residual trips, imposes a +0.3–1 m/s correction 07-22 has no use for.

  There is **no single σ that is safe across trips.** Per-window uncertainty
  (TASK 11) would be required, and it is feasible but unbuilt: the online
  saturation proxies cleanly separate the trips that need correction from those
  that do not — fraction of the rolling 30 s `v_spectral` above 15 m/s is
  0.05–0.25 on 07-24/25/26 and **exactly 0.00** on 07-22/23; `v_spectral` 20 s
  max is 14–18 vs 8. A gate on those leaves 07-22/23 untouched, but also
  suppresses enough genuine corrections that the pooled window MAE gain shrinks
  to ≈ 0.2 m/s.

### 8. Uncertainty (TASK 11, not implemented)

OOF residual-prediction RMSE ≈ 2.8 m/s overall, 3.1 m/s for true ≥ 15 — roughly
flat with speed (except the unreachable >20 bin) and with predicted-correction
magnitude. It is **not** flat by trip: bias −1.6 on held-out 07-26 (under-
corrects the trip that needs it most), +1.6 on held-out 07-25 (over-corrects a
negative-residual trip). A conservative `R_ml = R_spec + (2.8)²` is the minimum
honest inflation; a trip-adaptive term is what is actually missing.

---

## Verdict — **PARTIAL**

1. **Is the residual predictable on held-out trips?** Yes, weakly. CatBoost LOTO
   MAE 1.8–2.1 m/s vs 2.3–2.8 for predict-0 / trip-mean, R² +0.2–0.3, beats raw
   `v_spectral` on all five held-out trips, and **halves the 15–20 m/s error**
   (3.2 → 1.6). It regresses toward the trip-pool mean and cannot predict the
   > 20 m/s tail.
2. **Which features carry cross-trip information?** The **EKF/IMU velocity prior**
   (`v_prior − v_spectral` is the identifiable quantity when `v_spectral`
   saturates) and the **raw band-power representation + `v_spectral` short-window
   dynamics**. `v_spectral` level alone carries nothing.
3. **CatBoost vs Ridge?** CatBoost wins decisively — Ridge blows up cross-trip
   (MAE 3–4, R² negative), as every linear model has in this project.
4. **MLP vs CatBoost?** No. The MLP is unstable cross-trip and frequently worse
   than predict-0. A larger network is not justified.
5. **Residual or k target?** Residual — marginally better pooled, clearly better
   at low speed, no divide-by-small-`v_spectral`. `k` ties on the dense set.
6. **Does the corrected speed reduce the actual along-track lag?** Only with
   added measurement authority, and only where it is needed: 07-26 D/D_true
   0.83 → 0.86, median \|D_err\| 265 → 173 m, no topology regression. The same
   setting *hurts* 07-22. Net: real but conditional.
7. **Enough for a production ML correction?** **No.** The gain is not safely
   separable from the harm without per-window uncertainty gating, which this
   phase deliberately did not build. The wired bend-anchor + censor machinery
   still reaches 0.98 on 07-26 (with a bend); the ML correction's value would be
   on the ~4/5 of trips with **no** usable bend, at ~0.86, *if* gated.
8. **Limitation — capacity or observability?** **Observability / data.**
   GradientBoosting matches CatBoost, the MLP does not beat it, and LOTO is
   within 0.4 m/s of the random-split ceiling — the model is already near the
   information limit. The wall is five trips, an under-identified > 20 m/s
   regime, and a high-speed correction *magnitude* that does not fully transfer
   between trips (held-out 07-26 is under-corrected). More trips — especially
   fast ones with clean warm-ups — is the only thing that moves this to SUCCESS.

No production change. All flags remain `False`.
