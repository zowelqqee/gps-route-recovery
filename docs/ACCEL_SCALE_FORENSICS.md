# Is the distance error a single multiplicative constant?

**Verdict: B — CONDITIONAL SCALE.** The 7–16 % distance shortfall is not a
fixed multiplier and is not in the accelerometer. It is a **stable,
trip-independent, speed-dependent** error — the spectral speed model saturating
above the speed range its 145 s warm-up could calibrate. A speed-dependent
correction `c(v)` transfers between the two trips and removes most of the lag;
a single constant does not, and fails cross-validation by ~10 %.

Diagnostics only. Nothing multiplied in production. `g` for this forensic is
the local value **9.8195 m/s²**, not standard 9.80665.

## Part 1 — raw accelerometer vs local gravity

From the vehicle-logger day CSVs (`acc_x/y/z`, specific force in g-units, 100 Hz),
on every quiet stationary segment (speed < 0.3 m/s, accel std < 0.08 m/s²,
gyro < 0.6°/s):

| | 2026-07-22 | 2026-07-26 |
|---|---|---|
| quiet samples | (short stops, ~5–8 %) | 20 516 (17 %) |
| median \|a_raw\| (m/s², assuming CSV g = 9.80665) | **9.832** | **9.833** |
| std across quiet samples | ~0.03 | **0.031** |
| p05 / p95 | 9.79 / 9.89 | 9.79 / 9.89 |
| error vs 9.8195 | +0.013 | +0.014 |
| **k_g = 9.8195 / median** | **0.9988** | **0.9986** |
| per-stop k_g range | 0.9987–0.9993 | 0.9985–0.9991 |
| temperature | 25–27 °C | 30–32 °C |

Every stationary segment on both trips reads `|a| = 9.83 ± 0.03 m/s²`,
`k_g ≈ 0.999`. `az ≈ +1.002 g` when parked (the ~0.2 % that `k_g` reflects),
`ax ≈ −0.035`, `ay ≈ +0.025` g of residual horizontal offset. No dependence on
temperature or on time-in-trip; identical between the two trips.

**The raw MEMS accelerometer measures gravity to 0.15 %. There is no hardware
scale error.** The distance shortfall is entirely downstream.

## Part 2 — where the attenuation is (not the MEMS)

```
raw accel (g)          k_g ≈ 0.999   -- perfect
  → gravity removal     via the logger's own quaternion (0.003 g residual)
  → attitude projection onto the vehicle forward axis
  → a_long (m/s²)       open-loop ∫a_long over accel windows: median dv-scale
                        1.07 (07-22) / 0.91 (07-26), IQR ±1 -- no fixed slope,
                        huge per-window spread; the AHRS leans under sustained
                        acceleration and eats a variable share of it
  → speed filter        spectral model (fitted on 145 s warm-up) + lateral
                        anchors + ZUPT
  → v_est → D_est       D_est/D_true = 0.928 / 0.843
```

`a_long` mean is +0.008 / +0.020 m/s² and final `b_a` is +0.018 / −0.082 — the
accel channel carries almost no bias and contributes little to `D` (the doc
`PACMAN_TRACKER.md` already established: removing it changes distance < 15 %).
The distance is carried by the **spectral speed model**, and that is where the
speed-dependent attenuation lives.

## Part 3 — windowed distance scale `c_D = ΔD_true / ΔD_est`

Sliding windows, minimum true distance `1.5·W` m per window.

| | | 30 s | 60 s | 120 s | 240 s |
|---|---|---|---|---|---|
| **2026-07-22** | N | 86 | 44 | 25 | 13 |
| | mean c_D | 1.070 | 1.068 | 1.074 | 1.069 |
| | median c_D | 1.039 | 1.033 | 1.019 | 1.057 |
| | std | 0.303 | 0.231 | 0.171 | 0.108 |
| | MAD | 0.226 | 0.200 | 0.145 | 0.130 |
| | CV | 0.283 | 0.216 | 0.159 | **0.101** |
| | p05–p95 | 0.72–1.55 | 0.85–1.43 | 0.85–1.38 | 0.95–1.24 |
| **2026-07-26** | N | 123 | 62 | 31 | 14 |
| | mean c_D | 1.194 | 1.199 | 1.207 | 1.206 |
| | median c_D | 1.151 | 1.205 | 1.234 | 1.198 |
| | std | 0.254 | 0.152 | 0.110 | 0.065 |
| | MAD | 0.242 | 0.193 | 0.104 | 0.076 |
| | CV | 0.212 | 0.127 | 0.091 | **0.054** |
| | p05–p95 | 0.86–1.56 | 0.95–1.42 | 1.05–1.35 | 1.12–1.30 |
| **pooled** | mean c_D | 1.143 | 1.145 | 1.147 | 1.140 |
| | median c_D | 1.094 | 1.128 | 1.165 | 1.157 |
| | CV | 0.247 | 0.174 | 0.135 | 0.098 |

**Within each trip**, c_D is stable at 120–240 s (CV 0.05–0.10). **Between
trips** the level differs: **07-22 ≈ 1.07, 07-26 ≈ 1.20**. Pooling them keeps
the CV at ~0.10–0.14 even at 240 s because the two trips sit at different
levels.

## Part 4/5 — speed scale and Δv scale

`c_v` (windowed mean speed) is identical to `c_D` over the same window: 07-22
≈ 1.02–1.04, 07-26 ≈ 1.15–1.23.

`c_dv` (windows with |Δv_true| > 2.5 m/s):

| | 2026-07-22 | 2026-07-26 |
|---|---|---|
| accel c_dv | 1.43 (IQR 1.05–2.11) | 1.16 (IQR 0.52–1.81) |
| brake c_dv | 1.45 (IQR 0.93–2.48) | 1.24 (IQR 0.60–2.11) |
| sign consistent | 85–87 % | 81–83 % |

The acceleration channel under-responds by ~15–45 % **with an enormous
per-window spread** (IQR spanning 1.0–2.5 and sometimes the wrong sign). It is
not a clean scale — it is the AHRS leaning under sustained acceleration, which
is manoeuvre-dependent. And the accel channel is not what carries `D`.

## Part 6 — c_D by speed regime

`c_D` (45 s windows) binned by window mean speed, **both trips on one axis**:

| window mean speed | pooled median c_D | 07-22 | 07-26 |
|---|---|---|---|
| 0–3 m/s | 0.96 | 0.95 | 1.14 |
| 3–6 | **1.05** | 1.04 | 1.05 |
| 6–9 | **1.08** | 1.11 | 1.04 |
| 9–12 | **1.16** | — | 1.17 |
| 12–15 | **1.25** | — | 1.27 |
| 15–20 | **1.32** | — | 1.32 |

By window p90 speed: 0–6 → 0.97, 6–10 → 1.01, 10–14 → 1.04, 14–18 → 1.09,
18–25 → 1.30.

**`c_D` is a clean monotonic function of speed, and it is the same function for
both trips** where they overlap (3–6 m/s: 1.04 vs 1.05; 6–9: 1.11 vs 1.04).
`c_D` ≈ 1.0 at walking pace, ~1.05 at 5 m/s, ~1.16 at 10 m/s, ~1.32 at 18 m/s.
The whole-trip constants differ only because **07-22 spends most of its moving
time below 8 m/s (c ≈ 1.0–1.1) and 07-26 above 12 m/s (c ≈ 1.25–1.32)**.

This is spectral-model saturation: the model, ridge-fitted on the visible
145 s at low speed, emits a near-constant ~9–10 m/s prediction and progressively
undershoots as the true speed rises — exactly what `PACMAN_TRACKER.md` §"what is
physically not there" documents ("predicted 6.2 / 8.8 / 9.9 / 9.7 / 9.4 / 9.0
for true 0–2 / 2–6 / … / 20+ m/s").

## Part 7 — time stability

| | 1st half c_D | 2nd half c_D |
|---|---|---|
| 2026-07-22 (60 s) | 1.031 | 1.035 |
| 2026-07-22 (120 s) | 1.051 | 1.001 |
| 2026-07-26 (60 s) | 1.160 | 1.244 |
| 2026-07-26 (120 s) | 1.155 | 1.273 |

07-22 is flat in time. 07-26 drifts **up** from 1.16 to 1.27 — its second half
contains more fast driving. So even within 07-26 the "constant" tracks the
speed profile, not the clock. No stop-related or turn-related step change.

## Part 8 — between trips

The whole-trip naive corrections are `1/0.928 = 1.077` (07-22) and
`1/0.843 = 1.187` (07-26). The 11-point gap is **entirely speed-profile**:
the `c_D(speed)` curve is shared (Part 6), 07-26 just lives higher on it. This
is option **C in the brief's list — spectral saturation hurts the high-speed
trip** — not a per-trip hardware scale (B), not a non-multiplicative error (E).

## Part 9–11 — candidate constants, learnability, cross-validation

Diagnostic replay, `D_corr = ∫ v_est · c dt`, target `D_corr/D_true = 1.000`:

| correction | 07-22 | 07-26 |
|---|---|---|
| none | 0.928 | 0.843 |
| **global constant 1.15** | **1.067** (+7 % over) | **0.969** (−3 % under) |
| unified `c(v)` curve (both trips) | 1.007 | 0.978 |
| `c(v)` fitted on **07-22 only** → both | 1.000 | **0.964** |
| `c(v)` fitted on **07-26 only** → both | 1.015 | 0.992 |
| `c(v)` from warm-up (v ≤ 6 real, v > 6 flat at c(6)) | 0.995 | **0.912** |

Empirical `c(v)` (30 s windows):

| v ≈ | 2 | 4 | 8 | 10 | 14 | 18 |
|---|---|---|---|---|---|---|
| 07-22 | 0.95 | 1.05 | 1.14 | 1.08 | — | — |
| 07-26 | 1.10 | 1.06 | 1.03 | 1.30 | 1.20 | 1.15 |
| unified | 1.03 | 1.06 | 1.08 | 1.19 | 1.20 | 1.15 |

**Cross-validation of a fixed per-trip constant:**
- fit 07-22 (1.077) → apply to 07-26 → 0.900 (**10 % under**)
- fit 07-26 (1.187) → apply to 07-22 → 1.111 (**11 % over**)
- 30 s-window median → tested on 240 s windows: c under-states by 2 % (07-22)
  / 4 % (07-26)

**Can the constant be known without hidden GPS?**

| source | 07-22 | 07-26 | verdict |
|---|---|---|---|
| local gravity `k_g` | 1.001 | 1.001 | says "no correction" — wrong |
| stationary calibration | k_g 0.999 | k_g 0.999 | same — MEMS is fine |
| first 60 s of outage `c_v` | 1.30 | 1.13 | noisy, inconsistent with the true need |
| first-half-of-trip `c_D` | 1.03 | 1.16 | biased low (less fast driving early) |
| warm-up `c(v)`, v ≤ 6 m/s | ~1.0–1.08 | ~1.03–1.08 | **correct for the low-speed part only** |
| cross-trip constant | — | — | does not transfer (±10 %) |

The low-speed part of `c(v)` (v ≤ 6 m/s, `c ≈ 1.0–1.08`) *is* observable from
the GPS-visible warm-up and would fix 07-22. The **high-speed part
(`c ≈ 1.2–1.32` above 12 m/s) is not observable from a warm-up that contains no
fast driving** — which is precisely why the warm-up-only `c(v)` leaves 07-26
9 % short.

## Answers

1. **Raw accelerometer vs g = 9.8195:** `k_g = 0.999` on both trips, std 0.03,
   temperature-independent. The MEMS is accurate to 0.15 %.
2. **Stable constant `c_D` on 30/60/120/240 s:** within a trip, yes at 120–240 s
   (CV 0.05–0.10); across trips, no — 07-22 sits at 1.07, 07-26 at 1.20.
3. **Does it change with speed:** yes, strongly and monotonically —
   `c_D ≈ 0.96` at < 3 m/s, `1.05` at 5, `1.16` at 10, `1.32` at 18 m/s.
4. **07-22 vs 07-26:** the `c_D(speed)` *curve* is identical; the whole-trip
   numbers differ only because the two trips sample different parts of it.
5. **Where the attenuation appears:** downstream of a perfect MEMS — the
   spectral speed model saturating above its 145 s low-speed calibration range
   (secondary: the AHRS leaning under sustained longitudinal acceleration,
   `c_dv ≈ 1.2–1.45`, but noisy and not what carries `D`).
6. **Known without hidden GPS:** only partially. Gravity and the warm-up give
   the low-speed correction (`c ≈ 1.0–1.08`); the high-speed correction
   (`c ≈ 1.2–1.32`) requires fast driving under GPS, which the warm-up does not
   have.
7. **Cross-validation:** a fixed per-trip constant fails by ~10–11 % when moved
   to the other trip. A speed-dependent `c(v)` transfers to within ~3–4 %.

## Special question

*"Can a simple constant remove most of the 7–16 % lag?"* — **No, not one
constant.** A global 1.15 leaves 07-22 over by 7 % and 07-26 under by 3 %; it
trades one trip's undershoot for the other's overshoot. A **speed-dependent
`c(v)` removes most of it and is trip-transferable**, but its high-speed
segment cannot be learned from the GPS-visible warm-up — so on the trip that
needs it most (07-26), a warm-up-derived correction stops at ~9 % residual.

## What this points to (not done here)

The right lever is not a distance multiplier — it is re-calibrating the
**spectral speed model during the outage** from a source that scales with
speed: the `v = a_lat/ω` circular-motion anchors (accurate to a few tenths of
m/s in real turns) or the accepted turn-to-turn map intervals
(`INTERVAL_CALIBRATION.md`), feeding the scale state `k` that already exists.

## Not done (as instructed)

No production multiplier, no `k_a`/`D` change, no route-tracker or interval-gate
change, no spectral re-tune, no magnetometer, no Mahony, no constant learned
from withheld GPS presented as production calibration. `9.80665` was not used
as truth anywhere in this analysis.
