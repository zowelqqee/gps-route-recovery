# Can the spectral absolute-speed source be calibrated online without GPS?

**Verdict: C — the mechanism is sound, the information is not on these trips.**
A scalar spectral scale `k_s` in `v_true ≈ k_s · v_spectral` is the *right state
to learn* — on 07-26 a causal `k_s ≈ 1.30` would take `D/D_true` from 0.84 to
~1.00 and sextuple junction survival — but the two GPS-free anchors that could
set it (`v = a_lat/ω` turn anchors, accepted turn-to-turn map intervals) **do
not sample the high-speed cruising regime where the model saturates**, so an
online calibrator run causally converges to the wrong value (07-26) or
over-corrects (07-22). The machinery is implemented, fully gated, tested, and
**OFF by default** (`speed.spectral_scale_enabled = False`). Nothing in
production changed.

Diagnostics only. No hidden GPS in the calibration path. `k_s` is a **separate
side-state from `k_a`** (accel scale) and never absorbs the other's error.

---

## Phase 0 — is `c(v_spectral)` identifiable?

A correction indexed by the *observable* `v_spectral` (not by hidden `v_true`)
must be a function: one `v_spectral` → one multiplier. It is not.

Pooled over both trips, binning by `v_spectral` and looking at the true speed
that produced it:

| `v_spectral` bin | `v_true` p10–p90 | spread | `c_req = v_true/v_spectral` p10–p90 |
|---|---|---|---|
| 8–9 m/s | 7.0 – 17.5 | 10.5 m/s | 0.83 – 2.06 |
| 11–12 m/s | 9.1 – 21.2 | 12.1 m/s | 0.79 – 1.83 |
| 13–14 m/s | 12.4 – 20.4 | 8.0 m/s | 0.92 – 1.51 |

The median `c_req` curve indexed by `v_spectral` is **non-monotonic** — it peaks
near 1.4 around `v_spectral` 8–10 and falls back toward 1.0 by `v_spectral` 14+,
because the model's saturated output folds several true speeds onto the same
prediction. **A pointwise `c(v_spectral)` is not a usable correction.** A single
scalar (or slow-moving scalar) is the only well-posed online form.

## Phase 1 — do the raw features still carry high-speed information?

Yes, weakly, and **not transferably**.

| | 07-22 | 07-26 |
|---|---|---|
| warm-up model on the outage — RMSE / bias | 3.1 / −0.6 | **4.55 / −2.18** |
| ceiling (ridge refit on the outage w/ hidden GPS) — RMSE | 2.4 | **2.58** |
| ceiling recovers `v_true∈[15,20)` median pred | — | 16.4 (warm-up 13.3) |
| ceiling recovers `v_true∈[20,25)` median pred | — | 18.1 (warm-up 12.5) |
| cross-trip: model fitted on the *other* trip — RMSE | 3.18 | 5.96 (worse than warm-up) |
| partial corr. of features w/ `v_true` residual after removing `v_spectral` | ≤ 0.33 | ≤ 0.47 |

The band-power features *do* contain enough to halve the high-speed RMSE **if
you already have the outage's own GPS to refit** — but a model trained on one
trip makes 07-26 worse, and the residual partial correlation is 0.3–0.5. There
is signal, but not enough to justify a feature-recalibration model that isn't
just fitting one trip's noise. **A feature recalibrator (brief's Model D) was
not built** — Phase 14's honest options apply.

## Phase 2 — how many clean lateral anchors, and at what speed?

`v = a_lat/ω`, restricted to well-conditioned turns (`|ω| > 3·ω_min`, anchor
σ < 0.35·σ_max):

| | clean anchors | 0–6 m/s | 6–12 | 10–16 | 14–20 | median \|err\| |
|---|---|---|---|---|---|---|
| 07-22 | 17 | 8 | 7 | 2 | 0 | ~0.5 m/s |
| 07-26 | 7 | 0 | 2 | 4 | 4 | ~0.5 m/s |

The tracker's own (looser) production gate admits **189** k_s-eligible lateral
updates on 07-22 and **345** on 07-26. Either way, **every lateral anchor is
taken during a turn**, i.e. at 6–16 m/s where the spectral model is *not yet
saturated* — the regime that carries the 07-26 error (16–20 m/s sustained
cruising, throttle steady) produces no turn and therefore no lateral anchor.

## Phase 3 — do the map intervals constrain spectral scale?

`L_map ≈ k_s · ∫ v_spectral dt`, so `k_s ≈ L_map / ∫v_spectral dt`. Only turn
pairs that clear the anti-circularity endpoint-quality gate qualify.

| interval | dur | `L_map` | `∫v_spectral` | implied `k_s` | `∫v_true` | map quality |
|---|---|---|---|---|---|---|
| 07-22 ev0→ev1 | 93 s | 806 m | 594 m | **1.358** | 799 m | good (~1 % off) |
| 07-22 ev3→ev4 | 33 s | 184 m | 197 m | 0.938 | 210 m | **26 m / 12 % short** |
| 07-26 ev0→ev3 | 417 s | 3367 m | 3775 m | 0.892 (wrong sign) | 4107 m | **740 m / 18 % short** |

- **07-22** has exactly **one** trustworthy interval, and it implies `k_s ≈
  1.36` — but that interval covers the trip's single fast stretch, so applying
  its ratio to the whole (mostly slow) trip over-corrects.
- **07-26 has no usable map interval.** Its only long turn-pair has a committed
  route length 18 % short of truth, so the implied `k_s` is 0.89 — wrong
  direction. The interval innovation gate correctly rejects it; `k_s` is never
  moved by an interval on 07-26.

## Phase 4/6 — the online calibrator (Model A: scalar `k_s`)

Implemented in `speed.py`:

- `k_s` is a scalar side-state, `k_s(0)=1.0`, `σ_k_s(0)=0.15`, slow random walk
  `σ_rw = 0.004 /√s`, bounds `[0.7, 1.7]`, σ floor 0.03.
- Moved **only** by `spectral_scale_point()` (from a clean lateral anchor,
  `z = v_anchor/v_spectral`, requires `v_spectral ≥ 5 m/s`) and
  `spectral_scale_interval()` (from an accepted map interval,
  `z = L_map/∫v_spectral`, requires `∫v_spectral ≥ 30 m`). Scalar Kalman step,
  innovation-gated at 3.5 σ.
- `spectral_update()` applies `k_s·v_spectral` and **inflates** the measurement
  variance by `(v_spectral·σ_k_s)²` — an unconstrained `k_s` widens `σ_v`/`σ_D`
  rather than pretending to a correction it hasn't earned. This path **never**
  touches `k_s` (no self-calibration loop).
- **Interval Jacobian** `∂L_pred/∂k_s = ∫v_spectral dt`: analytic, exact
  (`L_pred` is linear in `k_s`), verified against finite difference in
  `test_interval_jacobian_dL_dk_s_is_the_integrated_spectral_speed`. The
  calibrator carries the length-domain σ into the `k_s` domain as
  `(σ_m/∫v_spectral)²`, which is `1/(∂L/∂k_s)²`.
- `k_a` is refused both a mean and an information update from every non-interval
  source (`_update(..., allow_scale=False)` zeros `K[KA]`), so enabling `k_s`
  cannot make `k_a` absorb spectral error. Confirmed: in every ablation below,
  `k_a` stays within 0.02 of 1.0 except where a real map interval moved it.

## Phase 10 — causal online ablation (production gates, no future leakage)

`A` baseline · `B` `k_s` from lateral anchors only · `C` `k_s` from map
intervals only · `D` both. Causal replay — each `k_s` update uses only anchors
at/before its timestamp.

| trip | run | D/D_true | endpoint ΔD | d30 / d60 / d120 (m) | route wrong | 1st wrong (s) | survival | lat / iv acc. | `k_s` ± σ | `k_a` |
|---|---|---|---|---|---|---|---|---|---|---|
| **07-22** | A | 0.930 | −7.0 % | 20.9 / 27.7 / 142 | 13 | 497 | **0.204** | 0 / 0 | 1.000 ± 0.00 | 1.000 |
| | B lat | 0.947 | −5.3 % | 21.9 / 27.1 / — | 13 | 476 | 0.155 | 189 / 0 | 1.048 ± 0.099 | 1.000 |
| | C iv | 1.043 | +4.3 % | 22.6 / 32.7 / — | 13 | 458 | 0.108 | 0 / 1 | 1.347 ± 0.139 | 1.020 |
| | D both | 0.967 | −3.3 % | 21.8 / 26.9 / — | 13 | 498 | 0.146 | 189 / 1 | 1.074 ± 0.095 | 1.004 |
| **07-26** | A | 0.835 | −16.5 % | 71.7 / 122 / 410 | 0 | — | 0.102 | 0 / 0 | 1.000 ± 0.00 | 1.000 |
| | B lat | 0.779 | −22.1 % | 87.1 / 160 / — | 0 | — | 0.070 | 345 / 0 | 1.019 ± 0.046 | 1.000 |
| | C iv | 0.834 | −16.6 % | 72.5 / 119 / — | 0 | — | 0.104 | 0 / **0** | 1.000 ± 0.198 | 1.000 |
| | D both | 0.780 | −22.0 % | 85.7 / 155 / — | 0 | — | 0.072 | 345 / 0 | 1.019 ± 0.046 | 1.000 |

**`k_s(t)` trajectories** (`stats["spectral_scale"]["trace"]`, sampled):

- **07-22 B**: rises to ~1.01 in warm-up, sags to **0.75** over the slow first
  half (low-speed turn anchors, where `v_anchor < v_spectral`), climbs back
  through 1.0–1.22 on the fast middle stretch, ends **1.048 ± 0.099**. Wide
  swings, high final σ — never converges.
- **07-22 C**: single interval update at t≈241 s, `k_s: 1.00 → 1.347`, σ 0.139.
- **07-26 B/D**: oscillates 0.75–1.10 for the whole trip tracking the turn
  speed, ends **1.019 ± 0.046** — a *confident wrong answer*, because 345 turn
  anchors at 6–16 m/s genuinely imply `k_s ≈ 1.0` there. The saturation regime
  never enters the estimator.
- **07-26 C**: no interval accepted; `k_s` stays 1.0, σ grows to 0.198 by random
  walk alone — **honest**: "not observed."

Route topology (`real_wrong = 13 / 0`) and first-wrong-junction time are
**unchanged** by every `k_s` variant — `k_s` shifts distance, not the route
graph, and the anti-circularity gates hold.

## Phase 10-H / Phase 4 — non-causal ceiling (what a *known* `k_s` would do)

Monkeypatch `build_imu_samples` to scale `spectral_speed`; no production change.
This uses outside knowledge of the right scalar — it is the **upper bound**, not
a deployable result.

| trip | correction | D/D_true | survival | d30 / d60 / d120 (m) |
|---|---|---|---|---|
| **07-22** | none | 0.931 | 0.204 | 20.9 / 27.7 / 142 |
| | scalar 1.10 | 0.961 | 0.162 | 20.6 / 26.3 / 123 |
| | scalar 1.20 | 0.991 | 0.161 | 20.8 / 26.6 / 104 |
| | scalar 1.35 | 1.037 | 0.144 | 22.3 / 31.1 / 115 |
| | piecewise `k(v_spectral)` | 0.96–0.97 | 0.17 | 20 / 25 / 103 |
| | **CEILING** `spectral := v_true` | **0.957** | **0.175** | 18.2 / 21.3 / 76 |
| **07-26** | none | 0.836 | 0.102 | 71.7 / 122 / 410 |
| | scalar 1.10 | 0.884 | 0.241 | 63 / 98 / 349 |
| | scalar 1.20 | 0.932 | 0.421 | 58 / 85 / 288 |
| | **scalar 1.35** | **1.005** | **0.666** | 57.5 / 81.5 / 305 |
| | piecewise mild | 0.943 | 0.463 | 57 / 81 / 270 |
| | **CEILING** `spectral := v_true` | 0.955 | 0.334 | 56 / 67 / 242 |

- **07-26: the lag is spectral scale.** A scalar ~1.30–1.35 removes essentially
  all of the distance error and lifts junction survival from 0.10 to 0.67. (The
  CEILING scores *lower* than scalar 1.35 only because replacing the spectral
  series with the noisier truth series adds per-sample variance the constant
  avoids — both confirm the diagnosis.)
- **07-22: the lag is not spectral.** The CEILING moves `D` only 0.931 → 0.957
  and *drops* survival 0.204 → 0.175. Every `k_s > 1` desynchronises the
  junction timing of a trip whose low-speed distance was already close. 07-22's
  residual belongs to the accel channel / AHRS lean (`ACCEL_SCALE_FORENSICS.md`
  Part 4–5) and route geometry, not the spectral scale.

## Phase 11 — 07-26 special: can causal calibration cut the 0.84 lag?

**No — not from the anchors this trip provides.** The scalar that would fix it
(~1.30) is roughly what a single clean cruising map interval would yield, and
07-26 has none (its one long turn-pair is on an 18 %-short committed route). Its
345 lateral anchors legitimately imply `k_s ≈ 1.0` because they all sit in the
unsaturated 6–16 m/s turn band. Run causally, `k_s` ends at 1.019 ± 0.046 and
`D/D_true` gets slightly *worse* (0.836 → 0.78). The information required is not
present during the outage.

## Phase 12 — 07-22 shallow fork: no distance self-confirmation

The `k_s` interval update takes the same `L_map` the route committed to, so it
must not be what confirms that route. It isn't: `real_wrong` and the
first-wrong-junction time are identical across A/B/C/D (13, t≈497 s), the
interval endpoint anti-circularity gate is upstream and unchanged, and
`test_map_interval_ks_update_needs_the_interval_machinery_to_raise_it` /
`test_grossly_wrong_interval_is_gated_out_of_k_s` pin this.

## Phase 13 — transfer

| transferable | not transferable |
|---|---|
| model **form** (scalar `k_s`, slow RW, anchor-gated scalar Kalman, min-speed gate, variance inflation) | the **value** — 07-22 wants ~1.0, 07-26 wants ~1.30 |
| priors `k_s(0)=1.0`, `σ=0.15`, bounds `[0.7,1.7]` | any fitted `c(v)` coefficients |
| the interval Jacobian `∫v_spectral dt` | — |

## Phase 16 — 07-26: why the committed interval is 3367 m when truth is ~4107 m, and the initial-GPS → turn-event anchor

**The 740 m is not lost in the route. It is one contiguous piece of the correct
road that the committed route had not yet driven at `t_b`, because endpoint B is
a road-curvature event force-matched to the graph junction 740 m upstream — and
that 740 m is exactly the odometer's spectral undershoot over the 4.1 km.**

### The interval ev0 → ev3

| | value |
|---|---|
| event A (id 0) | strong left, peak-rate 0.271 rad/s, `t_peak` 164.8, `d_event` 55.3 → junction node `1654267456` @ offset 39.2, residual −16 m, p 0.999 |
| event B (id 3) | left **42°** but peak-rate only **0.104 rad/s** spread over 578.3–589.0 s; `t_peak` 582.1, `d_event` 3505.8 → junction node `428122719` @ offset **3406.6**, residual **−89 m**, p **0.669**, **low-confidence + provisional** |
| committed A→B | 12 edges `1558·3883·13427·3093·729·4157·10458·528·1079·10266·10265·6964`, Σ len = **3367.4 m** |
| truth A→B (HMM map-match of withheld GPS) | **the same 12 edges + partial of edge `4326`**, graph distance **4104.2 m** (≈ ∫v_true 4106, ∫v_spectral 3775) |
| first divergence | none in topology — committed and truth edge lists are identical for all 12 edges; committed simply **stops one edge short** |

### Reconciliation of the 740 m

```
L_committed  (node 1654267456 → node 428122719, 12 full edges)   = 3367.4 m
+ edge 4326 travelled before the TRUE ev3 bend (truth s≈737)      = + 736.8 m
+ twin / parallel-carriageway geometry                            =     0.0 m
+ skipped edges / detour                                          =     0.0 m
+ route-history bookkeeping error                                 =     0.0 m
+ endpoint-offset error at A (init_s already subtracted in seed)  =     0.0 m
────────────────────────────────────────────────────────────────────────────
= truth A→B graph distance                                        = 4104.2 m
```

Every missing metre is **the first ~737 m of edge `4326`** ("Выборгское шоссе",
1038.8 m, a *curving* highway polyline). ev3's 42° is the car following that
curve — truth stays on edge 4326 from s≈300 (t≈555) to s≈1000 (t≈602) while its
heading rotates ~50°. `junction_turn(6964→4326) = 1.8°`; the only real turn at
node `428122719` is the 10 m stub `4325` (+90°, rejected). So the matcher bound
a **road-curvature event to the nearest upstream junction**, at the offset the
(spectral-undershooting) odometer reported — `d_event` 3506, and 0.16 × 4106 ≈
660–740 m is precisely the trip-wide odometer deficit. Truth reaches offset
3406.6 at **t≈500.8**, i.e. the junction was closed **81 s / 741 m early**.

### Does `real_wrong = 0` hide anything?

It hides the **−741 m junction-B position error** and the **curvature-as-junction
association**, but nothing else: no twin-edge substitution, no parallel
carriageway, no skipped geometry, no wrong road. The committed edge list *is*
truth's. `real_wrong` measures topology; the defect is arc-length alignment of
one junction, which it does not measure. (Side note: strong turn **ev1** (53°,
t 348.7) **expired unmatched** with no rollback — also highway curvature on the
Приозерское→Выборгское merge. 07-26's gyro events are dominated by road
curvature, not junctions, which is why it has so few usable anchors.)

### Initial-GPS → turn-event anchor

Start graph position is independent and exact: edge `7818` s 135.2 at `t_start`
144.8 (truth agrees to <1 m). `boundary_offset` of any later junction already
equals the committed map distance from that start position (the seed subtracts
`init_s`), so no start-side partial-edge term is needed. Only **two**
event-driven decisions exist on 07-26:

| start → event | L_map | ∫v_spectral | implied `k_s` | true `k_s` (hidden GPS) | ≥14 m/s frac | safe? |
|---|---|---|---|---|---|---|
| → ev0 (t 164.8, 20 s) | 39.2 m | 59.7 m | 0.66 | 0.57 | 0 % | **no** — 20 s, near-zero speed, ratio is the spectral floor not the scale |
| → ev3 (t 582.1, 437 s) | 3406.6 m | 3834.8 m | **0.888** | 1.082 | 20 % | **no** — endpoint B is provisional + low-confidence, the "turn" is road curvature (no junction evidence), and `L_map` is odometer-derived → calibrating the spectral scale on it is the forbidden circular loop |

`→ ev3` is the only candidate that reaches sustained high speed, and it is
unusable for the same reason the interval is short: the speed error mis-binds
its endpoint. Its implied `k_s = 0.888` points the **wrong way** (the true
requirement is ≥ 1.08 for that span, ≈ 1.30 for the trip).

### Causal replay (anchor used anyway, k_s applied only for t ≥ 582)

| run | D/D_true | survival |
|---|---|---|
| baseline | 0.836 | 0.103 |
| implied `k_s = 0.888` from t 582 | **0.805** (worse) | 0.103 |
| true `k_s = 1.082` from t 582 (hidden-GPS ref) | 0.859 | 0.103 |
| reference: scalar 1.30 whole outage | 0.983 | 0.690 |

Feeding the anchor's own number makes 07-26 **worse**; even its hidden-GPS-true
value barely helps, because it is a span average that under-weights the 18–25 m/s
cruise after ev3. **No safe initial-GPS → turn-event interval exists on 07-26** —
*via a junction*. Phase 17 revisits ev3 not as a junction but as a bend.

## Phase 17 — road curvature as a GPS-free intra-edge position anchor

**ev3 is not a junction turn — it is the bend of edge 4326, and its shape in the
edge polyline pins the car's along-edge position to ±1 m from gyro + map
geometry alone, no node and no GPS.** That recovers the 646 m the junction
mis-binding lost and, causally, lifts 07-26 `D/D_true` from 0.836 to 0.890
without touching route topology. The implied spectral scale for the span is
exactly right (1.080) but is a span average, not the trip's high-speed need.

Prototype only: `processor/src/geotrace/pacman_tracker/bend_anchor.py` is
standalone — nothing in the tracker imports it. Production is unchanged.

### The matcher

`match_curvature_event()` takes the active edge's arc-length + smoothed heading
profile and the bias-removed gyro yaw rate over the event window. Speed is
unknown, so it searches an affine `s(t) = s0 + v_bar·(t−t0)` (`v_bar ∈ [6,28]`
m/s) with `s0` over the **whole edge** — **no D / distance prior**. Score =
`−RMS(ψ_map(s(t)) − ψ_gyro(t)) − ½·|Δψ residual| + 1.5·corr(rate profiles)`. A
match is returned only if unique: the best `s0` must beat every `s0` more than
60 m away by ≥ 0.7.

### ev3 on edge 4326 — geometry + gyro only, no D prior, no hidden GPS

| | value |
|---|---|
| best `s0` / **`s_peak`** / `s_end` | 616 / **735** / 908 m |
| recovered `v_bar` | 17.5 m/s (the true cruise speed) |
| map Δψ / gyro Δψ | +49.6° / +48.2° |
| angle-shape RMS residual | **0.8°** |
| rate-profile correlation | 0.80 |
| uniqueness margin | **1.66** (best 0.82, 2nd-best >60 m away −0.83); **one** local maximum on the 1039 m edge |
| **truth `s` at `t_peak` (hidden GPS)** | **734 m → anchor error +1 m** |

### Intra-edge anchor measurement (diagnostic, not applied to production state)

| | value |
|---|---|
| `s_est` (tracker) at ev3 | ~89 m |
| `s_map_anchor` (bend) | 735 m |
| **`residual_s`** | **+646 m** — the missing along-edge distance |
| `σ_anchor` | ~40 m |
| active edge | 4326, **unchanged** — the anchor is a position on the already-committed edge, not a re-route |

### Classifier over every gyro event, both trips (`classify_event`, rules-first)

| truth ＼ class | junction | bend | reject |
|---|---|---|---|
| **junction** (8) | 5 | **0** | 3 |
| **bend** (1) | 0 | **1** | 0 |

- **ev3 → bend** (peak 0.10 rad/s, 10.7 s, fits edge geometry, no compatible junction successor).
- **Zero junctions misclassified as bends.** The five real sharp junction turns on 07-22 (peak 0.42–0.68 rad/s) all → junction; the bend matcher returns `None` for every one.
- ev1 07-26 (the other suspected curvature/merge) → **reject**, not bend — its 53° does not fit edge 729's polyline (`angle_resid` 29°).
- The 3 junction→reject are conservative (neither model fits the reconstructed active edge) — never a false anchor.

### TASK 9 — start-GPS → bend as a `k_s` anchor

| | value |
|---|---|
| `L_map` (start → bend, map geometry) | 4142 m |
| ∫ `v_spectral` dt (start → ev3) | 3835 m |
| **implied `k_s`** | **1.080** |
| truth distance start → bend (hidden GPS) | 4140 m |
| **true required `k_s`** | **1.080** (exact) |

This *is* a clean, odometer-independent high-speed absolute anchor (437 s span,
20 % ≥ 14 m/s, bend position from geometry not from D). But `k_s` 1.08 is the
**span average**; the 18–25 m/s cruise is mostly *after* ev3 and needs ~1.30, so
a `k_s` fixed here under-corrects the rest of the trip.

### TASK 8/11 — causal replay (anchor applied once at ev3 + settle, no future leakage)

| run | D/D_true | survival | topology |
|---|---|---|---|
| A baseline | 0.836 | 0.103 | — |
| **B bend position fix only (+646 m)** | **0.890** | 0.103 | unchanged |
| C bend `k_s` only (1.08 from t≈592) | 0.820 | 0.103 | unchanged |
| D position + `k_s` | 0.874 | 0.103 | unchanged |

The **position** anchor helps (+5.4 pp); the `k_s` from it does not (span
average too low for the post-ev3 cruise). Survival is unchanged because 07-26's
post-ev3 events are curvature too — there are no missed *junctions* left to
re-align, only distance, which is what improved.

### 07-22 (TASK 12)

Six gyro events, all six with `peak_rate` 0.42–0.68 rad/s. The matcher returns
`None` for every one (no edge-polyline bend fits a sharp compact turn); five
classify **junction**, one **reject** (t 913.7, a genuine junction the matcher
misses — a conservative miss, not a false anchor). **Zero safe bend anchors,
zero false anchors, no effect on 07-22** — in particular nothing fires near the
shallow fork, so it is untouched.

### Anti-circularity (TASK 10)

Route topology through ev3 was committed by the earlier sharp junction turns
(ev0 + straight-throughs), independent of D. The bend position comes from the
edge-4326 polyline shape vs the gyro shape, with **no D prior in the search**
(`test_no_distance_prior_is_used`), and it is unique (margin 1.66) — not a
"bad D pointed near bend X so we chose bend X" artefact. Only then is it used to
correct D. The forbidden loop (D → bend → D) is broken at the search: remove the
(non-existent) D prior and the same bend is still the unique answer.

## Phase 18 — bend anchors wired into `single_path`, and the bend's *local speed* as a high-speed `k_s` anchor

**A bend gives two GPS-free measurements from one event: an absolute along-edge
*position*, and — from the same time→arc-length fit — an absolute *local speed*.
The local speed is the high-speed calibration that was missing: at 07-26 ev3 the
bend fits `v_bar = 17.5 m/s` while the spectral model reads 14.1, so the *local*
`k_s` is 1.24 — exactly the correction the trip needs. Wired (gated OFF),
position + local-speed together take 07-26 `D/D_true` from 0.836 to ~0.95 with
route topology unchanged, and leave 07-22 (no bend) untouched.**

### Integration (all flags default `False` — production baseline unchanged)

| file | change |
|---|---|
| `config.py` | `single_path.bend_classification_enabled`, `bend_position_anchor_enabled`, `bend_local_speed_anchor_enabled` (+ shape/margin/regime tuning); `speed.bend_scale_enabled` (+ tuning) |
| `single_path.py` | `PhysicalTurn.is_bend`; `_classify_bends` (runs after `_ingest_turns`); `_junction_gap_deg`; `bend_anchors` list; `diagnostics()` exposes them |
| `speed.py` | `bend_scale` = **`k_high`**, a side-state separate from `k_s`/`k_a`; `_regime_weight(v_spectral)`; `bend_speed_anchor()`; `spectral_update` blends `k_low·(1−w) + k_high·w` |
| `tracker.py` | `_apply_bend_anchors()` — position via the shared `speed`+`offset_bias`+common-mode path (`k_a` protected), speed via `bend_speed_anchor` |

`_classify_bends`: a settled gyro event that is spread (`≥6 s`) and low-rate
(`peak ≤0.16 rad/s`), whose `d_event` lands on a committed edge (or ≤2 hops past
the frontier — the odometer lags most at a fast bend), whose polyline contains a
**unique** curvature match (`match_curvature_event`, no D prior), and for which
no reachable junction offers a compatible outgoing turn, is marked `is_bend` and
`consumed` — so it drives no junction, no straight-commit revision, no rollback
and no interval (**B11**).

### B1 — is the bend's local speed `v_bar` accurate? (07-26 ev3)

| | value |
|---|---|
| `v_bar` (shape fit) | **17.50 m/s** |
| `v_true` window mean / at peak (hidden GPS) | 17.66 / 17.83 → error **−0.16 / −0.33 m/s** |
| `σ_v_bar` (RMS-doubling half-width) | ~0.67 m/s |
| shape RMS / rate-corr / margin | 0.8° / 0.80 / 1.66 |

### B2 — the *local* high-speed `k_s`

| | value |
|---|---|
| `v_spectral` over the event (median) | 14.09 m/s |
| **`k_s_local = v_bar / v_spectral`** | **1.242 ± 0.258** |
| true local `k_s = v_true / v_spectral` (hidden GPS) | **1.254** |
| (start→bend *span-average* `k_s`, Phase 17) | 1.080 |

The local ratio lands squarely in the 1.2–1.35 the trip needs; the span-average
diluted it to 1.08 by mixing in the slow first half.

### B5 / B6 — is there an online saturation / OOD indicator? **No.**

Mahalanobis distance, standardized feature norm and ridge leverage of the
spectral feature windows, warm-up as the reference distribution:

| 07-26 `v_true` bin | spectral pred | `c_req` | Mahalanobis | feature-norm | leverage |
|---|---|---|---|---|---|
| 4–8 | 6.9 | 0.89 | 8.5 | 4.1 | 0.26 |
| 8–12 | 10.7 | 0.95 | 8.2 | 4.6 | 0.25 |
| 12–16 | 13.2 | 1.09 | 7.8 | 4.5 | 0.22 |
| 16–20 | 13.1 | **1.37** | 7.9 | 4.2 | 0.23 |
| 20–26 | 12.5 | **1.67** | 8.6 | 4.3 | 0.27 |

Every OOD score is **flat across speed** — the 16–25 m/s cruise that needs
`c_req ≈ 1.5` looks statistically identical to the 8 m/s driving that needs 1.0
(which is *why* the model saturates: the features stop separating speeds). A
detector (`maha > warm p95 · 1.5`) covers 2 % of the high-speed windows on 07-26
and its flagged `c_req` (1.04) equals its unflagged (1.12). **The only
online-available regime signal is `v_spectral` itself** — and it saturates.

### B7 — two-regime `k_s`, gated on the one signal there is

`spectral_update` now blends `k_low` (the existing `spectral_scale`, ~1 from
mid-speed lateral anchors) and `k_high` (`bend_scale`, from bend local-speed
anchors) by `w = clip((v_spectral − 9)/(14 − 9), 0, 1)`. `k_high` starts at 1.0
with σ 0.30 and **never shrinks without a bend anchor** — a trip with no bend
(07-22) keeps the wide prior and the blend stays at `k_low`. A bend anchor
never moves `k_low`, `k_s` or `k_a`. **B8** (a two-regime interval Jacobian
`∂L/∂k_low = ∫(1−w)v dt`, `∂L/∂k_high = ∫w v dt`) is *not* wired: 07-26 has no
usable interval and 07-22's single interval is a low-speed span — nothing to
calibrate `k_high` from an interval on these trips. Left as future work.

### B9 — causal ablations (monkeypatch `k` on future spectral; position via the shared machinery)

| 07-26 | D/D_true | survival | topology |
|---|---|---|---|
| A baseline | 0.836 | 0.103 | — |
| B bend position only | 0.889 | 0.103 | unchanged |
| C bend local-`k` (1.24) on all future spectral | 0.903 | 0.103 | unchanged |
| **D position + local-`k`** | **0.956** | **0.151** | unchanged |
| E position + `v_spectral`-gated `k` | 0.930 | 0.140 | unchanged |

Naive C (apply `k_local` to *all* spectral after ev3) works because on 07-26
every second after the bend is high-speed cruise; the `v_spectral` gate (E)
under-corrects the moderate-speed stretches and scores below D. On a trip that
slowed to city speed after a bend the gate would be the safer choice — hence it
is what the wired path uses.

### B10 — 07-22 safety

Zero bend anchors accepted (all six gyro events are sharp junction turns, peak
0.42–0.68 rad/s; the matcher returns `None` for each). `D/D_true` **0.950
unchanged**, survival unchanged, the shallow fork untouched. `k_high` stays
`1.0 ± 0.31` — never collapsed by a low-speed lateral anchor
(`test_lateral_anchor_cannot_collapse_k_high_uncertainty`).

### B9 (wired) — the real `single_path` path, flags on

| 07-26 | D/D_true | survival | evdriven | bend | `k_high` | 07-22 D/D_true |
|---|---|---|---|---|---|---|
| A baseline (flags off) | 0.836 | 0.103 | 2 | 0 | — | 0.950 |
| B classification only | 0.836 | 0.103 | **1** | 1 | — | 0.950 |
| C + position anchor | 0.889 | 0.103 | 1 | 1 | — | 0.950 |
| **D + `k_high` (bend speed)** | **0.927** | 0.101 | 1 | 1 | 1.19 ± 0.15 | **0.950** |
| E `k_high` only, no position | 0.869 | 0.101 | 1 | 1 | 1.19 | 0.950 |

`B` shows the classification working end to end: ev3 is bound to edge 4326 at
`s ≈ 735` (residual +646 m), consumed, and decision 12 is **no longer
event-driven** (`evdriven` 2 → 1) — it commits by geometry instead of matching
the curvature event to a wrong node, and no fake interval is produced.

`D` recovers 9.1 points of `D/D_true` (0.836 → 0.927) with route topology
unchanged. The wired `k_high` (1.19) is a touch below the true local `k_s`
(1.25) because the production denominator is the median `v_spectral` over the
event window and the regime blend `w(v_spectral)` is < 1 for the moderate-speed
stretches — so the wired gain is smaller than the flat-`k` monkeypatch's
`0.956`. That conservatism is the price of using the only online regime signal
there is.

**07-22 is byte-for-byte unchanged at 0.950 through every variant** — no bend
anchor is accepted (all six gyro events are sharp junction turns), `k_high`
stays `1.0 ± 0.31`, the shallow fork is untouched.

## Phase 19 — can we "hold" on the accelerometer after an anchor and starve the saturated spectral source?

**Hypothesis: after a trusted absolute-speed anchor, lean on `v_anchor + ∫a_long dt`
and suppress spectral for a while so it cannot drag `v` back to its saturated
value. Verdict: refuted for these trips.** The spectral drag is real but slow;
accel-only propagation is only useful for ~15–25 s after a *gentle* anchor and
~3 s after a stop; the anchor gaps on 07-26 are 75–130 s; and every suppression
schedule trades a steady small undershoot for a large mid-trip excursion that
desynchronises junctions. Nothing wired — diagnostics only, production defaults
unchanged.

### The spectral drag, measured (07-26, after ev3, `v_bend` 17.5 ≈ truth 17.8)

Per spectral update in the 60 s after ev3: gain `K_v ≈ 0.01`, `Δv ≈ −0.03 m/s`.
Tiny individually — but at ~2 updates/s they are relentless, and `v_true` really
is falling, so the fused estimate is pulled **below** truth and stays there:

| after ev3 | `v_true` | `v_fused` \|err\| | accel-only \|err\| | `D` err fused | `D` err accel-only |
|---|---|---|---|---|---|
| +5 s | 18.2 | 3.5 | **0.6** | −16 m | **+2 m** |
| +10 s | 13.4 | 5.3 | **0.8** | −41 m | **+4 m** |
| +20 s | 14.0 | 7.5 | **0.2** | −113 m | **+12 m** |
| +30 s | 16.1 | 0.4 | 2.2 | −130 m | +20 m |
| +45 s | 13.9 | 2.4 | 7.7 | −135 m | +89 m |
| +90 s | 16.9 | 0.1 | 7.9 | −164 m | +425 m |

Accel-only from the bend anchor is **dramatically better for ~20–30 s** (D error
+12 m vs the fused −113 m at +20 s), then runs away upward: an unmodelled
~+0.15 m/s² longitudinal residual accumulates with no stop to re-observe `b_a`.

### Accel-only useful horizon by anchor type (TASK 12)

Median seconds until accel-only error (propagated with the tracker's own `b_a(t)`)
first exceeds a threshold:

| anchor | \|v\|>1 | >2 | >3 | \|D\|>25 m | >50 m | >100 m |
|---|---|---|---|---|---|---|
| **ZUPT** (07-26 n=8) | 2.5 | 4.2 | 5.4 | 9.5 | 13 | 18 |
| **clean lateral** (07-26 n=3) | 18 | 21 | 25 | 26 | 32 | 48 |
| **bend** at `t_peak` (07-26) | ~10 | ~25 | — | ~15 | ~25 | ~45 |
| clean lateral (07-22 n=12) | 3.8 | 9 | 11 | 15 | 22 | 44 |
| ZUPT (07-22 n=13) | 4.7 | 7.2 | 9.1 | 13 | 19 | 35 |

- **After a stop, accel-only is useless within ~3 s** — the stop→go acceleration
  is exactly where `a_long` attenuates.
- **After a clean lateral or bend** (moderate speed, gentle dynamics) it holds
  ~15–25 s.

### Anchor cadence

Median gap between *any* absolute anchor: **07-26 = 75 s (p90 132 s)**,
**07-22 = 36 s (p90 66 s)**. 07-26's cruise stretches run 1–2 minutes with no
anchor — far past what accel-only can bridge.

### Suppression ablation (TASK 5/11) — all net-negative

| 07-26 (bend anchors on) | D/D_true | survival | d60 | d240 |
|---|---|---|---|---|
| baseline (Phase 18) | **0.927** | 0.101 | 107 | **332** |
| spectral OFF 10 s post-anchor | 0.932 | 0.133 | 182 | 693 |
| spectral OFF 30 s | 0.815 | 0.087 | 180 | 666 |
| spectral OFF 60 s | 0.936 | 0.149 | 188 | 693 |
| exponential decay (F0 25, τ 20 s) | 0.936 | 0.124 | 145 | 510 |
| uncertainty-gated (`σ_v` < 0.6 `σ_spectral`) | 0.917 | 0.094 | 114 | 360 |

07-22: **every** schedule ≤ the 0.950 baseline. The endpoint `D/D_true`
occasionally nudges up (0.927 → 0.936) but the intermediate errors 2× worse
(`d240` 332 → 510–693) and `evdriven` shifts (junction desync). The hold trades
a consistent small undershoot for a large excursion that lands near the right
total by luck. Uncertainty-gated is the least harmful and still loses.

Suppressing spectral **only after the one bend anchor** (leaving ZUPT/lateral
alone) is the same story in miniature: 07-26 endpoint 0.927 → 0.934, but
`d240` 332 → 391–693 depending on the schedule; 07-22 completely inert (no bend
fires). A marginal, noisy endpoint tick bought with a doubled mid-trip error.

### Why the lever is `k_high`, not suppression

The spectral drag is `~0.03 m/s` per update — slow enough that **correcting the
scale** (`k_high` from the bend's local speed, Phase 18) neutralises it: with
`k_high ≈ 1.19` the corrected spectral reads ~15.5 instead of ~13, so the
innovation against a true 16–18 is small and the drag mostly disappears.
Suppression instead removes the only continuous speed source during the exact
75–130 s cruise gaps where it is the only thing holding `v` together, and after
the frequent stops (where accel is worst) it is actively harmful.

### Answers (Phase 19)

1. **Does spectral fusion drag a correct post-anchor speed toward the saturated
   value?** Yes — slowly: `K_v ≈ 0.01`, `Δv ≈ −0.03 m/s` per update, ≈ −3 m/s
   over 60 s; fused `D` error after ev3 reaches −165 m.
2. **After bend `v ≈ 17.5`, how long is accel propagation accurate?** `D` error
   < 25 m for ~15–25 s, then it diverges (`b_a` not stationary on the cruise).
3. **After ZUPT?** ~3 s to 1 m/s, ~13 s to 25 m `D` error — the stop→go
   acceleration attenuates and there is nothing to reconstruct it from.
4. **After a clean lateral anchor?** ~18 s to 1 m/s, ~25 s to 25 m `D` error.
5. **Which suppression strategy works best?** **None.** All are net-negative on
   both trips; uncertainty-gated is least harmful (07-26 0.917 vs 0.927).
6. **07-26 `D/D_true` with anchor-hold?** Endpoint drifts to ~0.936 but
   intermediate error doubles; the honest figure stays **0.927** (Phase 18
   bend, no hold).
7. **Does it help without harming 07-22?** No — 07-22 degrades under every
   schedule.
8. **Practical interval between speed resets?** 07-26 **75 s** (p90 132 s),
   07-22 **36 s** (p90 66 s).

**The hypothesis is partly true but not actionable:** for ~15–25 s after a
clean lateral or bend anchor, accel-only *is* better than the spectral-dragged
fusion — but that window is far shorter than 07-26's anchor gaps, collapses to
~3 s after a stop, and suppressing spectral during it desynchronises junctions.
Phase 18's `k_high` captures the benefit (0.836 → 0.927) without the cost.

## Phase 20 — should the measured spectral calibration persist, and what invalidates it?

**Persistence: yes — and the wired `k_high` already does it.** It is a
calibration side-state, updated only by absolute-speed anchors; ordinary
spectral observations, ZUPT and elapsed time never move its value (only its
`σ` grows, slowly, via the random walk). **Physical invalidation: no.** Every
rule tested — reset `k_high` at a ZUPT, weaken it at a ZUPT, weaken it once
`∫|a_long| dt` passes a threshold — *degrades* 07-26 and does nothing for
07-22. On a Petersburg highway a stop at a light does **not** end the
high-speed spectral regime; the car resumes the same cruise, and there is no
post-stop anchor to re-establish `k`.

### Is `k_required(t) = v_true / v_spectral` stable within cruise episodes? (TASK 1, hidden GPS for evaluation only)

07-26 split into 13 continuous cruise episodes (≥ 8 s, bounded by stop / strong
accel / brake), windowed (3 s median) `k_required`:

| episode `v_true` | median `k_required` | within-episode σ | drift / 60 s |
|---|---|---|---|
| 13–16 m/s (×5) | 1.06 – 1.15 | 0.02 – 0.12 | < ±1.2 |
| 17–19 m/s (×4) | 1.41 – 1.60 | 0.07 – 0.16 | < ±1.4 |
| 19–22 m/s (×4) | 1.49 – 1.74 | 0.11 – 0.41 | −0.7 … +3.5 |

- **Within one continuous constant-speed cruise, `k_required` is roughly
  stable** (windowed σ ~0.05–0.15, small drift). Persisting it *there* is
  justified.
- **Across episodes it is not a constant — it tracks the episode's speed**
  (0.97 at 13–14 m/s → 1.74 at 21 m/s). This is the spectral saturation itself:
  the faster the car, the more the model under-reads, the bigger the correction
  needed. Median `k_required` across the 13 episodes: **1.29, σ 0.24, range
  0.97–1.74.**
- **Pointwise `k_required` is useless** — `v_spectral` carries ±3 m/s of
  high-frequency noise, so the ratio swings 0.9–1.5 within 5 s even at constant
  true speed. Only a 20–60 s window reveals the scale, by which time the speed
  may have changed.

### Trace after ev3 (TASK 10)

`v_bend` 17.5 was measured at ~17.8 m/s true. Over the next ~100 s the car
cruises 13–18 m/s (windowed `k_required` ~1.1–1.3 — the ev3 value is *about*
right), then stops at t≈697, then accelerates back to **20–22 m/s** where
`k_required` rises to **1.4–1.74** — which the ev3 anchor (1.24) *under*-corrects
and cannot fix. `∫|a_long| dt` reaches +63 m/s over the first 90 s while true
speed barely changes (~+0.7 m/s² unmodelled drift — same finding as Phase 19),
so it is not a usable "the regime changed" trigger.

### Invalidation ablation (TASK 4/5/11)

| 07-26 | D/D_true | survival | d240 | `k_high` end |
|---|---|---|---|---|
| **persist forever (wired)** | **0.927** | 0.101 | **332** | 1.19 |
| reset `k_high` at ZUPT | 0.894 | 0.101 | 418 | 1.00 |
| weaken `k_high` at ZUPT | 0.894 | 0.101 | 418 | 1.00 |
| weaken on `∫|a_long|` > 30 | 0.907 | 0.101 | 375 | 1.09 |
| weaken on `∫|a_long|` > 60 | 0.910 | 0.101 | 375 | 1.09 |
| weaken on `∫|a_long|` > 120 (never fires) | 0.927 | 0.101 | 332 | 1.19 |

07-22: **0.950 unchanged** through every variant (`k_high` never leaves 1.0,
no bend anchor). Topology unchanged everywhere (`evdriven` 1, wrong routes 0).

### Can physical dynamics detect when `k` went stale? (TASK 13)

**No.** The event that makes the ev3 `k` stale is the car going from ~17 to
~21 m/s cruise — a *speed*-band change. `v_spectral` saturates (17 and 21 m/s
both read ~13, so it cannot see it); `∫a_long` drifts; ZUPT fires at the stop
but the stop is not the regime boundary (the regime is the same either side).
There is no online signal that resolves which high-speed band the car is in.

### Answers (Phase 20)

1. **Is `k_required` stable within continuous cruise episodes?** Within one
   constant-speed episode, yes (windowed σ ~0.05–0.15). Across episodes, no —
   it is speed-dependent (0.97 → 1.74).
2. **How long is the measured `k ≈ 1.24` valid after ev3?** For cruise near its
   measurement speed (~17–18 m/s) — roughly the ~100 s to the next stop. It is
   not "wrong" there; it is simply too low for the 20+ m/s stretches later.
3. **What first makes it stale?** A speed-band change (≈17 → ≈21 m/s cruise) —
   **not** the stop, which is incidental (the car returns to high speed).
4. **Can that be detected online without GPS?** No — `v_spectral` saturates,
   `∫a_long` drifts, ZUPT is not the boundary.
5. **Best invalidation rule?** **None.** Persist-forever is optimal; every
   physical-invalidation rule loses.
6. **Does stateful `k` beat 0.927?** No — 0.927 *is* the persistent-`k` result
   (the wired `k_high`). Nothing added here improves it.
7. **How close to the flat-`k` ceiling 0.956?** The gap is not a persistence
   problem. 0.956 is a flat 1.24 applied everywhere; the true `k` runs
   0.97–1.74, so even that "ceiling" is wrong for the fast stretches. A single
   scalar — however persisted — cannot reach the speed-varying truth.
8. **Does 07-22 stay ≈ 0.950?** Yes, exactly, every variant.
9. **Topology unchanged?** Yes — `evdriven` 1, genuine wrong routes 0 in all
   variants.
10. **Production rule to eventually enable:** exactly the wired one — `k_high`
    a persistent calibration side-state, moved only by absolute-speed anchors,
    `σ` grows via a slow random walk but the value never auto-resets, blended
    into the spectral update by the `v_spectral` regime weight with its own
    variance in `R_corr`. **Do not add ZUPT- or accel-based invalidation.**
    The only valid invalidation is a *new, inconsistent* absolute-speed anchor,
    which `bend_speed_anchor`'s innovation gate already rejects or absorbs. All
    flags stay `False` by default pending trips with more than one bend.

**Verdict: persistence confirmed, physical invalidation refuted.** The wired
machinery is already the right stateful design; the remaining 0.927 → 0.956 gap
is the speed-dependence of `k_required`, which no single scalar and no online
regime signal can close on these trips.

## Phase 21 — `k = f(v_pred)` instead of `f(v_spectral)`: refuted (unstable)

**Idea:** index the spectral correction by the EKF's *predicted* speed (before
consuming the current spectral sample), so continuity keeps the tracker on the
right branch when `v_spectral` (saturated) cannot tell 17 from 21 m/s. **Result:
the diagnosis is right but the mechanism is an unstable positive-feedback
loop.** Diagnostic only, nothing wired.

### `k_required(v_true) = v_true / v_spectral` (TASK 2, hidden GPS for evaluation)

| `v_true` | 07-22 median | 07-26 median |
|---|---|---|
| 6–9 m/s | 1.16 | 0.86 |
| 9–12 | 1.39 | 0.99 |
| 12–14 | **1.64** | **1.01** |
| 14–16 | **1.91** | 1.13 |
| 16–18 | — | 1.27 |
| 18–20 | — | 1.52 |
| 20–22 | — | 1.62 |
| 22+ | — | 2.17 |

**The curve is not transferable.** At 12–14 m/s 07-22 needs `k = 1.64`, 07-26
needs `1.01` — each trip's spectral model is fitted on its own 145 s warm-up, so
its error-vs-speed shape is trip-specific. A hardcoded universal `f(v)` is dead
(cross-trip oracle: 07-22's curve on 07-26 → `D/D_true` **1.20**, a 20 %
over-correction).

### Oracle ceiling and the `v_pred` version (TASK 3 / 15C)

| 07-26 run | D/D_true | survival | v-MAE | v-MAE 18–22 m/s |
|---|---|---|---|---|
| baseline (wired `k_high`) | 0.927 | 0.10 | — | — |
| **oracle `k(v_true)`** (pool curve) | **0.99** | **0.57** | 2.4 | 2.8 |
| oracle `k(v_true)` (07-26 curve) | 0.96 | 0.45 | 2.4 | 2.9 |
| **`k(v_pred)`** (same good curve, evaluated at `v_pred`) | **1.08** | 0.11 | 4.4 | **6.7** |

A *perfect* speed-indexed correction has huge headroom (0.93 → 0.99, survival
0.10 → 0.57). Evaluated at `v_pred` it **over-corrects to +8 %** and does not
improve survival: `v_pred` is not near `v_true` (it is dragged down by the
saturated spectral and the IMU drift), so `k(v_pred)` picks the wrong point on
a monotone-rising curve, and — worse — `k(v_pred)·v_spectral` has **slope > 1**
in `v_pred` over the 16–22 m/s band.

### Stability (TASK 10 / 20 / 21)

Closed-loop gain `dv_new/dv_pred = 1 + K·(k'(v)·v_spectral − 1)` with the real
07-26 curve and `K ≈ 0.15`: **1.09–1.10 in the 16–22 m/s band** (> 1 =
amplifying). Synthetic saturated-cruise cases: `k(v_pred)` **runs away** to
+14 m/s speed error (vs −4 for flat `k`). Basin of attraction: initial speed
error of −3 … +3 m/s all converge to the *same* wrong fixed point (+14 m/s) —
the loop has no restoring force toward truth, only toward its own attractor.
07-22 (control): `k(v_pred)` degrades `D/D_true` 0.950 → 0.92–0.94.

### Answers (Phase 21)

1. **Already doing `k = f(v_state)`?** No — `_regime_weight` is indexed by raw
   `v_spectral`; `k` is a blend of two scalars, not `f(v)`. Genuinely new.
2. **The `k_required(v_true)` curve?** Monotone rising, trip-specific (table
   above); 07-26 0.86 → 2.17, 07-22 much steeper.
3. **Oracle ceiling?** `D/D_true` **0.99**, survival **0.57** on 07-26 — large.
4. **Does `v_pred` resolve 17-vs-21?** No — `v_pred` is dragged low by the
   saturated spectral, so it tracks neither branch; v-MAE in the 18–22 band is
   6.7 m/s.
5. **Feedback stable?** No — `dz/dv > 1` in the 16–22 band; synthetic runaway.
6. **Basin of attraction?** Zero — every initial error converges to the same
   wrong fixed point.
7. **Beats 0.927?** No — over-corrects to 1.08.
8. **Approaches 0.956?** No.
9. **07-22?** Degrades to 0.92–0.94.
10. **Universal `k(v)` transferable?** No — 12–14 m/s needs 1.64 (07-22) vs
    1.01 (07-26).
11. **Curve from sparse anchors?** The high-speed part is unobservable — the
    only anchors above 16 m/s are the single 07-26 bend; lateral anchors sit at
    6–16 m/s where 07-26's `k ≈ 1.0`.
12. **Justified production formulation?** None from this direction. The wired
    `k_high` (Phase 18) remains the best.

**Verdict: refuted.** Speed-indexed correction has real headroom but requires
`v_true`; substituting `v_pred` gives an unstable loop with no basin of
attraction, and the curve is not transferable between trips.

## Phase 22 — trusted map-event intervals (`L_map / ∫v_spectral`)

**Idea:** recognise two independent physical map locations, take
`k_interval = L_map / ∫v_spectral dt` over the span. **Verdict: sound, largely
already built — but on these trips there are not enough trusted events, and the
one 07-26 interval spans slow→fast so its `k` is a dilute average that hurts
when applied.** Nothing new wired.

### Trusted map events (bend classification on)

| | 07-26 | 07-22 |
|---|---|---|
| trusted events | **2** — junction ev0 (t165, `s` 39, +51°), **bend ev3** (t582, `s` 4142, margin 1.63) | 4 — all junctions (t147/241/647/680) |
| event-to-event intervals | **1** | 3 |
| safe-interval distance coverage | **43 %** | 82 % |
| safe-interval time coverage | 40 % | 50 % |
| gap between events | median 417 s, max 618 s | median 93 s, max 520 s |

**07-26 has one trusted map event per ~5 km** — the task's own bar for "too
sparse."

### The 07-26 interval (junction ev0 → bend ev3)

| | value |
|---|---|
| `L_map` (committed route) | 4102 m |
| **`L_true`** (HMM map-match) | **4104 m → err −2 m** |
| Δt | 417 s |
| `v_mean` | 9.8 m/s (true 9.8) |
| ∫`v_spectral` | 3775 m |
| **`k_interval`** | **1.087** (`k_true` 1.087) |

The bend endpoint *works* — `L_map` matches truth to 2 m, and the old circular
3367 m interval (Phase 16) is gone (ev3 is consumed as a bend, no junction
interval forms). **But `k_interval` = 1.09 is a span average** — the interval
runs from ev0 at ~13 m/s through the slow merge to ev3 at ~17.8 m/s, so it does
not isolate the 17–21 m/s saturation. There is no second trusted event in the
fast cruise to pair with ev3.

### Applying it causally

| 07-26 | D/D_true | survival | d240 |
|---|---|---|---|
| wired (bend position + `k_high`) | **0.927** | 0.101 | 332 |
| + junction→bend interval `k_s` | **0.819** | 0.100 | 438 |

Feeding the span-average `k` into `spectral_scale_interval` (which moves
`k_low`) over-corrects 07-26's low-speed portions and double-counts distance
against the bend position anchor. **Worse, not better.** 07-22 (control):
0.950 → 0.971 endpoint but survival 0.157 → 0.138 and `d240` 143 → 168 — its
three intervals imply inconsistent `k` (1.36, 1.14, 0.94; the last is on a
polyline the map gets 10 % wrong).

### Oracle-event ceiling (TASK 16)

If event recognition were perfect *and* events fell at the cruise boundaries,
an interval over the fast section would imply:

| oracle window | `v_mean_true` | `k_interval` |
|---|---|---|
| t740–990 (fast cruise) | 12.9 m/s | **1.38** |
| t854–990 | 12.5 | 1.48 |
| t955–1050 | 11.6 | 1.33 |

So a *well-placed* interval would give `k ≈ 1.38–1.48` — better than the wired
`k_high` 1.19. But no such events exist on 07-26 (only ev0 and ev3), and even
these oracle windows average ~13 m/s because 07-26's "fast" cruise keeps
dipping. Event *sparsity*, not event *recognition*, is the limit.

### Anti-circularity (TASK 5)

The bend **shape** match is D-independent (proven, tested). But
`_classify_bends` picks *which* edge to test from `ρ = d_event − offset_bias`
(the odometer). Injecting an extra ±1–3 m/s of D drift changes the endpoint
set — at −1 m/s ev3 is matched to a junction instead of a bend. **At the real
odometer lag ev3 classifies correctly** (the 2-hop frontier expansion added in
Phase 18 covers it), but the bend is a valid interval endpoint only within the
real lag range, not under arbitrary D error. Documented, not claimed as fully
D-independent.

### Answers (Phase 22)

1. **Safely identifiable events?** 07-26: junction ev0 + bend ev3 (2). 07-22:
   4 junctions.
2. **Independent intervals?** 07-26: **1**. 07-22: 3.
3. **Trip coverage?** 07-26: 43 % distance / 40 % time. 07-22: 82 % / 50 %.
4. **Map-length accuracy vs truth?** 07-26 interval: **−2 m** on 4104. 07-22:
   +14 / +54 / −20 m (the last on a 10 %-wrong polyline).
5. **Implied mean speeds?** 07-26: 9.8 m/s (span avg). 07-22: 8.6 / 3.7 / 5.7.
6. **Implied spectral `k`?** 07-26: **1.087**. 07-22: 1.36 / 1.14 / 0.94.
7. **Do they expose the 17–21 m/s saturation?** **No** — the one 07-26 interval
   is a slow→fast span average.
8. **Beats 0.927?** No — **0.819** (worse).
9. **Improves high-speed cruise?** No.
10. **07-22 safe?** Marginal — endpoint 0.950 → 0.971 but survival and `d240`
    regress; the inconsistent `k` values are a liability.
11. **Oracle-event ceiling?** `k ≈ 1.38–1.48` *if* events bracketed the cruise
    — but they don't exist.
12. **Frequent enough for Petersburg straight driving?** **No.** One trusted
    map event per ~5 km on 07-26, with 400–600 s gaps. Too sparse to be a
    primary anchor source for long-straight cruise.

**Verdict: the interval mechanism is sound and mostly already wired (interval
machinery + bend endpoints), but map events on these trips are too sparse, and
the single 07-26 interval spans slow→fast so its `k` (1.09) is a dilute average
that degrades the trip when applied. Nothing new wired.**

## Phase 23 — saturation-aware velocity fusion

Three ideas were tested. **Two refuted, one wired: a *censored* spectral update
once saturation is proven takes 07-26 `D/D_true` 0.927 → 0.982 with 07-22 exactly
unchanged.**

### Priority 1 — plateau crossing → short accel bridge: **refuted**

The reframing "recover only the last `Δv ≈ 8 m/s` during the acceleration" assumes
the accelerometer can measure that `Δv`. On this recorder it cannot — during
*sustained* acceleration the AHRS leans and projects most of the forward specific
force onto the (removed) gravity axis:

| 07-26 ramp | true `Δv` | `∫(a_long − b_a) dt` | implied `g_a` |
|---|---|---|---|
| post-ZUPT t712–752 (0 → 21.3) | 21.3 m/s | **3.26 m/s** | **6.5** |
| t789–798 (12.6 → 21.1) | 7.7 | **0.47 m/s** | **16.6** |
| t370–380 (14.7 → 22.1) | 7.2 | 6.9 m/s | 1.0 |

`g_a` is not a constant — 1 to 17 across three ramps on one trip. A pre-saturation
window (9 → 13 m/s) gives `g ≈ 3`; the harder part of the same ramp needs `g ≈ 7`.
Bridging from `v_cross` with any `g` (causal, oracle, or 1.30) reaches ~15 m/s
when truth is 21 — 6 m/s short, sometimes worse than the fused filter.
`v_spectral` is **fully frozen at ~13–15 across the whole ramp** (reads 13.4 at
true 9, 14.4 at true 16, 13.7 at true 21).

### Priority 3 — rolling curvature-speed matcher: **refuted**

The roads are geometrically straight. 07-26 cruise (t560–1050): committed edges
are Выборгское шоссе with total heading variation ≤ 17° per edge and `max|κ|`
≤ 0.005 rad/m. A rolling matcher (12 s window, 4 s hop) over 490 s finds **4
hits, all on edge 4326 at t578–590 — the ev3 bend itself.** 07-22 city grid
(Чкаловский проспект / Пионерская улица), 570 s: **zero hits.** "One bend per
5 km" is the actual geometry, not a detector artefact.

### Priority 2 — censored spectral update: **wired (gated OFF)**

Once a bend speed anchor proves the source under-reads at high speed (`v_bend`
exceeds the concurrent `v_spectral` by ≥ `spectral_saturation_margin_ms`), the
tracker records `spectral_saturation_known = True` and the plateau. From then on, with
`spectral_censor_enabled`, a spectral reading that

- sits within `spectral_censor_band_ms` of the plateau, **and**
- would pull `v` *down* by more than `spectral_censor_deadband_ms`

has its measurement variance multiplied by `spectral_censor_factor` (default 15).
A reading above the band, or one that lifts `v`, is untouched. The saturated
model reads ~the same 13 m/s whether the car is at 14 or 22, so a downward pull
there is genuinely uninformative; its noisy upper excursions (Phase 0: `v_spectral`
13–14 → `v_true` p90 ≈ 20) still ratchet `v` up.

| 07-26 | D/D_true | survival | evdriven | d120 | censored updates |
|---|---|---|---|---|---|
| bend position + `k_high` (Phase 18) | 0.927 | 0.101 | 1 | 332 | — |
| **+ censored, factor 8** | 0.971 | 0.142 | 1 | 293 | — |
| **+ censored, factor 15 (default)** | **0.982** | 0.148 | 1 | **293** | 3222 |
| + censored, factor 25 | 0.986 | 0.150 | 1 | 321 | — |
| + censored, factor 50 | 0.988 | 0.153 | 1 | 367 | — |

Higher factor keeps lifting the endpoint but grows the mid-trip error past ~15;
band (1–8) and deadband (0–1) barely matter. **07-22: 0.950 / survival 0.157 /
topology unchanged through every factor** — no bend anchor fires, so
`spectral_saturation_known` stays `False` and the path is completely inert
(`censored_updates = 0`). `evdriven` and rollbacks are unchanged everywhere.

The speed estimate is *noisier* under censoring (v-MAE 2.8 → 3.2: it now swings
13 → 25 → 8 with the spectral upper tail instead of sitting stuck at 13), but the
mean is right, so distance integrates correctly — v now actually reaches the
20–25 m/s band (t880 `v_est` 25.0, truth 22.3) instead of being pinned at 13.

This is the closest any GPS-free mechanism has come to the oracle `k(v_true)`
ceiling (`D/D_true` 0.99, Phase 21). It does **not** try to invert the saturated
signal — it stops the saturated signal from asserting a speed it does not have.

*(Phase 24 replaces the soft "×factor" form below with a `z = min(v, p)`
constraint model and decouples the trigger from `k_high`.)*

## Phase 24 — the saturated source is a *constraint* on `v`, not a measurement

Once `spectral_saturation_known` (permanent — a bend proved the model has a
plateau), a reading **near the plateau that would pull `v` down** is
`plateau + noise`, not a measurement, so its measurement variance is inflated by
`spectral_censor_factor` (15). A reading **above** the plateau band is the
source's right-skewed noise tail — which *does* track true speed (Phase 0:
`v_spectral` 13–14 → `v_true` p90 ≈ 20) — so it updates normally and lifts `v`.
The **saturation trigger is decoupled from `k_high`** (moved to
`_apply_bend_anchors`, needs only `bend_local_speed_anchor_enabled`): a bend can
prove the plateau without also being used to estimate a scale.

### 07-26 (wired)

| variant | D/D_true | survival | v-MAE | **v-bias** | p95 | v-max | max\|D err\| |
|---|---|---|---|---|---|---|---|
| bend position only | 0.889 | 0.103 | 2.97 | **−1.77** | 8.3 | 13.6 | 1318 |
| + `k_high` (Phase 18) | 0.927 | 0.101 | 2.80 | −1.33 | 7.3 | 13.6 | 332¹ |
| **+ censor, NO `k_high`** | **0.950** | 0.147 | 3.18 | **−0.64** | 9.5 | 13.8 | **729** |
| **+ `k_high` + censor** | **0.983** | 0.148 | 3.16 | **−0.17** | 9.3 | 13.6 | **704** |

¹ `d120`; the others are `max|D_est − D_true|` over the whole outage.

**07-22: 0.950 / survival 0.157 / topology unchanged through every variant** —
no bend fires, `spectral_saturation_known` stays `False`, the path is inert
(`censored_updates = 0`).

### What this says about the architecture

- **Censoring alone (no scale coefficient) does most of the work: 0.889 →
  0.950, speed bias −1.8 → −0.6, worst mid-trip `D` error 1318 m → 729 m.**
  That is the clean signal that the fix is physically right: the systematic
  undershoot was the illegitimate downward drag, not a missing gain.
- **`k_high` is the secondary polish: 0.950 → 0.983, bias → −0.17.** The
  project's central lever has moved from "estimate the right scale `k`" to
  "**once the plateau is known, stop reading it as a speed**". The bend is used
  first as *evidence of a plateau*, and only optionally as a scale sample.
- v-max stays at 13.6 (the down-weight, unlike an outright skip, keeps the
  filter well-behaved) and the worst intermediate `D` error nearly halves —
  the endpoint gain is not error cancellation.

**One line for the whole project: after saturation, spectral is a bound on `v`,
not a measurement of `v`.**

## Phase 25 — the strict `z = min(v, plateau)` form: refuted

**Idea:** split the state into `spectral_saturation_known` (permanent, the model
*has* a plateau) and "currently above the plateau" (per-update), and apply the
literal `z = min(v, p)` model — a reading in the plateau band only *bounds* `v`
from below by `k·p`: pull `v` up to that bound if it is under it, do nothing if
it already satisfies it, and — the strict part — **never trust a high reading in
the band as `v = reading`** (it is `p + noise`). **Result: worse.**

| 07-26 (wired) | D/D_true | v-bias | v-MAE | v-p95 | v-max | max\|D err\| |
|---|---|---|---|---|---|---|
| bend position only | 0.889 | −1.77 | 2.97 | 8.3 | 13.6 | 1318 |
| **Phase 24 down-censor + `k_high`** | **0.983** | −0.17 | 3.16 | 9.3 | 13.6 | 704 |
| Phase 24 down-censor, no `k_high` | 0.950 | −0.64 | 3.18 | 9.5 | 13.8 | 729 |
| Phase 25 strict, no `k_high` | 0.891 | −1.67 | 2.99 | 8.3 | 13.6 | 1293 |
| Phase 25 strict + `k_high` | 0.921 | −1.29 | 2.82 | 7.3 | 13.6 | 979 |

The strict form **caps `v` at `k·plateau`** (≈14 without `k_high`, ≈17 with),
because it forbids the one thing that was working: the source's noise is
**right-skewed**, and its upper tail *does* track true speed (Phase 0:
`v_spectral` 13–14 → `v_true` p90 ≈ 20). Phase 24's soft censor keeps trusting
those upward excursions (readings above `plateau + band`); the strict model
throws them away and the 20–25 m/s cruise is left ~6 points short.

So `z = min(v, p)` is the right *intuition* about the downward pull but the wrong
model for the *upward* information — the sensor is closer to
`z ≈ min(v, p) + right_skewed_noise(v)`, and the skew is usable.

**Kept from Phase 25:** the rename `spectral_saturated → spectral_saturation_known`
(it is a property of the trip's model, not the car's current speed), and the
saturation trigger decoupled from `k_high` (a bend proves the plateau whether or
not it also feeds a scale — moved to `_apply_bend_anchors`, needs only
`bend_local_speed_anchor_enabled`).

Config: `speed.spectral_censor_enabled` (+ `_factor` 15 / `_band_ms` 1.5 /
`_deadband_ms` 0.5 / `spectral_saturation_margin_ms` 1.0), default `False`.
Tests: `test_bend_integration.py` (19 total, 7 for censoring). Full suite:
**628 passed**.

## Phase 26 — the live along-route error, not the endpoint

The endpoint is nearly oracle (`D/D_true` 0.983). But the *live* trajectory on
07-26 is still wrong by up to **704 m** mid-outage. Phase 26 asks whether that
can be cut without touching topology or the endpoint.

### TASK 1 — where the 704 m is (trace, no new algorithm): **confirmed pre-bend**

Full 1 Hz trace of the best wired variant. The worst error is **−704 m at
elapsed 444 s** (estimate *behind* truth — the saturated-spectral undershoot),
and it accumulates almost monotonically over the whole pre-bend window
(el 60 s: −122, el 240 s: −350, el 440 s: −688) through the Выборгское шоссе
cruise where the spectral model is pinned near 13 and drags `v` down. The ev3
**bend position anchor at t≈590 corrects it in a single +660 m jump** (−688 →
−82). After the bend the live error stays bounded −2…−236 m, endpoint −190 m.
`|D_err|` p50/p90/p95 = 190 / 612 / 667 m, essentially all of the tail pre-bend.
So the hypothesis holds: **one unanchored pre-bend accumulation, fixed in one
step by the bend.**

### TASK 2 — earlier saturation trigger from lateral anchors: **refuted (no lead)**

`r_lat = v_lat − v_spectral` over the 313 well-conditioned high-speed lateral
anchors on 07-26. Pre-bend the conditioned mean is **−1.0 m/s** (the lateral
anchors do not see the source under-reading in the earlier cruise — they are
sparse, low-`ω`, and noisy-low). The first robustly-positive rolling `r_lat`
lands at el ~410 s, **3–12 s before the bend already fires**. A `k ≥ 10`-anchor
rolling rule never triggers on 07-22 (safe) but buys nothing on 07-26. The
lateral anchors simply do not sample the saturated regime early enough — the
standing project finding, re-confirmed. Nothing wired.

### TASK 3 — rolling spectral self-statistics: **transferable signal, unsafe to use**

A clean separator exists: the **fraction of a rolling 30 s raw-spectral window
above 15 m/s**. On 07-26 it is 0.09–0.28 pre-bend (up to 0.55 later); on 07-22
it is **exactly 0.00 in every window, the whole trip** — the car there never
sustains > 13 m/s so the spectral output never piles against its ceiling. A soft
`P_sat` from that fraction, scaling the existing down-censor, fires ~el 80 s on
07-26 (~360 s before the bend) and is provably inert on 07-22.

Wired prototype: 07-26 `D/D_true` 0.983 → 0.988, max `|D_err|` 704 → 633, p95
667 → 562. **But the committed route changes.** Arming the censor that early
lifts `v` enough, soon enough, that junction decisions move — the decision
sequence diverges from the baseline (extra edges 5318/5310/5297 at the trip
end), `identical route: False`. That is exactly the speed-policy-reroutes-the-car
circularity the architecture forbids, so the apparent `surv` gain
(0.148 → 0.355) is not creditable. **Refuted for production.** The censor is only
safe *after* the bend, when the route past that point is already determined and
the bend is itself the anchor. Nothing wired.

### TASK 4 — retrospective smoothing between the start and the bend: **wired (OFFLINE, gated OFF)**

Once the ev3 bend has landed there are two independent absolute along-route
positions (outage start, bend). The bend corrected the live distance *forward*;
the pre-bend history kept the full undershoot. Redistribute the bend's folded
correction (`delta_D_m`) *backward* over the pre-anchor trajectory ∝ time spent
moving (the saturated undershoot accrues ∝ cruise distance, so moving-time is
the right profile — `sigma_v²`- and residual-weighted variants concentrate the
fix and produce absurd 10–14 m/s implied speed corrections; uniform-moving does
not). The reconstructed history then joins the already-corrected future with no
step.

| 07-26 (wired) | D/D_true | endpoint | max\|D_err\| | p50 | p90 | p95 | v-bias | max Δv |
|---|---|---|---|---|---|---|---|---|
| B causal (down-censor + `k_high`) | 0.983 | −190 | 704 | 190 | 612 | 667 | −0.17 | — |
| **F + retro smoother (offline)** | **0.983** | **−190** | **394** | **112** | **222** | **262** | +0.46 | 2.1 m/s |

`d60 / d120 / d240` collapse −122/−248/−350 → −40/−60/−77. The endpoint,
`survival`, decision sequence and `k_high` are untouched. The residual 394 m is
a genuine *post*-bend causal excursion (el 975 s) — no later anchor exists to
smooth it against, so it is now the limiter.

**07-22 control:** no bend anchor → `retrospective_distance_smooth` returns
`None`, `retro_speed_trace` is `None`, every number identical to baseline
(0.950, 0 censored updates). Literally inert.

### TASK 5 — fixed-lag smoother: **negligible**

Existing `speed.fixed_lag_s`, swept 0 → 60 s: max `|D_err|` 704 → 677, p95
667 → 628. The bend correction is a single event ~450 s into the outage; a
≤ 60 s lag only reaches the last minute of a 7-minute accumulation. `survival`
unchanged (it does not reroute). Not worth a default; already available.

### TASK 6 — anti-circularity / leakage (the one wired mechanism, TASK 4)

| question | retro smoother |
|---|---|
| Uses withheld GPS? | No — only `delta_D_m` and `t_applied` from the bend anchor, and the emitted speed samples. |
| Uses future information? | Yes — the bend that lands ~450 s in. That is the whole point. |
| Offline / fixed-lag only? | Offline. Emitted as a *separate* `retro_speed_trace`; the causal `speed_trace` is byte-identical with the flag on or off. |
| Can the route choice confirm the correction? | No — it runs after the tracker, reads no hypotheses, writes no state; the committed route is already final. |
| Can it change topology? | No — it moves only the along-route scalar position of already-emitted samples. |
| Can the bend endpoint depend on bad `D`? | The bend *shape* match takes no `D` prior; edge-candidate selection uses the odometer, so a wrong route upstream could still mis-place it — but that risk is identical to the already-wired causal bend anchor and is not introduced here. |
| All production flags still OFF by default? | Yes — `retro_bend_smoothing_enabled=False`; the only unconditional change is one always-`NaN` diagnostic dict key. |

TASK 3's rolling-spectral trigger fails question 4/5 (it moves junction
decisions) and is **not wired** for that reason.

### TASK 7 — final ablation

| trip | variant | D/D_true | endpoint | max\|D_err\| | p50 | p90 | p95 | v-bias | v-p95 | surv | ev | d60 | d120 | d240 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 07-26 | A bend position only | 0.889 | −1265 | 1318 | 523 | 1246 | 1274 | −1.77 | 8.3 | 0.103 | 1 | −113 | −237 | −337 |
| 07-26 | B down-censor + `k_high` (causal best) | 0.983 | −190 | 704 | 190 | 612 | 667 | −0.17 | 9.3 | 0.148 | 1 | −122 | −248 | −350 |
| 07-26 | **F  B + retro smoother (offline)** | **0.983** | **−190** | **394** | **112** | **222** | **262** | +0.46 | 9.1 | 0.148 | 1 | −40 | −60 | −77 |
| 07-22 | A / B / F (all) | 0.950 | −161 | 219 | 133 | 191 | 198 | −0.23 | 5.3 | 0.157 | 4 | −114 | −66 | −64 |

07-22 is identical across A/B/F: no bend anchor → censor never triggers,
`retro_speed_trace` is `None`. The early lateral trigger (TASK 2) and rolling
trigger (TASK 3) are absent from the table because neither is wired (no lead
time; reroutes the car).

### Answers (Phase 26)

1. **Can the ~700 m live error be cut, keeping topology and ~0.98 endpoint?**
   Causally: **no** meaningful cut — every early trigger either has no lead time
   (lateral) or reroutes the car (rolling-spectral); fixed-lag ≤ 60 s gives ~4 %.
   Retrospectively: **yes** — whole-outage max 704 → 394 m, p90 612 → 222 m.
2. **Best zero-latency causal:** unchanged, `D/D_true` 0.983, max `|D_err|` 704 m.
3. **Best fixed-lag:** 60 s → 677 m. Diagnostic only.
4. **Best full retrospective:** `D/D_true` 0.983, max `|D_err|` 394 m,
   p50/p90/p95 = 112 / 222 / 262 m, max artificial Δv 2.1 m/s, `v` stays physical.
5. **What now limits quality:** the *post*-bend causal stretch. The outage ends
   with no further absolute anchor, so the last ~400 s of trajectory keeps its
   causal error and there is nothing (short of another bend or a map interval on
   that stretch, neither of which the geometry offers) to reconstruct against.

Files: `retro_smooth.py` (new), `tracker.py` (attach `retro_speed_trace`),
`single_path.py` (`ba["t_applied"]`), `config.py`
(`single_path.retro_bend_smoothing_enabled`, default `False`). Tests:
`test_retro_smooth.py` (9). Full suite: **637 passed**.

## Phase 27 — is the pre-bend error an AHRS gravity/orientation fault? **No — refuted**

Hypothesis: the recorder's accelerometer-levelled AHRS mistakes sustained
longitudinal acceleration for a nose-up tilt, leans into it, and the gravity
subtraction then removes the acceleration — and a gyro-dominant orientation
during dynamic acceleration would recover the lost Δv and cut the ~700 m
pre-bend error.

### TASK 1 — the lean is real, gyro-confirmed…

Three acceleration windows on 07-26, raw 100 Hz:

| window | true Δv | recorder `userAccel`·fwd | production `a_long` (+ `leveling_correction` τ=20) | recorder pitch change | gyro-integrated pitch change |
|---|---|---|---|---|---|
| t712–752 | +21.3 | +15.4 | −4.1 | **−5.9°** | +1.5° |
| t789–798 | +8.9 | +6.7 | +5.0 | **−3.4°** | +0.7° |
| t330–344 | +7.1 | +5.6 | +0.5 | **−5.6°** | +0.7° |
| t370–380 *(good)* | +8.3 | +3.0 | +3.3 | −1.1° | −0.3° |

The recorder pitch drops 3–6° during each hard acceleration while the gyro sees
essentially no rotation — a genuine false lean, confirmed by the gyro. On
t712–752 the swallowed component is ≈ g·sin(−5.9°) ≈ −1.0 m/s² ≈ −30 m/s over
the window.

### …but it does not cause the pre-bend error, and it is already mostly handled

Two things break the hypothesis:

1. **The recorder's own `userAcceleration` channel already recovers 72–82 % of
   true Δv on the *bad* windows** (t712 +15.4/+21.3, t789 +6.7/+8.9,
   t330 +5.6/+7.1). The *good* window t370 recovers only 38 %. "Bad" and "good"
   are not the leaning ones vs the level ones.

2. **The pre-bend `D` error is nearly invariant to the entire accelerometer
   treatment.** Sweeping `leveling_recovery_tau_s` on the Phase 25/26 wired best:

   | `a_long` channel | 07-26 pre-bend max\|D_err\| | el 440 s | v-bias | D/D_true |
   |---|---|---|---|---|
   | `leveling_correction` τ=20 (production) | 684 | −688 | −0.17 | 0.983 |
   | τ=8 | 674 | −679 | −0.25 | 0.985 |
   | τ=40 | 701 | −705 | −0.22 | 0.985 |
   | τ=0 (add-back off) | 701 | −706 | −0.63 | 0.977 |
   | from-scratch Mahony (`attitude.enabled`) | **1455** | −1506 | −3.98 | **0.562** |

   ±15 m on a 684 m error across the whole plausible range. Spectral saturation
   sets the pre-bend speed; `spectral_update` overwrites `predict` four times a
   second, so the accelerometer is barely in the loop. This is Phase 23's
   finding reached from the AHRS side. Removing `leveling_correction` entirely
   (τ=0) makes the *bias* worse (−0.17 → −0.63), so the production add-back is
   net-helping in aggregate even though it over-corrects the step-like lean on
   individual hard-acceleration windows.

### TASK 2–4 — gyro-dominant orientation is worse, not better

Standalone gyro-propagated Δv on the windows (seeded from a pre-window rest
gravity), with the accel correction off / ZARU-only / static-gated:

| window | true | recorder | pure gyro | gyro + static-gate |
|---|---|---|---|---|
| t712–752 | +21.3 | +10.1 | **−35.3** | −25.5 |
| t789–798 | +8.9 | +6.5 | −2.6 | −0.5 |
| t370–380 | +8.3 | +4.6 | −10.6 | −0.7 |

Pure gyro orientation drifts **2.7° median / ~5° p90 over 20–40 s** (07-26),
which is g·sin θ ≈ 0.3–0.45 m/s² of spurious longitudinal force — tens of m/s
over a cruise ramp, worse than the lean it replaces. The from-scratch Mahony
filter on the full causal replay: 07-26 `D/D_true` 0.983 → **0.562**, 07-22
0.950 → **0.586** with `survival` 0.157 → 0.013 and shifted decision timing (a
topology risk on the control trip). Per the brief's own stop rule ("3–5 instead
of 21, < 2 instead of 7.7 ⇒ refuted"), the branch stops here.

### TASK 5 — grade vs false lean: the discriminant is too weak

Real body pitch-rate at a 3° grade change over 2 s at 15 m/s is ≈ 0.026 rad/s —
the same magnitude as the moving `|gyro_y|` p90 (0.026 / 0.049 rad/s on the two
trips). A "freeze the accel correction during sustained acceleration" gate
cannot tell a real grade transition from noise, and gyro-only between trusted
resets has already drifted several m/s within 20 s.

### TASK 8 — answers

1. **What does the AHRS mistake for gravity?** Sustained forward specific force
   — it leans pitch 3–6° nose-down-in-estimate during hard acceleration, gyro
   unmoving.
2. **How much longitudinal force is recovered?** By the recorder's own
   `userAcceleration`: 72–82 % on the bad windows. By the production pipeline
   (`+ leveling_correction`): variable, and *negative* on the worst window
   (over-correction of a step-like lean).
3. **Does a better AHRS cut the pre-bend `D` error?** No — it is invariant to
   the accelerometer channel within ±15 m of 684 m.
4. **Route topology / 07-22 control:** the from-scratch filter *does* perturb
   07-22 decisions — another reason it stays off.
5. **Hidden GPS in the path?** None — this was all diagnostic.

**Verdict:** the false AHRS lean is real but it is not the pre-bend error's
cause; the accelerometer is not the channel that sets the pre-bend speed
(spectral is), and gyro-dominant orientation drifts worse than the lean. Nothing
wired. `attitude.enabled` stays `False` (confirmed harmful), `leveling_recovery_tau_s`
stays `20.0`. No code change; **637 passed** unchanged.

## Phase 28 — inverse-variance fusion of an IMU velocity with spectral? **Already there — refuted**

Hypothesis: carry a parallel IMU speed `v_imu ± σ_imu` (reset at each trusted
anchor, integrated forward with a *growing* variance) and Gaussian-fuse it with
`v_spec ± σ_spec`, so that while `σ_imu` is still small the saturated spectral
cannot pin `v` to 13–14 m/s.

### The EKF already *is* that fusion

`spectral_update` calls `_update(H = [0,1,0,0,0], innovation = k·v_spec − v, R = r)`.
That update is, exactly:

```
S    = P[v,v] + r
K_v  = P[v,v] / (P[v,v] + r)
v   := v + K_v · (k·v_spec − v)  ==  (1 − K_v)·v_pred  +  K_v·(k·v_spec)
```

so `K_v` **is** the inverse-variance weight `σ_imu² / (σ_imu² + σ_spec²)` with
`σ_imu² = P[v,v]` and `σ_spec² = r`. `v_pred` (the value `x[V]` holds when the
update lands) **is** `v_imu`: `predict` integrated `a_long − b_a` into it, and
`P[v,v]` was grown by `Q` (`accel_noise_ms2`, `accel_bias_rw`) on every step
since the last measurement — a variance that grows with elapsed time, seeded and
shrunk by ZUPT / lateral / bend just as the hypothesis asks. There is no missing
fusion layer to add. A second hand-rolled `v_imu` built by re-integrating
`a_long − b_a` from an anchor would reuse the same accelerometer, the same `b_a`
state and the same anchors that `predict` already consumes — fusing it back in is
**double-counting the accelerometer**, not adding an independent sensor.

### Instrumenting the real update (07-26, pre-bend)

| quantity | pre-bend median |
|---|---|
| `σ_pred = sqrt(P[v,v])` (`σ_imu`) | **4.5 m/s** (p10 1.6, p90 5.2) |
| `sqrt(r)` (`σ_spec`) | ~39 m/s |
| Kalman gain `K_v` (spectral's weight) | **0.013** (p90 0.017) |
| `|Δv|` per spectral update | 0.022 m/s |
| net spectral Δv over the whole pre-bend | **−4.3 m/s** (−49 down, +45 up) |

The IMU prior already gets **98.7 %** of the weight on every update. Spectral is
not dragging a good prior down — its net pull over 440 s is −4.3 m/s. Where the
truth is 20–24 m/s and the spectral reads 12–15, `v_pred` is *itself* already
only ~16 m/s: the accelerometer under-integrates the ramp (Phase 27), and no
fusion weight recovers speed the prior never had.

### The weight is already optimal — sweeping it only hurts

Global `R_spec` multiplier (>1 ⇒ trust the IMU prior more):

| ×R | 07-26 D/D | pre-bend max\|D_err\| | 07-22 D/D | 07-22 ev |
|---|---|---|---|---|
| 0.3 (trust spectral more) | 0.995 | 707 | 0.921 | 5 ⚠ |
| **1.0 (current)** | **0.983** | **684** | **0.950** | **4** |
| 3.0 | 0.968 | 707 | 0.947 | 4 |
| 10 | 0.853 | 785 | 0.937 | 4 |
| 100 (trust IMU prior) | 0.847 | 1023 | 0.923 | 5 ⚠ |

Pre-bend error is flat at ~684–707 m across a 300× range of the fusion weight,
best at the current setting; pushing toward the IMU prior breaks topology on both
trips (`ev` moves). A pure-IMU shadow (spectral fully disabled, ZUPT + lateral +
bend only): pre-bend v-MAE **10.9 m/s**, D error **1919 m** — 3× worse than with
spectral. Spectral is the stabiliser, not the problem.

### σ calibration (TASK 8)

Shadow `σ_imu` pre-bend coverage: 1σ 0.74, 2σ 0.92, 3σ 0.96 (targets
0.68 / 0.95 / 0.997) — mildly optimistic but roughly honest; it grows to 63 m/s.
Not the failure mode.

### Answers (Phase 28)

1. **Can accel-integrated velocity be a temporary uncertain prior so the
   saturated spectral doesn't pin `v` down?** It already is exactly that —
   `v_pred` with covariance `P[v,v]` — and it already carries 98.7 % of the
   weight. The saturated spectral contributes −4.3 m/s net over the pre-bend.
2. **Does the existing EKF already do the inverse-variance fusion, so the real
   problem is wrong `P_v` / `R_spec`?** Yes to the first half — the Kalman update is
   the inverse-variance blend. But `P_v` and `R_spec` are *not* miscalibrated:
   `σ_imu` ≈ 4.5 m/s is honest, and every reweighting (0.3× … 100×) leaves the
   pre-bend error at ~700 m or makes it worse. The pre-bend error is the prior
   being ~6 m/s low on fast cruise — an accelerometer limit (Phase 27) and a
   high-speed-regime unobservability (Phases 20–23), not a fusion-layer gap.

**Verdict: refuted, nothing wired.** No second filter — the EKF's measurement
update is already the fusion, and building a parallel `v_imu` to feed it would
double-count the accelerometer. No code change; **637 passed**.

## Phase 29 — acceleration attenuation audit: **refuted**

Hypothesis: some stage between the raw accelerometer and `EKF.predict` — a
filter, deadband, clip, the leveling transform, the bias subtraction — is
systematically eating the car's slow longitudinal acceleration, and that is the
~6 m/s pre-bend deficit.

### The pipeline (production, `attitude.enabled = False`)

| # | stage | file | DC behaviour |
|---|---|---|---|
| 1 | raw `userAcceleration` (device, ~100 Hz) | `MotionSample` | recorder AHRS already removed gravity (Phase 27 lean lives here) |
| 2 | `R_recorder @ uacc` → world | `motion_model.py:281` | rotation, `corr = 1.000`, lossless |
| 3 | `+ leveling_correction(τ=20)` on `a_world[:,:2]` | `motion_model.py:296` | mean 0.086 → 0.067 m/s²; `corr 0.90`; **±8 m/s over 30 s windows, mean +0.57** |
| 4 | project on `forward_world` | `motion_model.py:305` | dot product, lossless |
| 5 | `_robust_normalize` (Hampel 0.5 s, z 4.5, floor 0.12) | `motion_model.py:306` | **DC gain 1.000**, every frequency 1.000, ramp Δv 1.000, 0.1 % samples clipped |
| 6 | `_rolling_mean` | `motion_model.py:308` | **disabled** (`accel_smooth_window_s = 0`) |
| 7 | bin to 0.1 s, `a_long = median(bin)` | `motion_model.py:400` | median of ~10 ≈ mean |
| 8 | `EKF.predict`: `a = k_a·(a_long − b_a)` | `speed.py:503‑508` | `k_a` **exactly 1.0000** all pre-bend (no interval moved it); `b_a` ∈ [−1.11, +0.68], mean −0.05 |

### Δv accounting — 07-26 pre-bend acceleration windows

Retention `= Δv_stage / Δv_truth`, summed over the pre-bend windows, split by
sign:

| | truth | raw `userAccel` | +leveling | +Hampel | EKF input | after `k_a·(·−b_a)` |
|---|---|---|---|---|---|---|
| **acceleration** | +60.8 | +57.1 (**94 %**) | +20.0 (33 %) | +20.0 (33 %) | +20.1 (33 %) | +19.4 (32 %) |
| **braking** | −44.2 | −33.0 (75 %) | −14.1 (32 %) | −14.1 (32 %) | −14.3 (32 %) | +8.1 (−18 %) |

Per-window the raw signal ranges **6 %–444 %** of truth — it is extremely noisy;
the 94 %/75 % totals are cancellation, not fidelity. `leveling_correction`
appears to drop retention to ~33 %, but §3 above shows its true operator: a
near-zero-mean ±8 m/s / 30 s perturbation (net DC −0.02 m/s²). The "loss" is that
high-variance term landing negative on these particular windows, not a filter
with a low DC gain. Hampel, binning and `k_a` are all unity.

### Causal bypass (TASK 6) — nothing helps, everything hurts

| variant | 07-26 pre-bend max\|D_err\| | 07-26 D/D | 07-22 pre-bend | 07-22 `survival` / `ev` |
|---|---|---|---|---|
| **A baseline** | **684** | 0.983 | **219** | 0.157 / 4 |
| B `leveling_recovery_tau_s = 0` | 701 | 0.977 | 212 | 0.222 / 4 ⚠ |
| C raw world accel (no leveling, no Hampel) | 712 | 0.983 | 213 | 0.188 / 4 ⚠ |
| D raw device-forward accel | 712 | 0.983 | 207 | 0.189 / 4 ⚠ |
| E freeze `b_a = 0` | **795** | **0.823** | 243 | 0.219 / **5** 💥 |
| F τ=0 + freeze `b_a` | 701 | 0.973 | 203 | 0.242 / **5** ⚠ |
| G raw world + freeze `b_a` | 720 | 0.977 | 199 | 0.240 / **5** ⚠ |

Baseline is the *best* pre-bend number. Every bypass is ≥ 684 on 07-26 and shifts
07-22 decisions (`survival` moves, `ev` 4 → 5 whenever `b_a` is frozen).
**Pre-bend v-bias stays −1.5 … −2.5 m/s in every variant** — invariant to the
entire accelerometer treatment. `b_a` is load-bearing compensation for the raw
channel's per-window error (freezing it → `D/D` 0.823, topology break), not a
thief; its mean over the pre-bend is ≈ 0.

### Verdict (TASK 7 / TASK 9): **refuted — case D, with a note of A**

1. **Where is Δv lost?** Nowhere fixable between the recorder and `predict`.
   Stages 2, 4–7 are unity; `k_a = 1`; `leveling_correction` is a wash
   (±8 m/s zero-mean, net −0.02 m/s²); `b_a` is compensation, mean ≈ 0.
2. **Per stage?** raw `userAcceleration` carries 94 % / 75 % of accel / braking
   Δv (summed) but with 6–444 % per-window scatter; leveling's *apparent* 33 %
   is variance, not DC gain; Hampel 100 %; `k_a` 100 %.
3. **Is software filtering the cause of the deficit?** **No.** No filter has a
   DC-gain problem; the one non-trivial operator (`leveling_correction`) removed
   entirely leaves the pre-bend error *worse* (684 → 701) and perturbs 07-22.
4. **Causal bypass?** All seven variants ≥ baseline on 07-26; `b_a`-freeze
   variants break 07-22 topology. Nothing survives TASK 8.
5. **Production change?** **None.** The pre-bend deficit is the recorder AHRS
   lean (Phase 27, upstream in the sensor) plus spectral saturation setting the
   level (Phase 28) — not a preprocessing attenuator. No code change;
   **637 passed**.

## Tests

`processor/tests/pacman/test_spectral_scale.py` (18) +
`test_interval_jacobians.py` (3). Cover: `k_s` unchanged without an anchor;
ordinary `spectral_update` cannot reach the `k_s` Kalman; clean lateral anchor
moves `k_s` the right way; below-min-speed and wrong-sign anchors rejected;
clean interval moves `k_s` toward `L/∫v`; analytic vs finite-diff Jacobian; long
cruising interval constrains `k_s` while `k_a` stays unobserved; `k_a`/`k_s`
separate states; causal trace is append-only and time-stamped; no hidden-GPS
tokens in the calibration path; unconstrained `k_s` keeps wide σ; bounds; ZUPT
still pins `v=0`; interval `k_s` only callable from the gated single-path path.

Phase 18 integration: `processor/tests/pacman/test_bend_integration.py` (19, 7 for censoring - saturation trigger, down-censoring near the plateau, upward readings trusted, k_a/k_s/topology untouched).
Cover: a curve inside one edge classifies as a bend (not a junction) and the
event is consumed so it drives no decision; classification off by default;
the manager's bend match takes no distance prior; `bend_speed_anchor` moves
`k_high` and never `k_s`/`k_a`; `k_high` stays wide and at 1 without a bend
anchor; an ordinary spectral update cannot move `k_high`; a low-speed lateral
anchor cannot collapse `k_high`'s uncertainty; the regime weight uses only the
spectral output; `spectral_update` blends `k_low`/`k_high` by regime; no
hidden-GPS tokens in the bend path; (Phase 20) `k_high` persists through a stop
and a long quiet stretch with only its `σ` growing, and `zero_velocity` does
not touch the calibration state.

Phase 17 prototype: `processor/tests/pacman/test_bend_anchor.py` (9). Cover: a
smooth bend on one edge is matched at its true along-edge position with the
drive speed recovered; the matcher takes no distance prior; repeated identical
bends on one edge are rejected as not unique; a high `min_margin` rejects any
match; one event yields at most one `BendMatch`; a sharp compact turn on a
straight edge is not a bend and classifies as junction; the classifier calls a
spread low-rate geometry-matching event a bend and rejects when neither model
fits; `BendMatch` carries a uniqueness margin. Full suite: **637 passed**.

## Phase 18 answers

1. **Bend anchor safely integrated into `single_path` without false junction
   matches?** Yes. `_classify_bends` marks a spread/low-rate event with a
   unique curvature match and no compatible junction as `is_bend` + `consumed`;
   on 07-26 ev3 becomes a bend (decision 12 drops from event-driven to
   geometric), on 07-22 all six sharp turns stay junctions (zero false bends,
   confusion `bend→bend 1 / junction→junction 5 / junction→reject 3`). Gated,
   OFF by default; 619 tests green.
2. **Does ev3 still recover ~735 m with no D prior?** Yes — `s_map_anchor
   735 m` vs truth 734 m, from the same geometry-only matcher, now called
   inside the manager (`test_bend_match_in_the_manager_uses_no_distance_prior`).
3. **`v_bend` accuracy vs hidden truth?** `v_bar = 17.50 m/s` vs `v_true`
   17.66 (window) / 17.83 (peak) → **−0.16 / −0.33 m/s**, `σ ≈ 0.67`.
4. **`v_spectral` at ev3 and the local `k_s` it implies?** `v_spectral ≈ 14.1`
   (window median); `k_s_local = 17.5 / 14.1 = 1.24 ± 0.26`; true local 1.25.
   Squarely the 1.2–1.35 the trip needs (the span-average was only 1.08).
5. **Does naive causal use of that local `k_s` after ev3 beat 0.890?** Yes —
   monkeypatch flat `k = 1.24` on future spectral: `D/D_true 0.836 → 0.903`
   alone, **0.956** with the position anchor. Wired (regime-blended, window
   median denominator): **0.927**.
6. **Can an online spectral/OOD indicator find the saturated regime?** **No.**
   Mahalanobis distance, feature norm and ridge leverage are all flat across
   speed (the saturated regime is not OOD — that is why the model saturates).
   The only online regime signal is `v_spectral` itself, and it saturates near
   13 m/s; the wired blend gates on it for lack of anything better.
7. **Two-regime `k_s` causal performance / 07-22 safety?** 07-26: `k_low`
   (lateral, ~1) and `k_high` (bend, 1.19) blended by `w(v_spectral)` →
   `D/D_true 0.927`, topology unchanged. 07-22: no bend, `k_high` stays
   `1.0 ± 0.31`, `D/D_true 0.950` unchanged, shallow fork untouched.
8. **Final `D/D_true` / survival / route errors.** 07-26: 0.836 → **0.927**
   (position + `k_high`), survival 0.103 → 0.101, genuine wrong routes 0 → 0,
   `evdriven` 2 → 1 (one curvature event no longer mis-matched to a node).
   07-22: **0.950 / 0.158 / unchanged** — the mechanism is inert where there
   is no bend. (Monkeypatch ceiling for the flat correction: 0.956 / 0.151.)

## Answers to the brief (Phase 0–13)

> Answers 6, 7 and 11 below were the verdict **before Phase 17/18**. Phase 17
> then found a GPS-free anchor that *does* reach the high-speed regime — road
> curvature — and Phase 18 wired it: on 07-26 the bend at ev3 yields a local
> `k_s ≈ 1.24` and a `+646 m` position fix that take causal `D/D_true` from
> 0.836 to **0.927** with topology unchanged, and leave 07-22 at 0.950.

1. **Is `v_spectral` alone sufficient to recover `v_true`?** No. It saturates
   above ~10 m/s and folds multiple true speeds onto one prediction (Phase 0);
   `c(v_spectral)` is non-monotonic and not a function.
2. **Do raw features retain high-speed info?** Weakly (partial corr ≤ 0.47) and
   non-transferably. Enough to halve RMSE *with the outage's own GPS*, not
   enough for a GPS-free feature recalibrator.
3. **Clean lateral anchors per trip, and at what speed?** 07-22: 17, mostly
   0–12 m/s. 07-26: 7, spread 6–20 m/s. **All during turns** — none in the
   steady high-speed cruise that carries the error. Production gate: 189 / 345.
4. **Clean map intervals?** 07-22: one (implies `k_s ≈ 1.36`, but only covers
   the fast stretch). 07-26: **none** (only candidate is on an 18 %-short
   route).
5. **Which calibration model wins?** For an online GPS-free calibrator: **none**
   (verdict C). Scalar `k_s` (Model A) is the only well-posed form; affine and
   piecewise add parameters the anchors can't constrain; a feature recalibrator
   fits one trip's noise.
6. **Causal GPS-free improvement to `D/D_true`?** 07-22: 0.930 → 0.95–0.97
   (lateral) but survival 0.204 → 0.15 — net negative. 07-26: 0.836 → 0.78
   (worse). **Causal calibration does not help on either trip.**
7. **Can 07-26 improve from ~0.84 without hidden GPS?** Not causally with these
   anchors. A *known* scalar ~1.30 would (→ ~1.00, survival 0.67) — the
   mechanism is right, the observability isn't there.
8. **`k_s(t)` over the trip?** 07-22: swings 0.75 → 1.22 → 1.05, never
   converges. 07-26: oscillates 0.75–1.10, ends 1.019 ± 0.046 (confidently
   wrong) or stays 1.0 ± 0.198 (interval-only, honest).
9. **Does `σ_k_s` reflect unobserved high-speed regions?** Yes. Interval-only
   07-26 keeps `σ_k_s` growing to 0.20 with no anchor; the variance-inflation in
   `spectral_update` passes that straight into `σ_v`/`σ_D`. It does **not**
   vanish without an anchor in the relevant band.
10. **Does improved `D` also improve junction timing?** On 07-26 in the ceiling
    runs, yes — survival 0.10 → 0.67 with route topology unchanged. Causally,
    no, because the causal `k_s` is wrong.
11. **What stays fundamentally unobservable?** The spectral model's gain in the
    16–20 m/s sustained-cruise regime. No GPS-free absolute-speed anchor exists
    there: turns (lateral anchors) happen slower, and a trustworthy long map
    interval requires a correct committed route length, which 07-26 doesn't
    have between its usable turns. The warm-up has no fast driving. This is the
    same gap `ACCEL_SCALE_FORENSICS.md` reached from the distance side.

## Not done (as instructed)

No hardcoded `c(v_true)`; no correction fitted on withheld GPS presented as
production; no global 1.15; `k_a` cannot absorb spectral error (`k_s` is a
separate state); route choice never reused to calibrate `D`; no magnetometer;
no Mahony; no neural model; no widened route tolerances; no silent high-speed
extrapolation (`k_s` bounds + min-speed gate); calibration uncertainty does not
vanish without anchors. `spectral_scale_enabled = False` in the default config —
**production behaviour is unchanged.**

Phase 17/18: no hardcoded bend position, speed or `k_s`; the bend match takes
no D prior (removing it is a no-op — there is none); D chooses no bend and no
bend confirms a route (bend selection is geometry + gyro only, and topology was
already committed by the earlier sharp turns); no OOD detector was invented
where the features do not support one; the two-regime interval Jacobian (B8)
was **not** wired — neither trip has an interval that could calibrate `k_high`;
`bend_classification_enabled`, `bend_position_anchor_enabled`,
`bend_local_speed_anchor_enabled` and `speed.bend_scale_enabled` are all
`False` by default. `bend_anchor.py` is still not imported by the tracker
except through those gated paths.
