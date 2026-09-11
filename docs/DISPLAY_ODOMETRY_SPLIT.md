# Phase 32 — route/topology odometry vs display/position odometry

**Diagnostic only. Nothing in the tracker is modified. No hidden GPS in the
zero-latency branch (withheld GPS is used only to score the result and, in
TASK 8, as a would-be D-independent anchor).**

The idea: Phase 31 showed a learned/monotone spectral-residual correction
transfers at the window level but destabilises localisation *because it moves
`D`, which moves junction timing and route decisions*. So **freeze topology on
the current conservative odometer `D_route`, and run a separate `D_position`
for the displayed point only** — allowed to use the more aggressive corrected
velocity, forbidden from touching edge choice, branch weights, junction timing,
commits, rollback, turn matching or pruning.

Scripts: `tools/phase32_display_branch.py`, `phase32_alltrip.py`,
`phase32_task68.py`, `phase32_plots.py`.
Artefacts: `docs/plots/phase32/` (`task6_*.csv`, `task7_1d_alltrip.csv`,
`task10_bound_sweep.csv`, `task9_worstcase.csv`, `task8_parallel.csv`,
`phase32_summary.png`).

---

## Architecture (TASK 1–4, 12)

```
sensors ─┬─ baseline tracker ─→ D_route ─→ topology / edge commits   (UNCHANGED)
         └─ corrected velocity ─→ D_position ─→ displayed Pacman point only
```

`D_position` is computed **entirely post-hoc** from the baseline run's
`speed_trace` plus a gated velocity correction:
`v_position(t) = v_route(t) + α(t)·max(0, v_ml(t) − v_route(t))`
(one-directional — the premise is spectral *saturation*, i.e. under-read, so the
display branch may only add speed), then
`D_position(t) = D_route(t) + ∫ (v_position − v_route) dt`.

**There is no feedback path**, so every TASK 12 anti-circularity question is
answered by construction: `D_position` cannot change an edge (NO), a branch
weight (NO), junction timing (NO); the corrected model uses no hidden GPS in the
production path (NO); an anchor cannot pick the route by which position branch
is closer (NO); a D-independent route anchor *may* correct `D_position` after
the route has independently committed (YES). Route decisions on `rf-07-26` are
identical with and without the experiment — the same baseline run produces them
(`topology_rf-07-26.json`).

Correction sources (both from Phase 31, **no new training**):
* **iso** — `IsotonicRegression` of `residual` on `v_prior − v_spectral`, fit
  leave-one-trip-out on the other sessions.
* **cb** — the Phase 31 CatBoost honest residual model, same LOTO protocol.
Gate `α`: observable saturation evidence only — rolling fraction of `v_spectral`
above 15 m/s, and `v_spectral` 20 s max. Binary (`α∈{0,1}`) or soft (`α∈[0,1]`).

Excess past the committed frontier (TASK 4): on the trips tested `D_route`
stays within the committed edge chain, so the clamp/buffer never actually
triggers; the diagnostic keeps `excess_position_distance` separately and
releases it as the route grows, but it is not load-bearing here.

---

## TASK 6 — the 07-26 experiment (`task6_full_rf-07-26.csv`)

Baseline (`D_route`): max |D_err| **724 m**, p95 626, median 266, D/D_true
0.835, v-MAE 2.61, the point is *never* closer to truth than itself.

| run | D/D_true | max \|D_err\| | p95 | median | v-MAE | closer-time |
|---|---|---|---|---|---|---|
| **A** baseline `D_route` | 0.835 | 724 | 626 | 266 | 2.61 | — |
| **C** cb-soft, unbounded | 0.896 | 456 | 418 | 140 | 2.13 | 91 % |
| **D** iso-soft, unbounded | 0.915 | 372 | 342 | 111 | 2.07 | 90 % |
| **E** iso-binary, unbounded | 0.939 | **296** | 283 | **59** | 1.98 | 89 % |
| iso-soft, bounded \|Δ\|≤250 | 0.892 | 474 | 376 | 111 | 2.07 | 90 % |
| iso-soft, bounded \|Δ\|≤300 | 0.90 | ~424 | — | — | — | — |

**Unbounded, the displayed point on 07-26 goes from 724 m max lag to ~300 m,
median 266 → 59, with topology structurally untouched** — it clears the
"strong result < 300–400 m" bar. Max *lead* stays ≤ 25 m (it never runs ahead).
The bounded-safe version (below) reaches ~420–520 m.

30-minute cut `07-26-s1` (baseline max 1936, D/D_true 0.885 — a much worse route):
iso-binary unbounded → max 1116, median 389. Big absolute cut, still large,
because that route is genuinely lost, not just lagging.

---

## TASK 7 — all-trip evaluation: does freezing topology remove the Phase 31 regressions? (`task7_1d_alltrip.csv`)

23 trips; 8 have no map coverage but the 1-D `D_position vs D_true` test needs
no graph. Fair set = 20 trips (exclude `07-24-s6` 2× extrapolation, `07-26-s0`
broken mount, `07-28-s2`/`07-30-s1` routes already 1.4–6.5× too long).

**Unbounded, iso-soft:** helped 8/20, hurt 4/20, **worst regression +1809 m**
(`07-27-s0`), best −644 m (`07-26-s1`), median Δmax −2 m.

**The regressions did NOT go away — but their cause split in two:**

| Phase 31 regressor | Phase 31 replay | Phase 32 frozen topology | cause |
|---|---|---|---|
| `07-31-s1` | 74 → 472 m (catastrophic) | 74 → **93 m** (cb-soft), max 393 → 246 | **was topology** — fixed |
| `07-28-s1` | 263 → 564 m | 260 → **1075 m**, max 820 → 2400 | **NOT topology** — `D_position` diverges on its own |
| `07-27-s0` (new) | — | max 960 → 2769 m | same as `07-28-s1` |

**The decisive result (panel b of the plot): the correction helps a trip iff
its baseline odometer is under-reading (`D_route/D_true < 1`), and hurts it iff
the baseline is already at or above truth.** Every green (helped) point has
D/D_true ≤ 0.97; every red (hurt) point has D/D_true ≥ 1.08. The saturation
gate fires on "spectral is saturated", which is **necessary but not sufficient**
for "the odometer is behind" — `07-27-s0` and `07-28-s1` saturate *and* their
`D_route` already over-reads, so adding speed is pure harm, topology frozen or
not. There is no online observable for the sign of the current `D_err` — the
same wall as `SPECTRAL_CALIBRATION_FORENSICS.md`.

Trips with no saturation (`07-22-s0`, `07-24-s5`, `07-27-s1`, `07-26-s2/s3`):
the gate never fires, `D_position ≡ D_route` byte-for-byte. The "leave
non-saturation trips alone" requirement is met.

---

## TASK 10 — bounded display correction (`task10_bound_sweep.csv`)

`D_display = D_route + clip(D_position − D_route, −B, +B)`. This is the salvage:
it caps the worst-case regression at exactly `B` by construction.

| B [m] | median Δmax (20 trips) | worst Δmax | n helped | n hurt | 07-26 max \|D_err\| |
|---|---|---|---|---|---|
| 100 | −19 | +100 | 11 | 2 | 624 |
| **200** | **−19** | **+200** | **11** | **2** | **524** |
| 300 | −10 | +300 | 10 | 2 | 424 |
| 500 | −2 | +500 | 9 | 4 | 372 |
| ∞ | −2 | +1809 | 9 | 4 | 372 |

`B ≈ 200–300 m` is a genuinely transferable trade (not tuned on 07-26): it
helps 10–11 of 20 trips, leaves ~7 untouched, mildly hurts 2 (trips whose
baseline error was already < 200 m), and on 07-26 cuts max lag from 724 to
**424–524 m (−28 to −41 %)**. To reach 07-26's < 300 m you need `B ≥ 450`,
which re-opens the cross-trip downside.

---

## TASK 8 — parallel longitudinal hypotheses (`task8_parallel.csv`)

Run `D_base` and `D_corr` on one topology; at elapsed-time checkpoints score
which was closer to the withheld GPS (diagnostic only — never an online
selector). Median oracle "choose at first anchor" gain is only 1.6 % *pooled*
(most trips: the two are identical), **but on the trips where they differ the
winner is consistent within a trip**:

* under-reading trips — `D_corr` wins essentially every checkpoint
  (`rf-07-26` 7/0, `07-24-s0` 25/0, `07-31-s0` 22/0, `07-27-s3` 9/0);
* over-reading trips — `D_base` wins every checkpoint
  (`07-25-s0` 26/1, `07-27-s0` 25/0, `07-28-s1` 18/6).

Oracle gain on the ambiguous trips: 30–67 %. **So one early D-independent
anchor (start GPS, a bend position anchor, a trusted map anchor) would let a
2-state longitudinal bank pick the right branch per trip** — this is the most
promising route past the sign-of-`D_err` wall, and it does not need hidden GPS.
07-26's own bend anchor lands at t ≈ 590 s, i.e. near the end of the outage, so
for 07-26 specifically the anchor would arrive late.

---

## TASK 11 — uncertainty

The two branches **share the accelerometer channel, ZUPT, the bias state and
the spectral model**, so `σ_D_position` is not independent of `σ_D_route` and
must not be presented as such. Both branches are badly overconfident against
the hidden truth (crude `|z| ≤ 2` coverage ≈ 0.24 vs target 0.95) — consistent
with every prior phase; the display point is "weak" far more often than any σ
it carries would admit, and the UI must treat it that way.

---

## TASK 13 — Verdict

### A. 07-26 live point
Unbounded correction: max \|D_err\| **724 → ~300 m**, median **266 → 59 m**,
D/D_true 0.835 → 0.94, v-MAE 2.6 → 2.0, closer to truth ~90 % of the outage,
topology byte-identical. Bounded-safe (`B = 250`): **724 → ~474 m** (−34 %),
median 266 → 111.

### B. cross-trip safety (20-trip fair set)
| | helped | unchanged | hurt | worst regression |
|---|---|---|---|---|
| unbounded | 8 | ~8 | 4 | **+1809 m** |
| bounded `B = 200` | 11 | ~7 | 2 | **+200 m** (capped) |

Freezing topology **fixed the Phase 31 `07-31-s1` blow-up** (its damage was
route-decision timing) but **not `07-28-s1` / `07-27-s0`** — those diverge on
`D_position` alone because their baseline odometer already over-reads and the
saturation gate still fires.

### C. topology independence
Confirmed — structural (post-hoc, zero feedback). Route decision list on
`rf-07-26` unchanged. All TASK 12 anti-circularity answers hold.

### Overall: **PARTIAL — NO for the unbounded idea, conditional YES for a bounded display-only correction.**

* The split is the right architecture and it removes the *topology-induced*
  instability from Phase 31.
* But the unbounded correction that delivers the strong 07-26 result
  (< 300 m) regresses other trips by **+1800 m** even with topology frozen,
  because "spectral saturates" ≠ "odometer is behind" and there is no online
  signal for the difference. By the phase's own bar ("if other trips still get
  +300–500 m error at fixed topology → honest verdict NO") the unbounded path
  is **NO**.
* A **bounded display correction `B ≈ 200–300 m`** is defensible and
  transferable: 07-26 improves ~30–40 %, the worst cross-trip regression is
  capped at `B`, non-saturation trips are byte-identical. It is a diagnostic-
  grade display aid, not a localisation fix.
* The real way forward is **TASK 8's direction**: keep a 2-state longitudinal
  bank on the frozen topology and let the *first* D-independent anchor choose
  the branch per trip — the winner is consistent within a trip, and no hidden
  GPS is required.

### Implementation discipline (not yet done — pending a decision)
If pursued: config flag `display.position_branch_enabled` (default False);
separate `position_speed_trace` / `position_distance_trace`; `speed_trace` and
committed route unchanged; bounded correction with `B` config; the gate and the
isotonic map are the only new state; tests for (a) edge sequence invariant to
`v_position`, (b) excess cannot choose an uncommitted edge, (c) anchor touches
only the position branch, (d) baseline trace identical with the flag off,
(e) no hidden-GPS token. Existing 223 pacman tests stay green.
