# Making the map calibrate the odometer

Follow-up to [`EVENT_ALIGNED_JUNCTIONS.md`](EVENT_ALIGNED_JUNCTIONS.md). Event
alignment fixed *route selection*: on 2026-07-22 the committed single path now
follows ~20 junctions correctly (was 5), on 2026-07-26 it makes zero genuine
route errors. What is left is the odometer - 2026-07-26 is topologically
perfect and still ~16 % short in distance - and one genuinely ambiguous fork.

Three tasks, nothing else touched: beam, `single_path`, event alignment, the
anti-circularity gates, the production ban on hidden GPS, Mahony off. 582 tests
pass; the beam oracle ceiling is unchanged (0.996 / 1.000).

## Task 1 — the shallow fork, delayed local commitment

**2026-07-22, incoming edge 14321**: true next 23129 (map turn −38.5°), chosen
23127 (map turn −12.4°). The two branches are a real local ambiguity.

### What was built

- **Soft turn tier.** `detect_turns` at a lower threshold (rate > 0.035 rad/s,
  ∫ > 14°) finds gentle turns the strong detector misses - a driver taking a
  38° corner wide integrates to ~24° below the strong threshold. A soft
  `PhysicalTurn` steers a junction only if that junction offers a moderate
  (16–62°) successor turning the soft event's way, and the decision is left
  **provisional**.
- **Provisional forks.** A low-confidence fork whose runner-up is a
  *substantively different* direction (≥ 22° apart) is held provisional: the
  greedy best still propagates as the one active branch, the runner-up is kept
  live, and the next strong turn events are asked to choose. Bounds:
  `provisional_max_events_to_resolve` = 2, `provisional_max_alternatives` = 2.
- **Sequence scorer** (`_sequence_branch_score`): for each branch, the best
  bounded graph path that explains the ordered list of subsequent strong turn
  angles - each matched to one junction, every other junction near-straight.
  Uses only signed turn angle, direction, count and graph legality. It reads
  no `d_event`, no `offset_bias`, no `D` - verified by a source test.
- **Late-turn revision.** A turn event that settles after the odometer has
  already walked past its junction (the lag is tens of seconds here) flags that
  committed decision provisional so downstream intervals do not trust the chain
  through it. The active branch is not rewound - only its confidence corrected.

### Forensic — does the next event resolve the 14321 fork?

The physical turn onto 23129 is at t_peak ≈ 528 s, ∫ ≈ −24° - a **soft** turn,
below the strong threshold. The tracker committed the 14321 junction at
t ≈ 493 s, ~35 s before the car actually turned, so the crossing window showed
−5°.

The next **strong** turn is ev at t_peak ≈ 647 s, +85° - ~120 s and several
junctions later. Both branches thread a graph path consistent with
"straight → +85°":

| branch | shortest consistent path | per-turn angle fit |
|---|---|---|
| 23127 (greedy) | 23127 → 23109 → 23111 → 23112 | −0.4 |
| 23129 (**truth**) | 23129 → 23126 → 23124 → 23120 → 23121 | −0.5 |

Extending to two and three subsequent strong turns ([+85, −92, +121]) does not
separate them either - the graph around this area is dense enough that both
branches produce plausible signed-turn sequences. The only feature that *would*
separate them is the elapsed-time-vs-map-length consistency, which is a
D-agreement discriminator and is off-limits.

**Answer to question 1: no.** The next independent turn event does not resolve
the 14321 fork on independent evidence. The machinery is in place and it
correctly declines to switch (`provisional_switched` = 0). Per the brief -
*"если не выбирает — не подгонять"* - it is not forced. The 14321 fork remains
the first genuine route error on 2026-07-22, on both production and oracle D.

(The soft tier and late-turn revision also cannot rescue *this* fork: by the
time the −24° soft turn is detected, the lagging tracker is one or two edges
past the junction, so the flag lands on a downstream decision. They do
correctly mark other low-confidence junctions provisional, which gates
downstream intervals.)

## Task 2 — long intervals: a quality gate, not a length cutoff

The old rejection was flat: `duration > 240 s` or `distance > 3000 m`. That
threw away the single most useful measurement for accelerometer scale. It is
replaced by `_interval_reject_reason`, which asks what those cutoffs were
standing in for:

| gate | what it protects against |
|---|---|
| endpoint A / B `local_probability` ≥ 0.85, direction match | the turn decisively picked *one* junction successor, not a coin toss (raw angle residual is only a diagnostic - a cut corner leaves 20–35° between the driver's line and the map's node) |
| endpoint residual ≤ 2.5 σ_pos + 80 m | the odometer places the junction inside the bounded match tolerance |
| `interior_low_confidence` = 0 | no shaky junction on the committed chain between the endpoints |
| `upstream_unresolved_forks` = 0 | no unresolved fork *before* endpoint A - the whole chain, and this interval's map length, could be on the wrong branch |
| `rolled_back_inside` = false | no contradiction rewound inside the segment |
| `|discrepancy|` ≤ 120 m | beyond this the likely explanation is a wrong route, not a wrong odometer |
| sanity bounds: 12 000 m / 1200 s | numerical / runtime only, far above the old values |

`IntervalQuality` diagnostics (per candidate, in the report): endpoint angle z,
endpoint local probability, endpoint residual, interior junction count,
interior low-confidence count, upstream unresolved forks, rolled-back-inside,
unambiguous-path, both-endpoints-turn-anchored, independent-event-count.

### The 2026-07-26 interval — 3367 m / 417 s

| criterion | value | verdict |
|---|---|---|
| endpoint A (ev at t 165, +51°) | matched edge 1558 (**truth**), p 0.999, residual −16 m | **clean** |
| endpoint B (ev at t 590, +42°) | matched edge 4326 (**truth**), **p 0.67**, residual −115 m, gyro +42° vs map +2° | **weak** |
| interior | 11 junctions, **1 low-confidence** (the 07-26 shallow fork, `1079→10266`) | fails |
| committed path A→B | single chain | ok |
| discrepancy | +99 m over 3367 m (+3 %) | ok |

**Answer to question 2:** the 2026-07-26 interval is **not** safe to use - but
not because it is 417 s long. Endpoint B is a low-confidence turn match
(p 0.67, the +42° gyro event matches no successor of that junction well) and
there is an unresolved low-confidence fork on the committed chain between the
endpoints. Both are real reasons the map length A→B could be wrong. A clean
long interval on other data would now pass
(`test_a_long_clean_interval_passes_the_quality_gate`).

## Task 3 — where the interval innovation actually goes

### The measurement equation (audited, correct)

```
state       x = [D, v, b_a, D_A, k_a]           D_A frozen at open_interval
measurement z = L_map
h(x)          = D - D_A          H = [1, 0, 0, -1, 0]
innovation    = L_map - (D - D_A)
```

This is `Δ D` over the interval, **not** absolute `D_B`
(`test_interval_is_integrated_distance_not_average_terminal_speed`). Propagation
Jacobians, finite-difference verified
(`test_interval_jacobians.py`):

```
∂(ΔD_step)/∂k_a = 0.5 · (a_meas - b_a) · dt²        # zero when not accelerating
∂(ΔD_step)/∂b_a = -0.5 · k_a · dt²
```

### Why +56 m to D but only ~+2 % to k_a — the audit

2026-07-22, interval ev0→ev1 (806 m map, 93 s):

| quantity | value |
|---|---|
| `Var(D_B − D_A)` (filter's own uncertainty over the segment) | 3281 m² (σ ≈ 57 m) |
| `σ_m` (map length + endpoint localization + timing) | 40 m |
| `S` | 4861, innovation 84 m ⇒ **z = 1.2** |
| `P[k_a, D_B−D_A]` accumulated since the anchor | **+1.18** |
| `P[b_a, D_B−D_A]` | +0.7 |
| `P[v, D_B−D_A]` | +2.1 |
| Kalman gains | `K_D = 0.68`, `K_v = 4e-4`, `K_ba = 1.4e-4`, `K_ka = 2.4e-4` |
| deltas | `ΔD = +56 m`, `Δv = +0.04`, `Δb_a = +0.012`, `Δk_a = +0.020` |

`K_i = P[i, D_B−D_A] / S`. The innovation informs a state only in proportion to
the covariance that state has *built with the interval quantity since the
anchor*. Over this segment the car mostly **cruises**: `a_meas − b_a ≈ 0` on
average, so `∂(ΔD)/∂k_a` accumulates almost nothing and `P[k_a, D_B−D_A]` is
+1.2 against a `Var(D_B−D_A)` of 3281. The filter honestly cannot tell a scale
error from a bias error from accumulated noise over a constant-speed stretch,
so 68 % of the innovation lands in D (the aggregate) and the small slivers of
correlation nudge v, b_a, k_a.

`Δk_a = +0.020` on interval 1 is *not* negligible; the headline "+1.1 %" is the
**net** after interval 2 (184 m, on the post-fork wrong route) pulls it back
`−0.009`. `σ_k_a` moves 0.251 → 0.246 → the interval does reduce scale
uncertainty, just slightly.

**The lever for real scale observability is a long interval that spans
acceleration** - stops, slow-downs at junctions. That is exactly what a clean
2026-07-26 interval would give, and why Task 2 matters. On this data no such
clean interval exists.

### Route-state consistency after a D correction

**Answer to question 5: the interval feedback did not damage the route.** The
committed decision sequence is byte-identical with intervals on and off
(`test_route_state_stays_consistent_after_an_interval_D_correction`,
and on the real 2026-07-22 trip: `real_wrong_local_decisions` = 13 both ways,
every incoming→chosen pair identical). The ~4-point drop in `truth_survival`
(edge-at-tick) is a **phase artifact**: correcting the odometer's *magnitude*
shifts the wall-clock time at which the unchanged route crosses each junction
relative to the 1 Hz output grid. `distance_ratio` (0.93 → 0.95) and
`real_wrong` (13 → 13) are the metrics that reflect what happened.

The mechanics that keep it consistent when the interval lands:

- `hs.offset_bias += (Δ − take)` and route position advances by `take`
  (`interval_offset_bias_fraction` of Δ, clipped) - because the interval is a
  better estimate of the same odometer-vs-map disagreement the per-junction
  event matches track in `offset_bias`. Folding it in rather than adding it is
  what prevents the double correction
  (`test_route_state_stays_consistent_after_an_interval_D_correction`).
- `hs.map_anchor_distance += Δ`, `manager._last_commit_distance += Δ`, and every
  unconsumed future turn's frozen `d_event += Δ` - all odometer-frame, all move
  with D so the next junction match and crossing tolerance are computed against
  a D that did not jump out from under them.

### Fixed-lag

`apply_interval` with `event_t=None` is a forward update on the current state;
`_update` additionally conditions the retained 12 s history through the stored
cross-covariances (the scalar fixed-lag Kalman update, unchanged). The interval
spans far more than the retained history, which the design permits - states
older than the lag are immutable outputs, and the global `D / v / b_a / k_a`
correction carries the interval's information forward from there. Tests at lag
0/5/10/20/30 s bound what is revised.

## Ablation matrix

`runs/iteration3/`. A: event-aligned only. B: + soft tier + provisional forks.
C: + intervals with the *old* hard 3000 m / 240 s gate. D: + intervals with the
quality gate. E: D with `interval_offset_bias_fraction` = 0. F: all.

`ccj` = correct consecutive junctions; `real wrong` = genuine wrong route
decisions (twin-edge artifacts excluded); `d60` = mean integrated distance
error over 60 s windows, m.

### 2026-07-22

| run | real wrong | first wrong t | ccj | survival | D/D_true | d60 | k_a | k_a σ | iv applied | soft | prov forks | switched |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A  event-aligned | 13 | 497 s | 20 | 0.204 | 0.930 | 28 | 1.000 | – | 0/3 | 0 | 2 | 0 |
| B  + soft + provisional | 13 | 497 s | 20 | 0.204 | 0.930 | 28 | 1.000 | – | 0/3 | 0 | 3 | 0 |
| C  + intervals, old hard gate | 13 | 476 s | 20 | 0.158 | **0.950** | 27 | **1.012** | 0.250 | 2/3 | 0 | 1 | 0 |
| D  + intervals, quality gate | 13 | 476 s | 20 | 0.158 | **0.950** | 27 | **1.012** | 0.250 | 2/3 | 0 | 1 | 0 |
| E  D with offset-bias fraction 0 | 13 | 476 s | 20 | 0.158 | 0.950 | 27 | 1.012 | 0.250 | 2/3 | 0 | 1 | 0 |
| F  all | 13 | 476 s | 20 | 0.158 | 0.950 | 27 | 1.012 | 0.250 | 2/3 | 0 | 1 | 0 |
| A  oracle D | 15 | 532 s | 20 | 0.298 | 1.000 | 0 | – | – | 0/1 | 0 | 2 | 0 |
| F  oracle D | 15 | 534 s | 20 | 0.298 | 1.000 | 0 | – | 0.250 | 0/2 | 1 | 4 | 0 |

### 2026-07-26

| run | real wrong | ccj | survival | D/D_true | d60 | k_a | k_a σ | iv applied |
|---|---|---|---|---|---|---|---|---|
| A  event-aligned | 0 | 30 | 0.104 | 0.837 | 119 | 1.000 | – | 0/1 |
| B  + soft + provisional | 0 | 30 | 0.104 | 0.837 | 119 | 1.000 | – | 0/1 |
| C  + intervals, old hard gate | 0 | 30 | 0.105 | 0.837 | 118 | 1.000 | 0.258 | 0/1 |
| D  + intervals, quality gate | 0 | 30 | 0.105 | 0.837 | 118 | 1.000 | 0.258 | 0/1 |
| F  all | 0 | 30 | 0.105 | 0.837 | 118 | 1.000 | 0.258 | 0/1 |
| A  oracle D | 0 | 39 | 0.857 | 1.000 | 0 | – | – | – |
| F  oracle D | 0 | 39 | 0.853 | 1.000 | 0 | – | 0.250 | 0/2 |

**Reading the matrix:**

- **B (Task 1) is metric-identical to A.** The soft tier and provisional forks
  flag one more junction on 07-22 (`prov forks` 2 → 3) and never switch. The
  14321 fork is not resolvable on independent evidence; the machinery correctly
  leaves the greedy choice standing. No harm.
- **C vs D (Task 2) accept/reject the same intervals** on this data - but for
  different reasons. Old gate: "duration 407 s outside bound", "length 3367 m
  outside bound". Quality gate: "segment drift −152 m (a wrong route)" for the
  07-22 candidate spanning the shallow fork, "endpoint B p 0.67" for the
  07-26 candidate. The quality gate would admit a clean long interval; none
  exists here.
- **C/D/E/F (Task 3) are identical:** two short clean intervals on 07-22 move
  `D/D_true` 0.930 → 0.950, `k_a` 1.000 → 1.012, `k_a σ` 0.258 → 0.250. `d60`
  is flat (28 → 27). `real wrong` stays 13 - the committed route is unchanged;
  the `survival` drop 0.204 → 0.158 is the edge-at-tick phase artifact
  (§ Task 3). `offset_bias` fraction (E vs D) changes nothing measurable.
- **2026-07-26 gets no interval at all** on production D - and it is the trip
  that needs one. Its odometer is far enough off that only ev0 and one
  widely-spaced later event match, and that later event is a weak anchor. `d60`
  stays ~118 m, `k_a` stays 1.000.
- Beam and its oracle ceiling (0.996 / 1.000) untouched; oracle-D single_path
  unchanged by any of A–F except the phase artifact.

**Selected production configuration: A (event alignment on, intervals off).**
The interval path is real and correctly signed but marginal on this data, and
stays behind `intervals.enabled`. B's provisional machinery is safe to leave on
(`soft_turn_enabled`, `provisional_fork_enabled` default true) - it never
switches wrongly and its flags feed the interval gate - but delivers no route
gain here.

## Answers

1. **Shallow fork 14321 resolved by the next TurnEvent?** No - both branches
   thread a consistent signed-turn sequence; the machinery correctly does not
   force a switch.
2. **Is the 2026-07-26 3367 m / 417 s interval safe?** No, and now for a
   defensible reason: endpoint B is a p 0.67 turn match with a −115 m residual,
   and a low-confidence fork sits on the committed chain. Not "417 > 240".
3. **Why did the old interval move D +62 m but k_a only +1.1 %?** The interval
   spans mostly-constant-speed road, so the covariance `k_a` builds with the
   interval quantity is ~0.04 % of the segment's distance variance; the
   innovation lands where the variance is - in D. Interval 1 alone moves k_a
   +2 %; interval 2 on the wrong-route stretch cancels most of it.
4. **After the corrected attribution, where does the innovation go?** `K_D`
   0.68, and v / b_a / k_a in proportion to their (small) accumulated
   covariance with `D_B − D_A`. It is correct EKF behaviour, not a bug - the
   only way to move more into k_a is a clean interval that spans acceleration.
5. **Did interval feedback break route survival, and is it fixed?** It never
   broke the route (identical decision sequence); the metric drop was a
   timing-phase artifact. `offset_bias` folding, `map_anchor_distance` /
   `d_event` shifting keep every odometer-frame quantity coherent with the
   revised D, with no double correction.
6. **D error reduction without hidden GPS.** 2026-07-22: `distance_ratio`
   0.93 → 0.95 (two clean short intervals). 2026-07-26: **0** - no clean
   interval exists on that trip; its odometer is far enough off that only the
   two widely-spaced turn events match, and one of them is a weak anchor.

## Not done (as instructed)

Mahony, spectral tuning, hardcoded k_a, GPS-derived scale, removing beam,
parallel hypotheses, global tolerance widening, accepting weak intervals for
metrics, aggressive rollback, weakening anti-circularity, D teleport,
whole-trip probability products.
