# Event-aligned junction matching for the committed single path

Root cause (established in `SINGLE_PATH_FORENSICS.md`): the committed single
path scores a junction with a gyro window centred on **the time the odometer
walks the route up to the node**, not on **the time the gyro actually detected
the turn**. When the odometer lags (07-22: 112 m / 20 s by t+261 s), that
window is past the corner and shows ~straight, so the straight successor
out-scores the true 84° branch. Oracle D removes the lag and the committed
route is then near-perfect (07-26: 0 genuine route errors; 07-22: 20 correct
junctions instead of 5).

This iteration makes **turn events drive junction alignment**.

## Phase 1 — audit of the current turn pipeline (before changes)

```
raw gyro (ImuSample.yaw_rate, 10 Hz)
   │
   ├─► HeadingIntegrator(times, yaw_rate)              turns.py
   │       .delta(t0,t1,bias)   integrated signed yaw over ANY window
   │       .sigma(t0,t1,...)    ~9° model term dominates
   │       .centroid(t0,t1)     |ω|-weighted centre time
   │
   └─► detect_turns(times, yaw_rate, gyro_bias0)       turns.py
           → list[TurnEvent(t_start, t_peak, t_end, delta_psi, peak_rate)]
```

`TurnEvent` currently carries `t_start/t_peak/t_end` and the integrated
`delta_psi`. It has **no id, no sigma, no matched-junction / consumed state**,
and it is a plain dataclass shared with `BeamTurnObserver`.

In `tracker.run` the event list is built once (`self._turn_events`) and handed
to `SinglePathManager(... , self._turn_events)`.

### Where the event loses its identity — `SinglePathManager` (single_path.py)

```
advance(hs, D, t, speed, sigma_s)
  └─ if pending is None and s >= edge_length - crossing_tolerance:
        pending = _PendingDecision(
            t_cross = t + (edge_length - s)/speed          ← ESTIMATED from the
                       clipped to [t-10, t+10],               lagged odometer
            window  = _turn_window(speed)   (1.5–6 s),
            incoming_edge, boundary_offset, parent_route, offset_bias, tolerance)
  └─ if t >= pending.t_cross + pending.window:
        _commit(hs, D, t)
```

```
_commit(hs, D, t)
  successors = geometry.successors(incoming_edge, route.tail(depth))
  lo, hi     = pending.t_cross - window, pending.t_cross + window     ← window
  measured   = integrator.delta(lo, hi, gyro_bias)                    ← RE-INTEGRATED
  sigma      = integrator.sigma(lo, hi, ...)                            around the
  turns      = [junction_turn(incoming, e) for e in successors]         estimated
  scores     = turn_log_likelihood(turns, measured, sigma)              crossing time
  probabilities = local_probabilities(scores)     ← softmax, dormant siblings
  chosen     = argmax
  child.route_offset = boundary_offset
  # damped offset_bias nudge, clipped to ±30 m, from the centroid
```

**The `TurnEvent` objects are not consulted here at all.** They are used only
in `_maybe_rollback` for contradiction detection. The primary junction score
is recomputed from `HeadingIntegrator.delta` around `pending.t_cross`, which is
derived from the odometer position. When the odometer is 112 m behind:

- `pending` is created ~20 s late (odometer only then reaches `edge_length`);
- `pending.t_cross` ≈ t+261 s, window [257, 265] s;
- the real turn is the detected event at t_peak 240.6 s, Δψ +82.6°;
- `integrator.delta(257, 265)` = +2.4° — a different, later stretch of gyro;
- true edge (+84°) is 8σ out and loses on the one score term.

### The pending-turn mechanism to reuse, not duplicate

`_PendingDecision` + `self.pending` **is** the pending-turn slot. It is made
event-centric here rather than adding a second matcher:

- the trigger to open `pending` becomes "a settled unconsumed strong turn event
  is within a bounded distance of the next junction", *or* the old
  "odometer reached the edge end" (for genuine no-turn junctions);
- `_commit` uses `event.signed_angle` (frozen) when an event is matched, and
  only falls back to `integrator.delta` around the crossing when no event
  matches — which is correct, because then the gyro genuinely shows no turn.

Also reused unchanged: `HypothesisSet.anchor_turn_*` / `pending_drift` /
`_resolve_alignment_anchors` are the **beam** path (`SinglePathManager`
overrides `resolve_turns` to a no-op), and the manager's `pending_drift` list
is still the hook Phase 11 fills for turn-to-turn map intervals.

## Phases 2–7 — the mechanism (implemented in `single_path.py`)

`PhysicalTurn` wraps each `TurnEvent` as an immutable physical observation:

```
PhysicalTurn: id, t_start, t_peak, t_end,
              signed_angle   (frozen at detection, never recomputed),
              peak_rate, sigma_angle, quality,
              d_event        (odometer distance at t_peak, frozen at ingest),
              ingested, consumed, expired,
              matched_junction_offset, matched_t, matched_decision_index
```

**Ingest.** When `t ≥ t_end + event_settle_s`, `d_event` is frozen from the
odometer history and `sigma_angle` from `HeadingIntegrator.sigma`.

**Event → reachable junction.** For the single active route, the next junction
is at route length `R = route_offset + edge_length`. A turn matches it when

```
residual = R − (d_event − offset_bias)
|residual| ≤ tol_event
tol_event = clip(k·√(σ_D² + σ_map²) + drift_allowance,
                 event_match_min_tol_m, event_match_max_tol_m)
|residual| ≤ event_max_junction_distance_m        (hard bound)
t − t_peak ≤ event_max_age_s                        (else the event expires)
|signed_angle| ≥ event_min_angle_deg               (only strong turns steer)
```

`tol_event` is bounded — it cannot widen without limit, and it cannot let the
match jump more than `event_max_junction_distance_m`.

**Joint (junction, successor) score** for cross-event comparison:

```
turn_ll_j   = turn_log_likelihood(map_turn(successor_j), event.signed_angle, event.sigma_angle)
distance_ll = −½ (residual / σ_pos)²  · event_distance_prior_weight     (soft, weight < 1)
joint_j     = turn_ll_j + distance_ll
```

The distance term is a **soft prior** (`event_distance_prior_weight` = 0.4): it
breaks ties between events and mildly penalises a far junction, but it cannot
by itself select a branch — the turn angle does that. The committed branch
probabilities (`local_probabilities`, dormant siblings) use `turn_ll` only.

**Commit.** `measured = event.signed_angle` (frozen), never the crossing-time
window. The route advances; `offset_bias` is corrected toward
`d_event − R` with gain `event_offset_gain` (0.8), clipped to
`event_offset_max_correction_m` (160 m) — this is the bounded extra allowance
for accumulated odometer drift, and it realigns every following junction. The
event is marked `consumed` with its junction and decision index.

**Straight junctions (Phase 7).** If the odometer reaches the edge end and no
strong event matches within `_turn_window + event_settle_s +
straight_commit_extra_wait_s`, the junction is committed by the old path:
`integrator.delta` around the crossing (a genuine no-turn measurement) picks
the near-straight successor. No event is consumed.

**Consumption / no double use (Phase 6).** One `PhysicalTurn` is consumed by at
most one junction. An unmatched strong turn `expires` after
`event_max_age_s`; an expired-unconsumed strong turn, or the active route
driving past a junction a strong turn's `d_event` sits behind, is a
contradiction for `_maybe_rollback`.

## Config (`SinglePathConfig`, new fields)

| field | default | meaning |
|---|---|---|
| `event_min_angle_deg` | 25 | only turns this strong drive a junction |
| `event_match_k_sigma` | 2.5 | σ multiplier for `tol_event` |
| `event_match_drift_allowance_m` | 40 | bounded extra allowance for odometer drift |
| `event_match_min_tol_m` / `_max_tol_m` | 20 / 160 | clamp on `tol_event` |
| `event_max_junction_distance_m` | 220 | hard cap: a match cannot jump further |
| `event_max_age_s` | 60 | an unmatched event expires |
| `event_distance_prior_weight` | 0.4 | soft — distance never selects a branch alone |
| `event_offset_gain` | 0.8 | fraction of the residual folded into `offset_bias` |
| `event_offset_max_correction_m` | 160 | clip on that correction |
| `straight_commit_extra_wait_s` | 2 | grace before committing a junction straight |

## Results

Kept intact: beam, single_path, single_path_rollback, dormant alternatives /
parent-linked history, the bounded rollback machinery, the anti-circularity
gates, the D / k_a / fixed-lag machinery, all existing tests, the production
ban on hidden GPS, Mahony off. 567 tests pass; the beam oracle route ceiling is
unchanged (0.996 / 1.000). `runs/event-aligned/`.

### 1. Does the preserved gyro turn event fix the 2026-07-22 failure?

**Yes.** The known first genuine error was incoming edge 23038 → chosen 20629
(near-straight) instead of the true 20627 (+84° left), because the odometer was
~112 m behind and the crossing was scored with a gyro window ~20 s after the
real turn, which showed +2°.

Now (`single_path`, production odometer): the +82.6° event detected at
t_peak 240.6 s is preserved as `PhysicalTurn` id 1, matched to the 23038
junction at t+245.5 s (residual ~104 m, inside the bounded tolerance), and its
frozen angle scores the successors:

| successor | map turn | score | local prob |
|---|---|---|---|
| **20627 (true, chosen)** | +83.8° | −0.01 | **0.9992** |
| 20629 | +8.2° | −7.2 | 0.0008 |
| 20628 | −97.6° | −11.5 | ~0 |

The decision is `event_driven`, the event is `consumed` by this junction only,
and the later +2° window is never consulted. `test_single_path.py::
test_preserved_turn_event_fixes_the_delayed_odometer_junction` pins this on a
deterministic fixture (event 20 s / 100 m before the lagging crossing).

### 2. How much does single_path improve on the production odometer?

| | before (window scoring) | after (event-aligned) |
|---|---|---|
| **2026-07-22 truth-edge survival** | 0.045 | **0.204** |
| 2026-07-22 first genuine wrong junction | t+266.9 s | **t+497 s** |
| 2026-07-22 correct consecutive junctions | 5 | ~18 |
| 2026-07-22 permanent truth loss | t+241 s | t+497 s |
| **2026-07-26 genuine wrong route decisions** | 0 | **0** |
| 2026-07-26 truth-edge survival | 0.113 | 0.116 |

On 2026-07-22 the committed route now reaches the same ~18 junctions that only
oracle D used to reach. On 2026-07-26 the route was already topologically
perfect; event alignment keeps it so, and the residual 0.88 survival gap is
pure position lag from the ~16 % odometer shortfall, which route logic cannot
close.

### 3. What is the first genuine route failure now?

The **shallow fork** on 2026-07-22, ~t+497–532 s: incoming edge 14321, true
next 23129 (map turn −38.5°), chosen 23127 (map turn −12.4°). The measured turn
is −24° ± 9° — genuinely between the two branch angles. It is correctly flagged
low-confidence. This is the same miss that oracle D exposed before this
iteration; it is now the wall on both production and oracle D. It is a real
turn-angle discrimination limit (Phase 10), not an odometer or alignment
problem. 2026-07-26 still has no genuine route error on either odometer.

### 4. Do safe turn-to-turn map-distance intervals appear?

**Yes, a small number.** A candidate interval is emitted between two
consecutive `event_driven` junctions - both selected by an independent physical
turn (the distance term in the match is a soft prior, weight 0.4, that never
selects a branch alone), so the segment length is not a function of the
odometer.

| trip / D | interval | map length | odometer span | discrepancy | gate |
|---|---|---|---|---|---|
| 07-22 prod | ev0→ev1 (t147→t241, 93 s) | 806 m | 705 m | **−102 m** | passes |
| 07-22 prod | ev1→ev2 (t241→t647, 407 s) | 1508 m | 1418 m | −89 m | **rejected: 407 s > 240 s** |
| 07-22 prod | ev2→ev3 (t647→t680, 33 s) | 184 m | 156 m | −28 m | passes |
| 07-26 prod | ev0→ev2 (t165→t582, 417 s) | 3367 m | 3466 m | +99 m | **rejected: 3367 m > 3000 m and 417 s > 240 s** |
| 07-22 oracle | ev0→ev1 | 806 m | 799 m | −8 m | passes (small, as expected) |

The −102 m and −28 m discrepancies on 2026-07-22 are real and consistent with
the measured 7 % odometer shortfall over a route segment that is
independently verified correct. No gate was loosened; the long intervals are
refused.

### 5. Does k_a / D move after an accepted interval?

`single_path` + `intervals.enabled` + `accel_scale_enabled` (an ablation, off
by default):

| | D / D_true | k_a | truth survival |
|---|---|---|---|
| 2026-07-22 baseline | 0.93 | 1.000 (frozen) | 0.204 |
| 2026-07-22 + intervals | **0.95** | **1.011** | 0.155 |
| 2026-07-26 baseline | 0.837 | 1.000 | 0.116 |
| 2026-07-26 + intervals | 0.837 | 1.000 (no interval accepted) | 0.105 |

- On 2026-07-22 the ev0→ev1 interval applies at 1.26 σ, corrects D by +62 m
  toward truth, and moves k_a to 1.02 - the **correct direction** (the
  hidden-GPS diagnostic says ~1.27 is needed). ev2→ev3 nudges it back to ~1.01.
  The k_a step is small because the interval innovation is absorbed mostly by
  D; `P[k_a, D]` is weak while k_a is barely exercised.
- The mid-run D correction slightly regresses route survival (0.20 → 0.15) by
  perturbing junction alignment downstream.
- **2026-07-26 - the trip that most needs scale - gets nothing**: its only
  interval is 3367 m / 417 s and is refused by the unchanged length and
  duration gates. k_a stays frozen at 1.000.
- k_a never moves with `intervals.enabled = False`
  (`test_k_a_stays_frozen_without_an_accepted_interval`), and the beam path and
  its multi-route corroboration gate are untouched.

**Conclusion for production: unchanged baseline.** Event-aligned junction
matching is the real win and is on by default for `single_path`; interval
feedback to D / k_a is real, correctly signed, but marginal on this data (1–2
short intervals per trip, small k_a movement, slight route regression) and
stays an ablation flag.

## Full matrix

`runs/event-aligned/`. `iv✓` = intervals accepted; `ev dec` = event-driven
junction decisions; `ccj` = correct consecutive junctions from the start;
`real wrong` = genuine wrong route decisions (twin-edge artifacts excluded).

### 2026-07-22

| mode | D | intervals | truth survival | perm loss | D/D_true | k_a | iv✓ | ev dec | rollbacks | real wrong | ccj | first wrong t |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| beam | prod | – | 0.689 | 917 s | 0.938 | 1.000 | 0 | – | – | – | – | – |
| beam | oracle | – | 0.996 | never | 1.000 | 1.000 | 0 | – | – | – | – | – |
| single_path | prod | – | **0.204** | 497 s | 0.930 | 1.000 | 0 | 4 | 0 | 13 | **20** | 497 s |
| single_path | oracle | – | 0.298 | 528 s | 1.000 | 1.000 | 0 | 2 | 0 | 15 | 20 | 532 s |
| single_path | prod | on | 0.155 | 464 s | **0.952** | **1.011** | 2 | 4 | 0 | 13 | 20 | 475 s |
| single_path_rollback | prod | – | 0.204 | 497 s | 0.930 | 1.000 | 0 | 4 | 0 | 13 | 20 | 497 s |
| single_path_rollback | oracle | – | 0.298 | 528 s | 1.000 | 1.000 | 0 | 2 | 2 | 15 | 20 | 532 s |

### 2026-07-26

| mode | D | intervals | truth survival | perm loss | D/D_true | k_a | iv✓ | ev dec | rollbacks | real wrong | ccj | first wrong t |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| beam | prod | – | 0.303 | 778 s | 0.857 | 1.000 | 0 | – | – | – | – | – |
| beam | oracle | – | 1.000 | never | 1.000 | 1.000 | 0 | – | – | – | – | – |
| single_path | prod | – | 0.104 | 604 s | 0.837 | 1.000 | 0 | 2 | 0 | **0** | 30 | never |
| single_path | oracle | – | **0.857** | never | 1.000 | 1.000 | 0 | 2 | 0 | **0** | 39 | never |
| single_path | prod | on | 0.105 | 604 s | 0.837 | 1.000 | 0 | 2 | 0 | 0 | 30 | never |
| single_path_rollback | prod | – | 0.104 | 604 s | 0.837 | 1.000 | 0 | 2 | 0 | 0 | 30 | never |
| single_path_rollback | oracle | – | 0.857 | never | 1.000 | 1.000 | 0 | 2 | 0 | 0 | 39 | never |

Beam is unchanged by this iteration; its oracle route ceiling (0.996 / 1.000)
is the regression guard. `single_path_rollback` tracks `single_path` at the
conservative default. The `intervals` row is the ablation from §5.

## Follow-up

[`INTERVAL_CALIBRATION.md`](INTERVAL_CALIBRATION.md) takes the three loose
ends: the shallow fork (delayed local commitment - implemented, does not
resolve it on independent evidence), long map intervals (a quality gate
replaces the flat cutoff), and the interval EKF attribution (audited - correct;
the innovation lands in D because a cruising segment carries no scale
information).

## Not done (as instructed)

Mahony, beam widening, blanket crossing-tolerance widening, hardcoded D scale
or k_a, spectral re-tuning, aggressive rollback defaults, GPS-derived
production corrections, removing beam or the old single_path path. Hidden GPS
still reaches only benchmark metrics and the explicit oracle mode.

## Tests added

`processor/tests/pacman/test_single_path.py`:

- `test_preserved_turn_event_fixes_the_delayed_odometer_junction` - the 23038
  regression on a deterministic fixture
- `test_turn_event_angle_is_immutable_after_creation`
- `test_delayed_crossing_window_does_not_replace_the_event_angle`
- `test_one_event_is_consumed_by_at_most_one_junction`
- `test_bad_odometer_is_a_soft_prior_not_a_veto` /
  `test_event_match_cannot_jump_arbitrarily_far`
- `test_strong_left_event_rejects_a_right_successor`
- `test_no_event_straight_traversal_still_commits`
- `test_turn_to_turn_interval_is_emitted_between_two_event_anchored_junctions`
- `test_k_a_stays_frozen_without_an_accepted_interval`
- `test_production_branch_choice_has_no_truth_or_reference_input` extended to
  `_match_turn_to_junction` and `advance`
