# Single-path forensics: why the one committed route first turns wrong

This iteration answers two questions and nothing else:

1. **Why does a single committed active route first choose the wrong branch?**
2. **Can bounded DFS/backtracking repair that one local miss without
   re-opening thousands of simultaneous hypotheses?**

The speed stack is not touched. Beam is not removed. Hidden GPS is used only
for benchmark truth labels and for the oracle-distance diagnostic run.

## Diagnostic fixes finished first

Two measurement bugs were closed before any analysis:

1. **A dead end no longer ends the replay.** When the active route runs out of
   road it stays a live length-1 population, frozen on its last edge, and the
   tracker keeps emitting frames to the end of the outage. The run is scored as
   stuck/wrong for the remainder instead of terminating early with an empty
   set. `single_path` on 2026-07-22 previously stopped at t+495 s; it now runs
   the full t+1055 s. (`tests/pacman/test_single_path.py::
   test_dead_end_does_not_end_the_replay`.)

2. **The truth label for a junction is the real next edge, not the GPS edge at
   the moment of a late decision.** `_single_path_evaluation` now walks the
   physical route traversal (`_truth_traversal`, the ordered list of distinct
   map-matched edges) in lockstep with the decisions. The true successor of a
   decision's incoming edge is the next entry in that traversal, with no
   reference to the tracker's crossing time — which, when the odometer lags, is
   already one or two junctions further on. The old code labelled the first
   2026-07-22 error at t+184.7 s (incoming 20640); that decision is in fact
   correct (20640 → 20635). The first genuine error is at t+266.9 s.
   (`test_truth_traversal_and_divergence_cause`.)

A third correction fell out of the analysis: the map-matched reference snaps a
few stretches to an OSM edge the drivable graph does not connect to (a
carriageway split — same street name, ~7–10 m from the real successor node).
The committed route continuing via the connected twin is not a route error.
`_twin_edge_artifact` flags these; `real_wrong_decision_count` excludes them.

## The first genuine wrong decision — 2026-07-22, t+266.9 s

Full forensic row (`single_path`, production odometer):

| field | value |
|---|---|
| decision time / crossing time | t+266.9 s / t+260.9 s |
| incoming active edge | 23038 (Пионерская ул., 168 m) |
| along-edge position at decision | end of edge (crossing) |
| estimated D at the turn | 835.5 m |
| oracle D at the turn *(diagnostic column only)* | 947.8 m |
| **D error** | **−112.3 m (odometer 12 % short)** |
| distance to junction along active route | +9.85 m (tolerance ±58.3 m) |
| **gyro turn event that really carried the car off 23038** | t_start 238.8 s, t_peak **240.6 s**, t_end 244.3 s, Δψ **+82.6°**, peak rate +0.56 rad/s — a clean, unambiguous left |
| truth left edge 23038 at | t+240.05 s (coincident with that event) |
| **decision was made** | **20.3 s / 112 m after the real turn** |
| measured Δψ in the tracker's crossing window [~257,265] s | **+2.4° ± 9.2°** (no turn — the window is past the corner) |

Outgoing candidates (turn Student-t, dof 4, σ 9.2°):

| edge | signed map turn | z = (map − measured)/σ | log-score | local prob |
|---|---|---|---|---|
| **20629** (chosen) | +8.2° | +0.59 | −0.21 | **0.9988** |
| 20627 (**true**) | +83.8° | +8.34 | −7.28 | 0.00085 |
| 20628 | −97.6° | −10.2 | −8.26 | 0.00032 |

- chosen edge: 20629
- true next edge: 20627 — **present in the candidate set, ranked 2nd**
- what killed it: the single score component (turn-angle mismatch). The true
  edge needs an 84° left; the misaligned window shows 2°, so it is 8.3 σ out.

**Cause: turn/junction temporal misalignment driven by an undershooting
odometer.** The gyro event is not weak or ambiguous — it is 82.6° at 0.56
rad/s. But the active route, 112 m behind the car, only walks up to the 23038
junction 20 s after the turn is over, and by then the crossing window contains
straight driving. The local score is *confident* (p = 0.9988) and *wrong*,
because the confidence is computed from the same misaligned measurement.

### A/B — production odometer vs oracle distance

Same tracker, same real gyro, distance handed over from the withheld GPS
(diagnostic only; the production input builder has no path that constructs it).

| | production D | oracle D |
|---|---|---|
| **2026-07-22** first genuine wrong junction | t+266.9 s (23038, the turn above) | **t+531.9 s** |
| 2026-07-22 correct consecutive junctions | 5 | **20** (t+152 → t+489) |
| 2026-07-22 truth-edge survival | 4.5 % | 29.0 % |
| 2026-07-22 permanent truth loss | t+241 s | t+528 s |
| **2026-07-26** genuine wrong route decisions (whole outage) | **0** | **0** |
| 2026-07-26 truth-edge survival | 11.3 % | **86.0 %** |
| 2026-07-26 permanent truth loss | t+604 s | **never** |
| 2026-07-26 D / D_true | 0.837 | 1.000 |

Reading:

- **2026-07-26 is decisive for question 1.** The committed single path makes
  **zero genuine route errors** across the entire outage on *both* runs. The
  only difference oracle D makes is positional: with production D the route is
  topologically correct but up to ~1.9 km behind in time-position, so the
  edge-at-tick survival metric reads 11 %; with oracle D it reads 86 % and the
  truth is never permanently lost. On this trip the route logic is already
  right and the odometer is the whole problem.

- **2026-07-22 has a residual on top of the odometer.** With oracle D the path
  is perfect through 20 junctions, then makes one genuine error at t+531.9 s:
  incoming 14321, true next 23129 (map turn −38.5°), chosen 23127 (map turn
  −12.4°). Measured Δψ = **−24.0° ± 9.2°** — it sits *between* the two branch
  angles. z is +1.3 for the chosen and −1.6 for the true edge; local
  probabilities 0.59 vs 0.30; the decision is already flagged low-confidence.
  This is a genuine turn-angle discrimination limit: the driver's line through
  the fork (−24°) is not close enough to either map angle, and ±9° of
  "driver vs map" sigma cannot separate a −12° fork from a −38° fork. Once it
  diverges here it never re-acquires.

So: **question 1 answer is case 1 (turn/junction temporal alignment from a bad
odometer) as the dominant cause, with a secondary case 3 (local turn-angle
scoring at a shallow fork) that only surfaces once the odometer is removed.**
The true candidate is almost always generated (case 4 is not the problem — the
only "successor not generated" rows are the map-match twin-edge artifacts). The
true candidate, when it loses, loses on the one turn-angle score term, and it
loses because the measurement feeding that term is either mistimed (case 1) or
genuinely between two options (case 3).

## Question 2 — bounded DFS/backtracking

The rollback machinery keeps one active route and, at each decision, immutable
dormant siblings (edge, local score/probability, input-state snapshot,
low-confidence flag). On a contradiction it rewinds to a recent decision with
an unused sibling, activates the next-best one, discards everything committed
after it, and continues. Bounds: `rollback_max_depth`,
`rollback_max_alternatives_per_junction`, `rollback_replay_window_s`,
`rollback_max_age_s`. It is depth-first with a single active branch — never a
parallel beam.

Two trigger regimes were measured.

**Conservative (shipped default, `rollback_require_low_confidence = True`).**
Only rewinds decisions the local score was already unsure about.

- 2026-07-22 production: **0 rollbacks.** None of the wrong decisions are
  low-confidence — they are confidently wrong (p ≥ 0.999), because the
  misaligned turn window makes "straight" look certain.
- 2026-07-22 oracle D: 2 rollbacks fire, late in the trip. Truth survival
  **unchanged** (29.0 %), first divergence **unchanged** (t+531.9 s).
- **Rollbacks that repaired a previously-wrong local decision: 0.**

**Aggressive (`rollback_require_low_confidence = False`, direction-matched
sibling, depth 3).** A decisive unexplained strong gyro turn may rewind a
confident decision.

- It fires often (3–7 times per trip) and is **strictly harmful**:
  2026-07-22 oracle D survival 29.0 % → 7.8 %; 2026-07-26 oracle D
  86.0 % → 33.4 %.
- Why: an unexplained turn tells you *a* junction was wrong, not *which* one.
  With the odometer lagging, the contradiction often belongs to a junction the
  tracker has not reached yet, or one already outside the depth window. And the
  "replay" after a rewind is a crude fast-forward — it re-commits the
  intermediate junctions against the same mistimed gyro windows, so it swaps
  one committed error for another.

**Answer to question 2: no.** Neither regime repairs the failure. The
conservative one never usefully fires because the error does not look
uncertain locally; the aggressive one fires but, without a real buffered-IMU
replay that re-derives D and re-scores every intermediate junction with
correctly-aligned windows, it makes route survival worse. Backtracking on the
route while the odometer stays wrong is rearranging the symptom.

## Matrix

`runs/iteration-forensic/`, default config, review map, both review trips.
`D` handed over from withheld GPS in the oracle-distance rows only.

`D end` / `D_true end` are metres of distance at the end of the outage;
`route len` is the committed decision count; `dormant` is the number of stored
unused siblings; `wrong` / `real wrong` are wrong local decisions before /
after removing the map-match twin-edge artifacts.

| trip | mode | D | runtime s | surv | top1 | first loss | perm loss | dead ends | D end | D_true end | intervals | route len | dormant | rb | wrong | real wrong | 1st real wrong t |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 07-22 | beam | prod | 45 | 0.689 | 0.023 | 172 | 917 | — | 2971 | 3194 | 0 | — | — | — | — | — | — |
| 07-22 | beam | oracle | 63 | **0.996** | 0.075 | 748 | never | — | 3194 | 3194 | 0 | — | — | — | — | — | — |
| 07-22 | single_path | prod | 1.5 | 0.045 | 0.045 | 147 | 241 | 1 | 2971 | 3194 | 0 | 11 | 14 | 0 | 6 | 6 | **267** |
| 07-22 | single_path | oracle | 1.0 | 0.290 | 0.290 | 147 | 528 | 0 | 3194 | 3194 | 0 | 35 | 32 | 0 | 18 | 15 | **532** |
| 07-22 | single_path_rollback | prod | 1.7 | 0.045 | 0.045 | 147 | 241 | 1 | 2971 | 3194 | 0 | 11 | 14 | **0** | 6 | 6 | 267 |
| 07-22 | single_path_rollback | oracle | 1.2 | 0.290 | 0.290 | 147 | 528 | 0 | 3194 | 3194 | 0 | 33 | 32 | **2** | 16 | 13 | 532 |
| 07-26 | beam | prod | 75 | 0.303 | 0.046 | 164 | 778 | — | 9797 | 11436 | 0 | — | — | — | — | — | — |
| 07-26 | beam | oracle | 88 | **1.000** | 0.077 | never | never | — | 11436 | 11436 | 0 | — | — | — | — | — | — |
| 07-26 | single_path | prod | 1.5 | 0.113 | 0.113 | 166 | 604 | 0 | 9578 | 11436 | 0 | 31 | 26 | 0 | 3 | **0** | never |
| 07-26 | single_path | oracle | 1.1 | **0.860** | 0.860 | 166 | never | 0 | 11436 | 11436 | 0 | 39 | 31 | 0 | 3 | **0** | never |
| 07-26 | single_path_rollback | prod | 1.6 | 0.113 | 0.113 | 166 | 604 | 0 | 9578 | 11436 | 0 | 31 | 26 | 0 | 3 | 0 | never |
| 07-26 | single_path_rollback | oracle | 1.1 | 0.860 | 0.860 | 166 | never | 0 | 11436 | 11436 | 0 | 39 | 31 | 0 | 3 | 0 | never |

No map-distance interval passes the independence gate on either trip
(`intervals` = 0 everywhere), unchanged from the previous iteration.

**Backtracking metric — rollbacks that repaired a previously-wrong local
decision: 0 of 2.** The 2 conservative rollbacks on 07-22 oracle D fired at
low-confidence decisions late in the run; neither moved truth survival or the
first-divergence time.

## Beam turn-event diagnostics, production vs oracle D

`turn_diagnostics` in each `beam` report, snapshot after each strong gyro turn
settles (`settle_s` 6.5). 2026-07-22:

| turn (Δψ) | | prod: uniq hist b→a | prod eff-hyp after | prod top1 | prod truth rank / mass | | oracle eff-hyp after | oracle top1 | oracle truth rank / mass |
|---|---|---|---|---|---|---|---|---|---|
| t+240.6 (+83°) | | 998→1517 | 26.7 | 0.32 | none / 0 | | 64.5 | 0.17 | 349 / 4e-5 |
| t+647.4 (+78°) | | 5995→5990 | 3012 | 0.05 | 1016 / 1.5e-3 | | **7.2** | **0.54** | 487 / 9e-5 |
| t+680.0 (−92°) | | 3579→4058 | 2054 | 0.003 | 1164 / 2.3e-4 | | **76.6** | **0.49** | 3717 / 2e-7 |
| t+913.7 (−115°) | | 5994→40 | 2.4 | 0.70 | none / 0 | | **2.8** | 0.53 | 1062 / 2e-8 |

The task's headline numbers (uniq hist 3579→4058, 461/4142 compatible signed
turns, top1 0.29 %, eff-hyp 2054, truth rank 1164, 344 deduplicatable) are the
**production** row for the t+680 turn and reproduce exactly.

What oracle D changes: the belief goes from **thousands** of effective
hypotheses to **single digits / tens** — a perfect odometer makes the turn
scoring decisive. But the truth stays low-ranked (487, 3717, 1062) with
~zero mass. **Oracle D makes the beam sharp and confident, not correct.** The
diffuse belief in production is partly an odometer problem, but the *ranking*
failure underneath it is not — with perfect distance the mass concentrates on
the wrong parallel route. `beam` still scores 0.996 / 1.000 truth *survival*
because the 6000-wide, 25-nat beam keeps the true edge alive in the long tail;
it just never ranks it (top1 0.075 / 0.077).

## Where this leaves the committed-path tracker

The committed path is not the weak link on this data — on 2026-07-26 it is
already exactly right. What it needs before it can hand map-distance intervals
to the D / k_a filter is:

1. **Odometer alignment at the turn, not the odometer's own timestamp.** The
   junction crossing should be scored with the gyro window centred on the
   *detected turn event* nearest the expected junction, not on the time the
   lagged D happens to walk the route up to the node. The turn event at
   t+240.6 s is the correct anchor; the decision fired it 20 s late.
2. **A real replay** if backtracking is ever revisited: buffered IMU
   re-consumed from the rewind point, D re-propagated, every intermediate
   junction re-scored — not the current fast-forward.
3. **More than instantaneous turn angle** at shallow forks (the t+531.9 s
   miss): turn *count* and turn *sequence* along the candidate routes.

Only (1) is a prerequisite for feeding intervals; (2) and (3) are separate.

**Follow-up (implemented):** (1) is done — see
[`EVENT_ALIGNED_JUNCTIONS.md`](EVENT_ALIGNED_JUNCTIONS.md). Junction scoring is
now driven by the preserved gyro turn event, matched to a reachable junction
despite odometer error. 2026-07-22 production single_path survival 0.045 →
0.204, first genuine wrong junction t+267 → t+497 s (the shallow fork of
Phase 3 / point 3 above). Turn-to-turn map-distance intervals are emitted and
can feed D / k_a behind a flag; on this data they stay off.
