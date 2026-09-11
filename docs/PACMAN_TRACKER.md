# Pacman tracker — road-locked route recovery

A second, independent implementation of the GPS-outage core, in
`processor/src/geotrace/pacman_tracker/`. It shares the project's
infrastructure — trip loader, `LocalFrame`, GraphML loading and clipping,
`RoadNetwork`, the IMU front end in `motion_model.build_imu_stream` — and shares
no algorithm with `road_ekf.py` or `particle_filter.py`, which stay in place
untouched as the reference.

## Two ideas

**A car is on a road.** So track *which road and how far along it*, not a free
`(x, y)` the map is then asked to forgive.

**There is one car.** So speed and distance are estimated **once**, globally,
and the road hypotheses consume them. They are not properties of a road, and no
hypothesis is entitled to its own opinion about how fast the car was going.

The second idea is a correction. The first version gave every hypothesis its own
`v`, and every hypothesis pulled that `v` toward whatever its own road's
curvature implied — so the map set the speed, and the speed then chose the map.
It also meant along-road uncertainty had to be represented by *spawning
siblings* along the road, which multiplied through every junction: 571 000
branches and 310 000 hypotheses culled by the beam limit on one 20-minute trip.

```
                              IMU
                               │
        ┌──────────────┬───────┴───────┬──────────────────┐
        ▼              ▼               ▼                  ▼
  stop detector   longitudinal   a_lat / yaw_rate   spectral model
        │            accel             │                  │
     v = 0          Δv between      precise v         approximate
   b_a, b_g          anchors          anchor            v ± σ
        └──────────────┴───────┬───────┴──────────────────┘
                               ▼
                    GLOBAL SPEED FILTER      [ D, v, b_a ]
                       D ± σ_D,  v ± σ_v
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
     Pacman propagation                    turn events
    s_i = D − route_offset_i        integrated yaw vs map angle
              └──────── road-graph scoring ─────┘
                               │
                      merge equivalent states
                               ▼
                       route probability
```

## The speed filter (`speed.py`)

    x = [ D, v, b_a ]         plus a global gyro bias, estimated at stops

    D    distance travelled since the outage began, m
    v    vehicle speed, m/s (signed)
    b_a  longitudinal accelerometer bias, m/s²

    v' = v + (a_x − b_a)·dt
    D' = D + v·dt + ½(a_x − b_a)·dt²

Four sources, in descending order of authority.

**Stop** — `v = 0` at σ 0.3 m/s, plus a re-measurement of both biases. The
strongest anchor there is, and GPS-free: a stopped car's longitudinal
acceleration and yaw rate stop *varying* (accel std 0.008–0.011 m/s² parked
against 0.08–0.30 driving). It is gated on physics, not statistics — a car
cannot be stopped now if it was doing 12 m/s during the second the detector
itself called quiet, because a quiet stretch is one in which the accelerometer
reported no braking.

**`v = a_lat / ω`** — circular motion. `a_lat = v²κ` and `ω = vκ`, so their
ratio is the speed, with no map, no integration and no fitted calibration. No
threshold decides whether it is usable; its variance does:

    σ_v² = σ_a²/ω² + a_lat²·σ_ω²/ω⁴ + σ_model²

so a barely perceptible drift in the steering yields a measurement too vague to
matter and a real turn yields a sharp one.

**Spectral model** (`spectral.py`) — speed regressed on log band powers of the
6-axis IMU over 2.56 s windows, ridge-fitted on the visible GPS with the sigma
measured on a held-out tail of it. Mediocre and continuous: it is what holds the
estimate together between turns.

**Longitudinal accelerometer** — propagation *between* anchors only. Its zero
point moves 0.65 m/s² between parked and driving on this recorder, so it is
never allowed to set an absolute speed.

Deliberately absent: the map. `v = ω/κ` is not used, for reasons measured below.

## Hypotheses (`state.py`)

    hypothesis = ( edge, route, route_offset, log-weight )
    s_i        = D − route_offset_i          σ_s = σ_D, shared

`route_offset` is the length of the route up to the start of the current edge. A
hypothesis whose route is 80 m shorter simply sits 80 m further along its
current edge. Along-road uncertainty is one number, so there is **no splitting
along the road** — the mechanism that was multiplying through every junction is
simply not needed.

## Turn events (`turns.py`, `manager.py`)

Branching is scored by the turn actually taken:

```
              left +87°
                 /
   ─────────────●─────────────  straight +3°
                 \
              right −91°

   gyro integrated over the crossing:  −84°
   → right strongly favoured, straight and left are not
```

This compares a number the gyro measures well — integrated yaw over a few
seconds, drift-free at that timescale — against a number the map knows well, the
angle between two edges at a node. Neither side needs the polyline's shape to be
right, and neither needs an instantaneous speed. The window is sized from the
car's own speed, the score is Student-t so one wide-swung corner is not fatal,
and measuring *no* turn is evidence too: it rules out turning.

Curvature matching (`curvature.py`) survives as a **scorer only** — how well
does this road explain the gyro — and moves weights, never the speed.

## Merging and pruning

Two hypotheses on the same directed edge at the same distance along it have the
same future whatever their pasts, so one representative carries the combined
mass (log-sum-exp) and the better route. This is what makes the population scale
with the number of distinct road states rather than exponentially with junctions
crossed. Pruning keeps everything within 25 nats of the best and needs 12
consecutive strikes before removing anything: the leading edge of the belief is
exactly the part that scores slightly worse and exactly the part that must
survive.

## Corridors (`corridor.py`)

`[s − k·σ_D, s + k·σ_D]` walked along each hypothesis's own route, buffered
±6 m, back onto the edge it came from and forward until the first fork, which it
will not guess past. `CONFIDENT` additionally requires the belief to occupy few
effective streets — `exp(H)` over per-edge mass — because a belief spread over
twenty streets is not confident whatever its favourite one looks like.

## Results on the two review recordings

Both are real 20-minute drives; the recorder gave the reconstruction the first
~145 s of GPS and withheld every later fix into `reference-samples.jsonl`. The
outage is 1055 s long in both. Run with `runs/review-map.graphml`, default
config, `runs/pacman/`.

| | 2026-07-22 | 2026-07-26 |
|---|---|---|
| distance actually driven (withheld GPS) | 3 194 m | 11 451 m |
| edges in the map clip | 26 860 | 15 432 |
| **ground-truth edge survival** | **55.4 %** | **80.6 %** |
| survival in top-1 | 0.4 % | 11.1 % |
| survival in top-3 | 1.8 % | 17.7 % |
| survival in top-5 | 2.7 % | 23.0 % |
| first ground-truth loss | t+39 s | t+27 s |
| permanent ground-truth loss | t+601 s | never |
| top-1 position error, median | 997 m | 1 486 m |
| top-1 position error, mean | 1 227 m | 3 661 m |
| some hypothesis within 25 m of truth | 4.1 % | 11.7 % |
| corridor coverage | 1.0 % | 24.4 % |
| corridor area, median | 3 243 m² | 18 181 m² |
| CONFIDENT / AMBIGUOUS / LOW_CONFIDENCE ticks | 10 / 996 / 48 | 19 / 570 / 466 |
| final population | 2 509 | 3 971 |
| tracker runtime | 62 s | 96 s |
| vibration odometer σ / calibration ρ | 5.1 m/s / 0.62 | 9.5 m/s / 0.64 |

For scale, the existing `road_ekf` / particle pipeline on the same two windows
reports display-track mean errors of 482–500 m and 4 320–4 418 m
(`runs/review-20min/summary.json`). Mean position error is *not* the metric this
rewrite optimises and the two are not directly comparable — the old pipeline
reports one track, this one reports a belief — but the numbers are the same
order, and the survival column is what is new.

### How the numbers got there

Every step below was driven by a specific measurement, not by sweeping
coefficients. The survival rate is the column to read.

| change | 07-22 | 07-26 |
|---|---|---|
| first working version | 2.2 % | 3.9 % |
| + odometer scale-error state, honest σ_s | 2.2 % | 3.9 % |
| + split the belief along the road early (σ_s > 12 m, every 3 s) | 2.2 % | 3.9 % |
| + drop the `−½·log S` term from the score | **49.1 %** | 2.6 % |
| + beam 1500 → 6000 | 54.8 % | 2.8 % |
| + reversing rail at 1.5 m/s, prune margin 14 → 25 nats | 55.4 % | **80.6 %** |

## Why the correct Pacman died

The death report (`report.json` → `death_report`, and the full per-step
`ground_truth_trace.json`) answers this directly for each run. Both deaths have
a cause, and neither is "the score preferred another street".

**2026-07-22, t+39 s.** The true hypothesis was *not* pruned. At t = 183.5 s it
was alive at rank 8, log-weight −1.4 against a −25 threshold, normalised RMS
0.21 — a better fit than the model expects. One tick later the car crossed onto
edge 23040 and no hypothesis was there. The belief was behind: `v` = 7.1 m/s
against a true 12–14, `σ_s` = 17 m against a real along-road error of ~60 m. It
died by being *outrun*. Two hypotheses further along the same street had been
deleted seconds earlier at −27.8 and −25.2 nats with unremarkable residuals
(RMS 0.53, 0.52) — the leading edge of the belief is exactly the part that
scores slightly worse and exactly the part that had to survive.

**2026-07-26, t+27 s.** This one *was* a pruning death, with a mechanical cause.
The true-edge hypothesis sat 70 nats below the leader with 74 consecutive
strikes. Before the fix its speed was **−2.6 m/s**: the under-reading odometer
plus the accelerometer's phantom deceleration had it reversing. Speed enters the
measurement as `ω_map = v·κ`, so a negative `v` predicts the road turning *the
wrong way*, and the hypothesis then contradicted the gyro on every bend and was
convicted of being on the wrong street. Railing reversing at 1.5 m/s took this
trip from 2.8 % to 80.6 % survival and moved permanent loss from t+27 s to
never.

## What is physically not there

Three findings about the data itself, measured against the withheld GPS. None
is a tuning problem and none should be hidden by tuning.

**The accelerometer cannot produce distance.** Its zero point differs by about
0.65 m/s² between parked (+0.45) and driving (−0.19) — it is not a bias but the
attitude filter's tilt error — and it wanders ±0.3 m/s² within a hundred
seconds. Open-loop integration gives +18 m/s of speed error and 19 km of
distance error over a 3 km drive. Removing the accelerometer channel entirely
changes the reconstructed distance by under 15 %: on this recorder it is
contributing almost nothing either way.

**The vibration odometer is a stop detector, not a speedometer.** Its headline
correlation of 0.62–0.64 is computed over all samples and comes almost entirely
from telling stopped apart from moving — which the ZUPT detector already does.
*Conditional on the car moving*, its correlation with speed is **0.13** on 07-22
and **−0.01** on 07-26. It emits a near-constant value for every speed above
walking pace:

| true speed (07-26) | 0–2 | 2–6 | 6–10 | 10–14 | 14–20 | 20+ |
|---|---|---|---|---|---|---|
| predicted | 6.2 | 8.8 | 9.9 | 9.7 | 9.4 | 9.0 |

This is *not* a calibration-coverage problem: 07-26's visible window has p90 =
15.4 m/s and 53 % of it above 10 m/s, against 61 % in the outage. The window
covers the range; the features simply do not carry the speed. Its integrated
distance nevertheless lands close on 07-22 (3 151 m of a true 3 194 m), which is
a coincidence of that trip's speed profile — a constant predictor at the right
mean integrates to the right total while knowing nothing about *when* the car
was fast, which is exactly what along-road position needs.

**Map curvature can only rescue it on 1.5 % of steps.** On the true road, |κ|
has a median of 0.00000 and a p90 of 0.00057 rad/m — Petersburg streets are
straight. Only 1.5 % of moving steps have |κ| > 0.002, and even there
`v = ω/κ` has a median error of −5.8 m/s, because the map's smoothed corner is
tighter than the line a driver actually takes.

**But the IMU does contain a good speedometer, and it is not either of those.**
A car in a turn is in circular motion: `a_lat = v²κ` and `ω = vκ`, so
`v = a_lat/ω` — no map, no integration, no fitted calibration. Measured against
the withheld GPS at |ω| > 0.05 rad/s with 1 s smoothing:

| | 2026-07-22 | 2026-07-26 |
|---|---|---|
| availability | 7.1 % of samples | 7.3 % |
| correlation with true speed | **0.91** | **0.88** |
| median error | −0.28 m/s | −0.73 m/s |
| within 3 m/s | 97 % | 78 % |
| at \|ω\| > 0.2: correlation / within 3 m/s | 0.92 / 100 % | 0.99 / 100 % |

The small negative bias is tyre slip and body roll taking a share of the lateral
acceleration out of the horizontal plane. This is now the measurement the speed
estimate rests on (`MotionEstimator.lateral_speed_update`); the vibration
odometer is demoted to holding the speed off zero between turns.

**Top-1 is not recoverable from this data.** Survival is 55 % and 81 %; top-1
survival is 0.4 % and 11 %. On a Saint Petersburg grid, several parallel streets
of the same length and the same (near-zero) curvature produce the *same*
predicted yaw-rate profile at the same speed. The gyro cannot separate them —
not because the matcher is weak, but because the measurement genuinely does not
distinguish them, which is the same fact the straight-road regression test pins
down deliberately. The right output there is several narrow corridors labelled
AMBIGUOUS, which is what it produces: 996 of 1054 ticks on 07-22.

## What is still wrong

* **Top-1 selection is near-chance** (0.4 % / 11 %). Ranking is currently driven
  by gyro residuals alone. Turn *counts* and turn *sequence* along a route carry
  much more discriminating power than instantaneous rate matching and are not
  used at all yet.
* **07-22 loses the truth permanently at t+601 s** and never gets it back. The
  belief and the car are in different parts of the city by then; there is no
  re-acquisition mechanism.
* **Corridor coverage is 1 % on 07-22** while edge survival is 55 %. The truth is
  in the *population* but not in the four corridors drawn. The corridor builder
  reports the top few edges by mass; with a belief spread over dozens of streets
  that is a fair summary of the leader and a bad summary of the belief.
* **`non_finite_dropped` is 37 and 256.** The guard catches them and records
  them, but hypotheses whose covariance arithmetic comes apart are a bug, not a
  fact about the data. Not yet chased down.
* **Runtime is 62–96 s per 20-minute trip** at a 6000 beam, dominated by
  branching and merging in Python. Fine offline, far too slow for a phone.
* **The odometer is calibrated once, on the visible window.** A 145 s warmup that
  contains no fast driving cannot calibrate the top of the speed range, which is
  exactly 07-26's problem. Recalibrating during the outage from
  curvature-derived speeds would be the obvious next step - the scale state `k`
  is already there to receive it.
* **Reversing across a junction** transfers to the route's own parent edge only.
  A hypothesis that reverses onto an edge it did not arrive from is not modelled.
* **The committed single path first turned wrong because the odometer lags, not
  because the branch score is weak** — now fixed by event-aligned junction
  matching ([`EVENT_ALIGNED_JUNCTIONS.md`](EVENT_ALIGNED_JUNCTIONS.md)). A
  detected gyro turn is preserved as an immutable observation and matched to a
  reachable junction despite odometer error; its frozen angle scores the
  successors instead of a gyro window at the lagged crossing time. 07-22
  production single_path survival 0.045 → 0.204, first genuine wrong junction
  t+267 → t+497 s. Forensics of the original failure remain in
  [`SINGLE_PATH_FORENSICS.md`](SINGLE_PATH_FORENSICS.md).
* **The remaining single_path wall is a shallow fork** (07-22 t+497 s): a −24°
  driven turn between −12° and −38° map branches. A genuine local ambiguity.
  Delayed local commitment (soft turn tier, provisional forks, sequence
  scoring - [`INTERVAL_CALIBRATION.md`](INTERVAL_CALIBRATION.md) Task 1) is
  implemented and correctly does **not** resolve it: both branches thread a
  consistent signed-turn sequence, and the only separating feature is
  elapsed-time-vs-map-length, which is a D-agreement discriminator and off
  limits. Not forced.
* **Turn-to-turn map-distance intervals calibrate the odometer, marginally.**
  [`INTERVAL_CALIBRATION.md`](INTERVAL_CALIBRATION.md): the quality gate
  (endpoint turn-match confidence, interior forks, drift) replaces the old flat
  duration/distance cutoff. On 07-22 two short clean intervals move
  `D/D_true` 0.93 → 0.95 and `k_a` 1.00 → 1.01; the innovation lands mostly in
  D because a cruising segment carries almost no scale information. 07-26 - the
  trip that needs scale - has no clean interval. Behind `intervals.enabled`,
  off by default.
* **Bounded DFS backtracking does not repair a confident wrong junction** — the
  wrong branch does not look uncertain, and rewinding the route while D is
  wrong swaps one committed error for another. `single_path_rollback` stays a
  conservative superset of `single_path`.

## Tests

`processor/tests/pacman/`, 56 tests, ~4 s.

| file | what it holds the code to |
|---|---|
| `test_roadmap.py` | κ = 1/R on an arc, 0 on a straight; smoothing spreads a corner but preserves `∫κ ds = Δθ`; junction turns are signed and wrapped; the integral is independent of sampling step |
| `test_straight_road.py` | **the regression this rewrite exists for.** On `κ = 0`: `σ_s` does not shrink, `s` does not move, three hypotheses 100 m apart stay exactly equal in weight — and the gyro bias *is* still learned. Plus the converse: a bend and a junction bump do carry along-road information |
| `test_scale_error.py` | `σ_s` grows ~linearly with distance, not as its square root; turning the scale prior off collapses it; a bend measures the scale; a straight road teaches it nothing |
| `test_motion.py` | constant acceleration integrates exactly; bias is subtracted; uncertainty grows open-loop; ZUPT pulls `v` to 0 **without** collapsing the variance; the envelope only fires outside the band; signed velocity works; the stop detector separates idling from driving and needs a sustained quiet stretch |
| `test_junction.py` | the A/B/C/D junction from the brief: every legal successor is spawned, the gyro picks the branch actually driven (>0.8 of the mass), going straight picks straight, **one bad sample does not kill the correct branch**, turn restrictions are honoured |
| `test_manager.py` | merge preserves total mass; hypotheses that disagree are not merged; pruning needs sustained badness; a recovering hypothesis is not pruned; beam keeps the best; the population floor holds; splitting preserves mean and variance; dead ends drop; reversing returns to the previous edge; prune events record why |
| `test_corridor.py` | corridor length tracks `σ_s`; a 100 m σ gives a ribbon, not a 200 m disc; a split belief is AMBIGUOUS with narrow branches; a hopeless belief says LOW_CONFIDENCE instead of widening; **a belief spread over many streets is never CONFIDENT**; the corridor spills back along its own route and stops at a fork |
| `test_leakage.py` | **hidden GPS cannot reach the reconstruction.** Scramble the withheld fixes kilometres away, or delete them entirely, and the frame-by-frame output is byte-identical. Includes a guard-the-guard test (the scenario really does exercise the tracker) and its converse (corrupting the reference *does* change the metrics, so the first test is not passing vacuously) |

The leakage property was also checked end to end on both real 20-minute
recordings, not only on the synthetic stand-in. Each trip was reconstructed
three times — as recorded, with `reference-samples.jsonl` scrambled by ~5 km
with a random speed and course per fix, and with it deleted outright — and the
three runs give one identical SHA-256 over all 1056 output frames:

| trip | digest over all three runs |
|---|---|
| 2026-07-22 | `e6c4ef19f9b649b6654f10b6dcd5ef05c5494598f6b46b42a2d07f23ae6db7f5` |
| 2026-07-26 | `f12ed4190df4baa18f8ea567716ee832735e1df942b20ce9cc67e3cc929fb5a1` |

```bash
cd processor && .venv/bin/python -m pytest tests/pacman -q
```

## Running it

```bash
processor/.venv/bin/python -m geotrace.pacman_tracker.benchmark \
  --trip runs/review-20min/2026-07-22/trip \
  --trip runs/review-20min/2026-07-26/trip \
  --graph runs/review-map.graphml \
  --output runs/pacman
```

```bash
cd processor && .venv/bin/python -m pytest tests/pacman -q
```
