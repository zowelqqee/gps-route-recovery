# Magnetometer forensics — can it give an independent absolute heading?

**Verdict: C — NOT USEFUL** as a reliable route/junction anchor, with a narrow
"occasionally useful" asterisk on 2026-07-22 that does not survive contact with
the one place it is needed.

Investigation only. Nothing is wired into the EKF, route scoring, `single_path`,
or map intervals. Production architecture unchanged; 582 tests still pass.

## The headline facts

1. **The two review trips carry no magnetometer at all.** `samples.jsonl` for
   2026-07-22 and 2026-07-26 has `motion` (accel, gyro, gravity, quaternion)
   and `location` only. `MotionSample` *can* hold `magnetic_field` /
   `magnetic_accuracy`, and the day-log importer deliberately drops them
   ("none is used, and carrying them would double the memory a session needs" —
   `live_logs.py`).
2. **The raw day logs do have it.** `live_logs/imu_logs/2026-07-22_IMU_logs.csv`
   and `..._07-26_...csv` — the vehicle-logger box's own files — contain
   `mag_x, mag_y, mag_z` at **100 Hz**, alongside `acc_*`, `gyr_*`,
   `roll, pitch, yaw`, `quat_*`, `temp`, `pressure`. This analysis reads the
   magnetometer straight from those CSVs and aligns it to trip monotonic time
   via the window epoch in `metadata.json`.
3. **The recorder does not use the magnetometer.** Its `yaw` column is the
   plain integral of `gyr_z` (correlation with `∫gyr_z` = 0.999999, max
   divergence 0.22° over 300 s), and `attitude_source` is
   `accelerometer_levelled_ahrs`. So the magnetometer is genuinely independent
   and unused — the question is whether it *could* help.
4. **After the best calibration available without GPS, magnetic heading error
   is ~18° median / ~31° p90 on 2026-07-22 and ~30° median / ~58° p90 on
   2026-07-26** — both far outside "useful", and at the 2026-07-22 shallow fork
   the magnetometer points at the *wrong* branch.

## Phase 1 — raw data audit

| property | value |
|---|---|
| fields | `mag_x, mag_y, mag_z` (day CSV only; absent from imported trips) |
| units | µT-scale (see Phase 2); **raw / uncalibrated** |
| sample rate | 100 Hz (review trips downsample IMU to 50 Hz and drop mag) |
| timestamps | UTC epoch ms, shared with `acc/gyr`; trip clock offset to GPS 4.85 s (07-22) / 4.9 s (07-26), gyro-vs-course correlation 0.75 / 0.96 |
| coordinate frame | device/body frame, same as `acc`/`gyr`; mount is rigid, tilt ~1.5–2° (from `gravity_device`) |
| dropouts | none — 0 stale/held triples, 1–4 corrupt CSV rows out of 300 k+ |
| clipping | none — no axis pins at a rail |
| already calibrated? | **no** — hard-iron offset is larger than the field itself (Phase 4) |
| recorder AHRS uses mag? | **no** — `yaw` = ∫`gyr_z` exactly |
| magnetic accuracy / status flags | **none** in the vehicle-logger CSV (iPhone recordings have `magnetic_accuracy`; this recorder is a wired box, not a phone) |
| hard/soft-iron metadata | **none** |
| device frame fixed to car frame | yes — `mount_estimate` coherence 0.99, heading spread ≤ 2° |

## Phase 2 — field quality (trip windows only)

`B = ‖(mx,my,mz)‖`. Saint Petersburg total field ≈ 52 µT.

| | 2026-07-22 | 2026-07-26 |
|---|---|---|
| median \|B\| | **43.5** | **27.6** |
| p05 / p95 \|B\| | 28.6 / 54.7 | 23.6 / 44.4 |
| min / max \|B\| (in window) | 16.8 / 139.7 | 17.0 / 156.8 |
| robust band (3·MAD) exceeded by | 0.5 % of samples | **7.5 %** |
| rolling 4 s std(\|B\|), median / p90 | 1.1 / 6.3 | 0.8 / 2.6 |
| \|B\| stopped vs moving (median) | 39.8 vs 44.2 | 32.5 vs 26.8 |
| \|B\| by speed bin 0-2 / 6-10 / 10-15 m/s | 41 / 45 / **31** | 32 / 26 / 29 |
| \|B\| by 200 s bin | 34,46,46,47,47,34 | 37,29,26,31,25,25 |
| local std(\|B\|) turning vs straight | 1.3 vs 0.8 | 1.3 vs 0.7 |
| sudden jumps > 5 µT / sample | 59 | 38 |

Both trips read **well below the true field** and vary ±20–50 %. `|B|` changes
with speed, with stop/go, and drifts over the trip. The 2026-07-26 field is
both weaker and dirtier (7.5 % gross outliers, `mag_y` excursions to −150).
This is severe vehicle-body contamination, not a clean Earth field.

## Phase 3–4 — calibration feasibility

Tilt-compensated with the recorder's `roll/pitch` (tilt is tiny, so this barely
matters). Orientation coverage: **42 %** of 10° yaw bins on 07-22, **58 %** on
07-26 — a car does not rotate freely, so a full 3-D ellipsoid fit is poorly
observable, as expected. The 2-D horizontal analysis:

| | 2026-07-22 | 2026-07-26 |
|---|---|---|
| crude 3-D hard-iron offset (µT) | (38, −20, −9) | (28, −39, −19) |
| axis-scale anisotropy | 3.5 | **8.5** |
| 2-D horizontal ellipse axis ratio | 2.1 | 2.4 |
| radius std after best 2-D ellipse fit (ideal 0) | **0.75** | 0.35 |
| raw heading 1 s-window noise, straight driving | 2.4° | 1.6° |

The hard-iron offset is comparable to or larger than the field. `mag_y` carries
a scale 3–8× the other axes — a switching disturbance, not mild soft iron. The
2-D ellipse fits, but the **residual radius std of 0.35–0.75** means that even
after the best fixed 2-D hard+soft-iron correction the field vector still points
in the wrong direction by a large, varying amount: the distortion is not a
fixed ellipse.

Short-term heading *noise* is only 1.6–2.4°. The sensor is precise. The problem
is entirely bias.

## Phase 5–6 — heading error vs independent references, and bias stability

Reference heading = withheld GPS course at speed > 4 m/s (diagnostic truth
only). Calibration (2-D ellipse + one rotation absorbing mounting yaw +
declination + hard-iron rotation) fitted on the **first 145 s** — the real
GPS-visible window — then held for the rest of the outage.

| method (calib from t < 145 s) | avail | MAE | p90 | p95 | circ bias | <20° |
|---|---|---|---|---|---|---|
| **2026-07-22** raw heading | 35 % | 27° | 64° | 74° | −8° | 31 % |
| 2026-07-22 hard-iron only | 35 % | 59° | 115° | 128° | +8° | 11 % |
| 2026-07-22 **2-D ellipse** | 35 % | **18°** | **31°** | 41° | −1° | 62 % |
| **2026-07-26** raw heading | 73 % | 140° | 174° | 177° | −135° | 11 % |
| 2026-07-26 hard-iron only | 73 % | 66° | 117° | 127° | +34° | 24 % |
| 2026-07-26 **2-D ellipse** | 73 % | **30°** | **58°** | 70° | −24° | 15 % |

**Bias stability** (2-D ellipse, one global rotation, moving windows):

| window | 2026-07-22 bias range / std | 2026-07-26 bias range / std |
|---|---|---|
| 30 s | [−39°, +14°] / 13° | [−177°, +173°] / **66°** |
| 60 s | [−31°, +10°] / 12° | [−174°, +173°] / **75°** |
| 120 s | [−20°, +10°] / 11° | [−178°, +1°] / 58° |

2026-07-22: the bias sits within ±15° for the first ~9 minutes, then **swings
to −31° around t+600 s** (MAE 40°+ for ~2 minutes) before recovering.
2026-07-26: the bias is unusable — it wanders across the full circle and
includes a **sustained ~180° reversal from t+600 to t+720 s** (MAE ~165° for
two minutes).

`corr(bias, |B|)` is +0.15 (07-22) / +0.33 (07-26) — weakly related to field
magnitude, nowhere near predictive. `corr(bias, speed)` ≈ 0.

**The two trips, same recorder, behave completely differently.** 2026-07-22 is
marginal; 2026-07-26 is hopeless. Per-trip reliability cannot be assumed.

### The diagnostic ceiling

If the calibration rotation could be re-fitted every 60 s against continuous
truth (impossible during an outage), the *sensor* is capable of:

| | MAE | p90 |
|---|---|---|
| 2026-07-22 ceiling | 8° | 27° |
| 2026-07-26 ceiling | **2°** | 41° |

So 2026-07-26's magnetometer is *locally* excellent — its error is almost
entirely a slow spatial bias that tracks where the car is (different magnetic
environments along the route). Tracking that needs an absolute reference the
outage does not provide. 2026-07-22's ceiling is worse (8°), meaning it has a
genuine ~8° short-timescale distortion no calibration removes, plus a fat
p90 tail.

## Phase 7 — GPS-free disturbance detector

Candidate quality signal from mag alone: `|B|` deviation from a 15 s trailing
median < 10 %, `|d|B|/dt|` < 3 µT/s, and `|ψ̇_mag − (−gyr_z)|` (mag heading rate
vs gyro rate, 2 s smoothed) < 4°/s.

| | all moving | quality-gated |
|---|---|---|
| **2026-07-22** avail / MAE / p90 | 35 % / 18° / 31° | **1 % / 15° / 28°** |
| **2026-07-26** avail / MAE / p90 | 73 % / 30° / 58° | 3 % / 29° / 54° |

The gate helps 2026-07-22 a little — it trims the tail — but only down to
15° MAE / 28° p90, and at ~1 % availability that is one or two heading samples
per 20-minute trip. On 2026-07-26 the gate cannot find good samples: the
distortion is not a transient burst, it is a persistent slow bias, so the mag
vs gyro-rate consistency check passes while the *absolute* heading is 30°+ off.

## Phase 8 — the 2026-07-22 shallow fork

`incoming 14321` → candidate `23127` (map turn −12.4°, start course +50.6°) or
`23129` (**truth**, map turn −38.4°, start course +76.7°). Gyro integrated turn
≈ −24°. Truth leaves 14321 at t ≈ 527 s (the odometer had already committed the
junction 34 s earlier).

| | value |
|---|---|
| calibrated magnetic heading, t+3…12 s after the real turn | **+48°** |
| candidate 23127 start course / mag error to it | +51° / **−2°** |
| candidate 23129 (truth) start course / mag error to it | +77° / **−29°** |
| magnetic Δheading across the turn | **+12°** (wrong sign) |
| gyro Δheading across the turn | −24° (correct) |
| \|B\| at the fork vs 2-min nominal | +3 % (looks clean) |

**The magnetometer picks 23127 — the wrong branch — by a 27° margin, and its
relative heading change has the wrong sign.** The bias at this point in the
trip (~t+527, near the t+600 excursion) is enough to invert a 26°-apart fork.
`|B|` looks clean, so a disturbance gate would *pass* this sample and hand the
route manager confident wrong evidence.

**Answer: no, the magnetometer could not resolve the shallow fork. It is worse
than useless here — it would actively mislead.**

## Phase 9 — 2026-07-26 endpoint opportunities

2026-07-26's route topology is already correct; its weak points are turn-event
endpoints with low local confidence (e.g. the p 0.67 match at t+590 s in
`INTERVAL_CALIBRATION.md`). With the 145 s-window calibration, magnetic heading
error at the three matched turn events is −22° / −8° / −21°, and the median
moving error after t+200 s is ~30–100°. Adding this to endpoint scoring would
inject 20–100° of noise with no way (without truth) to know which samples are
the good ones.

| | gyro-only weak endpoints | improved by mag | correct improvements | false-confidence added |
|---|---|---|---|---|
| 2026-07-26 | ~3 (per iteration-3 analysis) | **0** | 0 | would add on most |

## Phase 10–11 — frame and declination

The device→vehicle yaw is already pinned by the gyro warm-up: `mount_yaw_deg`
−1.55° (07-22) / −4.11° (07-26), coherence 0.99, heading spread ≤ 2°. The
gyro world frame → map course offset (`world_yaw_offset_deg`, −118.4° / +127.7°)
comes from the GPS warm-up window. Saint Petersburg magnetic declination for
2026 (WMM) ≈ +11.4° E.

Decomposition of the magnetic heading:

```
ψ_map  ≈  ψ_mag_sensor  +  mounting_yaw  +  declination  +  calibration_rotation
             (measured)     (~−2 to −4°,      (~+11.4°,       (hard-iron ellipse
                             known from gyro)  known)          rotation — NOT
                                                               identifiable
                                                               without truth,
                                                               and not constant)
```

`mounting_yaw` and `declination` are known. The **calibration rotation is the
unidentifiable term**: it is entangled with the hard/soft-iron ellipse, it is
not constant (Phase 6), and there is no GPS-free way to fix it or track it
during the outage. The gyro + GPS-window already establish device→map heading
to < 2° — the magnetometer is not needed for the mounting problem and cannot
beat it.

## Required tables — summary

### Per trip, heading error (GPS-course reference, speed > 4 m/s)

| calib method (from t<145 s) | trip | avail | MAE | p90 | p95 |
|---|---|---|---|---|---|
| raw | 07-22 | 35 % | 27° | 64° | 74° |
| hard-iron | 07-22 | 35 % | 59° | 115° | 128° |
| 2-D ellipse | 07-22 | 35 % | 18° | 31° | 41° |
| 2-D ellipse + quality gate | 07-22 | 1 % | 15° | 28° | 31° |
| raw | 07-26 | 73 % | 140° | 174° | 177° |
| hard-iron | 07-26 | 73 % | 66° | 117° | 127° |
| 2-D ellipse | 07-26 | 73 % | 30° | 58° | 70° |
| 2-D ellipse + quality gate | 07-26 | 3 % | 29° | 54° | 57° |

### 2026-07-22 shallow fork

| | heading (course frame) | mag error |
|---|---|---|
| calibrated magnetic (post-turn) | +48° | — |
| candidate 23127 | +51° | −2° |
| candidate 23129 (**truth**) | +77° | **−29°** |
| gyro-only winner | 23129 correct branch not separated (fork stands) | |
| mag-only winner | **23127 (wrong)** | |
| fused winner | 23127 (mag drags it wrong) | |

## The two questions

**Is there a stable independent absolute-heading signal usable as a rare
quality-gated anchor?**
No. 2026-07-22 after the only calibration available without GPS gives 18° MAE
/ 31° p90; the quality gate trims that to 15° / 28° but at ~1 % availability and
with a bias that swings 40° mid-trip. 2026-07-26 — the trip that actually needs
anchors — is unusable at any calibration (30–140° MAE) because its bias wanders
across the whole circle, including a two-minute 180° reversal, and the
disturbance is a slow spatial bias that no GPS-free gate can catch. And the
signal is not even in the trip data format.

**Could it resolve the 2026-07-22 shallow fork or raise 2026-07-26 endpoint
confidence?**
No. At the shallow fork the calibrated magnetic heading is 27° closer to the
*wrong* candidate and its relative turn has the wrong sign; a `|B|` gate would
pass the sample. On 2026-07-26 it would add 20–100° of heading noise to
endpoints with no way to gate it.

## What would change the answer

- A magnetometer that reaches the reconstruction (re-import day logs keeping
  `mag_*`).
- A pre-trip full-rotation calibration drive to observe the 3-D ellipsoid.
- An in-car magnetic environment quiet enough that the hard-iron vector is
  actually constant — which neither of these two trips has.

None of these are available now, and 2026-07-26's slow spatial bias suggests
even a good pre-trip calibration would not survive the drive.
