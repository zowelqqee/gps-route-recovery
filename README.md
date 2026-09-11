# gps-route-recovery

Recovering where a car actually went when its GPS lied.

**Road-coordinate redesign (2026-09-08):** The Python default is now `road-ekf`: a bank of EKFs on directed road edges, with short fixed-width corridors and explicit ambiguity. See [the model, limits, and validation commands](docs/ROAD_EKF.md). The previous particle/display tracker remains available as `--algorithm road-particle-filter`. The iOS implementation has not yet been migrated.

**Pacman tracker (2026-09-08):** A second, independent reconstruction core lives in `processor/src/geotrace/pacman_tracker/`, benchmarkable side by side with the existing filters, which are untouched. It drops free `(x, y)` for a road-locked state — which edge, how far along it — and scores hypotheses by matching the gyro's yaw rate against the map's curvature times speed. See [the design, the measurements behind it, and the results](docs/PACMAN_TRACKER.md). Run it with `geotrace pacman --trip ... --graph ...`.

**Independent audit (2026-09-08):** See [findings, fixes, and reproducible validation](docs/INDEPENDENT_REVIEW.md). The Python engine is an experimental offline prototype. Historical tuning tables below predate corrections to calibration isolation, physics, filtering and metrics; they are not current accuracy or coverage guarantees.

In Saint Petersburg a phone's GPS will, in the course of one drive, drop out in
a tunnel, jump several kilometres across the Neva, sit on a constant offset down
a canyon street, drift slowly, and come back after an outage with a handful of
confidently-wrong fixes. This repository records a drive on an iPhone and then
reconstructs the route from the inertial sensors, constrained to the
OpenStreetMap road graph.

Two parts:

* **`ios/GeoTraceLab`** — a SwiftUI app that records GPS + CoreMotion, draws the
  route on MapKit, photographs a street sign after you stop, runs Vision OCR on
  it, and exports the whole trip as a single zip.
* **`processor/`** — a Python package (`geotrace`) that reads that export and
  runs a bank of road-coordinate EKFs. The older dual tracker and its
  free-space parking heuristic remain available only in the legacy algorithm.

## What this does not claim

These are load-bearing, not boilerplate:

* **The reconstructed route is a probabilistic estimate, not a measurement.**
  It is a filtered road hypothesis given the IMU and the road graph. It is not where
  the car provably was.
* **After a junction the answer is several roads with probabilities**, not one
  averaged line. The output is a set of corridors — "branch A, p = 0.71; branch
  B, p = 0.24" — never a single circle, never a convex hull that would fill the
  courtyards between two independent streets, and never a point halfway between
  them.
* **A real GPS outage leaves no ground truth.** Position error can only be
  measured against a synthetically corrupted trip or real GPS deliberately
  withheld before inference. On a real failure every error field in `metrics.json` is `null`,
  and the report says so rather than printing a comforting number.
* **A photograph does not identify an address.** The app runs OCR and records
  the last position it believed in. Placing a car from a facade needs a database
  of geo-referenced images, which this repository does not have and does not
  fake — `VisualPlaceRecognizer` is a documented interface with a stub that
  returns nothing.
* **The bundled parking zones are test polygons**, not the official Saint
  Petersburg paid-parking boundaries.

---

## Layout

```text
gps-route-recovery/
├── ios/GeoTraceLab/          SwiftUI recorder (XcodeGen project)
├── processor/                the geotrace Python package + tests
├── schemas/trip.schema.json  the on-disk trip format
├── sample-data/              a ready-made trip, a photo, test parking zones
├── live_logs/                day files from the in-car logger (GPS + IMU)
├── cache/spb-center.graphml  central Saint Petersburg, ~1800 edges
└── runs/                     output of the commands below
```

---

## Install

Python 3.11+ is required (3.11 is what this was developed and tested against).

```bash
cd gps-route-recovery
python3.11 -m venv processor/.venv
processor/.venv/bin/pip install --upgrade pip
processor/.venv/bin/pip install -e "processor[dev]"
```

Add the venv to your `PATH`, or prefix commands with `processor/.venv/bin/`.
Everything below assumes the latter.

Check it:

```bash
processor/.venv/bin/geotrace --version
processor/.venv/bin/geotrace --help
```

---

## Quick start — the whole pipeline, no iPhone needed

Four commands. The road graph and a sample trip are already in the repository,
so this needs no network access.

```bash
processor/.venv/bin/geotrace inject-fault --trip sample-data/trip-001 --output runs/demo --scenario --start 45 --duration 25 --dropout-duration 45 --seed 42
```

```bash
processor/.venv/bin/geotrace reconstruct --trip runs/demo --graph cache/spb-center.graphml --algorithm road-particle-filter --particles 5000 --confidence 0.95 --seed 42 --parking-zones sample-data/parking-zones.geojson
```

```bash
processor/.venv/bin/geotrace report --run runs/demo
```

```bash
open runs/demo/results/report.html
```

`--scenario` applies the composite failure the problem describes: GPS first
drifts off by a constant offset, then disappears for 45 s, then returns with
four false fixes before it settles.

### What comes out

```text
runs/demo/results/
├── reconstructed-route.geojson      the estimate
├── parking-tracker.geojson          terminal manoeuvre, endpoint and confidence polygon
├── tracking-result.json             schema v2 dual-tracker result; final position is parking_tracker
├── uncertainty-polygons.geojson     95% corridors, one feature per branch
├── corrupted-gps.geojson            what the receiver claimed
├── reference-gps.geojson            the clean track (synthetic trips only)
├── baseline-ekf-dead-reckoning.geojson
├── baseline-last-known-position.geojson
├── metrics.json                     computed numbers, not placeholders
├── diagnostics.json                 every gate decision, every rejected fix
└── report.html                      map, legend, layer switches, error chart
```

Measured on the bundled sample trip (300 s through central Saint Petersburg,
composite failure, 5000 particles, seed 42):

| algorithm | mean error | median | p95 | max | at outage end |
| --- | --- | --- | --- | --- | --- |
| `road_particle_filter` | **17.0 m** | 7.9 m | **67.2 m** | **185.7 m** | 5.0 m |
| `ekf_dead_reckoning` | 184.9 m | 7.9 m | 956.1 m | 1048.3 m | 0.0 m |
| `last_known_position` | 244.8 m | 0.0 m | 1405.8 m | 1543.8 m | 0.0 m |

95% polygon coverage 1.00, top-1 branch accuracy 1.00, up to 2 branches held
simultaneously, 6.2% of good fixes rejected, 13.8% of the injected false fixes
accepted, trust restored 3.0 s after GPS came back, and the parking zone
identified as `test-zone-01` (p = 1.00, `confident`).

The two baselines look deceptively good on *median* error because they are
exact whenever GPS is healthy; the p95 and max columns are where the outage
lives, and that is the whole problem.

---

## Recording a real trip

### 1. Build the app

Needs macOS with Xcode 15+ (developed against Xcode 26 / Swift 6.2) and
[XcodeGen](https://github.com/yonaskolb/XcodeGen) (`brew install xcodegen`).

```bash
cd ios/GeoTraceLab && xcodegen generate && open GeoTraceLab.xcodeproj
```

Select your iPhone, set your own signing team, and run. Deployment target is
iOS 17. There are no third-party dependencies.

### 2. Record

1. Mount the phone in a windscreen or vent holder.
2. **Start trip** shows the calibration sheet: hold still ~6 s (this measures
   sensor bias, which is the single biggest source of drift once GPS goes), then
   drive straight for a short stretch (this ties the gyro heading to a real
   compass direction without trusting the magnetometer, which a car body
   distorts). Recording begins automatically only after this confirmation; an
   uncalibrated recording is intentionally not allowed.
3. Drive. The map draws the track live; the header shows GPS accuracy, elapsed
   time and both sample counts.
4. **Finish trip.**
5. **Photo** — enabled only after the trip ends or after a confirmed standstill,
   never while moving. Vision OCR runs on the shot and the recognised lines are
   saved beside it.

### 3. Export

**Export**, or swipe a row in **Trips**, produces `trip-<UUID>.zip` through the
share sheet — AirDrop it, save it to Files, or drag it off the device in Finder
(file sharing is enabled). No manual editing is needed: the processor opens the
zip directly.

### 4. Process

```bash
processor/.venv/bin/geotrace reconstruct --trip ~/Downloads/trip-<UUID>.zip --graph cache/spb-center.graphml --algorithm road-particle-filter --particles 5000 --seed 42
```

The ordinary `reconstruct` command always runs both trackers. `RoadTracker`
remains constrained to OSM; `ParkingTracker` owns the final parked-car position,
its confidence polygon and status. On a real trip there is no reference track, so the error fields are `null` by
design. To measure the algorithms, corrupt a good trip yourself first:

```bash
processor/.venv/bin/geotrace inject-fault --trip ~/Downloads/trip-<UUID>.zip --output runs/mine-broken --fault dropout --start 60 --duration 45
```

### 5. Look at the results on the phone

**Results** imports `reconstructed-route.geojson`, `corrupted-gps.geojson` and
`uncertainty-polygons.geojson` back into the app, so the same MapKit view shows
the original track (green), the corrupted one (red), the reconstruction (blue)
and the 95% corridors, each toggleable.

---

## Recording with the vehicle logger

The second recorder in this repository is not a phone. It is a box wired into
the car that writes one file per day per sensor, in `live_logs/`:

```text
live_logs/
├── gps_logs/2026-07-22_GPS_logs.csv                 timestamp_ms,lat,lon,speed,course   10 Hz
├── gps_logs/2026-07-22_GPS_GNRMC_original_logs.txt  the raw NMEA the receiver emitted
└── imu_logs/2026-07-22_IMU_logs.csv                 acc, gyr, mag, roll/pitch/yaw, quat  100 Hz
```

### Sparse RFID validation logs

`real_tests/` contains a different, GPS-free validation source: one SI-unit
IMU day file and sparse RFID checkpoints.  The replay exporter uses earlier
checkpoints only to calibrate the fixed mount and spectral model, then withholds
every RFID point on the evaluated circuit.  Errors are calculated at those
physical checkpoints after tracking; the legal OSM path drawn between them in
the panel is visualization, not continuous ground truth.

```bash
cd replay-ui
npm run data:real
```

The generated replay appears as `09-08 · RFID` in the trip selector.  The
machine-readable diagnostic report is written to
`runs/real-tests-rfid/2026-09-08/report.json`.

```bash
processor/.venv/bin/geotrace import-live --logs live_logs --day 2026-07-22 --list
```

```text
2026-07-26: 4 session(s)
  [0] 2026-07-26T15:24:13Z .. 2026-07-26T15:59:58Z   2145 s   16615 fixes   1255 s moving   181.73 km  28 GPS jumps
  [1] 2026-07-26T17:17:12Z .. 2026-07-26T18:15:42Z   3510 s   32005 fixes   2112 s moving    22.99 km
```

A day is several drives, not one trip, and the listing is where a bad one shows
itself: session 0 above reports 182 km in 21 minutes of driving because the
receiver teleported 28 times. That distance is the honest sum of what it
claimed, and the jump count sits next to it rather than being cleaned away.

```bash
processor/.venv/bin/geotrace import-live --logs live_logs --day 2026-07-22 --session 0 --output runs/live-0722
```

```bash
processor/.venv/bin/geotrace reconstruct --trip runs/live-0722 --graph cache/spb.graphml --algorithm road-particle-filter --particles 5000 --seed 42
```

### GPS at the start and then nothing

The logger's GPS actually runs for the whole drive. The import does not keep it
that way. Only the opening `--gps-warmup` seconds of *driving* go into
`samples.jsonl`; every later fix is moved into `reference-samples.jsonl`, which
no filter ever opens. So the reconstruction faces exactly the case this
repository is about — a place to start from and then only inertia — while the
fixes it was not allowed to see remain as a yardstick to score it against.

This is a better yardstick than the synthetic corruption the rest of the
repository uses, and a worse one than a survey. Those withheld fixes are a real
receiver's output with a real receiver's few metres of error, so `metrics.json`
says so in its own words rather than reusing the sentence written for injected
faults. `--keep-all-gps` imports the whole track instead, as a control: with
GPS present the tracker follows it to about a metre, which is what makes the
import itself believable.

The stationary minutes before the car pulls away are not counted against the
warm-up — a parked car's course is noise — but a fixed `--pre-roll` of them is
kept, because that is where the accelerometer and gyro bias are measured.

The results directory is the same one every other run produces, so
`corrupted-gps.geojson` keeps its name here and holds the warm-up fixes: nothing
was corrupted, that file is simply the GPS the filters were given.
`report.html` says so — its banner, its legend and the layer switch all name the
green track as real GPS that was withheld, not as a synthetic reference, because
a report that claimed the recording had been corrupted on purpose would be
describing something that did not happen.

### Three things the format does not tell you

Each of these was measured, and each one is silently wrong if assumed:

* **The accelerometer reports specific force.** At rest it reads +1 g on Z,
  where CoreMotion reads ~0 and keeps gravity in its own channel. Gravity is
  removed with the logger's own attitude quaternion, which leaves 0.003 g on a
  standing car — the tilt of the bracket is real, and the quaternion knows
  about it. The gyro is in degrees per second, and `yaw` is nothing but the
  integral of `gyr_z`: the magnetometer is not fused into it anywhere
  (correlation with the column is 1.000), so the heading is relative and it
  drifts.

* **The two files are not on the same clock.** The GPS file runs 4.5–4.9 s
  behind the IMU file on five of the six days checked, stable within a session
  and sharply peaked — the correlation that finds it falls from 0.78 at the
  peak to 0.23 two seconds away. This is the largest single error in the
  import: at 12 m/s it seeds the filter sixty metres down the road from where
  the car was. `import-live` recovers it by matching the gyro's yaw rate
  against the derivative of the GPS course, which are the same physical
  quantity measured twice. With the lag left in, the mount angle below measures
  at a coherence of 0.23; with it removed, 0.99.

* **The gyro's world frame points nowhere in particular.** Its yaw origin is
  wherever the logger happened to boot, so it differs every session (−34°,
  −119°, +124°, 177° on four of the days). The import measures two angles in
  the warm-up window and writes them into the trip's `MountCalibration`: the
  rotation from that frame onto East/North, and the angle between the box's own
  axes and the car's. The second is the sanity check — it comes out between −1°
  and +4° on every day, which is to say the box is bolted in facing forwards.

The mount rotation is measured from acceleration, but not by differentiating
the GPS velocity vector — at 10 Hz that derivative is mostly receiver noise.
A car's acceleration in its own frame is two quantities that *are* each clean:
`dv/dt` from the GPS speed, and the centripetal term `v * yaw_rate` whose yaw
rate comes from the gyro. Comparing the angle of that predicted vector with the
measured one gives the rotation directly.

### What it costs to lose GPS here

Session 0 of 2026-07-22 is 42 minutes and 8 km through northern Saint
Petersburg. With 120 s of GPS at the start, the remaining 23 742 fixes withheld,
and 5000 particles — that is **forty minutes of unaided inertia**:

| algorithm | mean | median | p95 | max |
| --- | --- | --- | --- | --- |
| `road_particle_filter` | **912 m** | 938 m | **1747 m** | **2131 m** |
| `ekf_dead_reckoning` | 1009 m | 1068 m | 1470 m | 1483 m |
| `last_known_position` | 1497 m | 1696 m | 2213 m | 2223 m |

Error against the withheld track, measured from the moment GPS stops: 183 m at
30 s, 474 m at 1 min, 336 m at 5 min, 582 m at 15 min, 1366 m at 30 min. The 95%
corridor contains the true position essentially always — but by the end it is a
2 km-wide corridor, which is an honest statement of ignorance rather than a
useful constraint. Forty minutes is well past what this sensor set supports.

Over the first seven minutes of the same session (`--max-duration 420`, 4.5
minutes unaided, 2000 particles), which is nearer the regime the method is for:

| algorithm | mean | median | p95 | max |
| --- | --- | --- | --- | --- |
| `road_particle_filter` | **359 m** | 350 m | **531 m** | **556 m** |
| `ekf_dead_reckoning` | 446 m | 459 m | 714 m | 716 m |
| `last_known_position` | 801 m | 861 m | 1148 m | 1150 m |

Decomposed against the reference's own direction of travel, the error is almost
entirely **along-track and behind**: −117 m at 25 s into the outage, −357 m at
50 s, with under 3 m of cross-track error in the same stretch. The heading is
fine. The car is simply not where the filter thinks along the road it correctly
identified, because it accelerated from 5 m/s to 15 m/s the moment GPS went
away and nothing told the filter about it.

### Speed is not observable from this IMU

That is not a tuning failure, and the measurement says so. Instantaneously the
longitudinal acceleration is a good signal — correlated 0.95 with `dv/dt` from
GPS on 2026-07-22, once the clock offset is removed. Integrated, it is useless:

| day | corr(a_long, dv/dt) | rms error of the integral over 60 s | rms of the true change |
| --- | --- | --- | --- |
| 2026-07-22 | 0.95 | 9.1 m/s | 4.3 m/s |
| 2026-07-23 | 0.73 | 24.2 m/s | 3.1 m/s |
| 2026-07-25 | 0.55 | 21.8 m/s | 5.1 m/s |
| 2026-07-26 | 0.91 | 9.9 m/s | 8.8 m/s |
| 2026-07-29 | 0.54 | 6.4 m/s | 5.8 m/s |
| 2026-07-30 | 0.44 | 14.3 m/s | 3.0 m/s |

On every day, integrating the accelerometer to get the speed change over a
minute is worse than predicting no change at all. The high-frequency content is
real; the low-frequency content — the only part integration cares about — is
attitude-filter artefact and bias. Holding roll and pitch fixed from the still
period instead of using the per-sample quaternion is worse again (free
integration reaches 77 m/s after eight minutes, against 24 m/s for the
quaternion).

This is why `MotionConfig.accel_deadband_ms2` is set high enough to coast at the
last known speed rather than integrate, and why lowering it on this data makes
the reconstruction worse, not better (sweeping 0.0 → 1.0 moves the mean error
from 617 m to 359 m).

### Undoing what the attitude filter took

The loss above is not noise, and no filter removes it — boosting the low
frequencies back by ×2 and ×3 makes the 60 s integral worse in proportion
(5.4 → 8.9 → 13.0 m/s), because what lives down there is drift, not signal.

But the missing signal is recoverable, and the gyro is what makes it so. The
quaternion changes for two reasons: rotations that actually happened, which the
gyro also reports, and the filter's own levelling, which it does not. Subtract
one angular velocity from the other and what remains is the lean alone;
integrate that and you have the tilt error the filter is carrying; rotate
gravity by it and you have the acceleration it swallowed. `motion_model.
leveling_correction` does exactly that, and `--leveling-tau` turns it on.

It works, in the sense that can be checked independently: fitting a free scale
factor on top of the recovered amount returns 0.75–1.27 across five recorded
days — near one, i.e. what comes back is what went missing, not a tuned fudge.
The transfer function flattens accordingly:

| smoothing | gain before | gain after |
| --- | --- | --- |
| 2 s | 0.86 | 0.99 |
| 15 s | 0.73 | 1.05 |
| 30 s | 0.55 | 1.01 |
| 60 s | 0.49 | 0.94 |

**And it does not fix free integration — it makes it worse.** Restoring the low
frequencies restores the drift sharing that band: integrating the corrected
signal for 30 minutes reaches 180 km of distance error against 61 km for the
raw one. The AHRS's damage was also, accidentally, a brake.

It pays off only where something bounds the horizon, which in city driving is a
zero-velocity update at every traffic light.

### Standing still is a measurement

A stopped car reports its speed exactly, for free, and on these recordings that
happens constantly: 34 stops of two seconds or more in 40 minutes, with a
median of 22 s of driving between them and never more than 145 s.

`zupt_requires_gps` existed because the old stillness test could not tell a stop
from a straight cruise — both have near-zero acceleration and near-zero yaw
rate. Road vibration can: a parked car only idles, at around 0.01 g, while a
moving one is being shaken at 0.03–0.07 g. `zupt_vibration_g` adds that as a
necessary third condition.

It is still not sufficient on its own. Across five days the conjunction admits
genuine motion at 2 m/s, and on one day at 10 m/s over a smooth road. What
makes the update safe without GPS is the test already beside it: an update that
contradicts the filter's own speed estimate is refused, not applied.

There is a second flaw here, real but not the one it looks like. The update is
applied on every filter step, so a thirty-second red light folds the same
measurement in three hundred times, and those samples are not independent.
`zupt_min_interval_s` rate-limits that — and rate-limiting was measured and left
off by default, because on this data it fixed nothing (the overconfidence below
came from somewhere else) and cost 65 m of accuracy by throwing away stop
evidence the filter was using.

### They pay off only where the motion model is the bottleneck

Both mechanisms are real, both are implemented, both are off by default. The
measurements say why, and they do not say the same thing on every recording.
20-minute windows, GPS withheld after 120 s, `road_particle_filter` at 2000
particles:

**2026-07-22, where the default already works:**

| | error | p95 | EKF baseline | corridor | coverage |
| --- | --- | --- | --- | --- | --- |
| default | **390 m** | **741 m** | 1009 m | 1998 m | 0.95 |
| ZUPT without GPS | 488 m | 862 m | 1081 m | 1998 m | 0.95 |
| levelling recovery | 834 m | 1295 m | 1362 m | 1998 m | 0.61 |
| levelling + deadband 0.2 | 408 m | 679 m | 1240 m | 1998 m | 0.79 |
| all three + `accel_noise` 1.2 | 474 m | 862 m | **875 m** | 1998 m | 0.94 |

**2026-07-26, where it does not** — the default is 4511 m out with a corridor
that holds the true position 4% of the time:

| | error | p95 | EKF baseline | corridor | coverage |
| --- | --- | --- | --- | --- | --- |
| default | 4511 m | 9476 m | 4483 m | 1538 m | 0.04 |
| ZUPT without GPS | 4511 m | 9476 m | 4483 m | 1538 m | 0.04 |
| levelling + deadband 0.2 | 4525 m | 9471 m | 4484 m | 636 m | 0.02 |
| all three + `accel_noise` 1.2 | **4108 m** | **9104 m** | **2951 m** | 1998 m | **0.40** |

So: on a recording the road graph can carry, the corrections cost accuracy; on
one it cannot, the full combination takes 34% off the dead-reckoning error and
lifts the corridor's coverage tenfold. Neither is a default, and the difference
between those two outcomes is not a tuning detail - it is which of two error
sources happens to dominate.

Four things this pins down, each worth knowing on its own:

* **Only the whole combination works.** Every partial configuration is a loss on
  both days. The levelling recovery alone is the worst of them, because it
  restores signal and drift together (see above) with nothing bounding the
  integration; the deadband and the extra process noise are what make it safe,
  not optional extras beside it.

* **The accuracy the recovery seems to buy is overconfidence.** On the short
  window it cut the error from 359 m to 306 m while the corridor shrank from
  364 m to 158 m and coverage fell 0.81 to 0.10. The filter had been handed a
  strong acceleration signal with none of that signal's own uncertainty
  attached. Restoring it (`accel_noise` 0.35 to 1.2) restores coverage and
  takes most of the accuracy back with it - the honest trade, and the reason
  the last row of each table is the only one worth using.

* **The road graph had already absorbed the win, on 07-22.** The correction does
  fix the motion model: dead reckoning covers 81% of the true distance against
  37% before, and along-track error drops from 301 m to 196 m. But at twenty
  minutes the particle filter's error is dominated by which street it committed
  to, not by how far along it it thinks it went. On 07-26, where it commits
  wrong regardless, the better motion model is what is left to help.

* **The vibration test needs a threshold nobody has.** Standing vibration ranges
  from 0.009 g to 0.052 g across five days, and the fixed 0.02 g rejected every
  genuine stop on 07-26 - which is why its ZUPT row is identical to the
  default. Calibrating per trip from the stationary pre-roll does not rescue
  it: on 07-23 the parked car shakes at 0.050 g and the moving one at 0.002 g.
  Some days a car idles harder than it drives.

The measurement that says the levelling recovery is physically right - a free
scale factor returning 0.75-1.27, a transfer function flattened to 1.0 - is not
the measurement that says it helps. Whether it helps depends on the recording,
so it is a flag and not a default.

Recovering the along-track error properly still needs an odometer — wheel speed
over CAN — not better filtering of this accelerometer.

### Smoothing hides the noise, not the bias

`--accel-smooth` boxcar-averages the world-frame acceleration before anything
else touches it. It is a different mechanism from the levelling recovery above
— it removes noise, not tilt — but it fails the exact same way, on the same
recording, for the same reason: it makes particles agree with each other more
without making them more correct, and the corridor reports that agreement as
confidence.

One window, session 0 of 2026-07-22, GPS withheld after 120 s, 5000 particles,
`road_particle_filter`:

| `--accel-smooth` | `--accel-deadband` | mean error | p95 | `coverage_95` | mean corridor area |
| --- | --- | --- | --- | --- | --- |
| off | 1.0 (default) | **428.8 m** | 784.3 m | 0.95 | 8.2 km² |
| 5 s | 1.0 (default) | 445.2 m | 920.4 m | **0.12** | 1.35 km² |
| 5 s | 0.2 | 439.9 m | 1024.8 m | 0.87 | 8.9 km² |
| 5 s | 0.5 | 432.3 m | 767.1 m | 0.87 | 8.2 km² |
| 5 s | 1.5 | 436.6 m | 920.3 m | 0.97 | 9.3 km² |
| 5 s | 2.0 | 439.7 m | 858.2 m | 0.95 | 9.5 km² |

Smoothing on its own (row 2) is a trap: the point estimate gets *worse* (429 m
→ 445 m) while the corridor shrinks sixfold and its coverage collapses from
0.95 to 0.12 — a tight, confident, wrong answer, for the same reason the
levelling recovery above needed a compensating `accel_noise` raise. Raising
the deadband back up alongside the smoothing (rows 3-6) restores coverage to
0.87-0.97, but it does not buy back accuracy: every deadband from 0.2 to 2.0
combined with 5 s smoothing lands within about 2% of the no-smoothing
baseline, never below it.

So on this recording `--accel-smooth` earns its keep only as camouflage
removal — paired with a raised deadband it stops the corridor from lying, it
does not make the reconstruction better. If it is used at all, it has to be
paired with `--accel-deadband` at 1.5 or higher; at the 1.0 default the two
combine into the worst row in the table. That said, this is one 17-minute
window on one day out of six recorded, and the dip in coverage lands exactly
on the default deadband value (0.87 → 0.87 → **0.12** → 0.97 → 0.95) in a way
that is not fully explained — it may be this trip's own geometry rather than
a clean threshold. `--accel-smooth 5 --accel-deadband 2.0` is retained as an experimental pair.
The independent multi-day review does not establish it as a generally better
configuration. See the audit report for updated measurements.

---

## A larger map

`cache/spb-center.graphml` covers roughly 59.915–59.950 N, 30.32–30.39 E. For a
trip outside that, download once:

```bash
processor/.venv/bin/geotrace download-map --place "Saint Petersburg, Russia" --output cache/spb.graphml
```

or, much faster, a bounding box:

```bash
processor/.venv/bin/geotrace download-map --bbox 59.99 59.82 30.55 30.15 --output cache/spb.graphml
```

The graph is cached as GraphML and is never re-downloaded; `reconstruct` clips
it to `--radius` metres around the trip origin. OpenStreetMap only, no paid
APIs, no panorama scraping.

---

## CLI

```text
geotrace download-map   --place | --bbox N S E W   --output cache/spb.graphml
geotrace simulate       [--graph G] --output DIR [--duration S] [--stop-at S] [--seed N]
geotrace inject-fault   --trip DIR --output DIR   --fault ... | --scenario
geotrace reconstruct    --trip DIR [--graph G] [--algorithm A] [--particles N]
                        [--confidence 0.95] [--seed 42] [--parking-zones F]
                        [--allow-simulated]
geotrace report         --run DIR [--output report.html] [--polygon-stride N]
geotrace inspect        --trip DIR
geotrace reconstruct    ... [--leveling-tau S] [--zupt-without-gps]
                        [--zupt-vibration G] [--accel-deadband MS2]
                        [--accel-smooth S]
geotrace import-live    --logs DIR [--day D] [--list] [--session N] --output DIR
                        [--gps-warmup S] [--keep-all-gps] [--imu-rate HZ]
                        [--pre-roll S] [--max-duration S]
                        [--clock-align warmup|session|none] [--clock-offset S]
```

Every subcommand has `--help`.

**Algorithms** (`--algorithm`), all three computed on every run so the metrics
always contain a real comparison:

* `last-known-position` — hold the last trusted fix.
* `ekf-dead-reckoning` — EKF over `[E, N, v, psi, b_a, b_omega]`, no road graph.
* `road-particle-filter` — the main one.

**Faults** (`--fault`): `dropout`, `offset` (`--east`/`--north`), `drift`
(`--drift-east`/`--drift-north`), `jumps` (`--sigma`), `false_recovery`
(`--count`), or `--scenario` for the composite failure. `--accuracy` makes the
corrupted fixes claim to be accurate, which is the point: a receiver that has
just teleported the car 4 km still reports 12 m. What was applied is recorded in
`faults.json`, and the clean track is preserved in `reference-samples.jsonl`.

`--allow-simulated` accepts fixes CoreLocation flagged as simulated by software.
Needed only for trips recorded in the iOS Simulator, where every fix carries the
flag; never use it on real data.

Every threshold lives in `processor/src/geotrace/config.py` and can be overridden
wholesale with `--config my.json`.

---

## How it works

### State

```text
X = [E, N, v, psi, b_a, b_omega]
```

Position in a local metric frame (azimuthal-equidistant, centred on the first
trusted fix — exact against the WGS84 geodesic, unlike the flat R = 6371 km
sphere, which is 0.3% out at this latitude), forward speed, heading in radians
CCW from east, and the two sensor biases.

```text
psi_{t+1} = wrap(psi_t + w_hat dt)
psi_bar   = psi_t + 0.5 w_hat dt
E_{t+1}   = E_t + v dt cos(psi_bar) + 0.5 a_hat dt^2 cos(psi_bar)
N_{t+1}   = N_t + v dt sin(psi_bar) + 0.5 a_hat dt^2 sin(psi_bar)
v_{t+1}   = max(0, v_t + a_hat dt)
```

with `a_hat = a_parallel - b_a`, `w_hat = w - b_omega`, and `a_parallel` the
longitudinal component of the world-frame acceleration obtained by rotating
CoreMotion's device-frame vector through its attitude quaternion. Integration
across a timestamp gap larger than `motion.max_gap_s` is refused outright rather
than silently inventing position.

### Detecting the failure

A single `horizontalAccuracy` is not evidence, so every fix is tested against
several independent things: a physical gate
(`d_max = v dt + 0.5 a_max dt^2 + m`), a Mahalanobis gate
(`D^2 = r^T S^-1 r` against `chi^2(2)` at p = 0.99), speed consistency, course
consistency (ignored below 3 m/s, where CoreLocation course is noise), distance
to the nearest drivable road, and the recent history. All gates are evaluated
even after one fails, so a rejection is diagnosable afterwards.

That drives `TRUSTED → SUSPECT → LOST → RECOVERING → TRUSTED`. Two details
matter more than they look:

* Returning to TRUSTED needs several *consecutive consistent* fixes. That is
  what rejects the scatter a receiver emits in the first seconds after it
  re-acquires.
* In LOST and RECOVERING the Mahalanobis gate is **not** applied, and course and
  speed are checked against the *previous accepted fix* rather than the filter.
  Dead reckoning that has run free for a minute is not a valid reference; gating
  returning fixes against it makes the filter defend its own drift and never
  recover. When trust does return, both filters are re-anchored on the recovered
  fix instead of being blended with the stale solution.

### The road particle filter

Each particle is `(edge, distance along edge, speed, heading, b_a, b_omega, w)`,
so it is always *on a road* and the belief after a junction is a set of
distinct branches rather than a blob. At a junction the outgoing edge is sampled
from

```text
P(e' | p) ∝ exp( -wrap(theta_e' - psi)^2 / (2 sigma_turn^2) ) · P_route(e')
```

which is where the gyro decides the turn. One-way streets need no penalty term —
they simply have no reverse edge — and U-turns are excluded unless the car
actually stopped.

```text
w~ = w · L_GPS · L_psi · L_v · L_map
N_eff = 1 / sum(w_i^2),  systematic resampling when N_eff < N/2
```

Two things keep it honest over a long outage. The per-particle accelerometer
bias is only observable through the GPS *speed* likelihood, so that term is
tight and the bias random walk is small — otherwise whatever was learned before
the outage is forgotten within seconds and the along-track estimate runs away by
hundreds of metres. And while GPS is TRUSTED a small fraction of the worst
particles is replaced with fresh ones drawn around the fix, so a filter that
committed to the wrong branch during an outage can climb back out.

### Uncertainty polygons

Take the smallest set of particles carrying at least `gamma` of the mass, group
them by connected branch, and build one corridor per branch as a union of
buffers around the occupied road segments, with
`r = r_min + k·sigma_perp` and `sigma_perp` growing while GPS is unavailable.
Explicitly not a convex hull. On the bundled fork test the corridor union is
2.8× smaller than the hull over the same particles, and the midpoint between the
two branches — open ground the hull would happily claim — is outside every
polygon.

While the cloud still straddles the junction the branches genuinely are one
connected Y-shaped region, and it is reported as one component. That is honest,
not a bug.

The point estimate is the weighted mean *within the highest-probability branch*,
snapped to the carriageway, so it never lands between two roads.

---

## Metrics

`metrics.json` contains mean / median / 95th percentile / maximum position
error and the error at the end of the outage; 95% polygon coverage and mean
area; top-1 branch accuracy and top-3 branch recall; the fraction of good fixes
wrongly rejected and of false fixes wrongly accepted; parking-zone probability
and decision; and how long trust took to return. Everything is computed — there
are no placeholders — and everything that needs a reference track is `null` when
there isn't one.

---

## Tests

```bash
cd processor && ../processor/.venv/bin/python -m pytest -q
```

**262 tests, all passing.** They cover coordinate round-tripping, angle
normalisation, straight-line / turning / stationary state transitions, bias
handling, the analytic Jacobian against finite differences, the physical and
Mahalanobis gates, every state-machine transition, systematic resampling and
weight normalisation, particles crossing a junction, the impossibility of
driving up a one-way street, multi-branch polygons, the absence of a convex
hull, parking-zone probability, seed reproducibility, and the guard against
reconstructing against a road graph that does not cover the roads driven.

The integration test builds the fork graph from the brief:

```text
          branch A
         /
start -- junction
         \
          branch B
```

drives onto branch A, kills GPS *before* the junction so only the gyro sees the
turn, and asserts that branch A ends up with the higher probability — and that
driving onto B flips the answer, so the filter is reading the gyro rather than
favouring one road.

iOS tests:

```bash
cd ios/GeoTraceLab && xcodebuild -project GeoTraceLab.xcodeproj -scheme GeoTraceLab -destination 'platform=iOS Simulator,name=iPhone 16' test
```

**50 tests, all passing** on the iOS 26 simulator: the wire format against the
brief's worked examples, fix validity, the shared monotonic timebase, JSONL
writing at a sustained 50 Hz, zip export, GeoJSON import, and the heading
conventions.

---

## What has actually been verified

Run on this machine (macOS 26, Xcode 26, Swift 6.2, Python 3.11.14):

* the Python suite — 376 tests, passing;
* the iOS app builds for the simulator, and its 50 tests pass;
* the app was launched on an iPhone 16 simulator, permissions granted, a trip
  recorded against a simulated drive through central Saint Petersburg, and
  finished — 76 GPS fixes written;
* that untouched recording was then read by `geotrace inspect` (76 lines, 0
  malformed, 0 rejected) and reconstructed, which is the export→process contract
  the whole thing depends on;
* the full CLI chain `simulate → inject-fault → reconstruct → report` on both
  the synthetic grid and the real OpenStreetMap graph, producing the numbers in
  the table above;
* `report.html` rendered and checked in a browser.

Not verified, and needing a real device:

* **CoreMotion.** The iOS Simulator has no motion hardware, so the app correctly
  reports "no motion sensors on this device" and records zero motion samples.
  The synthetic-trip results quoted above come from *simulated* IMU data
  generated by `geotrace simulate`, which renders a physically consistent
  50 Hz stream from a known route. The IMU pipeline has never seen a real
  **phone's** accelerometer — though it has now seen a real vehicle logger's,
  through `import-live`, and the section above says what that cost.
* **Camera and Vision OCR** on real photographs. The OCR *parsing* is tested,
  but no real street sign has been through `VNRecognizeTextRequest` here.
* Background location while the screen is locked, and battery cost over a long
  drive.
* Signing and installation on a physical iPhone.

Expect the real-device numbers to be worse than the table above, and the live
logger says by how much: metres of error over a synthetic 45 s dropout, hundreds
of metres over a real four-minute one. The synthetic IMU is cleaner than
anything bolted into a car, which is exactly the regime where the road
constraint earns its keep and the EKF baseline degrades fastest.

## Limitations

* No turn restrictions or traffic lights in the route prior — the only prior is
  a mild preference for staying on a larger road.
* The road graph is static; roadworks and closures are invisible.
* Visual place recognition is an interface with a deliberately empty stub.
* The parking zones in `sample-data/` are invented for testing.
* No CAN or OBD-II. Wheel speed would remove most of the along-track drift that
  currently dominates the error during an outage — and on the vehicle-logger
  data it is not an optimisation but the only route, because that IMU's
  longitudinal channel cannot be integrated for speed at all (see above).
* The vehicle logger's own clock offset is measured per session and assumed
  constant across it. It held to a quarter of a second over four quarters of a
  2.5-hour drive, but nothing enforces that, and a logger that drifted would
  be silently absorbed into the reconstruction.
* `import-live` reads a whole day's IMU file to find its sessions and again to
  cut one out. On a 900 MB day that is about ten seconds; it is not incremental
  and there is no index.
* `geotrace report` re-runs the reconstruction to rebuild the result object it
  renders from, so on a 42-minute live trip it costs the eight minutes of
  `reconstruct` a second time before it draws anything. Fine on a 300 s
  synthetic trip, tiresome here.
* `sample-data/trip-001/samples.jsonl` is ~6.5 MB, because 300 s of 50 Hz motion
  data is 15000 samples. Real trips grow at roughly 1.3 MB per minute.
