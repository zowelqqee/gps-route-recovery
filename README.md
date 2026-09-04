# gps-route-recovery

Recovering where a car actually went when its GPS lied.

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
  runs a dual tracker: `RoadTracker` reconstructs the OSM-constrained route,
  while `ParkingTracker` estimates the final free-space parking manoeuvre.

## What this does not claim

These are load-bearing, not boilerplate:

* **The reconstructed route is a probabilistic estimate, not a measurement.**
  It is the most likely path given the IMU and the road graph. It is not where
  the car provably was.
* **After a junction the answer is several roads with probabilities**, not one
  averaged line. The output is a set of corridors — "branch A, p = 0.71; branch
  B, p = 0.24" — never a single circle, never a convex hull that would fill the
  courtyards between two independent streets, and never a point halfway between
  them.
* **A real GPS outage leaves no ground truth.** Position error can only be
  measured against a synthetically corrupted trip, where the clean track was
  kept aside. On a real failure every error field in `metrics.json` is `null`,
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

* the Python suite — 262 tests, passing;
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
  Every reconstruction result quoted here comes from *simulated* IMU data
  generated by `geotrace simulate`, which renders a physically consistent
  50 Hz stream from a known route. The IMU pipeline has never seen a real
  phone's accelerometer.
* **Camera and Vision OCR** on real photographs. The OCR *parsing* is tested,
  but no real street sign has been through `VNRecognizeTextRequest` here.
* Background location while the screen is locked, and battery cost over a long
  drive.
* Signing and installation on a physical iPhone.

Expect the real-device numbers to be worse than the table above. The synthetic
IMU is cleaner than a phone in a vibrating cradle, which is exactly the regime
where the road constraint earns its keep and the EKF baseline degrades fastest.

## Limitations

* No turn restrictions or traffic lights in the route prior — the only prior is
  a mild preference for staying on a larger road.
* The road graph is static; roadworks and closures are invisible.
* Visual place recognition is an interface with a deliberately empty stub.
* The parking zones in `sample-data/` are invented for testing.
* No CAN or OBD-II. Wheel speed would remove most of the along-track drift that
  currently dominates the error during an outage.
* `sample-data/trip-001/samples.jsonl` is ~6.5 MB, because 300 s of 50 Hz motion
  data is 15000 samples. Real trips grow at roughly 1.3 MB per minute.
