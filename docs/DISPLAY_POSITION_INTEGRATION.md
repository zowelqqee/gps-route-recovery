# Phase 32 display-position branch — production integration

The Phase 32 iso-binary result (`DISPLAY_ODOMETRY_SPLIT.md`) is now a runtime
component wired into the tracker and surfaced in the replay panel. It is
**off by default**; the panel/demo config turns it on.

## Runtime flow

```
IMU  ─► GlobalSpeedTracker ─► v_route, D_route ─► SinglePathManager ─► committed route
 │                                                     (topology, junctions, commits)
 │                                                            │  (unchanged, byte-identical)
 └─► DisplayPositionBranch ──────────────────────────────────►│
        v_prior (pre-spectral) , v_spectral                   │
        │  frozen iso-binary estimator                        │
        │  residual_hat = isotonic(mean_3s(v_prior−v_spectral))│
        │  gate = frac_30s(v_spec>15)>0.02  or  max_20s(v_spec)>15
        │  v_position = v_route + gate·max(0, (v_spec+residual_hat) − v_route)
        │  D_position = D_route + ∫(v_position − v_route) dt
        ▼
   locate(): project D_position onto the ALREADY-committed edge chain
             → (edge, s, lat, lon); clamp at the committed frontier,
               report excess_position_distance, release it as the route grows
        ▼
   result.position_trace  ─►  benchmark position_trace.json  ─►  export_replay.py  ─►  panel marker
```

There is **no feedback path**. `DisplayPositionBranch` is never read by any
tracker method; `_display_locate` only reads `hs` (the committed route). With
the flag off, `position_trace` is `None` and every byte of `frames` /
`speed_trace` / decisions is identical (`test_display_position.py`).

## Files changed

| file | change |
|---|---|
| `processor/src/geotrace/pacman_tracker/display_position.py` | **new** — `DisplayResidualModel` (loads the frozen artifact), `DisplayPositionBranch` (`step` per IMU sample, `locate` per output tick), `DisplayPositionSample` |
| `processor/src/geotrace/pacman_tracker/data/display_residual_iso.json` | **new** — the frozen Phase 32 iso-binary isotonic curve + gate constants (full-pool and leave-07-26-out grids). Built by `tools/phase32_export_model.py`; not retrained at runtime |
| `processor/src/geotrace/pacman_tracker/config.py` | `DisplayConfig` (`position_branch_enabled` default `False`, `leave_0726_out`, `model_path`); `PacmanConfig.display` |
| `processor/src/geotrace/pacman_tracker/tracker.py` | init the branch (single-path, flag on, non-oracle only); feed it `(t, dt, v_route, v_prior, v_spectral, D_route)` per step; emit `position_trace`; `stats["display_position"]`; `_display_locate` helper; `TrackerResult.position_trace` |
| `processor/src/geotrace/pacman_tracker/benchmark.py` | `--display-position` / `--display-leave-0726-out` flags; writes `position_trace.json` |
| `processor/tests/pacman/test_display_position.py` | **new** — 13 tests (invariants + rf-07-26 regression) |
| `replay-ui/scripts/export_replay.py` | reads `position_trace.json`; the on-map Pacman is the corrected display position, the conservative route point is kept as `route`; `meta.display` summary |
| `replay-ui/app/page.tsx` | header pill + DISTANCE-panel rows for the display estimator (baseline route error, gate state, ΔD, excess). No change to markers/trail/controls logic |
| `.claude/launch.json` | `replay-ui` dev-server entry (Node 24 via nvm) |

Run: `runs/pacman-display/2026-07-26/` (benchmark with `--display-position
--display-leave-0726-out`).

## rf-07-26 acceptance (`tools/phase32_acceptance.py`, `test_display_position.py`)

**Corrected display odometer `D_position` vs withheld GPS** — reproduces the
Phase 32 diagnostic within plumbing tolerance:

| metric | baseline `D_route` | `D_position` (runtime) | Phase 32 diagnostic |
|---|---|---|---|
| median \|D − D_true\| | 266 m | **57 m** | 59 m |
| p95 | 625 m | **285 m** | 283 m |
| max | 732 m | **296 m** | 296 m |
| D/D_true | 0.834 | **0.939** | 0.939 |
| closer to truth than route | — | 88 % of the outage | ~89 % |

The one intentional deviation from the raw diagnostic: none. The diagnostic did
not veto the correction during a stop, and neither does the runtime (the
gate's ~20 s `v_spectral` memory keeps it briefly active into a stop); adding a
stationary veto moved 07-26 to ~95 m, so it was left out — matching the proven
estimator exactly.

**Topology:** `speed_trace`, committed edge sequence (14 edges), the 13 junction
decisions and `real_wrong = 0` are byte-identical to the flag-off run.

**Full-pool model** (the shipped default, 07-26 in-sample): median 50 m,
max 281 m, D/D_true 0.945.

## The rendered marker vs the odometer

The *odometer* `D_position` (how far the car has travelled) is at 57 m. The
*on-map marker* is placed by projecting `D_position` onto the committed edge
chain, and **the marker may not cross the committed frontier** (the end of the
last edge the route has committed) — TASK 4's "no guessing the road ahead".

On 07-26 the committed route is extended edge-by-edge as `D_route` (the
*conservative*, undershooting odometer) reaches each junction, so the frontier
runs ~100–250 m behind where `D_position` wants the marker. Between junctions
the marker holds at the frontier (`excess_position_distance` grows, shown in the
panel); at each junction the route commits the next edge and the marker snaps
forward. Net rendered-marker error on 07-26:

| | median | p95 | max | closer than baseline |
|---|---|---|---|---|
| rendered marker (2-D, frontier-clamped) | **180 m** | 570 m | 839 m | 88 % of ticks |
| baseline route point | 268 m | 638 m | 839 m | — |

So the marker is a third closer than today's and tracks the true point tightly
right after every junction; the residual lag is the committed route itself
being paced by the conservative odometer, which the display branch is not
allowed to change. Relaxing this (rolling the marker onto a junction's single
unambiguous successor) is a small, defensible follow-up but was out of scope.

## Panel

```bash
cd replay-ui && npm install          # once (needs Node >= 22; nvm use 24)
python3 scripts/export_replay.py     # regenerates public/data/replay-07-26.json
npm run dev                          # http://localhost:3000
```

The header shows `DISPLAY · iso-binary (Phase 32, frozen) · median 58 m vs
route 266 m`. The DISTANCE panel shows the corrected along-route error, the
baseline route error, the saturation-gate state, ΔD and any frontier excess.
The truth circle is replay-only and never enters the estimator.

## Not done (deliberately)

- No 2-state longitudinal bank / anchor selector / bounded fallback — the
  architecture leaves room for them (`DisplayConfig`, separate traces) but
  Phase 32's verdict says a *bounded* correction is the safe cross-trip form and
  the unbounded one shipped here is 07-26-tuned in spirit. It stays **display
  only** and **off by default**; do not route it into localisation.
- No retro-smoothing in the live marker (Phase 26 is a separate replay mode).
