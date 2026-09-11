# Phase 37 — measurement-authority guards

Date: 2026-09-11

## Result

Three causal, GPS-free guards are now production defaults. They change only
measurement authority; the EKF state, covariance update, prediction model and
Phase-35 timestamp behavior are unchanged.

1. **False-ZUPT guard.** An IMU-variance stop claim is rejected when the
   independent spectral source still reports at least 5 m/s. A rejected stop is
   processed as a moving sample, so spectral/lateral measurements are not lost.
2. **Lateral consensus guard.** A biased-low `a_lat / omega` measurement is
   softly down-weighted only when both the pre-update EKF speed and the spectral
   speed exceed it by more than 4 m/s. Variance inflation grows smoothly and is
   capped at 25; the anchor is never hard-rejected.
3. **Pre-bend spectral saturation guard.** A downward spectral update in the
   observed 11–16 m/s plateau is given 8x measurement variance when the EKF is
   at least 1.5 m/s faster and the longitudinal channel is not reporting
   braking. Upward updates and braking-time downward updates are unchanged.

All three mechanisms expose counters in `speed.stats()`:

- `zupt_motion_rejected`
- `lateral_consensus_downweighted`
- `spectral_guarded`

## Review-trip ablation

GPS truth was used only after each replay for scoring. It was never visible to
the filter or guards. The selected configuration is `all_f8_db4`.

| trip | variant | median abs D | p95 | max | endpoint | speed MAE |
|---|---|---:|---:|---:|---:|---:|
| rf-07-22 | baseline | 103.0 m | 136.5 m | 145.8 m | -141.8 m | 1.99 m/s |
| rf-07-22 | guards | **99.0 m** | **130.6 m** | **139.9 m** | **-135.7 m** | 1.99 m/s |
| rf-07-26 | baseline | 265.4 m | 624.7 m | 731.6 m | -731.6 m | 2.63 m/s |
| rf-07-26 | guards | **126.0 m** | **409.1 m** | **472.5 m** | **-472.5 m** | **2.23 m/s** |

On rf-07-26 the guards rejected exactly the 72 false-ZUPT ticks found in Phase
36, down-weighted 92 of 535 lateral updates, and guarded 844 spectral plateau
updates. On rf-07-22 no ZUPT was rejected; only 11 lateral and nine spectral
updates were softened.

The independent rf-map-covered 07-23 session changed only slightly: max error
74.5 -> 75.2 m and endpoint 50.4 -> 51.1 m. The 07-24 and 07-25 independent
sessions cannot be evaluated with `runs/review-map.graphml` because their last
visible fixes have no drivable edge within the configured 45 m initialization
radius; the ablation records those skips explicitly rather than silently
changing the map or initialization gate.

## Bad-interval changes on rf-07-26

| interval | baseline error change | guarded error change |
|---|---:|---:|
| 27–115 s | -258.2 m | **-120.9 m** |
| 212–304 s | -356.2 m | **-298.0 m** |
| 316–390 s | -63.4 m | **-38.5 m** |
| 436–455 s | -107.6 m | **-63.2 m** |

The main spectral episode is improved, not eliminated. The guard correctly
removes authority from a saturated observation but cannot reconstruct the
missing 20–22 m/s speed by itself; that still requires an independent absolute
high-speed anchor or a better spectral model.

## Topology gate

The real `PacmanTracker` was run in single-path mode with guards off/on. For
both rf-07-22 and rf-07-26, every chosen edge and the complete final edge list
were identical. Decision counts and real-wrong-decision counts were also
unchanged (26/6 on 07-22, 13/0 on 07-26). The improvement is therefore an
odometry change, not a route-switch artefact.

## Verification and artifacts

- Full processor suite: **659 passed** in 147.14 s.
- `tools/phase37_authority_guards.py`
- `tools/phase37_topology_gate.py`
- `docs/plots/phase37/authority_guard_ablation.json`
- `docs/plots/phase37/topology_gate.json`

Phase 36 now explicitly disables these guards in its local replay config, so
its pre-fix attribution remains exactly reproducible after the production
defaults changed.
