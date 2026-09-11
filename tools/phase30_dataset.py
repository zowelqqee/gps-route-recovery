#!/usr/bin/env python3
"""Phase 30 - learned spectral residual correction: dataset builder.

Replays the BASELINE speed EKF (predict + ZUPT + lateral + spectral_update, all
calibration extras OFF) for every review trip, reconstructs the raw spectral
speed, band powers and IMU channels on a common 0.5 s grid, and tiles windows.

Target:   residual = mean(v_true) - mean(v_spectral)   over the window
          k        = mean(v_true) / mean(v_spectral)

Hidden GPS (reference-samples) is used ONLY for the target / evaluation columns
(v_true_*). Every `feat_*` column is available during a real GPS outage.

Outputs (docs/plots/phase30/):
    windows_5s.csv        non-overlapping 5 s windows, all trips
    windows_2s.csv        denser 2 s windows
    grid_<trip>.csv       the raw 0.5 s grid per trip (for the replay diagnostics)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from geotrace.loader import load_trip
from geotrace.coordinates import LocalFrame
from geotrace.pacman_tracker.config import PacmanConfig
from geotrace.pacman_tracker.spectral import extract_features, BANDS
from geotrace.pacman_tracker.speed import GlobalSpeedTracker
from geotrace.pacman_tracker.tracker import (
    _fit_spectral, _lateral_channel, _initial_heading, _initial_biases,
    build_imu_samples,
)

ROOT = Path("/Users/arseniyabramidze/delimobil/gps-route-recovery")
OUT = ROOT / "docs" / "plots" / "phase30"
OUT.mkdir(parents=True, exist_ok=True)

TRIPS = {
    "07-22": ROOT / "runs/review-final/2026-07-22/trip",
    "07-23": ROOT / "runs/review-final/2026-07-23/trip",
    "07-24": ROOT / "runs/review-final/2026-07-24/trip",
    "07-25": ROOT / "runs/review-final/2026-07-25/trip",
    "07-26": ROOT / "runs/review-final/2026-07-26/trip",
}

GRID_DT = 0.5
NB = len(BANDS)  # 6


# --------------------------------------------------------------------------
def replay_trip(tag: str, trip_dir: Path) -> pd.DataFrame:
    trip, _ = load_trip(trip_dir)
    cfg = PacmanConfig()  # baseline: spectral_scale / bend / censor all off

    visible = trip.usable_locations
    frame = LocalFrame(visible[0].latitude, visible[0].longitude)
    heading0 = _initial_heading(visible, frame)
    t_visible_end = visible[-1].monotonic_time

    model = _fit_spectral(trip, visible, cfg)
    if not model.fitted:
        raise SystemExit(f"{tag}: spectral model did not fit: {model.reason}")

    samples = build_imu_samples(trip, cfg, heading0, model)
    accel_bias0, gyro_bias0, _ = _initial_biases(samples, visible, t_visible_end)
    speed0 = float(visible[-1].speed) if visible[-1].has_valid_speed else 0.0
    t_start = float(t_visible_end)

    spd = GlobalSpeedTracker(cfg.speed, v0=speed0, gyro_bias0=gyro_bias0,
                             accel_bias0=accel_bias0)

    steps = [s for s in samples if s.t > t_start]
    rec = []
    for s in steps:
        spd.current_t = s.t
        spd.predict(s.a_long, s.dt, s.shock, s.gap)
        if s.stationary:
            spd.zero_velocity(s.a_long, s.dt, s.yaw_rate, s.stationary_run_s)
            v_prior = spd.speed
            pvv_prior = spd.P[1, 1]
        else:
            spd.lateral_anchor(s.a_lat, s.yaw_rate_smooth, s.dt, s.shock,
                               spectral_speed=s.spectral_speed)
            v_prior = spd.speed            # after predict + lateral, before spectral
            pvv_prior = spd.P[1, 1]
            if math.isfinite(s.spectral_speed):
                spd.spectral_update(s.spectral_speed, s.spectral_sigma, s.dt)
        rec.append(dict(
            t=s.t, dt=s.dt,
            v_ekf=spd.speed, sigma_v=spd.sigma_speed, pvv=spd.P[1, 1],
            v_prior=v_prior, pvv_prior=pvv_prior,
            D=spd.distance,
            b_a=float(spd.x[2]),
            a_long=s.a_long,
            v_spec=float(s.spectral_speed),
            sigma_spec=float(s.spectral_sigma),
            yaw_rate=s.yaw_rate, yaw_rate_smooth=s.yaw_rate_smooth,
            a_lat=s.a_lat,
            stationary=bool(s.stationary),
            accel_std=s.accel_std, gyro_std=s.gyro_std,
        ))
    rp = pd.DataFrame(rec)

    # ---- v_true : withheld GPS Doppler speed (+ visible-window speed) --------
    tv = [(f.monotonic_time, float(f.speed)) for f in visible if f.has_valid_speed]
    tv += [(f.monotonic_time, float(f.speed)) for f in trip.reference_locations
           if f.is_usable and f.has_valid_speed]
    tv.sort()
    vt_t = np.array([x[0] for x in tv])
    vt_v = np.array([x[1] for x in tv])

    # ---- band powers -------------------------------------------------------
    raw_t, a_lat_raw, omega_raw, accel, gyro = _lateral_channel(trip)
    feats = extract_features(raw_t, accel, gyro, cfg.spectral_window_s, cfg.spectral_hop_s)
    fv, ft = feats.values, feats.times

    def bp(ch, b):
        return fv[:, ch * (NB + 1) + b]

    accel_low = np.mean([bp(c, 0) for c in (0, 1, 2)], axis=0)
    accel_b1 = np.mean([bp(c, 1) for c in (0, 1, 2)], axis=0)
    accel_b2 = np.mean([bp(c, 2) for c in (0, 1, 2)], axis=0)
    accel_mid = np.mean([bp(c, 3) for c in (0, 1, 2)], axis=0)
    accel_b4 = np.mean([bp(c, 4) for c in (0, 1, 2)], axis=0)
    accel_high = np.mean([bp(c, 5) for c in (0, 1, 2)], axis=0)
    accel_tot = np.mean([bp(c, 6) for c in (0, 1, 2)], axis=0)
    gyro_tot = np.mean([bp(c, 6) for c in (3, 4, 5)], axis=0)
    # spectral tilt: high band minus low band (log ratio) - rises with speed
    band_tilt = accel_high - accel_low

    a_mag = np.linalg.norm(accel, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)

    # ---- common 0.5 s grid ------------------------------------------------
    t_lo = max(rp.t.iloc[0], ft[0], vt_t[0])
    t_hi = min(rp.t.iloc[-1], ft[-1], vt_t[-1])
    grid = np.arange(t_lo, t_hi, GRID_DT)

    def ig(x, y):
        return np.interp(grid, x, y)

    g = pd.DataFrame(dict(
        t=grid, trip=tag, visible_end=t_visible_end,
        outage=(grid > t_visible_end),
        v_true=ig(vt_t, vt_v),
        v_spec=ig(rp.t, rp.v_spec),
        v_ekf=ig(rp.t, rp.v_ekf),
        v_prior=ig(rp.t, rp.v_prior),
        sigma_v=ig(rp.t, rp.sigma_v),
        pvv=ig(rp.t, rp.pvv),
        pvv_prior=ig(rp.t, rp.pvv_prior),
        sigma_spec=ig(rp.t, rp.sigma_spec),
        D=ig(rp.t, rp.D),
        b_a=ig(rp.t, rp.b_a),
        a_long=ig(rp.t, rp.a_long),
        a_long_debias=ig(rp.t, rp.a_long - rp.b_a),
        yaw_rate=ig(rp.t, rp.yaw_rate),
        a_lat=ig(rp.t, rp.a_lat),
        accel_std=ig(rp.t, rp.accel_std),
        gyro_std=ig(rp.t, rp.gyro_std),
        stationary=ig(rp.t, rp.stationary.astype(float)),
        a_mag=ig(raw_t, a_mag),
        gyro_mag=ig(raw_t, gyro_mag),
        vib_low=ig(ft, accel_low), vib_b1=ig(ft, accel_b1), vib_b2=ig(ft, accel_b2),
        vib_mid=ig(ft, accel_mid), vib_b4=ig(ft, accel_b4), vib_high=ig(ft, accel_high),
        vib_total=ig(ft, accel_tot), gyro_power=ig(ft, gyro_tot),
        band_tilt=ig(ft, band_tilt),
    ))
    g["dep_sigma_spec"] = float(model.deployment_sigma_ms)
    g["spec_gap"] = float(model.generalisation_gap)
    return g


# --------------------------------------------------------------------------
def _slope(t, y):
    if len(t) < 2 or np.ptp(t) < 1e-6:
        return 0.0
    return float(np.polyfit(t - t[0], y, 1)[0])


def make_windows(g: pd.DataFrame, window_s: float, hop_s: float | None = None) -> pd.DataFrame:
    n = int(round(window_s / GRID_DT))
    hop = n if hop_s is None else max(1, int(round(hop_s / GRID_DT)))
    tt = g.t.to_numpy()
    rows = []
    for i0 in range(0, len(g) - n + 1, hop):
        sl = slice(i0, i0 + n)
        w = g.iloc[sl]
        t = tt[sl]
        vt = w.v_true.to_numpy()
        vs = w.v_spec.to_numpy()
        if not (np.all(np.isfinite(vt)) and np.all(np.isfinite(vs))):
            continue
        vt_m, vs_m = float(vt.mean()), float(vs.mean())
        t_end = float(t[-1] + GRID_DT)
        outage = t_end <= g.visible_end.iloc[0] and "visible" or "outage"
        regime = "visible" if t_end <= g.visible_end.iloc[0] else "outage"

        # history windows ending at this window's start (no lookahead)
        i_end = i0 + n
        hist = g.iloc[:i_end]
        ht = hist.t.to_numpy()

        def integ(col, secs):
            m = ht >= (t_end - secs)
            if m.sum() < 2:
                return 0.0
            return float(np.trapezoid(hist[col].to_numpy()[m], ht[m]))

        def wstat(col, secs, fn):
            m = ht >= (t_end - secs)
            if m.sum() < 2:
                return 0.0
            return float(fn(hist[col].to_numpy()[m]))

        vpr = w.v_prior.to_numpy()
        r = dict(
            trip=g.trip.iloc[0], window_s=window_s,
            t_start=round(float(t[0]), 2), t_end=round(t_end, 2),
            regime=regime,
            t_since_outage=round(max(0.0, t_end - float(g.visible_end.iloc[0])), 1),
            # ---- TARGETS (hidden GPS) ----
            v_true_mean=round(vt_m, 4), v_true_std=round(float(vt.std()), 4),
            v_true_min=round(float(vt.min()), 4), v_true_max=round(float(vt.max()), 4),
            v_true_range=round(float(vt.max() - vt.min()), 4),
            residual=round(vt_m - vs_m, 5),
            k=round(vt_m / vs_m, 5) if vs_m > 1e-6 else np.nan,
            # ---- observable features ----
            feat_v_spec=round(vs_m, 4),
            feat_v_spec_std=round(float(vs.std()), 4),
            feat_v_ekf=round(float(w.v_ekf.mean()), 4),
            feat_v_prior=round(float(vpr.mean()), 4),
            feat_vprior_minus_vspec=round(float(vpr.mean()) - vs_m, 4),
            feat_vekf_minus_vspec=round(float(w.v_ekf.mean()) - vs_m, 4),
            feat_pvv=round(float(w.pvv.mean()), 5),
            feat_sqrt_pvv=round(float(np.sqrt(np.clip(w.pvv.mean(), 0, None))), 5),
            feat_sigma_spec=round(float(w.sigma_spec.mean()), 4),
            feat_dep_sigma_spec=round(float(g.dep_sigma_spec.iloc[0]), 4),
            feat_spec_gap=round(float(g.spec_gap.iloc[0]), 4),
            feat_a_long=round(float(w.a_long.mean()), 5),
            feat_b_a=round(float(w.b_a.mean()), 5),
            feat_a_long_debias=round(float(w.a_long_debias.mean()), 5),
            feat_a_long_debias_std=round(float(w.a_long_debias.std()), 5),
            feat_a_long_debias_rms=round(float(np.sqrt(np.mean(w.a_long_debias.to_numpy()**2))), 5),
            feat_pos_accel_frac=round(float((w.a_long_debias.to_numpy() > 0.3).mean()), 4),
            feat_brake_frac=round(float((w.a_long_debias.to_numpy() < -0.3).mean()), 4),
            # temporal IMU integrals of (a_long - b_a)
            feat_dv_imu_2s=round(integ("a_long_debias", 2.0), 4),
            feat_dv_imu_5s=round(integ("a_long_debias", 5.0), 4),
            feat_dv_imu_10s=round(integ("a_long_debias", 10.0), 4),
            feat_dv_imu_20s=round(integ("a_long_debias", 20.0), 4),
            # slopes of v_prior
            feat_vprior_slope_5s=round(wstat("v_prior", 5.0, lambda a: _slope(np.arange(len(a))*GRID_DT, a)), 5),
            feat_vprior_slope_10s=round(wstat("v_prior", 10.0, lambda a: _slope(np.arange(len(a))*GRID_DT, a)), 5),
            feat_vprior_slope_20s=round(wstat("v_prior", 20.0, lambda a: _slope(np.arange(len(a))*GRID_DT, a)), 5),
            # a_long stats over history
            feat_along_mean_5s=round(wstat("a_long_debias", 5.0, np.mean), 5),
            feat_along_rms_10s=round(wstat("a_long_debias", 10.0, lambda a: np.sqrt(np.mean(a**2))), 5),
            feat_along_rms_20s=round(wstat("a_long_debias", 20.0, lambda a: np.sqrt(np.mean(a**2))), 5),
            # spectral temporal behaviour
            feat_vspec_slope_5s=round(wstat("v_spec", 5.0, lambda a: _slope(np.arange(len(a))*GRID_DT, a)), 5),
            feat_vspec_slope_10s=round(wstat("v_spec", 10.0, lambda a: _slope(np.arange(len(a))*GRID_DT, a)), 5),
            feat_vspec_slope_20s=round(wstat("v_spec", 20.0, lambda a: _slope(np.arange(len(a))*GRID_DT, a)), 5),
            feat_vspec_std_10s=round(wstat("v_spec", 10.0, np.std), 5),
            feat_vspec_std_20s=round(wstat("v_spec", 20.0, np.std), 5),
            feat_vspec_max_20s=round(wstat("v_spec", 20.0, np.max), 5),
            feat_vspec_frac_gt15_30s=round(wstat("v_spec", 30.0, lambda a: np.mean(a > 15.0)), 5),
            feat_vspec_frac_gt13_30s=round(wstat("v_spec", 30.0, lambda a: np.mean(a > 13.0)), 5),
            # saturation indicator: v_spec in plateau band while v_prior climbing
            feat_sat_indicator=round(
                float((12.0 <= vs_m <= 15.5) and
                      wstat("v_prior", 10.0, lambda a: _slope(np.arange(len(a))*GRID_DT, a)) > 0.05), 3),
            # band powers (log)
            feat_vib_low=round(float(w.vib_low.mean()), 4),
            feat_vib_b1=round(float(w.vib_b1.mean()), 4),
            feat_vib_b2=round(float(w.vib_b2.mean()), 4),
            feat_vib_mid=round(float(w.vib_mid.mean()), 4),
            feat_vib_b4=round(float(w.vib_b4.mean()), 4),
            feat_vib_high=round(float(w.vib_high.mean()), 4),
            feat_vib_total=round(float(w.vib_total.mean()), 4),
            feat_gyro_power=round(float(w.gyro_power.mean()), 4),
            feat_band_tilt=round(float(w.band_tilt.mean()), 4),
            # motion / context
            feat_gyro_mag_mean=round(float(w.gyro_mag.mean()), 5),
            feat_gyro_mag_std_10s=round(wstat("gyro_mag", 10.0, np.std), 5),
            feat_yaw_abs_mean=round(float(np.abs(w.yaw_rate.to_numpy()).mean()), 5),
            feat_yaw_abs_max_20s=round(wstat("yaw_rate", 20.0, lambda a: np.max(np.abs(a))), 5),
            feat_a_mag_mean=round(float(w.a_mag.mean()), 5),
            feat_stationary_frac=round(float(w.stationary.mean()), 4),
            # bookkeeping (NOT features - audit)
            _v_spec_std=round(float(vs.std()), 4),
            _steady=bool((vt.max() - vt.min()) < 2.0),
            _moving=bool(vt_m > 1.0 and vs_m > 1.0),
            _reliable_vspec=bool(vs_m >= 2.0),
        )
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> int:
    grids = {}
    for tag, d in TRIPS.items():
        print(f"replaying {tag} ...", flush=True)
        g = replay_trip(tag, d)
        g.to_csv(OUT / f"grid_{tag}.csv", index=False)
        grids[tag] = g

    for ws, name in ((5.0, "windows_5s.csv"), (2.0, "windows_2s.csv")):
        parts = [make_windows(g, ws) for g in grids.values()]
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(OUT / name, index=False)
        out = df[df.regime == "outage"]
        print(f"\n{name}: {len(df)} rows ({len(out)} outage)")
        for tag in TRIPS:
            s = out[out.trip == tag]
            sm = s[s._steady & s._moving & s._reliable_vspec]
            print(f"  {tag}: outage={len(s):4d}  steady&moving&reliable={len(sm):4d}  "
                  f"v_true {s.v_true_mean.min():.1f}-{s.v_true_mean.max():.1f}  "
                  f"resid mean {s.residual.mean():+.2f} std {s.residual.std():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
