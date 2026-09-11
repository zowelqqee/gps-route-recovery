#!/usr/bin/env python3
import warnings, sys
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/Users/arseniyabramidze/delimobil/gps-route-recovery/docs/plots/phase31")

fig, ax = plt.subplots(1, 3, figsize=(19, 5.4))

# (a) regime improvement: full 23 vs clean 21
r_full = [(-4.3, -13.7, -3.4, 9.4, -11.6, 10.7)]   # from honest_out 2s all-23
r_clean = [19.8, 14.9, 33.1, 32.2, 16.2, 11.8]     # 2s no-extrap
regs = ["0-5", "5-10", "10-15", "15-20", "20-25", "25+"]
x = np.arange(len(regs))
ax[0].bar(x - 0.2, r_full[0], 0.4, label="all 23 trips", color="#d62728")
ax[0].bar(x + 0.2, r_clean, 0.4, label="21 in-regime trips", color="#2ca02c")
ax[0].axhline(0, color="k", lw=.8)
ax[0].set_xticks(x); ax[0].set_xticklabels(regs)
ax[0].set_ylabel("corrected-speed MAE improvement  [%]")
ax[0].set_xlabel("true-speed regime [m/s]")
ax[0].set_title("(a) CatBoost residual, trip-LOTO\nfull pool vs pathological trips removed")
ax[0].legend()
ax[0].grid(alpha=.25, axis="y")

# (b) scaling curve
sc = pd.read_csv(OUT / "task8_scaling_clean.csv").drop_duplicates("n_train")
ax[1].errorbar(sc.n_train, sc.csMAE, yerr=sc.csMAE_sd, marker="o", label="all speeds", color="#1f77b4")
ax[1].errorbar(sc.n_train, sc.csMAE_15_25, yerr=sc.csMAE_15_25_sd, marker="s",
               label="15-25 m/s", color="#ff7f0e")
ax[1].axhline(3.536, color="#1f77b4", ls="--", lw=1, label="baseline v_spec (all)")
ax[1].axhline(4.550, color="#ff7f0e", ls="--", lw=1, label="baseline v_spec (15-25)")
ax[1].set_xlabel("number of training trips")
ax[1].set_ylabel("held-out corrected-speed MAE [m/s]")
ax[1].set_title("(b) train-size scaling (7 held-out trips)\nflattens by ~8 trips")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=.25)

# (c) per-trip replay delta D_ratio and medDerr for gated (C) vs baseline
rp = pd.read_csv(OUT / "task12_replay.csv")
rp = rp.sort_values("sat_mean")
xx = np.arange(len(rp))
dmed = (rp.C_bin_gate_medDerr - rp.A_baseline_medDerr)
ax[2].bar(xx, dmed, color=np.where(dmed < 0, "#2ca02c", "#d62728"))
ax[2].axhline(0, color="k", lw=.8)
ax[2].set_xticks(xx)
ax[2].set_xticklabels([f"{t}\n(sat {s:.2f})" for t, s in zip(rp.trip, rp.sat_mean)],
                      rotation=45, ha="right", fontsize=7)
ax[2].set_ylabel("median |D_err| change,  gated ML - baseline  [m]")
ax[2].set_title("(c) offline replay: gated ML effect on along-track lag\n(negative = better; inconsistent)")
ax[2].grid(alpha=.25, axis="y")

fig.suptitle("Phase 31 - large-scale cross-trip validation (23 trips, leave-one-trip-out)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "phase31_summary.png", dpi=130)
print("wrote", OUT / "phase31_summary.png")
