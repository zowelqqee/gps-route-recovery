#!/usr/bin/env python3
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from phase32_alltrip import corrected, trip_series

OUT = Path("/Users/arseniyabramidze/delimobil/gps-route-recovery/docs/plots/phase32")
fig, ax = plt.subplots(1, 3, figsize=(19, 5.4))

# (a) 07-26 longitudinal traces
b = corrected("rf-07-26", "iso", "soft")
bb = corrected("rf-07-26", "iso", "soft", bound=250)
T = b["t"] - b["t"][0]
ax[0].plot(T, b["D_true"], "k", lw=2, label="D_true (withheld GPS)")
ax[0].plot(T, b["D_route"], "#d62728", lw=1.6, label="D_route (baseline, topology)")
ax[0].plot(T, b["D_pos"], "#2ca02c", lw=1.6, label="D_position (iso-soft, unbounded)")
ax[0].plot(T, bb["D_pos"], "#1f77b4", lw=1.4, ls="--", label="D_position (bounded |Δ|≤250 m)")
ax[0].set_xlabel("outage elapsed [s]"); ax[0].set_ylabel("distance travelled [m]")
ax[0].set_title("(a) 07-26: display odometry vs route odometry\nmax |D_err|: 724 → 372 (unbounded) / 474 (bounded)")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.25)

# (b) cross-trip: Δmax|D_err| vs baseline D/D_true
df = pd.read_csv(OUT / "task7_1d_alltrip.csv")
s = df[(df.variant == "iso-soft") & ~df.broken & ~df.extrap].copy()
s["dmax"] = s.max_pos - s.max_route
ax[1].scatter(s.Dr_route, s.dmax, s=60, c=np.where(s.dmax < -15, "#2ca02c",
              np.where(s.dmax > 15, "#d62728", "#999")))
for _, r in s.iterrows():
    if abs(r.dmax) > 100:
        ax[1].annotate(r.trip, (r.Dr_route, r.dmax), fontsize=7, xytext=(3, 3),
                       textcoords="offset points")
ax[1].axhline(0, color="k", lw=.8); ax[1].axvline(1.0, color="grey", ls=":", lw=1)
ax[1].set_xlabel("baseline D_route / D_true"); ax[1].set_ylabel("Δ max |D_err|,  D_position − D_route  [m]")
ax[1].set_title("(b) correction helps iff baseline UNDER-reads (D/D_true<1)\nunbounded; no online signal for this")
ax[1].grid(alpha=.25)

# (c) bound sweep tradeoff
bd = pd.read_csv(OUT / "task10_bound_sweep.csv")
bd = bd[bd.gate == "soft"].merge(df[["trip", "broken", "extrap"]].drop_duplicates(), on="trip")
bd = bd[~bd.broken & ~bd.extrap]
Bs = [50, 100, 150, 200, 300, 500]
med = [(bd[bd.B == B].max_pos - bd[bd.B == B].max_route).median() for B in Bs]
worst = [(bd[bd.B == B].max_pos - bd[bd.B == B].max_route).max() for B in Bs]
rf = [bd[(bd.B == B) & (bd.trip == "rf-07-26")].max_pos.iloc[0] for B in Bs]
ax[2].plot(Bs, med, "-o", label="median Δmax across 20 trips", color="#2ca02c")
ax[2].plot(Bs, worst, "-s", label="worst-trip Δmax (= +B, capped)", color="#d62728")
ax[2].plot(Bs, [r - 724 for r in rf], "-^", label="07-26 Δmax (from 724 m)", color="#1f77b4")
ax[2].axhline(0, color="k", lw=.8)
ax[2].set_xlabel("correction bound B  [m]"); ax[2].set_ylabel("Δ max |D_err|  [m]")
ax[2].set_title("(c) bounded display correction: the safety/benefit trade\nB≈200–300 m: 07-26 −200…−300 m, others capped at +B")
ax[2].legend(fontsize=8); ax[2].grid(alpha=.25)

fig.suptitle("Phase 32 — route/topology odometry vs display/position odometry (23 trips)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "phase32_summary.png", dpi=130)
print("wrote", OUT / "phase32_summary.png")
