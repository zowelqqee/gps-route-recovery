#!/usr/bin/env python3
"""Plot the shipped panel data: corrected display odometry vs route odometry
vs truth, and the rendered-marker error."""
import json
from math import radians
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/Users/arseniyabramidze/delimobil/gps-route-recovery")
d = json.loads((ROOT / "replay-ui/public/data/replay-07-26.json").read_text())
F = [f for f in d["frames"] if "display" in f]


def hav(a, b, c, e):
    R = 6371000.0
    p1, p2 = radians(a), radians(c)
    dp, dl = radians(c - a), radians(e - b)
    x = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(min(1.0, np.sqrt(x)))


el = np.array([f["elapsed"] for f in F])
d_pos = np.array([f["dEst"] for f in F])
d_route = np.array([f["dRoute"] for f in F])
d_true = np.array([f["dTrue"] for f in F])
gate = np.array([f["display"]["gateActive"] for f in F])
excess = np.array([f["display"]["excessM"] for f in F])
marker_err = np.array([hav(f["tracker"]["lat"], f["tracker"]["lon"],
                           f["truth"]["lat"], f["truth"]["lon"]) for f in F])
route_pt_err = np.array([hav(f["route"]["lat"], f["route"]["lon"],
                             f["truth"]["lat"], f["truth"]["lon"]) for f in F])

fig, ax = plt.subplots(1, 3, figsize=(19, 5.2))

ax[0].plot(el, d_true, "k", lw=2, label="D_true (withheld GPS)")
ax[0].plot(el, d_route, "#d62728", lw=1.6, label="D_route (baseline odometer, topology)")
ax[0].plot(el, d_pos, "#2ca02c", lw=1.6, label="D_position (corrected display odometer)")
ax[0].fill_between(el, 0, d_true.max(), where=gate, color="#2ca02c", alpha=0.06,
                   step="mid", label="saturation gate ON")
ax[0].set_xlabel("outage elapsed [s]"); ax[0].set_ylabel("distance travelled [m]")
ax[0].set_title("(a) 07-26 runtime: corrected display odometry\n"
                "|D_position − D_true|  median 57 m   (route: 266 m)")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.25)

ax[1].plot(el, np.abs(d_pos - d_true), "#2ca02c", lw=1.4, label="|D_position − D_true|  (odometer)")
ax[1].plot(el, marker_err, "#1f77b4", lw=1.4, label="rendered marker vs truth  (2-D, clamped to frontier)")
ax[1].plot(el, route_pt_err, "#d62728", lw=1.2, alpha=.8, label="baseline route point vs truth")
ax[1].set_xlabel("outage elapsed [s]"); ax[1].set_ylabel("position error [m]")
ax[1].set_title("(b) odometer error 57 m; rendered marker 180 m\n"
                "(median) — the committed-route frontier paces the marker")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.25)

ax[2].plot(el, excess, "#ff7f0e", lw=1.4)
ax[2].fill_between(el, 0, excess, color="#ff7f0e", alpha=.2)
ax[2].set_xlabel("outage elapsed [s]"); ax[2].set_ylabel("excess held at frontier [m]")
ax[2].set_title("(c) excess_position_distance: the marker holds this far\n"
                "short of D_position and releases it at each junction")
ax[2].grid(alpha=.25)

fig.suptitle("Phase 32 display-position branch — wired into the tracker runtime (rf-07-26, topology byte-identical)",
             fontsize=13)
fig.tight_layout()
out = ROOT / "docs/plots/phase32/phase32_panel_runtime.png"
fig.savefig(out, dpi=130)
print("wrote", out)
