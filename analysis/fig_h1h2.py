"""Supplementary Fig. S1: decoy-variable FDR estimates vs the true FDP in simulation (TCE network, 50 replicates).
Usage: python analysis/fig_h1h2.py SIM_CALIBRATION_ALL_P_CSV OUT_PNG"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BLUE, ORANGE, VIOLET, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#6250d6", "#1baf7a", "#222222", "#6b6b6b"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED})
d = pd.read_csv(sys.argv[1])
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
sets = [("clean", "Clean", BLUE), ("confounded", "Latent confounding", ORANGE), ("inactive30", "30% inactive guides", VIOLET),
        ("misassign5", "5% misassigned cells", AQUA)]
mk = {20: "o", 40: "s", 80: "^"}
for s, lab, c in sets:
    for p in (20, 40, 80):
        g = d[(d.setting == s) & (d.p == p)].groupby("top_k")[["true_FDP", "est_perm", "est_knockoff"]].mean()
        ax[0].scatter(g.true_FDP, g.est_perm, marker=mk[p], s=18, color=c, label=lab if p == 20 else None, zorder=3)
ax[0].plot([0, 1], [0, 1], ls="--", color=MUTED, lw=0.8)
ax[0].fill_between([0, 1], [-0.11, 0.89], [0.11, 1.11], color="#e9e8e3", lw=0, zorder=1)
ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1)
ax[0].set_xlabel("True FDP (mean of 50 replicates)"); ax[0].set_ylabel("Permutation-decoy FDR estimate")
ax[0].set_title("A  No systematic artefact\n(shaded: ±0.11; ○ p = 20, □ 40, △ 80; top 20/40/80/160 edges)", loc="left", fontsize=8)
ax[0].legend(frameon=False, fontsize=7, loc="upper left")
rates = [("off0", 0), ("off05", 5), ("off10", 10), ("off15", 15), ("off25", 25)]
for p, ls in [(20, ":"), (40, "-"), (80, "--")]:
    t = [d[(d.setting == s) & (d.top_k == 20) & (d.p == p)] for s, _ in rates]
    x = [r for _, r in rates]
    ax[1].plot(x, [v.true_FDP.mean() for v in t], ls=ls, marker="o", ms=4, color=ORANGE, label=f"True FDP, p = {p}")
    ax[1].plot(x, [v.est_perm.mean() for v in t], ls=ls, marker="s", ms=4, color=BLUE, label=f"Decoy estimate, p = {p}")
ax[1].set_xlabel("Guides with seed off-targets (%)"); ax[1].set_ylabel("FDP of top-20 edges"); ax[1].set_ylim(-0.02, 0.7)
ax[1].set_title("B  Seed off-targets are invisible to decoys", loc="left", fontsize=8)
ax[1].legend(frameon=False, fontsize=6.5, loc="upper left", ncol=2)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=300)
