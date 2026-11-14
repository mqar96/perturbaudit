import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C1, C2, C3 = "#2a78d6", "#eb6834", "#7a5fc2"; INK, MUTED = "#2b2b2a", "#77766f"
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5df",
                     "axes.axisbelow": True, "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK})
d = pd.read_csv("derived/sim_calib3/calibration_results.csv")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
# A: calibration, no-violation settings
sets = [("clean", "Clean"), ("confounded", "Confounded"), ("inactive30", "30% inactive"), ("misassign5", "5% misassigned")]
for i, (s, lab) in enumerate(sets):
    g = d[d.setting == s].groupby("top_k")[["true_FDP", "est_perm", "est_knockoff"]].mean()
    ax[0].plot(g.true_FDP, g.est_perm, "o-", color=[C1, C3, C2, MUTED][i], ms=4, lw=2, label=lab)
ax[0].plot([0, 0.7], [0, 0.7], ls="--", color=MUTED, lw=1); ax[0].fill_between([0, 0.7], [0, 0.525], [0, 0.875], color="#e6e5df", alpha=0.6, lw=0)
ax[0].set_xlabel("True FDP (mean of 50 replicates)"); ax[0].set_ylabel("Decoy-estimated FDR"); ax[0].set_xlim(0, 0.7); ax[0].set_ylim(0, 0.7)
ax[0].set_title("A  Decoys are calibrated without systematic artefacts\n(shaded: ±25% band; points: top 20/40/80/160 edges)", loc="left", fontsize=9.5)
ax[0].legend(frameon=False, fontsize=8, loc="upper left")
# B: dose response
rates = [("off0", 0), ("off05", 5), ("off10", 10), ("off15", 15), ("off25", 25)]
t20 = [d[(d.setting == s) & (d.top_k == 20)] for s, _ in rates]
x = [r for _, r in rates]
ax[1].errorbar(x, [t.true_FDP.mean() for t in t20], yerr=[t.true_FDP.std() / np.sqrt(len(t)) for t in t20], fmt="o-", color=C2, lw=2, ms=5, capsize=3, label="True FDP")
ax[1].errorbar(x, [t.est_perm.mean() for t in t20], yerr=[t.est_perm.std() / np.sqrt(len(t)) for t in t20], fmt="s-", color=C1, lw=2, ms=5, capsize=3, label="Decoy estimate")
ax[1].set_xlabel("Guides with seed off-targets (%)"); ax[1].set_ylabel("FDP of top-20 edges"); ax[1].set_ylim(-0.02, 0.7)
ax[1].set_title("B  Systematic artefacts are invisible to decoys", loc="left", fontsize=9.5)
ax[1].legend(frameon=False, fontsize=8.5, loc="upper left")
fig.tight_layout(); fig.savefig("project/02_PerturbSeq_Falsification_Toolkit/05_Development_Validation/figures/fig_h1h2_calibration.png", dpi=200)
