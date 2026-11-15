"""Supplementary Fig. S2: end-to-end audit benchmark in simulation.
Usage: python analysis/fig_sim_audit.py SIM_AUDIT_RESULTS_CSV OUT_PNG"""
import sys
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C1, C2, C3 = "#2a78d6", "#eb6834", "#a8a7a0"; INK, MUTED = "#2b2b2a", "#77766f"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
                     "axes.axisbelow": True, "grid.color": "#e6e5df", "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK})
D = pd.read_csv(sys.argv[1])
sets = ["clean", "confounded", "offtarget15", "offtarget15_withM4", "inactive30", "misassign5"]; labs = ["Clean", "Latent\nconfounding", "15% off-target\n(no seed map)", "15% off-target\n+ seed map (M4)", "30% inactive\nguides", "5% mis-\nassignment"]
meths = [("gbm_grn", "GBM-GRN"), ("dotears", "dotears"), ("inspre", "inspre"), ("tce", "TCE (knockdown network; truth = descendants)")]
fig, axes = plt.subplots(2, 2, figsize=(13, 7.6), sharey=True); axes = axes.ravel()
for ax, (m, ml) in zip(axes, meths):
    for i, s in enumerate(sets):
        d = D[(D.setting == s) & (D.method == m)]
        for j, (col, c, lab) in enumerate([("precision_all", C3, "All top-40 edges"), ("precision_audited", C1, "Audited"), ("precision_flagged", C2, "Flagged")]):
            v = d[col].dropna(); mu = v.mean(); se = v.std() / np.sqrt(max(len(v), 1))
            ax.bar(i + (j - 1) * 0.27, mu, 0.26, yerr=se, color=c, capsize=2, label=lab if i == 0 else None)
        ax.text(i, 1.03, f"keeps {d.true_kept.mean():.0%}", ha="center", fontsize=7.5, color=MUTED)
    ax.set_xticks(range(len(sets)), labs, fontsize=7.5); ax.set_ylim(0, 1.1)
    ax.set_title(ml, loc="left", fontsize=10)
axes[0].set_ylabel("Precision (fraction true)"); axes[2].set_ylabel("Precision (fraction true)")
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=8.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.005))
fig.suptitle("Simulation, 20 replicates per setting: audit verdicts vs known truth ('keeps' = share of true top-40 edges passing the audit)", fontsize=9, x=0.01, ha="left", color=MUTED)
fig.tight_layout(rect=(0, 0.06, 1, 0.97)); fig.savefig(sys.argv[2], dpi=200)
