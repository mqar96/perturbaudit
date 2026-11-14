import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C1, C2 = "#2a78d6", "#eb6834"; INK, MUTED = "#2b2b2a", "#77766f"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
                     "grid.color": "#e6e5df", "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK})
M = pd.read_csv("derived/multidraw_k562_p100e/multidraw_metrics.csv"); S = pd.read_csv("derived/multidraw_k562_p100e/multidraw_stability.csv")
order = ["tce", "gbm_grn", "inspre", "dotears"]; lab = ["TCE", "GBM-GRN", "inspre", "dotears"]
m = M[M.top_k == 50].set_index("method").loc[order]; s = S[S.top_k == 50].set_index("method").loc[order]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
x = range(len(order))
ax[0].bar([i - 0.2 for i in x], m.unsupported_frac_mean, 0.38, yerr=m.unsupported_frac_std, color=C1, label="Unsupported", capsize=3)
ax[0].bar([i + 0.2 for i in x], m.reversed_frac_mean, 0.38, yerr=m.reversed_frac_std, color=C2, label="Reversed", capsize=3)
ax[0].text(0, 0.03, "0 by\nconstruction", ha="center", color=MUTED, fontsize=8); ax[0].set_xticks(list(x), lab); ax[0].set_ylabel("Fraction of top-50 edges"); ax[0].set_ylim(0, 1); ax[0].legend(frameon=False, loc="upper left")
ax[0].set_title("Intervention support (mean ± sd, 5 draws)", loc="left", fontsize=10)
ax[1].bar(x, s.jaccard_mean, 0.6, yerr=s.jaccard_sd, color=C1, capsize=3)
for i, v in zip(x, s.frac_edges_in_all_draws): ax[1].text(i, 0.93, f"{v:.0%} of edges\nin all 5", ha="center", color=MUTED, fontsize=8)
ax[1].set_xticks(list(x), lab); ax[1].set_ylabel("Pairwise Jaccard, top-50"); ax[1].set_ylim(0, 1.05)
ax[1].set_title("Stability across 80% subsamples", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig("project/02_PerturbSeq_Falsification_Toolkit/05_Development_Validation/figures/fig_multidraw_k562.png", dpi=200)
