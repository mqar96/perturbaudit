import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C1, C2 = "#2a78d6", "#eb6834"; INK, MUTED = "#2b2b2a", "#77766f"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
                     "axes.axisbelow": True, "grid.color": "#e6e5df", "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK})
E = pd.read_csv("derived/transfer_c9/transfer_edges.csv")
E = E[(E.panel == "k562_p300e") & (E.top_k == 100) & E.testable]
order = [("tce", "TCE"), ("gbm_grn", "GBM-GRN"), ("inspre", "inspre"), ("dotears", "dotears")]
fig, ax = plt.subplots(figsize=(6.6, 3.6))
for i, (m, lab) in enumerate(order):
    d = E[E.method == m]
    for j, (flag, col, name) in enumerate([(False, C1, "Supported in K562"), (True, C2, "Unsupported in K562")]):
        s = d[d.unsupported == flag]
        if not len(s): continue
        r = s.replicated.astype(float); p = r.mean(); se = np.sqrt(p * (1 - p) / len(r))
        x = i + (j - 0.5) * 0.38
        ax.bar(x, p, 0.36, yerr=se, color=col, capsize=3, label=name if i == 1 else None, zorder=3)
        ax.text(x, p + se + 0.02, f"n={len(r)}", ha="center", fontsize=7.5, color=MUTED)
ax.axhline(0.257, color=MUTED, lw=1, ls="--", label="Random gene pairs")
ax.axhline(0.405, color=MUTED, lw=1, ls=":", label="Random pairs with K562 |z|≥2")
ax.set_xlim(-0.55, 3.45); ax.set_xticks(range(4), [l for _, l in order]); ax.set_ylim(0, 1); ax.set_ylabel("Replicated in RPE1 knockdowns")
ax.set_title("K562 top-100 edges (p=300) tested in an independent cell line", loc="left", fontsize=10)
ax.legend(frameon=False, loc="upper right", fontsize=8.5)
fig.tight_layout(); fig.savefig("project/02_PerturbSeq_Falsification_Toolkit/05_Development_Validation/figures/fig_transfer_c9.png", dpi=200)
