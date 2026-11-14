import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C1, C2 = "#2a78d6", "#eb6834"; INK, MUTED = "#2b2b2a", "#77766f"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
                     "axes.axisbelow": True, "grid.color": "#e6e5df", "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK})
T = pd.concat([pd.read_csv(f) for f in ["derived/transfer_rpe1_generic.csv", "derived/transfer_hepg2.csv", "derived/transfer_jurkat.csv"]])
T = T[T.verdict.isin(["audited", "unsupported", "reversed"])]
combos = [("k562_p300e", "gbm_grn", "GBM-GRN\nK562"), ("k562_p300e", "dotears", "dotears\nK562"), ("k562_p300e", "inspre", "inspre\nK562"),
          ("k562_p300e", "adapre", "ADAPRE\nK562"), ("k562_p300e", "sdcd", "SDCD\nK562"), ("gwps_p788", "gbm_grn", "GBM-GRN\nGWPS"),
          ("gwps_p788", "inspre", "inspre\nGWPS"), ("gwps_p788", "adapre", "ADAPRE\nGWPS")]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.1), sharey=True)
summary = []
for ax, tgt in zip(axes, ["RPE1", "HepG2", "Jurkat"]):
    for i, (pn, m, lab) in enumerate(combos):
        d = T[(T.target == tgt) & (T.panel == pn) & (T.method == m)]
        a = d[d.verdict == "audited"]; f = d[d.verdict != "audited"]
        ra = float(a.replicated.iloc[0]) if len(a) else np.nan
        rf = float((f.replicated * f.n).sum() / f.n.sum()) if len(f) else np.nan
        ax.bar(i - 0.2, ra, 0.38, color=C1, label="Audited in K562" if i == 0 else None)
        ax.bar(i + 0.2, rf, 0.38, color=C2, label="Flagged in K562 (unsupported or reversed)" if i == 0 else None)
        summary.append(dict(target=tgt, source=pn, method=m, audited=ra, flagged=rf))
    ax.set_xticks(range(len(combos)), [c[2] for c in combos], fontsize=7.5); ax.set_ylim(0, 1)
    ax.set_title(f"Tested in {tgt}", loc="left", fontsize=10)
axes[0].set_ylabel("Edges replicated in the other cell line")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig("project/02_PerturbSeq_Falsification_Toolkit/05_Development_Validation/figures/fig_transfer_3lines.png", dpi=200)
pd.DataFrame(summary).to_csv("derived/transfer_3lines_summary.csv", index=False)
print(pd.DataFrame(summary).round(2).to_string(index=False))
