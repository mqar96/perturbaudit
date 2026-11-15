"""Cross-platform concordance, genome-wide seed-match off-targets (2 screens) and off-target/guide check power curves.
Fig. 3 and Fig. 4 are drawn by make_fig34.py from the frozen v0.2.1 tables.
Usage: python analysis/fig_final.py [FIG_DIR] [SIM_POWER_CSV]"""
import sys
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#2e9e6b", "#8a5cc2"; INK, MUTED = "#2b2b2a", "#77766f"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
                     "axes.axisbelow": True, "grid.color": "#e6e5df", "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK})
FIG = (sys.argv[1] if len(sys.argv) > 1 else "figures") + "/"
# 2 concordance
D = pd.read_csv("derived/cross_platform_pairs.csv.gz").dropna(subset=["zT"])  # from cross_platform.py
D["same"] = np.sign(D.zT) == np.sign(D.zK562)
cc = pd.DataFrame([dict(line=l, sign=D[D.line == l].same.mean(), rep=(D[D.line == l].same & (D[D.line == l].zT.abs() >= 3)).mean())
                   for l in ["Jurkat", "HepG2", "RPE1", "HCT116"]])
fig, ax = plt.subplots(figsize=(5.2, 3.4)); x = np.arange(4)
ax.bar(x - 0.2, cc.sign, 0.38, color=C1, label="Same sign"); ax.bar(x + 0.2, cc.rep, 0.38, color=C3, label="Same sign and |z| ≥ 3")
for i, (a, b) in enumerate(zip(cc.sign, cc.rep)): ax.text(i - 0.2, a + 0.02, f"{a:.2f}", ha="center", fontsize=7.5); ax.text(i + 0.2, b + 0.02, f"{b:.2f}", ha="center", fontsize=7.5)
ax.axhline(0.5, color=MUTED, lw=0.8, ls="--"); ax.set_xticks(x, ["Jurkat", "HepG2", "RPE1", "HCT116\n(Xaira)"]); ax.set_ylim(0, 1.05)
ax.set_ylabel("Strong K562 GWPS effects (|z| ≥ 5, n ≈ 8,900)"); ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_title("How well K562 effects carry over to each line", loc="left", fontsize=9.5)
fig.tight_layout(); fig.savefig(FIG + "fig_concordance_4lines.png", dpi=200); plt.close(fig)
# 3 genome-wide seed matches
H = pd.read_csv("derived/m4_genomewide_hct116_ranks.csv"); Gw = pd.read_csv("derived/m4_genomewide_gwps.csv", index_col=0)
bins = ["8-9", "10", "11", "12", "13", "14-18"]
fig, ax = plt.subplots(figsize=(5.6, 3.6))
h = H.set_index("bin").loc[bins]; g = Gw.loc[bins]
ax.plot(bins, 100 * h.gene_pct_lt_0p5pct, "o-", color=C1, lw=2, ms=6, label="HCT116 (Xaira, 18,294 perturbations)")
ax.plot(bins, 100 * g.bottom_0p5pct, "s-", color=C2, lw=2, ms=6, label="K562 GWPS (Replogle, 11,258)")
ax.axhline(0.5, color=MUTED, ls="--", lw=1); ax.text(5, 0.9, "no seed effect (0.5%)", ha="right", fontsize=7.5, color=MUTED)
ax.set_xlabel("Seed match length (bp, PAM-proximal) at a distant promoter (> 1 Mb)"); ax.set_ylabel("Matched gene in its bottom 0.5% (%)")
ax.set_title("Seed matches anywhere in the genome knock genes down", loc="left", fontsize=9.5); ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(FIG + "fig_m4_genomewide_seed.png", dpi=200); plt.close(fig)
# 4 power
P = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else "results/sim_power.csv"); A = P.groupby(["mode", "strength", "cells"]).sensitivity.mean().reset_index()
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for ax, (mode, title) in zip(axes, [("M4_seedmap", "M4 off-target (with seed map)"), ("M2_guides", "M2 guide-specific (no seed map)")]):
    for c, col in zip([50, 150, 400], [C2, C1, C3]):
        d = A[(A["mode"] == mode) & (A.cells == c)]; ax.plot(d.strength, d.sensitivity, "o-", color=col, lw=2, ms=5, label=f"{c} cells per perturbation")
    ax.set_title(title, loc="left", fontsize=9.5); ax.set_xlabel("Off-target effect strength"); ax.set_ylim(0, 1.05)
axes[0].set_ylabel("Artefact edges flagged (sensitivity)"); axes[1].legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(FIG + "fig_h3_power.png", dpi=200); plt.close(fig)
print("figures written")
