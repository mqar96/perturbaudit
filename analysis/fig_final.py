"""Final figures added 12 Nov: transfer to 4 lines with random baseline, cross-platform concordance, genome-wide
seed-match off-targets (2 screens), H3 power curves, enrichment over chance."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#2e9e6b", "#8a5cc2"; INK, MUTED = "#2b2b2a", "#77766f"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
                     "axes.axisbelow": True, "grid.color": "#e6e5df", "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK})
FIG = "project/02_PerturbSeq_Falsification_Toolkit/05_Development_Validation/figures/"
# 1 transfer, 4 lines
S = pd.read_csv("derived/transfer_4lines_summary.csv")
combos = [("k562_p300e", m) for m in ["gbm_grn", "dotears", "inspre", "adapre", "sdcd"]] + [("gwps_p788", m) for m in ["gbm_grn", "dotears", "inspre", "adapre"]]
lab = {"gbm_grn": "GBM-GRN", "dotears": "dotears", "inspre": "inspre", "adapre": "ADAPRE", "sdcd": "SDCD"}
fig, axes = plt.subplots(1, 4, figsize=(20, 4.4), sharey=True)
for ax, t in zip(axes, ["RPE1", "HepG2", "Jurkat", "HCT116"]):
    for i, (p, m) in enumerate(combos):
        r = S[(S.target == t) & (S.source == p) & (S.method == m)]
        if not len(r): continue
        r = r.iloc[0]
        ax.bar(i - 0.2, r.audited, 0.38, color=C1, label="Audited in K562" if i == 0 else None)
        ax.bar(i + 0.2, r.flagged, 0.38, color=C2, label="Flagged in K562" if i == 0 else None)
        ax.plot([i - 0.42, i + 0.42], [r.random] * 2, color=INK, lw=1.2, label="Random gene pairs" if i == 0 else None)
        if r.fisher_p < 0.05: ax.text(i, max(r.audited, r.flagged) + 0.03, "*", ha="center", fontsize=11)
    ax.set_xticks(range(len(combos)), [f"{lab[m]}\n{'K562' if p.startswith('k562') else 'GWPS'}" for p, m in combos], fontsize=6.5)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{t} ({'independent platform' if t == 'HCT116' else 'same platform as source'})", loc="left", fontsize=9.5)
axes[0].set_ylabel("Edges replicated in the other cell line"); axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.text(0.01, 0.01, "* Fisher P < 0.05, audited vs flagged", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(FIG + "fig_transfer_4lines.png", dpi=200); plt.close(fig)
# 2 concordance
cc = pd.DataFrame(dict(line=["Jurkat", "HepG2", "RPE1", "HCT116"], sign=[0.914, 0.873, 0.793, 0.69], rep=[0.648, 0.576, 0.525, 0.195]))
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
P = pd.read_csv("derived/sim_power.csv"); A = P.groupby(["mode", "strength", "cells"]).sensitivity.mean().reset_index()
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for ax, (mode, title) in zip(axes, [("M4_seedmap", "M4 off-target (with seed map)"), ("M2_guides", "M2 guide-specific (no seed map)")]):
    for c, col in zip([50, 150, 400], [C2, C1, C3]):
        d = A[(A["mode"] == mode) & (A.cells == c)]; ax.plot(d.strength, d.sensitivity, "o-", color=col, lw=2, ms=5, label=f"{c} cells per perturbation")
    ax.axhline(0.7, color=MUTED, ls="--", lw=1); ax.set_title(title, loc="left", fontsize=9.5); ax.set_xlabel("Off-target effect strength"); ax.set_ylim(0, 1.05)
axes[0].set_ylabel("Artefact edges flagged (sensitivity)"); axes[1].legend(frameon=False, fontsize=8, loc="lower right")
axes[0].text(2.4, 0.66, "plan target 70%", ha="right", fontsize=7.5, color=MUTED)
fig.tight_layout(); fig.savefig(FIG + "fig_h3_power.png", dpi=200); plt.close(fig)
# 5 enrichment over chance
E = pd.read_csv("derived/audited_vs_chance.csv").dropna(subset=["enrichment"])
E = E[E.top_k >= 20].sort_values("enrichment")
fig, ax = plt.subplots(figsize=(6.2, 0.22 * len(E) + 1))
ax.barh(range(len(E)), E.enrichment, color=[C1 if v >= 1.5 else (C2 if v < 1 else MUTED) for v in E.enrichment])
ax.axvline(1, color=INK, lw=1); ax.set_yticks(range(len(E)), [f"{m}  ·  {p}" for p, m in zip(E.panel, E.method)], fontsize=7)
ax.set_xlabel("Audited fraction ÷ chance (all ordered pairs)"); ax.grid(axis="x", color="#e6e5df"); ax.grid(axis="y", visible=False)
ax.set_title("Top-50 edges: enrichment of audited edges over chance", loc="left", fontsize=9.5)
ax.text(0.99, 0.02, "blue ≥ 1.5×   grey 1–1.5×   orange < 1× (worse than random pairs)", transform=ax.transAxes, ha="right", fontsize=7, color=MUTED)
fig.tight_layout(); fig.savefig(FIG + "fig_enrichment_over_chance.png", dpi=200); plt.close(fig)
print("figures written")
