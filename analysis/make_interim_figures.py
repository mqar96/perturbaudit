import json, glob, os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "project/02_PerturbSeq_Falsification_Toolkit/05_Development_Validation/figures"
os.makedirs(OUT, exist_ok=True)
C1, C2 = "#2a78d6", "#eb6834"
INK, MUTED = "#2b2b2a", "#77766f"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5df",
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False})

def bars(ax, labels, series, names, colors, ylog=False, annot=None):
    n = len(series); w = 0.8 / n; x = np.arange(len(labels))
    for k, (vals, nm, c) in enumerate(zip(series, names, colors)):
        pos = x - 0.4 + w * (k + 0.5)
        ax.bar(pos, vals, width=w * 0.92, color=c, label=nm)
        if annot is not None:
            for xi, v, a in zip(pos, vals, annot[k]):
                if np.isfinite(v):
                    ax.text(xi, v, a, ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    if ylog: ax.set_yscale("log")

m3 = pd.read_csv("derived/m3_cross_dataset.csv")
order = ["Replogle K562 essential", "Replogle K562 genome-wide", "Replogle RPE1 essential", "Zhu/Marson CD4 T cells (Stim8hr, guide 1)", "Frangieh melanoma (Cas9 KO)", "Papalexi THP-1 (Cas9 KO)"]
short = ["K562 ess.\nCRISPRi", "K562 GW\nCRISPRi", "RPE1\nCRISPRi", "CD4 T\nCRISPRi", "Frangieh\nCas9 KO", "Papalexi\nCas9 KO"]
s1, s2, a1, a2 = [], [], [], []
for d in order:
    r = m3[m3.dataset == d].set_index("window")
    for w, s, a in [("0-1 kb", s1, a1), ("1-10 kb", s2, a2)]:
        v = r.at[w, "enrichment"] if w in r.index else np.nan
        nn = int(r.at[w, "sig_in_window"]) if w in r.index else 0
        s.append(v if (np.isfinite(v) and v > 0 and nn >= 5) else np.nan); a.append(f"n={nn}")
fig, ax = plt.subplots(figsize=(7.2, 3.4))
bars(ax, short, [s1, s2], ["target within 0-1 kb", "target within 1-10 kb"], [C1, C2], ylog=True, annot=[a1, a2])
ax.axhline(1, color=MUTED, lw=0.8, ls="--")
for xi in [4, 5]:
    ax.text(xi, 1.6, "<5 edges\nwithin 10 kb\n(not shown)", ha="center", fontsize=7, color=MUTED)
ax.set_ylabel("Enrichment of strong effects\n(observed / distance-matched expected)")
ax.set_title("Genomic-neighbour effects are CRISPRi-specific (M3)", loc="left", fontsize=10, color=INK)
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_m3_neighbour_enrichment.png", dpi=200); plt.close(fig)

dv = json.load(open("derived/m2_donor_vs_guide.json"))
donor = np.mean([v["donor_split"]["frac_one_side"] for v in dv.values()]) * 100
guide = list(dv.values())[0]["guide_split"]["frac_one_side"] * 100
labels = ["CD4 CRISPRi\ndonor halves\n(same guides)", "CD4 CRISPRi\nguide 1 vs guide 2", "Frangieh Cas9 KO\nguide pairs"]
vals = [donor, guide, 1.1]
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.bar(range(3), vals, color=C1, width=0.6)
for i, v in enumerate(vals): ax.text(i, v + 0.8, f"{v:.1f}%", ha="center", fontsize=8, color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Strong edges (|z|>=8) seen in one\nhalf only (other half |z|<1), %")
ax.set_title("Guide-specific edges far exceed donor noise (M2)", loc="left", fontsize=10, color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_m2_guide_vs_donor.png", dpi=200); plt.close(fig)

m4 = pd.read_csv("derived/m4_hartman_gwps_pairs.csv")
m4 = m4[m4.n_edges_parent >= 20]
fig, ax = plt.subplots(figsize=(5.2, 3.2))
bins = np.linspace(0, 1, 21)
ax.hist(m4.frac_explained_null_median, bins=bins, color=C2, alpha=0.9, label="strength-matched random gene (median of 200)")
ax.hist(m4.frac_edges_explained, bins=bins, color=C1, alpha=0.85, label="Hartman off-target gene")
ax.set_xlabel("Share of the guide's edges that are sign-concordant\nedges of the comparison gene's own knockdown")
ax.set_ylabel("Off-target guides (n=%d)" % len(m4))
ax.set_title("Off-target guides inherit the off-target gene's edges (M4)", loc="left", fontsize=10, color=INK)
ax.legend(fontsize=7, loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_m4_offtarget_edges.png", dpi=200); plt.close(fig)

sp = pd.read_csv("project/02_PerturbSeq_Falsification_Toolkit/05_Development_Validation/results/sim_M5b_pipeline_dotears.csv")
sp = sp[sp.top_k == 40]
fig, ax = plt.subplots(figsize=(6.2, 3.2))
bars(ax, list(sp.setting), [sp.FDP_raw.values, sp.FDP_after.values], ["dotears top-40 edges", "after intervention-support filter (M5b)"], [C2, C1],
     annot=[[f"{v:.2f}" for v in sp.FDP_raw], [f"{v:.2f}" for v in sp.FDP_after]])
ax.set_ylabel("True false-discovery proportion"); ax.set_ylim(0, 1)
ax.set_title("Simulation: the screen's own interventions remove most false edges", loc="left", fontsize=10, color=INK)
ax.legend(fontsize=7, loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_sim_m5b.png", dpi=200); plt.close(fig)

rows = []
for P, lab in [("k562_p100", "K562"), ("rpe1_p100", "RPE1")]:
    a = pd.read_csv(f"derived/panels/{P}/audit_summary.csv")
    for m in ["dotears", "sdcd", "inspre", "gbm_grn"]:
        r = a[(a.method == m) & (a.top_k == 50)]
        rows.append((lab, m, (r.unsupported.iloc[0] / 50 * 100) if len(r) else np.nan))
df = pd.DataFrame(rows, columns=["data", "method", "unsupported"])
meths = ["dotears", "sdcd", "inspre", "gbm_grn"]; mlab = ["dotears", "SDCD", "inspre", "GBM-GRN\n(associative)"]
k = df[df.data == "K562"].set_index("method").reindex(meths).unsupported.values
r = df[df.data == "RPE1"].set_index("method").reindex(meths).unsupported.values
fig, ax = plt.subplots(figsize=(6.2, 3.2))
bars(ax, mlab, [k, r], ["K562 essential panel (p=100)", "RPE1 essential panel (p=100)"], [C1, C2],
     annot=[[("" if not np.isfinite(v) else f"{v:.0f}%") for v in k], [("" if not np.isfinite(v) else f"{v:.0f}%") for v in r]])
for i, v in enumerate(r):
    if not np.isfinite(v): ax.text(i + 0.2, 1, "not run", ha="center", fontsize=7, color=MUTED, rotation=90)
ax.set_ylabel("Top-50 edges not supported by the\nscreen's own knockdown of the parent, %")
ax.set_title("Real screens: many inferred edges lack intervention support (M5b)", loc="left", fontsize=10, color=INK)
ax.legend(fontsize=7, loc="upper right"); ax.set_ylim(0, 80)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_real_m5b_methods.png", dpi=200); plt.close(fig)
print(sorted(os.listdir(OUT)))
