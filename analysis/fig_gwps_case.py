import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#7a5fc2", "#d9d8d2"; INK, MUTED = "#2b2b2a", "#77766f"
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
                     "axes.axisbelow": True, "grid.color": "#e6e5df", "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK})
D = "derived/panels/gwps_p788"
fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.2), gridspec_kw=dict(width_ratios=[1.25, 1.1, 1]))
# A: verdict composition by top-k
ks = [25, 50, 100, 200, 500]
for m, lab, off in [("inspre", "inspre", -0.2), ("inspre_dag", "inspre (DAG)", 0.2)]:
    E = pd.read_csv(f"{D}/api_edges_{m}.csv")
    a = [(E.head(k).verdict == "audited").mean() for k in ks]
    r = [(E.head(k).verdict == "reversed").mean() for k in ks]
    u = [(E.head(k).verdict == "unsupported").mean() for k in ks]
    x = np.arange(len(ks)) + off
    ax[0].bar(x, a, 0.36, color=C1, label="Audited" if m == "inspre" else None)
    ax[0].bar(x, u, 0.36, bottom=a, color=C4, label="Unsupported" if m == "inspre" else None)
    ax[0].bar(x, r, 0.36, bottom=np.add(a, u), color=C2, label="Reversed" if m == "inspre" else None)
for i, k in enumerate(ks):
    ax[0].text(i - 0.2, 1.02, "N", ha="center", fontsize=7, color=MUTED); ax[0].text(i + 0.2, 1.02, "D", ha="center", fontsize=7, color=MUTED)
ax[0].set_xticks(range(len(ks)), [f"top {k}" for k in ks]); ax[0].set_ylim(0, 1.1); ax[0].set_ylabel("Fraction of edges")
ax[0].set_title("A  inspre on genome-wide K562 (788 genes)\nN = no DAG constraint, D = DAG (authors' setting)", loc="left", fontsize=9.5)
ax[0].legend(frameon=False, fontsize=8, loc="lower right", ncol=3, bbox_to_anchor=(1, -0.3))
# B: RPL3 hub
E = pd.read_csv(f"{D}/api_edges_inspre.csv")
h = E[(E.parent == "RPL3")].head(12)
y = np.arange(len(h))
ax[1].barh(y - 0.2, h.z_parent_kd_on_child, 0.38, color=C1, label="RPL3 knockdown → child")
ax[1].barh(y + 0.2, h.z_child_kd_on_parent, 0.38, color=C2, label="child knockdown → RPL3")
ax[1].set_yticks(y, h.child); ax[1].invert_yaxis(); ax[1].axvline(0, color=MUTED, lw=0.8)
ax[1].axvline(-3, color=MUTED, lw=0.8, ls=":"); ax[1].axvline(3, color=MUTED, lw=0.8, ls=":")
ax[1].set_xlabel("Knockdown effect (z)"); ax[1].grid(axis="x", color="#e6e5df"); ax[1].grid(axis="y", visible=False)
ax[1].set_title("B  inspre's top hub: RPL3 → ribosomal genes", loc="left", fontsize=9.5)
ax[1].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2)
# C: RPE1 confirmation
T = pd.read_csv("derived/gwps_p788_transfer_rpe1.csv")
T = T[(T.method == "inspre") & T.verdict.isin(["audited", "unsupported", "reversed"])].set_index("verdict").loc[["audited", "unsupported", "reversed"]]
x = np.arange(3)
ax[2].bar(x - 0.2, T.rep_forward, 0.38, color=C1, label="Forward edge replicates")
ax[2].bar(x + 0.2, T.rpe1_child_kd_moves_parent, 0.38, color=C2, label="Child KD moves parent")
for i, n in enumerate(T.n_testable_in_rpe1): ax[2].text(i, 0.95, f"n={n}", ha="center", fontsize=7.5, color=MUTED)
ax[2].set_xticks(x, ["Audited", "Unsupported", "Reversed"]); ax[2].set_ylim(0, 1); ax[2].set_ylabel("Fraction in RPE1 screen")
ax[2].set_title("C  Same inspre edges tested in RPE1", loc="left", fontsize=9.5)
ax[2].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.92))
fig.tight_layout(); fig.savefig("project/02_PerturbSeq_Falsification_Toolkit/05_Development_Validation/figures/fig_gwps_inspre_case.png", dpi=200)
