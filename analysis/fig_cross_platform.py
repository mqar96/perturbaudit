import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C = {"Jurkat": "#2a78d6", "HepG2": "#2e9e6b", "RPE1": "#8a5cc2", "HCT116": "#eb6834"}; INK, MUTED = "#2b2b2a", "#77766f"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e6e5df", "axes.axisbelow": True})
D = pd.read_csv("derived/cross_platform_pairs.csv.gz")
D["effect_bin"] = pd.cut(D.zK562.abs(), [5, 10, 20, 1e9], labels=["5–10", "10–20", "> 20"])
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
ax = axes[0]
for l in ["Jurkat", "HepG2", "RPE1", "HCT116"]:
    g = D[D.line == l].groupby("effect_bin", observed=True).rep.mean()
    ax.plot(range(3), g.values, "o-", color=C[l], lw=2, ms=6, label=l + (" (independent lab)" if l == "HCT116" else ""))
ax.set_xticks(range(3), ["5–10", "10–20", "> 20"]); ax.set_xlabel("Strength of the K562 effect (|z|)"); ax.set_ylabel("Replicated in the other line"); ax.set_ylim(0, 1)
ax.set_title("Bigger effects carry over better, but the platform gap stays", loc="left", fontsize=9.5); ax.legend(frameon=False, fontsize=8)
ax = axes[1]; D["kd"] = pd.cut(D.own_target, [-1e9, -10, -5, -3, 1e9], labels=["≤ −10", "−10 to −5", "−5 to −3", "> −3"])
for l in ["Jurkat", "HepG2", "RPE1", "HCT116"]:
    g = D[D.line == l].groupby("kd", observed=True).rep.agg(["mean", "size"]); g = g[g["size"] >= 20]
    ax.plot([["≤ −10", "−10 to −5", "−5 to −3", "> −3"].index(i) for i in g.index], g["mean"].values, "o-", color=C[l], lw=2, ms=6)
ax.set_xticks(range(4), ["≤ −10", "−10 to −5", "−5 to −3", "> −3"]); ax.set_xlabel("Knockdown of the parent gene in the target line (own z)"); ax.set_ylim(0, 1)
ax.set_title("Strong knockdown in HCT116 does not close the gap", loc="left", fontsize=9.5)
fig.tight_layout(); fig.savefig("project/02_PerturbSeq_Falsification_Toolkit/05_Development_Validation/figures/fig_cross_platform.png", dpi=200)
