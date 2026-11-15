"""Fig 3 (enrichment over chance) and Fig 4 (matched transfer) from frozen v0.2.1 tables.
Usage: python make_fig34.py FROZEN_DIR OUT_DIR
FROZEN_DIR holds audited_vs_chance.csv, transfer_matched_3lines.csv and transfer_matched_hct116.csv (CC BY-NC-SA record).
"""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FROZEN, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, VIOLET = "#2a78d6", "#eb6834", "#6250d6"
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e4e4e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False})

METHOD = {"adapre": "ADAPRE", "causalgrn": "CausalGRN", "dotears": "dotears", "gbm_grn": "GBM",
          "gies": "GIES", "inspre": "inspre", "inspre_default": "inspre (default)",
          "inspre_dag": "inspre (DAG)", "sdcd": "SDCD", "sdcd_full": "SDCD (all cells)", "tce": "TCE control"}
PANEL = {"gwps_p788": "K562 GW p788", "gwps_p1434": "K562 GW p1434", "k562_p300e": "K562 ess. p300",
         "k562_p100cg": "K562 p100 (CausalBench)", "k562_p50": "K562 p50", "rpe1_p100e": "RPE1 p100",
         "norman_p100": "Norman p100", "tian21i_p100": "Tian CRISPRi p100",
         "tian21a_p100": "Tian CRISPRa p100", "adamson_p100": "Adamson p100"}

# ---------------- Fig 3 ----------------
a = pd.read_csv(f"{FROZEN}/audited_vs_chance.csv")
a = a[(a.method != "tce") & (a.n_testable > 0)].copy()
order = list(PANEL)
a["pi"] = a.panel.map({p: i for i, p in enumerate(order)})
a = a.sort_values(["pi", "enrichment"], ascending=[True, False]).reset_index(drop=True)
floor = 0.02
a["plot_e"] = a.enrichment.clip(lower=floor)
fig, ax = plt.subplots(figsize=(3.6, 6.8))
y = np.arange(len(a))[::-1]
ax.axvline(1, color=MUTED, lw=0.8, ls="--", zorder=1)
for yi, (_, r) in zip(y, a.iterrows()):
    zero = r.enrichment <= 0
    ax.plot([1, r.plot_e], [yi, yi], color=GRID, lw=1.2, zorder=1)
    ax.scatter(r.plot_e, yi, s=22, marker="<" if zero else "o", color=BLUE,
               edgecolor="white", linewidth=0.6, zorder=3)
    ax.text(max(r.plot_e, 1) * 1.15, yi, f"{r.audited:.2f}", va="center", fontsize=5.5, color=MUTED)
ax.set_yticks(y)
ax.set_yticklabels([f"{PANEL[r.panel]} · {METHOD.get(r.method, r.method)}" for _, r in a.iterrows()],
                   fontsize=5.6)
# panel separators
for i in range(1, len(a)):
    if a.panel[i] != a.panel[i - 1]:
        ax.axhline(y[i] + 0.5, color=GRID, lw=0.6)
ax.set_xscale("log")
ax.set_xlim(floor * 0.8, 60)
ax.set_xticks([0.03, 0.1, 0.3, 1, 3, 10, 30])
ax.set_xticklabels(["0.03", "0.1", "0.3", "1", "3", "10", "30"])
ax.set_xlabel("Enrichment over all-pairs chance (log)")
ax.set_ylim(-0.8, len(a) - 0.2)
ax.tick_params(axis="y", length=0)
ax.set_title("◀ no audited edges (0×) · labels = audited fraction", fontsize=5.8, color=MUTED, loc="right")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/Fig3_enrichment.{ext}", dpi=400)
plt.close(fig)

# ---------------- Fig 4 ----------------
t = pd.concat([pd.read_csv(f"{FROZEN}/transfer_matched_3lines.csv"), pd.read_csv(f"{FROZEN}/transfer_matched_hct116.csv")])
t = t[t.method != "tce"].copy()
src_short = {"k562_p300e": "ess. p300", "gwps_p788": "GW p788", "gwps_p1434": "GW p1434"}
t["label"] = t.method.map(METHOD) + " · " + t.source.map(src_short)
targets = ["RPE1", "HepG2", "Jurkat", "HCT116"]
rows = t[t.target == "RPE1"].label.tolist()
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=True, sharey=True)
for ax, tg in zip(axes.flat, targets):
    d = t[t.target == tg].set_index("label").loc[rows]
    yy = np.arange(len(rows))[::-1]
    for yi, (_, r) in zip(yy, d.iterrows()):
        ax.plot([r.flagged, r.audited], [yi, yi], color=GRID, lw=1.6, zorder=1)
    ax.scatter(d.random_same_sign, yy, marker="|", s=60, color=MUTED, lw=1.2, zorder=2, label="random same-sign pairs")
    ax.scatter(d.matched_null, yy, marker="D", s=16, facecolor="white", edgecolor=INK, lw=0.8, zorder=3,
               label="effect-size-matched null")
    ax.scatter(d.flagged, yy, s=24, facecolor="white", edgecolor=ORANGE, lw=1.3, zorder=4, label="flagged in K562")
    ax.scatter(d.audited, yy, s=24, color=BLUE, edgecolor="white", lw=0.5, zorder=5, label="audited in K562")
    ax.scatter(d.effect_rank, yy, marker="x", s=20, color=VIOLET, lw=1.2, zorder=6,
               label="same-size set ranked by K562 |z|")
    plat = "Xaira X-Atlas/Orion (different platform)" if tg == "HCT116" else "Replogle 2022 (same platform)"
    ax.set_title(f"{tg}  ·  {plat}", fontsize=7, loc="left", color=INK)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_yticks(yy)
    ax.set_yticklabels(rows, fontsize=6)
for ax in axes[1]:
    ax.set_xlabel("Fraction replicating in target line (same sign, |z| ≥ 3)")
h, l = axes[0, 0].get_legend_handles_labels()
order_l = ["audited in K562", "flagged in K562", "effect-size-matched null", "same-size set ranked by K562 |z|",
           "random same-sign pairs"]
idx = [l.index(x) for x in order_l]
fig.legend([h[i] for i in idx], order_l, loc="lower center", ncol=5, frameon=False, fontsize=6,
           bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0, 0.04, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/Fig4_transfer_matched.{ext}", dpi=400)
plt.close(fig)
print("ok", len(a), "networks in Fig3")
