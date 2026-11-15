"""inspre on the K562 genome-wide 788-gene panel: verdicts by list length, the RPL3 hub, and the same edges tested in
RPE1, HepG2 and Jurkat. Audit settings are those of freeze_audits.py (batch-matched reference, guides ignored for the
genome-wide panel, 10-kb neighbour check, 200 decoys, seed 0).

Usage: python analysis/gwps_inspre_case.py PANELS_DIR GTF OUT_DIR NAME=TARGET_NPZ ...
Writes edges_top500_<method>.csv, transfer_by_verdict.csv and fig_gwps_inspre_case.png in OUT_DIR."""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from perturbaudit.screen import Screen
from perturbaudit.api import audit_network
from perturbaudit import proximity

PANELS, GTF, OUT = sys.argv[1:4]
targets = dict(a.split("=", 1) for a in sys.argv[4:])
os.makedirs(OUT, exist_ok=True)
K, METHODS = 500, ["inspre", "inspre_dag"]

z = np.load(os.path.join(PANELS, "gwps_p788", "panel.npz"), allow_pickle=True)
X, lab, names = z["X"].astype(np.float64), z["labels"].astype(str), [str(n) for n in z["names"]]
real = [i for i, n in enumerate(names) if not n.startswith("DECOY")]
keep = ~np.char.startswith(lab, "DECOY")
batch = z["batch"][keep] if "batch" in z.files and len(z["batch"]) else None
rn = [names[i] for i in real]
sc = Screen(X[keep][:, real], lab[keep], rn, "non-targeting", batch, None)
tss = proximity.load_tss_gtf(GTF)
E = {}
for m in METHODS:
    f = os.path.join(OUT, f"edges_top500_{m}.csv")
    if os.path.exists(f):
        E[m] = pd.read_csv(f); continue
    W = np.load(os.path.join(PANELS, "gwps_p788", f"W_{m}.npy"))[np.ix_(real, real)]
    E[m], _ = audit_network(sc, W, rn, top_k=K, n_decoys=200, seed=0, tss=tss)
    E[m].to_csv(f, index=False)
    print(m, E[m].head(50).verdict.value_counts().to_dict(), flush=True)

rows = []
for tname, path in targets.items():
    T = np.load(path, allow_pickle=True)
    Z = T["Z"].astype(np.float32); pi = {str(p): i for i, p in enumerate(T["perts"])}; gi = {}
    for i, g in enumerate(T["genes"]):
        gi.setdefault(str(g), i)
    zt = lambda a, b: float(Z[pi[a], gi[b]]) if a in pi and b in gi else np.nan
    e = E["inspre"].copy()
    e["zf"] = [zt(a, b) for a, b in zip(e.parent, e.child)]
    e["zr"] = [zt(b, a) for a, b in zip(e.parent, e.child)]
    e = e.dropna(subset=["zf", "zr"])
    e["forward"] = (e.zf.abs() >= 3) & (np.sign(e.zf) == np.sign(e.z_parent_kd_on_child))
    e["reverse"] = (e.zr.abs() >= 3) & (np.sign(e.zr) == np.sign(e.z_child_kd_on_parent))
    for v, g in e.groupby("verdict"):
        rows.append(dict(target=tname, method="inspre", top_k=K, verdict=v, n_testable=len(g),
                         forward_replicates=round(g.forward.mean(), 3), child_kd_moves_parent=round(g.reverse.mean(), 3)))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "transfer_by_verdict.csv"), index=False)
print(R.to_string(index=False))

# figure
BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#c9c8c2", "#222222", "#6b6b6b"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(11, 3.8), gridspec_kw=dict(width_ratios=[1.25, 1.1, 1.1]))
ks = [25, 50, 100, 200, 500]
for m, off in [("inspre", -0.2), ("inspre_dag", 0.2)]:
    e = E[m]
    t = [e.head(k)[~e.head(k).verdict.str.startswith("untestable")] for k in ks]
    a = [(x.verdict == "audited").mean() for x in t]; r = [(x.verdict == "reversed").mean() for x in t]
    u = [(x.verdict == "unsupported").mean() for x in t]
    xx = np.arange(len(ks)) + off
    ax[0].bar(xx, a, 0.36, color=BLUE, label="Audited" if m == "inspre" else None)
    ax[0].bar(xx, u, 0.36, bottom=a, color=GREY, label="Unsupported" if m == "inspre" else None)
    ax[0].bar(xx, r, 0.36, bottom=np.add(a, u), color=ORANGE, label="Reversed" if m == "inspre" else None)
for i in range(len(ks)):
    ax[0].text(i - 0.2, 1.02, "N", ha="center", fontsize=7, color=MUTED); ax[0].text(i + 0.2, 1.02, "D", ha="center", fontsize=7, color=MUTED)
ax[0].set_xticks(range(len(ks)), [f"top {k}" for k in ks]); ax[0].set_ylim(0, 1.1); ax[0].set_ylabel("Fraction of testable edges")
ax[0].set_title("A  inspre, K562 genome-wide (788 genes)\nN = no DAG constraint, D = DAG", loc="left", fontsize=8.5)
ax[0].legend(frameon=False, fontsize=7, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.12))
h = E["inspre"][E["inspre"].parent == "RPL3"].head(12)
y = np.arange(len(h))
ax[1].barh(y - 0.2, h.z_parent_kd_on_child, 0.38, color=BLUE, label="RPL3 knockdown → child")
ax[1].barh(y + 0.2, h.z_child_kd_on_parent, 0.38, color=ORANGE, label="child knockdown → RPL3")
ax[1].set_yticks(y, h.child); ax[1].invert_yaxis(); ax[1].axvline(0, color=MUTED, lw=0.8)
for v in (-3, 3):
    ax[1].axvline(v, color=MUTED, lw=0.8, ls=":")
ax[1].set_xlabel("Knockdown effect (z)")
ax[1].set_title("B  inspre's top hub: RPL3 → ribosomal genes", loc="left", fontsize=8.5)
ax[1].legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.15), ncol=2)
vs = ["audited", "unsupported", "reversed"]
tn = list(targets)
w = 0.8 / max(len(tn), 1)
for j, t in enumerate(tn):
    d = R[R.target == t].set_index("verdict").reindex(vs)
    ax[2].bar(np.arange(3) - 0.4 + w * (j + 0.5), d.child_kd_moves_parent, w * 0.9, color=[BLUE, "#6250d6", "#1baf7a"][j % 3], label=t)
ax[2].set_xticks(range(3), ["Audited", "Unsupported", "Reversed"]); ax[2].set_ylim(0, 1)
ax[2].set_ylabel("Child knockdown moves parent\n(same sign as K562, |z| ≥ 3)")
ax[2].set_title("C  Same inspre edges in other lines", loc="left", fontsize=8.5)
ax[2].legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_gwps_inspre_case.png"), dpi=300)
