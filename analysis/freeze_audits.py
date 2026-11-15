"""Frozen top-50 verdicts and like-for-like all-pairs chance baseline for every panel and inferred network.

Usage: python analysis/freeze_audits.py PANELS_DIR GTF OUT_DIR [panel ...]

Each panel directory holds panel.npz (X, labels, names, optional batch and guides) and W_<method>.npy networks.
The method audits and the chance baseline use the same Screen (same batch-matched reference and guides
policy), the same neighbour annotation and the same decoy settings, so enrichment = audited / chance compares
like with like. Guides are ignored for the K562 genome-wide panels (their elements target alternative TSSs).
"""
import os, sys, glob
import numpy as np
import pandas as pd
from perturbaudit.screen import Screen
from perturbaudit.api import audit_network
from perturbaudit import proximity

PANELS, GTF, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
only = set(sys.argv[4:])
TOP_K, N_DECOYS, SEED = 50, 200, 0
os.makedirs(OUT, exist_ok=True)
tss = proximity.load_tss_gtf(GTF)


def load(panel):
    z = np.load(os.path.join(PANELS, panel, "panel.npz"), allow_pickle=True)
    X, lab, names = z["X"].astype(np.float64), z["labels"].astype(str), [str(n) for n in z["names"]]
    real = [i for i, n in enumerate(names) if not n.startswith("DECOY")]
    keep = ~np.char.startswith(lab, "DECOY")
    batch = z["batch"][keep] if "batch" in z.files and len(z["batch"]) else None
    guides = z["guides"][keep] if ("guides" in z.files and len(z["guides"]) and not panel.startswith("gwps")) else None
    rn = [names[i] for i in real]
    sc = Screen(X[keep][:, real], lab[keep], rn, "non-targeting", batch, guides)
    return sc, real, rn


def frac(E):
    t = E[~E.verdict.astype(str).str.startswith("untestable")]
    return len(t), (round((t.verdict == "audited").mean(), 4) if len(t) else np.nan)


rows, chance_rows, cov_rows = [], [], []
for D in sorted(glob.glob(os.path.join(PANELS, "*", "panel.npz"))):
    panel = os.path.basename(os.path.dirname(D))
    if only and panel not in only:
        continue
    sc, real, rn = load(panel)
    cb, cg = sc.reference_coverage()
    cov_rows.append(dict(panel=panel, batches=len(cb), batches_without_reference=int((~cb.reference_ok).sum()),
                         min_controls_per_batch=int(cb.control_cells.min()), perturbed_cells=int(cb.perturbed_cells.sum()),
                         cells_excluded=int(cg.cells_without_reference.sum()), groups_untestable=int((~cg.testable).sum())))
    for f in sorted(glob.glob(os.path.join(PANELS, panel, "W_*.npy"))):
        m = os.path.basename(f)[2:-4]
        W = np.load(f)[np.ix_(real, real)]
        E, _ = audit_network(sc, W, rn, top_k=TOP_K, tss=tss, n_decoys=N_DECOYS, seed=SEED)
        E.insert(0, "method", m); E.insert(0, "panel", panel)
        E.to_csv(os.path.join(OUT, f"edges_{panel}_{m}.csv"), index=False)
        n, a = frac(E)
        vc = E.verdict.value_counts()
        rows.append(dict(panel=panel, method=m, top_k=len(E), n_testable=n, audited=a,
                         **{v: int(vc.get(v, 0)) for v in ["unsupported", "reversed", "neighbour", "guide_specific",
                                                            "untestable_ineffective_knockdown", "untestable_no_reference"]}))
        print(rows[-1], flush=True)
    A = np.ones((len(rn), len(rn))); np.fill_diagonal(A, 0)
    E, _ = audit_network(sc, A, rn, tss=tss, n_decoys=N_DECOYS, seed=SEED)
    n, a = frac(E)
    chance_rows.append(dict(panel=panel, genes=len(rn), testable_pairs=n, chance_audited=a))
    print(chance_rows[-1], flush=True)

V = pd.DataFrame(rows); C = pd.DataFrame(chance_rows)
V.to_csv(os.path.join(OUT, "verdicts_top50.csv"), index=False)
C.to_csv(os.path.join(OUT, "chance_all_pairs.csv"), index=False)
pd.DataFrame(cov_rows).to_csv(os.path.join(OUT, "reference_coverage.csv"), index=False)
M = V.merge(C, on="panel")
M["enrichment"] = (M.audited / M.chance_audited).round(2)
M.to_csv(os.path.join(OUT, "audited_vs_chance.csv"), index=False)
print(M.to_string())
