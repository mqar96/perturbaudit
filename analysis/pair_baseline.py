"""Chance baseline: audit ALL ordered gene pairs of a panel (what a random network would score)."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, "perturbaudit/src")
from perturbaudit.screen import Screen
from perturbaudit.api import audit_network
rows = []
for D in sys.argv[1:]:
    z = np.load(f"{D}/panel.npz", allow_pickle=True)
    X, lab, names = z["X"].astype(np.float64), z["labels"].astype(str), [str(n) for n in z["names"]]
    real = [i for i, n in enumerate(names) if not n.startswith("DECOY")]; keep = ~np.char.startswith(lab.astype(str), "DECOY")
    rn = [names[i] for i in real]; sc = Screen(X[keep][:, real], lab[keep], rn)
    A = np.ones((len(rn), len(rn))); np.fill_diagonal(A, 0)
    E, _ = audit_network(sc, A, rn, n_decoys=100)
    t = E[~E.verdict.str.startswith("untestable")]; v = t.verdict.value_counts(normalize=True)
    rows.append(dict(panel=os.path.basename(D.rstrip("/")), genes=len(rn), pairs=len(t), audited=round(v.get("audited", 0), 3),
                     unsupported=round(v.get("unsupported", 0), 3), reversed=round(v.get("reversed", 0), 3)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("derived/pair_baseline.csv", index=False)
