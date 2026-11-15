"""Audit every W_<method>.npy in a panel with perturbaudit.api. Usage: audit_panel_api.py PANEL_DIR [TOP_K]"""
import sys, glob, os, json
import numpy as np, pandas as pd

from perturbaudit.screen import Screen
from perturbaudit.api import audit_network
from perturbaudit import proximity
from perturbaudit.preflight import own_intervention_variance
D = sys.argv[1]; K = int(sys.argv[2]) if len(sys.argv) > 2 else 200
z = np.load(f"{D}/panel.npz", allow_pickle=True)
X, lab, names = z["X"].astype(np.float64), z["labels"].astype(str), [str(n) for n in z["names"]]
real = [i for i, n in enumerate(names) if not n.startswith("DECOY")]
keep = ~np.char.startswith(lab.astype(str), "DECOY")
guides = z["guides"] if ("guides" in z.files and len(z["guides"]) and not os.environ.get("NO_GUIDES")) else None
batch = z["batch"] if "batch" in z.files and len(z["batch"]) else None
rn = [names[i] for i in real]
sc = Screen(X[keep][:, real], lab[keep], rn, "non-targeting",
            batch[keep] if batch is not None else None, guides[keep] if guides is not None else None)
pf = own_intervention_variance(X[keep][:, real], lab[keep], rn)
pf.to_csv(f"{D}/preflight.csv", index=False)
from perturbaudit import offtarget
OT = os.environ.get('OFFTARGET_TABLE')
otm = offtarget.offtarget_map(offtarget.load_hartman(OT)) if OT else None
tss = proximity.load_tss_gtf("data/annot/gencode.v43.basic.gtf.gz")
rows = []
for f in sorted(glob.glob(f"{D}/W_*.npy")):
    m = os.path.basename(f)[2:-4]
    W = np.load(f)[np.ix_(real, real)]
    E, S = audit_network(sc, W, rn, top_k=K, tss=tss, offtarget_map=otm)
    E.to_csv(f"{D}/api_edges_{m}.csv", index=False)
    S.insert(0, "method", m); rows.append(S)
out = pd.concat(rows); out.insert(0, "panel", os.path.basename(D.rstrip("/")))
out["preflight_degenerate_genes"] = int(pf.degenerate.sum())
out.to_csv(f"{D}/api_summary.csv", index=False)
cols = ["method", "top_k", "n_testable", "audited_fraction_of_testable", "unsupported", "reversed", "offtarget", "neighbour", "guide_specific", "untestable_ineffective_knockdown"]
print(out[out.top_k.isin([50, 100])][[c for c in cols if c in out]].to_string(index=False))
