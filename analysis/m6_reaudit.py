"""Re-audit the clean networks of the M6 stress test with the current audit code.
Stress retention does not depend on the audit (it compares refitted networks with the clean one), so only the
clean network's verdicts are recomputed; per-edge stress survival is carried over from run_m6_stress.py.
Usage: python analysis/m6_reaudit.py PANEL_DIR OUT_CSV m6_edges_<method>.csv ..."""
import sys, os, numpy as np, pandas as pd
from perturbaudit.screen import Screen
from perturbaudit.api import audit_network

D, OUT, files = sys.argv[1], sys.argv[2], sys.argv[3:]
z = np.load(f"{D}/panel.npz", allow_pickle=True)
X, lab, names = z["X"].astype(np.float64), z["labels"].astype(str), [str(n) for n in z["names"]]
real = [i for i, n in enumerate(names) if not n.startswith("DECOY")]
keep = ~np.char.startswith(lab, "DECOY")
X, lab, names = X[keep][:, real], lab[keep], [names[i] for i in real]
sc = Screen(X, lab, names)
gi = {g: i for i, g in enumerate(names)}
rows = []
for f in files:
    M = os.path.basename(f)[len("m6_edges_"):-4]
    E0 = pd.read_csv(f)
    W = np.zeros((len(names), len(names)))
    for a, b, s in zip(E0.parent, E0.child, E0.score):
        W[gi[a], gi[b]] = abs(s)
    E, _ = audit_network(sc, W, names, top_k=len(E0), n_decoys=100)
    E = E.merge(E0[["parent", "child", "verdict", "stress_survival", "fragile"]].rename(columns={"verdict": "verdict_v020"}),
                on=["parent", "child"])
    assert len(E) == len(E0)
    E.to_csv(os.path.join(os.path.dirname(OUT), f"m6_edges_{M}_v021.csv"), index=False)
    fr = E.groupby("verdict").fragile.agg(["mean", "size"])
    rows.append(dict(method=M, n_edges=len(E), verdicts_changed=int((E.verdict != E.verdict_v020).sum()),
                     fragile_all=round(E.fragile.mean(), 3),
                     **{f"fragile_{v}": round(fr.loc[v, "mean"], 3) if v in fr.index else np.nan for v in ["audited", "unsupported", "reversed"]},
                     **{f"n_{v}": int(fr.loc[v, "size"]) if v in fr.index else 0 for v in ["audited", "unsupported", "reversed"]}))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
