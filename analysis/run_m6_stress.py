"""M6: label-noise stress test. Reassign x% of cells to a random other label (ambient misassignment range
1.5-9%, CLEANSER) and add simulated doublets (cell = mean of two cells, label of the first), refit, and report
how many of the clean network's top-K edges survive and which audit verdicts they had.
Usage: run_m6_stress.py PANEL_DIR METHOD [TOP_K=200] [REPS=3]"""
import sys, os, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perturbaudit.screen import Screen
from perturbaudit.api import audit_network
import run_method as RM

D, M = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 200
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
z = np.load(f"{D}/panel.npz", allow_pickle=True)
X, lab, names = z["X"].astype(np.float64), z["labels"].astype(str), [str(n) for n in z["names"]]
real = [i for i, n in enumerate(names) if not n.startswith("DECOY")]
keep = ~np.char.startswith(lab.astype(str), "DECOY")
X, lab, names = X[keep][:, real], lab[keep], [names[i] for i in real]
fn = {"tce": RM.run_tce, "inspre": RM.run_inspre, "gbm_grn": RM.run_gbm_grn, "dotears": RM.run_dotears, "sdcd": RM.run_sdcd}[M]

def top_edges(W):
    W = np.nan_to_num(np.abs(np.asarray(W, float))); np.fill_diagonal(W, 0)
    idx = [i for i in np.argsort(W.ravel())[::-1][:K] if W.ravel()[i] > 0]
    return [(names[i // len(names)], names[i % len(names)]) for i in idx]

W0 = fn(X, lab.astype(object), names)
E0, _ = audit_network(Screen(X, lab, names), W0, names, top_k=K, n_decoys=100)
base = list(zip(E0.parent, E0.child)); verdict = dict(zip(base, E0.verdict))
labels_all = np.unique(lab)
rows, surv = [], {e: [] for e in base}
for kind, rates in [("misassign", [0.015, 0.03, 0.06, 0.09]), ("doublet", [0.05, 0.10])]:
    for rate in rates:
        for r in range(REPS):
            rng = np.random.default_rng(1000 * r + int(rate * 1000))
            Xn, ln = X.copy(), lab.copy()
            n = len(ln); k = int(rate * n); idx = rng.choice(n, k, replace=False)
            if kind == "misassign":
                ln[idx] = rng.choice(labels_all, k)
            else:
                partner = rng.choice(n, k)
                Xn[idx] = (X[idx] + X[partner]) / 2
            S = set(top_edges(fn(Xn, ln.astype(object), names)))
            kept = [e in S for e in base]
            for e, kk in zip(base, kept): surv[e].append(kk)
            byv = pd.Series(kept, index=[verdict[e] for e in base]).groupby(level=0).mean()
            rows.append(dict(method=M, kind=kind, rate=rate, rep=r, retained=float(np.mean(kept)),
                             **{f"retained_{v}": float(byv.get(v, np.nan)) for v in ["audited", "unsupported", "reversed"]}))
            print(kind, rate, r, round(float(np.mean(kept)), 3), flush=True)
R = pd.DataFrame(rows); R.to_csv(f"{D}/m6_stress_{M}.csv", index=False)
E0["stress_survival"] = [float(np.mean(surv[e])) for e in base]
E0["fragile"] = E0.stress_survival < 0.5
E0.to_csv(f"{D}/m6_edges_{M}.csv", index=False)
print(R.groupby(["kind", "rate"])[["retained", "retained_audited", "retained_unsupported", "retained_reversed"]].mean().round(3).to_string())
print("fragile edges:", int(E0.fragile.sum()), "of", len(E0), "| fragile among audited:", round(E0[E0.verdict == "audited"].fragile.mean(), 3),
      "| among unsupported:", round(E0[E0.verdict == "unsupported"].fragile.mean(), 3) if (E0.verdict == "unsupported").any() else "n/a")
