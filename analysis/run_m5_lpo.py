"""M5 / H5: leave-perturbation-out test of audited vs unaudited networks at matched density.
For each of 10 folds over perturbed genes: fit the method without the held-out genes' cells, audit it on the
training screen only, then ask what each network implies for knocking down the held-out genes
(descendants should respond; non-descendants should not), scored against the held-out cells' knockdown effects.
Usage: run_m5_lpo.py PANEL_DIR METHOD [TOP_K=300] [FOLDS=10]"""
import sys, os, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perturbaudit.screen import Screen
from perturbaudit.api import audit_network
from perturbaudit import heldout
import run_method as RM

D, M = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 300
F = int(sys.argv[4]) if len(sys.argv) > 4 else 10
z = np.load(f"{D}/panel.npz", allow_pickle=True)
X, lab, names = z["X"].astype(np.float64), z["labels"].astype(str), [str(n) for n in z["names"]]
real = [i for i, n in enumerate(names) if not n.startswith("DECOY")]
keep = ~np.char.startswith(lab.astype(str), "DECOY")
X, lab, names = X[keep][:, real], lab[keep], [names[i] for i in real]
p = len(names); gi = {g: i for i, g in enumerate(names)}
full = Screen(X, lab, names)
Zfull = full.effects().reindex(index=names)  # held-out evaluation uses the held-out genes' own cells
SIG, NULL = 3.0, 2.0
fn = {"tce": RM.run_tce, "inspre": RM.run_inspre, "gbm_grn": RM.run_gbm_grn, "dotears": RM.run_dotears, "sdcd": RM.run_sdcd}[M]
rng = np.random.default_rng(0)
perturbed = [g for g in names if g in set(Zfull.dropna(how="all").index)]
folds = np.array_split(rng.permutation(perturbed), F)

def evaluate(G, held):
    """Implied-effect precision/recall on held-out perturbations for adjacency G (p x p bool)."""
    Dsc = heldout.descendants(G)
    tp = fp = fn_ = tn = 0
    for g in held:
        i = gi[g]; zrow = np.abs(Zfull.loc[g].values); m = np.ones(p, bool); m[i] = False
        resp = (zrow >= SIG) & m; nonresp = (zrow < NULL) & m; pred = Dsc[i] & m
        tp += (pred & resp).sum(); fp += (pred & nonresp).sum(); fn_ += (~pred & resp).sum(); tn += (~pred & nonresp).sum()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn_, 1); spec = tn / max(tn + fp, 1)
    return dict(precision=prec, recall=rec, specificity=spec, balanced_acc=(rec + spec) / 2, n_pred=int(tp + fp), n_edges=int(G.sum()))

rows = []
for f, held in enumerate(folds):
    held = set(held); tr = ~np.isin(lab, list(held))
    t = time.time()
    W = np.nan_to_num(np.abs(np.asarray(fn(X[tr], lab[tr].astype(object), names), dtype=float)))
    np.fill_diagonal(W, 0)
    sc = Screen(X[tr], lab[tr], names)
    E, _ = audit_network(sc, W, names, top_k=K, n_decoys=100, seed=f)
    flagged = E.verdict.isin(["unsupported", "reversed", "neighbour", "offtarget", "guide_specific", "unstable"])
    keepE = E[~flagged]
    Gfull = np.zeros((p, p), bool); Gaud = np.zeros((p, p), bool)
    for a, b in zip(E.parent, E.child): Gfull[gi[a], gi[b]] = True
    for a, b in zip(keepE.parent, keepE.child): Gaud[gi[a], gi[b]] = True
    n_a = int(Gaud.sum())
    Gmatch = np.zeros((p, p), bool)
    for a, b in zip(E.parent.iloc[:n_a], E.child.iloc[:n_a]): Gmatch[gi[a], gi[b]] = True
    rand = []
    for r in range(20):
        Rg = np.zeros((p, p), bool); idx = rng.choice(p * p, n_a, replace=False); Rg.flat[idx] = True; np.fill_diagonal(Rg, False)
        rand.append(evaluate(Rg, held))
    for name, G in [("full_topK", Gfull), ("audited", Gaud), ("unaudited_matched", Gmatch)]:
        rows.append(dict(panel=os.path.basename(D.rstrip("/")), method=M, fold=f, network=name, **evaluate(G, held)))
    rows.append(dict(panel=os.path.basename(D.rstrip("/")), method=M, fold=f, network="random_matched",
                     **{k: float(np.mean([r[k] for r in rand])) for k in rand[0]}))
    print(f"fold {f} {time.time()-t:.0f}s audited {n_a}/{len(E)}", flush=True)
R = pd.DataFrame(rows)
out = f"{D}/m5_lpo_{M}.csv"; R.to_csv(out, index=False)
S = R.groupby("network")[["precision", "recall", "balanced_acc", "n_edges", "n_pred"]].mean().round(3)
print(S.to_string())
