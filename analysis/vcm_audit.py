"""Edge-level audit of perturbation-response prediction models on unseen perturbations (K562 GWPS pseudo-bulk).
Test = 20% of perturbations with a transcriptional effect (energy test p < 0.001) whose target gene is measured.
A model's predicted edges for an unseen parent = its K most extreme predicted children. An edge is supported when the
real held-out value of that child is in the child's top/bottom 1% across all perturbations, with the predicted sign
(chance ~1%). Models: mean of training perturbations; linear gene-embedding baseline (Ahlmann-Eltze et al.);
co-expression kNN; RPE1 measured profile of the same gene (cross-cell-line 'oracle'); GEARS/others added from files."""
import os, sys, numpy as np, pandas as pd, anndata as ad
from numpy.linalg import svd
rng = np.random.default_rng(0); K = 20
a = ad.read_h5ad("data/replogle/K562_gwps_normalized_bulk_01.h5ad")
X = np.nan_to_num(np.asarray(a.X, dtype=np.float32), nan=0, posinf=0, neginf=0); X = np.clip(X, -10, 10)
gn = a.var.gene_name.astype(str).values if "gene_name" in a.var else a.var_names.astype(str).values
tg = np.array([s.split("_")[1] for s in a.obs_names]); ctrl = tg == "non-targeting"
gi = {}
for i, g in enumerate(gn): gi.setdefault(g, i)
eff = (a.obs.energy_test_p_value.values < 0.001) & ~ctrl & np.isin(tg, list(gi))
idx = np.flatnonzero(eff); rng.shuffle(idx); test = np.sort(idx[: len(idx) // 5])
train = np.setdiff1d(np.flatnonzero(~ctrl), test)
print("perturbations", (~ctrl).sum(), "effective", eff.sum(), "test", len(test), "genes", X.shape[1], flush=True)
nc = np.flatnonzero(~ctrl); lo = np.empty(X.shape[1], np.float32); hi = np.empty(X.shape[1], np.float32)
for c0 in range(0, X.shape[1], 500):
    blk = X[nc, c0:c0 + 500]; lo[c0:c0 + 500] = np.quantile(blk, 0.01, axis=0); hi[c0:c0 + 500] = np.quantile(blk, 0.99, axis=0)
obs_names = a.obs_names; del a; import gc; gc.collect()
Ytr = X[train]; mu = Ytr.mean(0)
preds = {}
preds["mean of training perturbations"] = np.tile(mu, (len(test), 1))
# linear gene-embedding baseline: gene embeddings from PCA of training matrix; ridge from target-gene embedding to profile
from sklearn.utils.extmath import randomized_svd
k = 50; U, S, Vt = randomized_svd(Ytr - mu, k, random_state=0); G = Vt.T * S; del U  # genes x k
Etr = np.array([G[gi[t]] if t in gi else np.zeros(k) for t in tg[train]]); Ete = np.array([G[gi[t]] for t in tg[test]])
lam = 0.1 * np.trace(Etr.T @ Etr) / k
W = np.linalg.solve(Etr.T @ Etr + lam * np.eye(k), Etr.T @ (Ytr - mu)); preds["linear gene-embedding"] = mu + Ete @ W
# kNN co-expression: average training profiles of the 20 perturbed genes most correlated (across training perturbations) with the target
Cg = (Ytr - mu); Cg /= (np.linalg.norm(Cg, axis=0) + 1e-9)
trg = [gi.get(t, -1) for t in tg[train]]; trg = np.array(trg); okt = trg >= 0
P = []
for t in tg[test]:
    c = Cg[:, gi[t]] @ Cg[:, trg[okt]]; nb = np.argsort(-c)[:20]; P.append(Ytr[okt][nb].mean(0))
preds["co-expression kNN"] = np.array(P)
# RPE1 profile of the same gene (other cell line, measured): only for test genes perturbed in RPE1
del Cg; gc.collect()
b = ad.read_h5ad("data/replogle/rpe1_normalized_bulk_01.h5ad"); XR = np.clip(np.nan_to_num(np.asarray(b.X, dtype=np.float32)), -10, 10)
gR = b.var.gene_name.astype(str).values if "gene_name" in b.var else b.var_names.astype(str).values
tR = {s.split("_")[1]: i for i, s in enumerate(b.obs_names)}; colR = np.array([list(gR).index(g) if g in set(gR) else -1 for g in gn]) if False else None
gRi = {g: i for i, g in enumerate(gR)}; cmap = np.array([gRi.get(g, -1) for g in gn])
PR = np.full((len(test), X.shape[1]), np.nan, np.float32)
for r, t in enumerate(tg[test]):
    if t in tR: PR[r, cmap >= 0] = XR[tR[t], cmap[cmap >= 0]]
preds["RPE1 measured (other cell line)"] = PR
extra = sorted(f for f in os.listdir("derived/vcm") if f.startswith("pred_") and f.endswith(".npz")) if os.path.isdir("derived/vcm") else []
for f in extra:
    z = np.load(f"derived/vcm/{f}", allow_pickle=True); M = np.full((len(test), X.shape[1]), np.nan, np.float32)
    pt = {p: i for i, p in enumerate(z["perts"])}; gz = {g: i for i, g in enumerate(z["genes"])}; cm = np.array([gz.get(g, -1) for g in gn])
    for r, t in enumerate(tg[test]):
        if t in pt: M[r, cm >= 0] = z["pred"][pt[t], cm[cm >= 0]]
    preds[str(z["name"])] = M
Yte = X[test]; rows = []
for name, Pm in preds.items():
    sup = []; corr = []; generic = []
    for r in range(len(test)):
        p = Pm[r]; m = ~np.isnan(p); m[gi[tg[test][r]]] = False
        if m.sum() < 100: continue
        d = (p - mu)[m]; cand = np.flatnonzero(m)[np.argsort(-np.abs(d))[:K]]; sgn = np.sign((p - mu)[cand])
        real = Yte[r, cand]; s = ((sgn > 0) & (real >= hi[cand])) | ((sgn < 0) & (real <= lo[cand]))
        sup.append(s.mean())
        corr.append(np.corrcoef((p - mu)[m], (Yte[r] - mu)[m])[0, 1])
        generic.append(cand)
    gc = pd.Series(np.concatenate(generic)).value_counts() if generic else pd.Series(dtype=int)
    rows.append(dict(model=name, test_perturbations=len(sup), edge_precision_at20=np.mean(sup), edge_precision_sd=np.std(sup), pearson_delta=np.nanmean(corr),
                     distinct_children_predicted=int(gc.size), top_child_share=float(gc.iloc[0] / len(sup)) if len(sup) else np.nan))
R = pd.DataFrame(rows); os.makedirs("derived/vcm", exist_ok=True); R.to_csv("derived/vcm/vcm_audit_summary.csv", index=False)
np.savez("derived/vcm/split.npz", test=np.array(tg[test], dtype=str), train=np.array(tg[train], dtype=str), genes=np.array(gn, dtype=str))
print(R.round(3).to_string(index=False)); print("chance edge precision ~0.01")
