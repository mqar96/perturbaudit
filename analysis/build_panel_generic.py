"""Build a method-ready panel from any single-guide Perturb-seq h5ad (scPerturb layout by default).
Usage: build_panel_generic.py H5AD OUTDIR P [--pert-col perturbation] [--control control] [--guide-col guide_id]
       [--batch-col gemgroup] [--max-cells 300] [--max-ctrl 5000] [--min-cells 30] [--kd-z 3] [--counts auto]"""
import argparse, json, os, sys
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad

from perturbaudit.screen import Screen
from perturbaudit import proximity

ap = argparse.ArgumentParser()
ap.add_argument("h5ad"); ap.add_argument("out"); ap.add_argument("p", type=int)
ap.add_argument("--pert-col", default="perturbation"); ap.add_argument("--control", default="control")
ap.add_argument("--guide-col", default="guide_id"); ap.add_argument("--batch-col", default=None)
ap.add_argument("--max-cells", type=int, default=300); ap.add_argument("--max-ctrl", type=int, default=5000)
ap.add_argument("--min-cells", type=int, default=30); ap.add_argument("--kd-z", type=float, default=3.0)
ap.add_argument("--counts", default="auto"); ap.add_argument("--label-prefix", action="store_true", help="gene = label before first underscore"); ap.add_argument("--control-regex", default=None); ap.add_argument("--ignore-nperts", action="store_true"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(a.seed)
A = ad.read_h5ad(a.h5ad)
obs = A.obs
pert = np.array([str(x) for x in obs[a.pert_col].values], dtype=object)
if a.control_regex:
    import re
    pert = np.where([bool(re.search(a.control_regex, x)) for x in pert], a.control, pert)
if a.label_prefix:
    pert = np.array([x if x == a.control else x.split("_")[0] for x in pert])
single = np.array(["_" not in p and "," not in p and ";" not in p and "|" not in p for p in pert]) | (pert == a.control)
if "nperts" in obs and not a.ignore_nperts:
    single &= (obs["nperts"].values <= 1) | (pert == a.control)
pert = np.where(pert == a.control, "non-targeting", pert)
genes = A.var_names.astype(str).values
gset = {g: i for i, g in enumerate(genes)}
counts = pd.Series(pert[single]).value_counts()
cand = [g for g, n in counts.items() if g != "non-targeting" and n >= a.min_cells and g in gset]
# subsample cells
keep = []
for g in cand + ["non-targeting"]:
    idx = np.flatnonzero(single & (pert == g))
    k = a.max_ctrl if g == "non-targeting" else a.max_cells
    keep.append(rng.choice(idx, min(k, len(idx)), replace=False))
keep = np.sort(np.concatenate(keep))
X = A.X[keep]
X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
is_counts = a.counts == "yes" or (a.counts == "auto" and np.allclose(X[:200].data, np.round(X[:200].data)))
umi = np.asarray(X.sum(1)).ravel()
if is_counts:
    X = sp.diags(1e4 / np.maximum(umi, 1)) @ X
    X.data = np.log1p(X.data)
lab = pert[keep]
guide = obs[a.guide_col].astype(str).values[keep] if a.guide_col and a.guide_col in obs else None
batch = obs[a.batch_col].astype(str).values[keep] if a.batch_col and a.batch_col in obs else None
cols = np.array([gset[g] for g in cand])
sc = Screen(X[:, cols], lab, cand, "non-targeting", batch)
kd = sc.knockdown()
mean_expr = np.asarray(X[lab == "non-targeting"][:, cols].mean(0)).ravel()
eff = [g for g in cand if abs(kd.get(g, 0)) >= a.kd_z]
eff = sorted(eff, key=lambda g: -mean_expr[cand.index(g)])[: a.p]
cols = np.array([gset[g] for g in eff])
m = np.isin(lab, eff + ["non-targeting"])
Xp = np.asarray(X[m][:, cols].toarray(), dtype=np.float32)
tss = proximity.load_tss_gtf("data/annot/gencode.v43.basic.gtf.gz")
near = [(x, y, int(proximity.tss_distance(x, y, tss))) for i, x in enumerate(eff) for y in eff[i + 1:] if proximity.tss_distance(x, y, tss) < 10000]
np.savez_compressed(f"{a.out}/panel.npz", X=Xp, labels=np.array(lab[m], dtype=object), names=np.array(eff, dtype=object), umi=umi[m],
                    guides=np.array(guide[m] if guide is not None else [], dtype=object),
                    batch=np.array(batch[m] if batch is not None else [], dtype=object))
meta = dict(source=os.path.basename(a.h5ad), p_real=len(eff), k_decoy=0, k_perm=0, k_knock=0, n_cells=int(m.sum()),
            n_obs=int((lab[m] == "non-targeting").sum()), counts_normalized=bool(is_counts), n_candidates=len(cand),
            n_effective=int(sum(abs(kd.get(g, 0)) >= a.kd_z for g in cand)),
            kd_sign_median=float(np.median([kd[g] for g in eff])) if eff else None,
            neighbour_pairs=near, offtarget_pairs=[], decoy_design="none (perturbaudit.api builds decoys from held-out controls)")
json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)
print(json.dumps({k: v for k, v in meta.items() if k != "neighbour_pairs"}), "neighbour_pairs", len(near))
