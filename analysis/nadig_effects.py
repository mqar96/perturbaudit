"""On the owner's PC: download Nadig 2025 HepG2 / Jurkat single-cell h5ad (GEO GSE264667), compute batch-matched
null-variance knockdown z for every perturbation x gene with perturbaudit.screen, save compact results, delete raw."""
import os, sys, time, urllib.request, numpy as np, pandas as pd
R = os.environ.get("WORK_DIR", ".")
import anndata as ad, scipy.sparse as sp
from perturbaudit.screen import Screen
for cell in sys.argv[1:]:
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/GSE264667_{cell}_raw_singlecell_01.h5ad"
    f = os.path.join(R, f"nadig_{cell}.h5ad"); t = time.time()
    if not os.path.exists(f):
        urllib.request.urlretrieve(url, f + ".part"); os.replace(f + ".part", f)
    print(cell, "downloaded", round(os.path.getsize(f) / 1e9, 2), "GB", round(time.time() - t), "s", flush=True)
    A = ad.read_h5ad(f)
    print(A.shape, list(A.obs.columns)[:20], flush=True)
    obs = A.obs
    gcol = "gene" if "gene" in obs else ("gene_symbol" if "gene_symbol" in obs else obs.columns[0])
    pert = obs[gcol].astype(str).values
    batch = obs["gem_group"].astype(str).values if "gem_group" in obs else None
    X = A.X.tocsr() if sp.issparse(A.X) else sp.csr_matrix(A.X)
    tot = np.asarray(X.sum(1)).ravel()
    X = sp.diags(1e4 / np.maximum(tot, 1)) @ X; X.data = np.log1p(X.data)
    genes = A.var["gene_name"].astype(str).values if "gene_name" in A.var else A.var_names.astype(str).values
    sc = Screen(X, pert, genes, "non-targeting", batch)
    Z = sc.effects(min_cells=20)
    n = pd.Series(sc.groups["target"]["n"], index=sc.groups["target"]["names"]).reindex(Z.index)
    out = os.path.join(R, f"nadig_{cell}_effects.npz")
    np.savez_compressed(out, Z=Z.values.astype(np.float16), perts=Z.index.values.astype(str), genes=np.array(genes, dtype=str), n=n.values)
    print(cell, "effects", Z.shape, "saved", round(os.path.getsize(out) / 1e6, 1), "MB", flush=True)
    del A, X, sc
    os.remove(f); print(cell, "raw deleted", flush=True)
