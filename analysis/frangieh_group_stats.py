import h5py, numpy as np, pandas as pd, time
f = h5py.File("data/scperturb/FrangiehIzar2021_RNA.h5ad", "r")
def col(name):
    o = f["obs"][name]
    if isinstance(o, h5py.Group):
        cats = np.array([x.decode() if isinstance(x, bytes) else x for x in o["categories"][:]], dtype=object)
        return cats[o["codes"][:]]
    v = o[:]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in v], dtype=object) if v.dtype.kind in "OS" else v
pert = col("perturbation"); nper = col("nperts"); ncounts = col("ncounts").astype(np.float64)
vn = f["var"]["_index"][:] if "_index" in f["var"] else f["var"][f["var"].attrs["_index"]][:]
genes = np.array([x.decode() if isinstance(x, bytes) else x for x in vn], dtype=object)
single = (nper == 1) | (pert == "control")
grp_names, grp = np.unique(np.where(single, pert, "__drop__"), return_inverse=True)
G = len(grp_names); n = len(pert)
counts = np.bincount(grp, minlength=G)
indptr = f["X"]["indptr"][:]; data = f["X"]["data"]; idx = f["X"]["indices"]
p = len(indptr) - 1
S1 = np.zeros((G, p), np.float64); S2 = np.zeros((G, p), np.float64)
t0 = time.time(); step = 400
for c0 in range(0, p, step):
    c1 = min(p, c0 + step)
    a, b = indptr[c0], indptr[c1]
    d = data[a:b].astype(np.float64); r = idx[a:b]
    v = np.log1p(d / ncounts[r] * 1e4)
    colid = np.repeat(np.arange(c0, c1), np.diff(indptr[c0:c1 + 1]))
    key = grp[r] * p + colid
    np.add.at(S1.ravel(), key, v)
    np.add.at(S2.ravel(), key, v * v)
    if (c0 // step) % 10 == 0:
        print(c1, "/", p, round(time.time() - t0), "s", flush=True)
np.savez_compressed("derived/frangieh_group_stats.npz", S1=S1, S2=S2, N=counts, groups=grp_names, genes=genes)
print("done", G, "groups")
