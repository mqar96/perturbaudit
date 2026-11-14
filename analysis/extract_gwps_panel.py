"""Extract a gene panel (cells x panel genes) from the scPerturb GWPS single-cell h5ad (dense, chunked).
Usage: extract_gwps_panel.py H5AD GENES_CSV OUTDIR [max_cells_per_pert=100] [max_ctrl=10000]"""
import sys, os, json, time
import numpy as np, pandas as pd, h5py
H5, GC, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
MAXC = int(sys.argv[4]) if len(sys.argv) > 4 else 100
MAXN = int(sys.argv[5]) if len(sys.argv) > 5 else 10000
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
h = h5py.File(H5, "r")
def cat(grp):
    g = h["obs"][grp]
    c = g["categories"][:].astype(str); return c[g["codes"][:]]
pert = cat("perturbation"); guide = cat("guide_id")
batch = h["obs"]["batch"][:]
umi = h["obs"]["UMI_count"][:] if "UMI_count" in h["obs"] else h["obs"]["ncounts"][:]
nperts = h["obs"]["nperts"][:]
vn = h["var"]["gene_name"]; vn = (vn["categories"][:].astype(str)[vn["codes"][:]] if isinstance(vn, h5py.Group) else vn[:].astype(str))
genes = pd.read_csv(GC).gene.astype(str).tolist()
col = {g: i for i, g in enumerate(vn)}
genes = [g for g in genes if g in col]
ctrl_label = "control" if (pert == "control").any() else "non-targeting"
sel = []
for g in genes:
    idx = np.flatnonzero((pert == g) & (nperts <= 1))
    sel.append(rng.choice(idx, min(MAXC, len(idx)), replace=False))
ci = np.flatnonzero(pert == ctrl_label)
sel.append(rng.choice(ci, min(MAXN, len(ci)), replace=False))
rows = np.sort(np.concatenate(sel))
cols = np.array(sorted(col[g] for g in genes)); gorder = [vn[c] for c in cols]
X = h["X"]; rb = X.chunks[0]
out = np.empty((len(rows), len(cols)), dtype=np.float32)
t = time.time(); pos = 0
blocks = np.unique(rows // rb)
for k, b in enumerate(blocks):
    r = rows[(rows >= b * rb) & (rows < (b + 1) * rb)]
    blk = X[b * rb: min((b + 1) * rb, X.shape[0]), :]
    out[pos:pos + len(r)] = blk[np.ix_(r - b * rb, cols)]
    pos += len(r)
    if k % 25 == 0: print(f"block {k}/{len(blocks)} {time.time()-t:.0f}s", flush=True)
lab = np.where(pert[rows] == ctrl_label, "non-targeting", pert[rows])
is_counts = bool(np.allclose(out[:500], np.round(out[:500])))
if is_counts:
    tot = umi[rows].astype(np.float64)
    out = np.log1p(out / np.maximum(tot[:, None], 1) * 1e4).astype(np.float32)
np.savez_compressed(f"{OUT}/panel.npz", X=out, labels=np.array(lab, dtype=object), names=np.array(gorder, dtype=object),
                    umi=umi[rows], guides=np.array(guide[rows], dtype=object), batch=np.array(batch[rows].astype(str), dtype=object))
sys.path.insert(0, "perturbaudit/src")
from perturbaudit import proximity
tss = proximity.load_tss_gtf("data/annot/gencode.v43.basic.gtf.gz")
near = [(x, y, int(proximity.tss_distance(x, y, tss))) for i, x in enumerate(gorder) for y in gorder[i + 1:] if proximity.tss_distance(x, y, tss) < 10000]
json.dump(dict(source=os.path.basename(H5), p_real=len(gorder), k_decoy=0, k_perm=0, k_knock=0, n_cells=int(len(rows)),
               n_obs=int((lab == "non-targeting").sum()), counts_normalized=is_counts, neighbour_pairs=near, offtarget_pairs=[],
               note="GWPS single-cell (scPerturb). counts->CP10k log1p using total UMI; panel = 788 strongest knockdowns (bulk own z, >=50 cells), matching inspre's GWPS panel size"),
          open(f"{OUT}/meta.json", "w"), indent=1)
print("done", out.shape, "seconds", round(time.time() - t), "neighbour pairs", len(near))
