"""Audit a published edge list against the K562 genome-wide screen's own knockdowns (pseudo-bulk).
z(parent KD -> child) = normalized bulk value x sqrt(n cells) (Replogle bulk values are per-cell z-normalized means
relative to non-targeting cells, so this is the knockdown z-statistic). Simplified rules, narrower than
perturbaudit.api: support = |z| >= 3 only (no decoy q-values), reversal = child knockdown |z| >= 5 (no q), neighbour
check only; no off-target, guide or stability checks.
Usage: audit_published_edges_gwps.py EDGES_CSV PARENT_COL CHILD_COL OUT_PREFIX [filter_expr]"""
import sys, numpy as np, pandas as pd, anndata as ad

from perturbaudit import proximity
E = pd.read_csv(sys.argv[1]); pc, cc, out = sys.argv[2], sys.argv[3], sys.argv[4]
if len(sys.argv) > 5: E = E.query(sys.argv[5])
b = ad.read_h5ad("data/replogle/K562_gwps_normalized_bulk_01.h5ad")
X = b.X if isinstance(b.X, np.ndarray) else b.X.toarray()
gene = np.array([x.split("_")[1] for x in b.obs_names.astype(str)]); n = b.obs.num_cells_filtered.values.astype(float)
best = {}
for r, (g, k) in enumerate(zip(gene, n)):
    if g not in best or k > n[best[g]]: best[g] = r
vn = b.var["gene_name"].astype(str).values; col = {g: i for i, g in enumerate(vn)}
def z(a, c):
    if a not in best or c not in col: return np.nan
    r = best[a]; return X[r, col[c]] * np.sqrt(n[r])
tss = proximity.load_tss_gtf("data/annot/gencode.v43.basic.gtf.gz")
rows = []
for a, c in zip(E[pc].astype(str), E[cc].astype(str)):
    zf, zr, kd = z(a, c), z(c, a), z(a, a)
    d = proximity.tss_distance(a, c, tss)
    if a not in best: v = "untestable_parent_not_perturbed"
    elif c not in col: v = "untestable_child_not_measured"
    elif abs(zf) >= 3: v = "neighbour" if d < 10000 else "audited"
    elif not np.isnan(kd) and abs(kd) < 2: v = "untestable_ineffective_knockdown"
    elif not np.isnan(zr) and abs(zr) >= 5: v = "reversed"
    else: v = "unsupported"
    rows.append(dict(z_parent_kd_on_child=zf, z_child_kd_on_parent=zr, parent_knockdown_z=kd, tss_distance=d, verdict=v))
E = pd.concat([E.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
E.to_csv(f"{out}_edges.csv", index=False)
vc = E.verdict.value_counts(); t = E[~E.verdict.str.startswith("untestable")]
print("n edges", len(E), "| testable", len(t), "| audited fraction of testable", round((t.verdict == "audited").mean(), 3))
print(vc.to_string())
