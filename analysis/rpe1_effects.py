"""RPE1 knockdown z-matrix (Replogle 2022 RPE1 essential screen) in the target-effects format used for transfer:
null-variance z against the pooled non-targeting cells; for each gene, its element with the most cells (at least 20). Input: per-element statistics from extract_replogle_sc.py (element_stats.npz, names.npz, nt_cells_lognorm.npy).
Usage: python analysis/rpe1_effects.py ELEMENT_STATS_DIR OUT_NPZ"""
import sys, numpy as np
from perturbaudit import decoy

D, out = sys.argv[1:3]
st = np.load(f"{D}/element_stats.npz"); nm = np.load(f"{D}/names.npz", allow_pickle=True)
nt = np.load(f"{D}/nt_cells_lognorm.npy", mmap_mode="r")
S1, N = st["S1"], st["N"]
el, eg, vn = nm["element"], nm["element_gene"], nm["var_gene_name"]
X = np.asarray(nt, dtype=np.float64)
Z = decoy.null_var_z(N, S1, X.mean(0), X.var(0, ddof=1), len(X))
best = {}
for i in np.flatnonzero(np.array(["non-targeting" not in s for s in el]) & (N >= 20)):
    g = eg[i]
    if g not in best or N[i] > N[best[g]]:
        best[g] = i
perts = sorted(best)
idx = [best[g] for g in perts]
np.savez_compressed(out, Z=Z[idx].astype(np.float16), perts=np.array(perts, dtype=str), genes=np.array(vn, dtype=str),
                    n=N[idx].astype(float))
print(len(perts), Z.shape[1])
