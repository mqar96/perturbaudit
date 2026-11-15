"""Post-process Orion HCT116 effects on the PC: (1) M4 seed-match analysis on the full matrix with the Hartman xaira table,
(2) compact effects file (all perturbations x panel genes + top-8000 NT-expressed genes) for transfer/M3 in the workspace."""
import os, numpy as np, pandas as pd
R = os.environ.get("WORK_DIR", ".")
log = open(os.path.join(R, "orion_post.log"), "w", buffering=1)
z = np.load(os.path.join(R, "orion_hct116_effects.npz"), allow_pickle=True)
Z = z["Z"]; perts = [str(p) for p in z["perts"]]; genes = np.array([str(g) for g in z["genes"]]); n = z["n"]
Z = np.nan_to_num(Z.astype(np.float32), nan=0.0, posinf=1000.0, neginf=-1000.0).clip(-1000, 1000)
st = np.load(os.path.join(R, "orion_state.npz"), allow_pickle=True); CN = st["CN"]; CS1 = st["CS1"]
mu = CS1[: len(CN)].sum(0) / max(CN.sum(), 1)
log.write(f"Z {Z.shape}; inf/huge cells {(np.abs(Z) >= 1000).sum()}; NT cells {CN.sum()}; median cells/pert {np.median(n)}\n")
pi = {p: i for i, p in enumerate(perts)}; gi = {}
for i, g in enumerate(genes): gi.setdefault(g, i)
# own-target knockdown
own = np.array([Z[pi[p], gi[p]] if p in gi else np.nan for p in perts])
log.write(f"own-target z median {np.nanmedian(own):.1f}; frac <= -3 {np.nanmean(own <= -3):.3f}\n")
# M4
h = pd.read_csv(os.path.join(R, "xaira_hartman.csv"))
rows = []
for r in h.itertuples():
    if r.gene not in pi or r.offtarget_gene not in gi: continue
    o = gi[r.offtarget_gene]
    if mu[o] < 0.05: continue
    rows.append((r.gene, r.offtarget_gene, r.seed_match_len, r.offtarget_log2fc, r.neighbor_rank, own[pi[r.gene]], Z[pi[r.gene], o]))
M = pd.DataFrame(rows, columns=["gene", "offtarget_gene", "seed_len", "hartman_lfc", "neighbor_rank", "own_z", "z_off"])
# null for each off-target gene: its z across all perturbations (robust centre/scale)
cols = sorted(set(M.offtarget_gene)); ci = [gi[g] for g in cols]
med = np.median(Z[:, ci], 0); mad = np.median(np.abs(Z[:, ci] - med), 0) * 1.4826 + 1e-6
mm = dict(zip(cols, med)); ms = dict(zip(cols, mad))
M["z_rel"] = [(v - mm[g]) / ms[g] for v, g in zip(M.z_off, M.offtarget_gene)]
M.to_csv(os.path.join(R, "orion_m4.csv"), index=False)
M["bin"] = pd.cut(M.seed_len, [-1, 0, 7, 10, 11, 13, 25], labels=["0", "5-7", "8-10", "11", "12-13", "14+"])
log.write(M.groupby("bin").agg(n=("z_off", "size"), mean_z=("z_off", "mean"), frac_lt_m3=("z_off", lambda d: float((d < -3).mean())), frac_rel_lt_m3=("z_rel", lambda d: float((d < -3).mean()))).to_string() + "\n")
# compact
pg = set(open(os.path.join(R, "panel_genes.txt")).read().split())
top = set(np.argsort(-mu)[:8000]); keep = sorted(top | {gi[g] for g in pg if g in gi} | {gi[p] for p in perts if p in gi and mu[gi[p]] >= 0.02})
np.savez_compressed(os.path.join(R, "orion_hct116_compact.npz"), Z=Z[:, keep].astype(np.float16), perts=np.array(perts, dtype=str), genes=genes[keep], n=n, mu=mu[keep], own=own)
log.write(f"compact genes {len(keep)}\nPOST DONE\n")
