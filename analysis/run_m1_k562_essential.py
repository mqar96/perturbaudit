import json, sys, numpy as np, pandas as pd

from perturbaudit import decoy

D = "derived/k562_essential"
st = np.load(f"{D}/element_stats.npz"); nm = np.load(f"{D}/names.npz", allow_pickle=True)
cm = np.load(f"{D}/cell_meta.npz")
nt = np.load(f"{D}/nt_cells_lognorm.npy", mmap_mode="r")
S1, S2, N, NG, gem_levels = st["S1"], st["S2"], st["N"], st["NG"], st["gem_levels"]
el, el_gene, vname = nm["element"], nm["element_gene"], nm["var_gene_name"]
is_nt_el = np.array(["non-targeting" in s for s in el])
real = np.flatnonzero(~is_nt_el & (N >= 20))

ref_mask = decoy.split_nt(cm["nt_gem"], 0.5, seed=1)
X_ref = np.asarray(nt[ref_mask], dtype=np.float64)
ref_mean = X_ref.mean(0); ref_var = X_ref.var(0, ddof=1); ref_n = len(X_ref)
del X_ref
X_pool = np.asarray(nt[~ref_mask], dtype=np.float32)
pool_gem = cm["nt_gem"][~ref_mask]

expressed = ref_mean > np.log1p(0.1)
STAT = sys.argv[1] if len(sys.argv) > 1 else "nullvar"
zf = (lambda n_, a_, b_: decoy.null_var_z(n_, a_, ref_mean, ref_var, ref_n)) if STAT == "nullvar" else (lambda n_, a_, b_: decoy.welch_z(n_, a_, b_, ref_mean, ref_var, ref_n))
Zr = zf(N[real], S1[real], S2[real])
self_mask = el_gene[real][:, None] == vname[None, :]
valid = (~self_mask) & expressed[None, :]
real_abs = np.abs(Zr)[valid]

res = {}
for label, matched in [("batch_matched", True), ("unmatched", False)]:
    M, pick = decoy.make_decoy_groups(pool_gem, NG[real], gem_levels, n_decoys=len(real), batch_matched=matched, seed=2)
    n, s1, s2 = decoy.group_moments(M, X_pool)
    Zd = zf(n, s1, s2)
    dec_abs = np.abs(Zd)[:, expressed].ravel()
    curve = decoy.target_decoy_fdr(real_abs, dec_abs, [3, 4, 5, 6, 8, 10, 15, 20])
    q = decoy.qvalues(real_abs, dec_abs)
    res[label] = dict(curve=curve, n_q05=int((q <= 0.05).sum()), n_q10=int((q <= 0.10).sum()),
                      decoy_tail_gt5=float((dec_abs > 5).mean()), gaussian_tail_gt5=5.7e-7)
    if matched:
        q_matched = q

print("real perturbations:", len(real), " expressed readout genes:", int(expressed.sum()), " tested pairs:", real_abs.size)
for k, v in res.items():
    print(f"\n== {k}: edges q<=0.05 {v['n_q05']:,}  q<=0.10 {v['n_q10']:,}  decoy P(|z|>5)={v['decoy_tail_gt5']:.2e} (Gaussian 5.7e-07)")
    print("  thr   real_edges  decoy_hits  est_FDR")
    for t, r, d, f in v["curve"]:
        print(f"  {t:4.0f}  {r:11,d}  {d:10,d}  {f:7.4f}")

Q = np.full(Zr.shape, np.nan); Q[valid] = q_matched
rows_i, cols_j = np.nonzero(valid & (Q <= 0.05))
edges = pd.DataFrame(dict(element=el[real][rows_i], parent=el_gene[real][rows_i], child=vname[cols_j],
                          z=Zr[rows_i, cols_j], q=Q[rows_i, cols_j]))
edges.to_parquet("derived/k562_essential/m1_tce_edges_q05_" + STAT + ".parquet")
json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in res.items()},
          open("derived/k562_essential/m1_results_" + STAT + ".json", "w"), indent=1, default=float)
print("\nsaved edges:", len(edges))
