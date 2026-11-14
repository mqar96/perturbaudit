import numpy as np, pandas as pd, anndata as ad, json

lib = pd.read_csv("repos/perturb_seed/guide_libraries/sgrna_replogle.csv")
lib["guide_id"] = lib["sgID_A"] + "|" + lib["sgID_B"]
g2el = dict(zip(lib["guide_id"], lib["unique sgRNA pair ID"]))
h = pd.read_csv("data/hartman/perturb_seed_reproduce/data/replogle_k562_2kb_filt_no_lfc_cutoff_off_target_candidates.csv.gz", low_memory=False)
hc = h[(h.seed_match_len >= 12) & (h.offtarget_log2fc < -0.5)].copy()
hc["element"] = hc["guide_id"].map(g2el)
hc["full_match"] = (hc.seed_match_len == 20) & (hc.guide_hamming_dist == 0)

a = ad.read_h5ad("data/replogle/K562_gwps_normalized_bulk_01.h5ad")
ok = a.obs["num_cells_filtered"].fillna(0).values >= 25
Z = np.asarray(a.X)[ok] * np.sqrt(a.obs["num_cells_filtered"].values[ok])[:, None]
Z = np.nan_to_num(Z)
els = np.asarray(a.obs_names[ok], dtype=object)
el_gene = np.array([e.split("_")[1] for e in els], dtype=object)
el_idx = {e: i for i, e in enumerate(els)}
vname = np.asarray(a.var["gene_name"].values, dtype=object)
gene_to_els = pd.Series(range(len(els))).groupby(el_gene).apply(list).to_dict()
strength = np.sqrt((Z ** 2).mean(1)) + 1e-9
THR = 5.0
rng = np.random.default_rng(0)

def edge_set(i, excl):
    m = (np.abs(Z[i]) > THR) & ~np.isin(vname, excl)
    return m

rows = []
for _, r in hc.iterrows():
    P = el_idx.get(r["element"])
    if P is None:
        continue
    O_list = gene_to_els.get(r["offtarget_gene"], [])
    O_list = [o for o in O_list if o != P]
    if not O_list:
        continue
    O = max(O_list, key=lambda o: strength[o])
    excl = [el_gene[P], r["offtarget_gene"]]
    mask = ~np.isin(vname, excl)
    cP = np.corrcoef(Z[P, mask], Z[O, mask])[0, 1]
    eP = edge_set(P, excl); eO = edge_set(O, excl)
    concord = eP & eO & (np.sign(Z[P]) == np.sign(Z[O]))
    expl = concord.sum() / max(eP.sum(), 1)
    s = strength[O]
    pool = np.flatnonzero((np.abs(strength - s) / s < 0.2) & (el_gene != el_gene[P]) & (el_gene != r["offtarget_gene"]))
    nulls_c, nulls_e = [], []
    for k in rng.choice(pool, min(200, len(pool)), replace=False):
        nulls_c.append(np.corrcoef(Z[P, mask], Z[k, mask])[0, 1])
        ek = edge_set(k, excl)
        nulls_e.append((eP & ek & (np.sign(Z[P]) == np.sign(Z[k]))).sum() / max(eP.sum(), 1))
    sib = [o for o in gene_to_els.get(el_gene[P], []) if o != P]
    c_sib = max([np.corrcoef(Z[P, mask], Z[o, mask])[0, 1] for o in sib], default=np.nan)
    rows.append(dict(parent_gene=el_gene[P], element=els[P], offtarget=r["offtarget_gene"], full_match=bool(r["full_match"]),
                     seed_len=int(r["seed_match_len"]), offtarget_log2fc=float(r["offtarget_log2fc"]),
                     n_edges_parent=int(eP.sum()), corr_parent_offtarget=cP, corr_null_median=float(np.median(nulls_c)),
                     corr_null_p=(np.sum(np.array(nulls_c) >= cP) + 1) / (len(nulls_c) + 1),
                     frac_edges_explained=expl, frac_explained_null_median=float(np.median(nulls_e)),
                     corr_parent_sibling_element=c_sib))
df = pd.DataFrame(rows).drop_duplicates(["element", "offtarget"])
df.to_csv("derived/m4_hartman_gwps_pairs.csv", index=False)
for lab, sub in [("all HC pairs", df), ("seed-only (not full 20bp match)", df[~df.full_match]), ("full 20bp (shared promoter)", df[df.full_match])]:
    sig = sub[sub.corr_null_p <= 0.01]
    print(f"\n== {lab}: n={len(sub)}")
    print(f"  median corr(parent, off-target KD) = {sub.corr_parent_offtarget.median():.3f}  vs null {sub.corr_null_median.median():.3f}")
    print(f"  pairs with corr beyond 99% of strength-matched nulls: {len(sig)} ({len(sig)/max(len(sub),1):.0%})")
    big = sub[sub.n_edges_parent >= 20]
    print(f"  parents with >=20 edges: {len(big)}; median share of edges shared with off-target KD = {big.frac_edges_explained.median():.2f} (null {big.frac_explained_null_median.median():.2f})")
    print(f"  parents with >=20 edges where >=30% of edges are shared with off-target KD: {(big.frac_edges_explained>=0.3).sum()}")
print("\nTop examples:")
print(df.sort_values("frac_edges_explained", ascending=False).head(12)[["parent_gene","offtarget","full_match","n_edges_parent","corr_parent_offtarget","corr_null_median","frac_edges_explained","frac_explained_null_median","corr_parent_sibling_element"]].round(2).to_string(index=False))
