import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "perturbaudit/src")
from perturbaudit import guidesplit as gs

D = "derived/cd4_by_guide"
o1 = pd.read_parquet(f"{D}/guide_1_obs.parquet"); o2 = pd.read_parquet(f"{D}/guide_2_obs.parquet")
var = pd.read_parquet(f"{D}/var.parquet")
Z1 = np.load(f"{D}/guide_1_zscore.npy", mmap_mode="r"); Z2 = np.load(f"{D}/guide_2_zscore.npy", mmap_mode="r")
i2 = {t: i for i, t in enumerate(o2.target_condition)}
pairs = [(i, i2[t]) for i, t in enumerate(o1.target_condition) if t in i2]
pairs = [(a, b) for a, b in pairs if o1.ontarget_significant.iat[a] and o2.ontarget_significant.iat[b]]
print("target x condition rows with both guides effective:", len(pairs))
a_idx = np.array([p[0] for p in pairs]); b_idx = np.array([p[1] for p in pairs])
z1 = np.asarray(Z1[np.sort(a_idx)], dtype=np.float32)[np.argsort(np.argsort(a_idx))]
z2 = np.asarray(Z2[np.sort(b_idx)], dtype=np.float32)[np.argsort(np.argsort(b_idx))]
vid = np.asarray(var.gene_id.values, dtype=object)
tgt = np.asarray(o1.target_contrast.values, dtype=object)[a_idx]
self_mask = tgt[:, None] == vid[None, :]
z1[self_mask] = 0; z2[self_mask] = 0

lib = pd.read_csv("data/marson/sgrna_library_metadata.suppl_table.csv")
near_cols = [c for c in lib.columns if "10kb" in c.lower() or "bidirectional" in c.lower()]
print("library neighbour columns:", near_cols[:6])

res = {}
for name, A, B in [("guide1->guide2", z1, z2), ("guide2->guide1", z2, z1)]:
    r = gs.replication_table(A, B, 5.0, 2.0)
    res[name] = r
    print(name, r, f"replication rate {r['replicated']/max(r['discovered'],1):.3f}")
d = (np.abs(z1) >= 5) | (np.abs(z2) >= 5)
Q, p = gs.heterogeneity(z1[d], z2[d])
q = gs.bh(p)
frac_het = float((q <= 0.05).mean())
print(f"edges discovered by either guide (|z|>=5): {d.sum():,}; significantly guide-specific (BH q<=0.05): {(q<=0.05).sum():,} ({frac_het:.1%})")
strong_both = (np.abs(z1) >= 5) & (np.abs(z2) >= 5) & (np.sign(z1) == np.sign(z2))
print(f"edges strong in both guides, same sign: {strong_both.sum():,}")
per_row_disc = (np.abs(z1) >= 5).sum(1); per_row_rep = ((np.abs(z1) >= 5) & (np.abs(z2) >= 2) & (np.sign(z1) == np.sign(z2))).sum(1)
rows_ok = per_row_disc >= 20
rate = per_row_rep[rows_ok] / per_row_disc[rows_ok]
print(f"perturbations with >=20 guide-1 edges: {rows_ok.sum():,}; median per-perturbation replication {np.median(rate):.2f}; with <50% replicated: {(rate<0.5).sum():,} ({(rate<0.5).mean():.1%})")
flag_cols = ["distal_offtarget_flag", "neighboring_gene_KD"]
for c in flag_cols:
    f1 = o1[c].values[a_idx].astype(bool) | o2[c].values[b_idx].astype(bool)
    if f1.any():
        print(f"  {c}: rows {f1.sum()}, median replication flagged {np.median(per_row_rep[rows_ok & f1]/np.maximum(per_row_disc[rows_ok & f1],1)) if (rows_ok&f1).any() else float('nan'):.2f} vs unflagged {np.median(per_row_rep[rows_ok & ~f1]/np.maximum(per_row_disc[rows_ok & ~f1],1)):.2f}")
out = pd.DataFrame(dict(target_condition=o1.target_condition.values[a_idx], gene=o1.target_contrast_gene_name.values[a_idx],
                        condition=o1.culture_condition.values[a_idx], n_edges_g1=per_row_disc,
                        n_rep_g2=per_row_rep, guide_corr_authors=o1.guide_correlation_all.values[a_idx]))
out.to_csv("derived/m2_cd4_per_perturbation.csv", index=False)
json.dump(dict(res=res, n_pairs=len(pairs), discovered_either=int(d.sum()), guide_specific_q05=int((q<=0.05).sum()), frac_guide_specific=frac_het,
               median_rep=float(np.median(rate)), n_rows_ge20=int(rows_ok.sum()), frac_rows_rep_lt50=float((rate<0.5).mean())),
          open("derived/m2_cd4_summary.json", "w"), indent=1)
