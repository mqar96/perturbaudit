import json, numpy as np, pandas as pd
DD = "derived/cd4_by_donors"; DG = "derived/cd4_by_guide"
splits = [("CE0006864_CE0008162", "CE0008678_CE0010866"), ("CE0006864_CE0008678", "CE0008162_CE0010866"), ("CE0006864_CE0010866", "CE0008162_CE0008678")]
var_d = pd.read_parquet(f"{DD}/var.parquet").gene_id.values
var_g = pd.read_parquet(f"{DG}/var.parquet").gene_id.values
common_genes = np.intersect1d(var_d, var_g)
cd = np.searchsorted(var_d, common_genes) if np.all(np.diff(var_d.astype(str).argsort()) != 0) else None
pos_d = {g: i for i, g in enumerate(var_d)}; pos_g = {g: i for i, g in enumerate(var_g)}
cols_d = np.array([pos_d[g] for g in common_genes]); cols_g = np.array([pos_g[g] for g in common_genes])

o1 = pd.read_parquet(f"{DG}/guide_1_obs.parquet"); o2 = pd.read_parquet(f"{DG}/guide_2_obs.parquet")
Z1 = np.load(f"{DG}/guide_1_zscore.npy", mmap_mode="r"); Z2 = np.load(f"{DG}/guide_2_zscore.npy", mmap_mode="r")
g1 = {t: i for i, t in enumerate(o1.target_condition)}; g2 = {t: i for i, t in enumerate(o2.target_condition)}
eff = {t for t, i in g1.items() if t in g2 and o1.ontarget_significant.iat[i] and o2.ontarget_significant.iat[g2[t]]}

def metrics(A, B, tgt_cols):
    A = A.copy(); B = B.copy()
    for r, c in enumerate(tgt_cols):
        if c >= 0: A[r, c] = 0; B[r, c] = 0
    out = {}
    for nm, X, Y in [("a->b", A, B), ("b->a", B, A)]:
        disc = np.abs(X) >= 5
        rep = disc & (np.abs(Y) >= 2) & (np.sign(X) == np.sign(Y))
        out[nm] = rep.sum() / max(disc.sum(), 1)
    strong = np.maximum(np.abs(A), np.abs(B)) >= 8
    one = ((np.abs(A) >= 8) & (np.abs(B) < 1)) | ((np.abs(B) >= 8) & (np.abs(A) < 1))
    return dict(replication=float(np.mean(list(out.values()))), strong=int(strong.sum()), one_side_only=int(one.sum()),
                frac_one_side=float(one.sum() / max(strong.sum(), 1)), n_disc=int((np.abs(A) >= 5).sum() + (np.abs(B) >= 5).sum()))

res = {}
for a, b in splits:
    oa = pd.read_parquet(f"{DD}/{a}_obs.parquet"); ob = pd.read_parquet(f"{DD}/{b}_obs.parquet")
    ia = {t: i for i, t in enumerate(oa.target_condition)}; ib = {t: i for i, t in enumerate(ob.target_condition)}
    tcs = sorted(t for t in ia if t in ib and t in eff)
    Za = np.load(f"{DD}/{a}_zscore.npy", mmap_mode="r"); Zb = np.load(f"{DD}/{b}_zscore.npy", mmap_mode="r")
    A = np.stack([np.asarray(Za[ia[t]], np.float32)[cols_d] for t in tcs]); B = np.stack([np.asarray(Zb[ib[t]], np.float32)[cols_d] for t in tcs])
    G1 = np.stack([np.asarray(Z1[g1[t]], np.float32)[cols_g] for t in tcs]); G2 = np.stack([np.asarray(Z2[g2[t]], np.float32)[cols_g] for t in tcs])
    cpos = {g: i for i, g in enumerate(common_genes)}
    tgt_cols = [cpos.get(o1.target_contrast.iat[g1[t]], -1) for t in tcs]
    res[f"{a}|{b}"] = dict(n_perturbations=len(tcs), donor_split=metrics(A, B, tgt_cols), guide_split=metrics(G1, G2, tgt_cols))
    print(f"\nsplit {a} vs {b}: {len(tcs)} shared perturbations")
    for k in ["donor_split", "guide_split"]:
        m = res[f"{a}|{b}"][k]
        print(f"  {k:12s} replication {m['replication']:.3f}  strong edges {m['strong']:,}  one-side-only {m['one_side_only']:,} ({m['frac_one_side']:.1%})")
json.dump(res, open("derived/m2_donor_vs_guide.json", "w"), indent=1)
