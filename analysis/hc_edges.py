"""High-confidence knockdown edges (resource): K562 GWPS 788-gene panel parent->child pairs that pass the audit checks
(supported |z| >= 5; TSS distance >= 10 kb; no parent-guide seed match >= 11 nt at the child's promoter in the
Replogle dual library map) and replicate (same sign, |z| >= 3) in RPE1, HepG2 and Jurkat; HCT116 reported alongside."""
import gzip, re, numpy as np, pandas as pd
D = pd.read_csv("derived/cross_platform_pairs.csv.gz")
W = D.pivot_table(index=["parent", "child"], columns="line", values="zT", aggfunc="first")
K = D.groupby(["parent", "child"]).zK562.first()
W = W.join(K)
tss = {}
with gzip.open("data/annot/gencode.v43.basic.gtf.gz", "rt") as f:
    for line in f:
        if line.startswith("#"): continue
        c = line.split("\t", 9)
        if c[2] != "gene": continue
        tss.setdefault(re.search(r'gene_name "([^"]+)', c[8]).group(1), (c[0], int(c[3]) if c[6] == "+" else int(c[4])))
def dist(a, b):
    u, v = tss.get(a), tss.get(b)
    return np.nan if u is None or v is None else (abs(u[1] - v[1]) if u[0] == v[0] else np.inf)
W = W.reset_index()
W["tss_distance"] = [dist(a, b) for a, b in zip(W.parent, W.child)]
seed = set()
for ch in pd.read_csv("derived/offtarget_maps/hCRISPRi_dual_Replogle_2022_seed_offtargets_ge10.csv.gz", chunksize=1_000_000, usecols=["target", "gene", "seed_len"]):
    ch = ch[ch.seed_len >= 11]; seed |= set(zip(ch.target, ch.gene))
W["seed_match_ge11"] = [(a, b) in seed for a, b in zip(W.parent, W.child)]
rev = D.set_index(["parent", "child"]).zK562.to_dict()
W["reverse_zK562"] = [rev.get((b, a), np.nan) for a, b in zip(W.parent, W.child)]
for l in ["RPE1", "HepG2", "Jurkat", "HCT116"]:
    W[f"rep_{l}"] = (np.sign(W[l]) == np.sign(W.zK562)) & (W[l].abs() >= 3)
W["n_same_platform_rep"] = W[["rep_RPE1", "rep_HepG2", "rep_Jurkat"]].sum(1)
W["tested_all3"] = W[["RPE1", "HepG2", "Jurkat"]].notna().all(1)
ok = (W.tss_distance >= 10_000) & ~W.seed_match_ge11
W["tier"] = np.where(ok & W.tested_all3 & (W.n_same_platform_rep == 3) & W.rep_HCT116, "A: 4 lines incl. independent platform",
            np.where(ok & W.tested_all3 & (W.n_same_platform_rep == 3), "B: 3 same-platform lines",
            np.where(ok & (W.n_same_platform_rep >= 2), "C: 2 lines", "not high-confidence")))
W = W.rename(columns={"RPE1": "z_RPE1", "HepG2": "z_HepG2", "Jurkat": "z_Jurkat", "HCT116": "z_HCT116"})
out = W[W.tier != "not high-confidence"].sort_values(["tier", "zK562"], key=lambda s: s if s.name == "tier" else -s.abs())
out.to_csv("derived/high_confidence_edges.csv", index=False)
print(W.tier.value_counts()); print("flagged neighbour:", (W.tss_distance < 10_000).sum(), " seed>=11:", W.seed_match_ge11.sum())
print("bidirectional (reverse |z|>=5) among tier A/B:", (out[out.tier.str[0].isin(["A","B"])].reverse_zK562.abs() >= 5).mean().round(3))
print("parents:", out.parent.nunique(), "children:", out.child.nunique())
print(out[out.tier.str.startswith("A")].parent.value_counts().head(12).to_dict())
print(out[out.tier.str.startswith("A")].head(10)[["parent","child","zK562","z_RPE1","z_HepG2","z_Jurkat","z_HCT116"]].round(1).to_string(index=False))
