"""Replication of the genome-wide seed-match test in Replogle K562 GWPS (normalized pseudo-bulk, 11,258 perturbations
x 8,248 genes). Scale-free statistic: rank of the matched gene's value within its own column (all perturbations).
Expected bottom-0.5% rate with no seed effect: 0.5%."""
import sys, numpy as np, pandas as pd, gzip, re, anndata as ad

from perturbaudit import seedscan as ss
h = pd.read_csv("data/hartman/perturb_seed_reproduce/data/replogle_k562_2kb_filt_no_lfc_cutoff_off_target_candidates.csv.gz", low_memory=False, usecols=["guide_id", "gene", "protospacer_sequence"])
G = h.dropna().drop_duplicates().rename(columns={"guide_id": "guide", "gene": "target", "protospacer_sequence": "protospacer"})
G["guide"] = G.guide + "::" + G.protospacer
S = ss.build_sites("data/seedscan/promoters_transcript_w1000.tsv")
R = ss.scan(G, S, min_len=8); del S
R = R.groupby(["target", "gene"], as_index=False).seed_len.max()
R.to_csv("data/seedscan/replogle_seedscan_tx_pairs.csv.gz", index=False)
a = ad.read_h5ad("data/replogle/K562_gwps_normalized_bulk_01.h5ad")
X = np.asarray(a.X, dtype=np.float32); tg = [s.split("_")[1] for s in a.obs_names]; gn = list(a.var.gene_name.astype(str))
pi = {}
for i, t in enumerate(tg): pi.setdefault(t, i)
gi = {}
for i, g in enumerate(gn): gi.setdefault(g, i)
tss = {}
with gzip.open("data/annot/gencode.v43.basic.gtf.gz", "rt") as f:
    for line in f:
        if line.startswith("#"): continue
        c = line.split("\t", 9)
        if c[2] != "gene": continue
        tss.setdefault(re.search(r'gene_name "([^"]+)', c[8]).group(1), (c[0], int(c[3]) if c[6] == "+" else int(c[4])))
def far(x, y):
    u, v = tss.get(x), tss.get(y)
    return u is not None and v is not None and (u[0] != v[0] or abs(u[1] - v[1]) >= 1_000_000)
R = R[R.target.isin(pi.keys()) & R.gene.isin(gi.keys()) & (R.target != R.gene)]
R = R[[far(x, y) for x, y in zip(R.target, R.gene)]].copy()
r_ = np.array([pi[t] for t in R.target]); c_ = np.array([gi[g] for g in R.gene]); pct = np.empty(len(R))
for c in np.unique(c_):
    m = c_ == c; col = np.sort(X[:, c]); pct[m] = np.searchsorted(col, X[r_[m], c], side="right") / len(col)
R["gene_pct"] = pct
rng = np.random.default_rng(0); base = []
keys = set(zip(R.target, R.gene)); tl = [t for t in pi if t in tss]; gl = [g for g in gi if g in tss]
cols_sorted = {}
while len(base) < 100_000:
    t, g = tl[rng.integers(len(tl))], gl[rng.integers(len(gl))]
    if t == g or (t, g) in keys or not far(t, g): continue
    c = gi[g]
    if c not in cols_sorted: cols_sorted[c] = np.sort(X[:, c])
    base.append(np.searchsorted(cols_sorted[c], X[pi[t], c], side="right") / X.shape[0])
base = np.array(base)
R["bin"] = pd.cut(R.seed_len, [7, 9, 10, 11, 12, 13, 18, 20], labels=["8-9", "10", "11", "12", "13", "14-18", "19-20"])
T = R.groupby("bin").agg(n=("gene_pct", "size"), bottom_0p5pct=("gene_pct", lambda x: float((x < 0.005).mean())), bottom_1pct=("gene_pct", lambda x: float((x < 0.01).mean())))
T.loc["random pairs"] = [len(base), float((base < 0.005).mean()), float((base < 0.01).mean())]
print(T.to_string()); T.to_csv("derived/m4_genomewide_gwps.csv")
