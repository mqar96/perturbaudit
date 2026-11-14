"""M4 genome-wide: do seed matches found by perturbaudit.seedscan anywhere in the genome (not only among the 20
nearest genes, as in Hartman's tables) predict knock-down of the matched gene in X-Atlas/Orion HCT116?
Pairs within 1 Mb of the target and the target itself are excluded. Baseline: random target-gene pairs."""
import numpy as np, pandas as pd, gzip, re
from scipy.stats import fisher_exact
T = np.load("data/orion/orion_hct116_compact.npz", allow_pickle=True)
Z = T["Z"]; perts = [str(p) for p in T["perts"]]; genes = [str(g) for g in T["genes"]]; mu = T["mu"]
pi = {p: i for i, p in enumerate(perts)}; gi = {}
for i, g in enumerate(genes): gi.setdefault(g, i)
expr = mu >= 0.1
tss = {}
with gzip.open("data/annot/gencode.v43.basic.gtf.gz", "rt") as f:
    for line in f:
        if line.startswith("#"): continue
        c = line.split("\t", 9)
        if c[2] != "gene": continue
        tss.setdefault(re.search(r'gene_name "([^"]+)', c[8]).group(1), (c[0], int(c[3]) if c[6] == "+" else int(c[4])))
def far(a, b):
    x, y = tss.get(a), tss.get(b)
    return x is not None and y is not None and (x[0] != y[0] or abs(x[1] - y[1]) >= 1_000_000)
best = {}
for ch in pd.read_csv("data/seedscan/xaira_seedscan_tx.csv.gz", chunksize=2_000_000, usecols=["target", "gene", "seed_len"]):
    ch = ch[ch.target.isin(pi.keys()) & ch.gene.isin(gi.keys()) & (ch.target != ch.gene)]
    g = ch.groupby(["target", "gene"]).seed_len.max()
    for k, v in g.items():
        if v > best.get(k, 0): best[k] = v
rows = [(t, g, L, float(Z[pi[t], gi[g]])) for (t, g), L in best.items() if expr[gi[g]] and far(t, g)]
D = pd.DataFrame(rows, columns=["target", "gene", "seed_len", "z"])
rng = np.random.default_rng(0); tl = [p for p in perts if p in tss]; ge = [g for g in genes if expr[gi[g]] and g in tss]
base = []
while len(base) < 200_000:
    a, b = tl[rng.integers(len(tl))], ge[rng.integers(len(ge))]
    if a != b and (a, b) not in best and far(a, b): base.append(float(Z[pi[a], gi[b]]))
base = np.array(base)
D["bin"] = pd.cut(D.seed_len, [7, 9, 10, 11, 12, 13, 18, 20], labels=["8-9", "10", "11", "12", "13", "14-18", "19-20"])
S = D.groupby("bin").agg(n=("z", "size"), frac_lt_m3=("z", lambda x: float((x < -3).mean())), median_z=("z", "median"))
S.loc["no match (random)"] = [len(base), float((base < -3).mean()), float(np.median(base))]
b_hit = int((base < -3).sum())
for lo in (11, 12, 13):
    h = D[D.seed_len >= lo]
    S.loc[f">={lo} vs random Fisher"] = [len(h), fisher_exact([[int((h.z < -3).sum()), int((h.z >= -3).sum())], [b_hit, len(base) - b_hit]], alternative="greater")[1], np.nan]
print(S.to_string()); S.to_csv("derived/m4_genomewide_hct116.csv")
D[D.seed_len >= 11].sort_values("z").to_csv("derived/m4_genomewide_hct116_pairs11.csv", index=False)
# --- confound control: rank of z(t,g) within gene g's column (all perturbations) and within target t's row
Zf = Z.astype(np.float32)
ix = [(pi[t], gi[g]) for t, g in zip(D.target, D.gene)]; r_ = np.array([a for a, b in ix]); c_ = np.array([b for a, b in ix])
colrank = np.empty(len(D)); rowrank = np.empty(len(D))
for c in np.unique(c_):
    m = c_ == c; col = np.sort(Zf[:, c]); colrank[m] = np.searchsorted(col, Zf[r_[m], c], side="right") / len(col)
ecols = np.flatnonzero(expr)
for r in np.unique(r_):
    m = r_ == r; row = np.sort(Zf[r, ecols]); rowrank[m] = np.searchsorted(row, Zf[r, c_[m]], side="right") / len(row)
D["gene_pct"] = colrank; D["target_pct"] = rowrank
C = D.groupby("bin").agg(n=("z", "size"), gene_pct_lt_0p5pct=("gene_pct", lambda x: float((x < 0.005).mean())),
                          target_pct_lt_0p5pct=("target_pct", lambda x: float((x < 0.005).mean())),
                          both_lt_1pct=("z", lambda x: float(((D.loc[x.index, "gene_pct"] < 0.01) & (D.loc[x.index, "target_pct"] < 0.01)).mean())))
print("expected under no seed effect: 0.005, 0.005, 0.0001"); print(C.to_string()); C.to_csv("derived/m4_genomewide_hct116_ranks.csv")
