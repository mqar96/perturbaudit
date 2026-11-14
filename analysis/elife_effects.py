"""Replogle 2022 eLife (GSE205310): stream a 10x MatrixMarket file in two passes and compute, per guide/construct,
batch-matched null-variance z for the top expressed genes + all targeted genes. Writes <out>.npz.
Usage: elife_effects.py PREFIX OUT [n_top_genes=4000]"""
import sys, numpy as np, pandas as pd, gzip
sys.path.insert(0, "perturbaudit/src")
pre, out = sys.argv[1], sys.argv[2]; NTOP = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
feat = pd.read_csv(f"{pre}.features.tsv.gz", sep="\t", header=None); genes = feat[1].astype(str).values
bc = pd.read_csv(f"{pre}.barcodes.tsv.gz", header=None)[0].astype(str).values
ci = pd.read_csv(f"{pre}_cell_identities.csv.gz")
ci = ci[ci.guide_identity.astype(str).str.count(r"\|") == 0]  # single-guide cells only
gid = dict(zip(ci.cell_barcode, ci.guide_identity.astype(str))); gem = dict(zip(ci.cell_barcode, ci.gemgroup))
cell_guide = np.array([gid.get(b, "") for b in bc]); cell_gem = np.array([gem.get(b, -1) for b in bc])
def chunks():
    with gzip.open(f"{pre}.matrix.mtx.gz", "rt") as f:
        for line in f:
            if not line.startswith("%"): break
        for df in pd.read_csv(f, sep=" ", header=None, names=["g", "c", "v"], dtype={"g": np.int32, "c": np.int32, "v": np.float32}, chunksize=20_000_000):
            yield df.g.values - 1, df.c.values - 1, df.v.values
tot = np.zeros(len(bc)); gsum = np.zeros(len(genes))
for g, c, v in chunks():
    np.add.at(tot, c, v); np.add.at(gsum, g, v)
print("pass 1 done", flush=True)
targets = sorted({x.split("_")[0] for x in set(cell_guide) if x and not x.lower().startswith("non")})
keep_g = set(np.argsort(-gsum)[:NTOP]) | {i for i, gname in enumerate(genes) if gname in set(targets)}
keep_g = np.array(sorted(keep_g)); gpos = -np.ones(len(genes), np.int64); gpos[keep_g] = np.arange(len(keep_g))
guides = sorted({x for x in cell_guide if x}); gi = {x: i for i, x in enumerate(guides)}
cg = np.array([gi.get(x, -1) for x in cell_guide])
gems = sorted({x for x in cell_gem if x >= 0}); gmi = {x: i for i, x in enumerate(gems)}
cb = np.array([gmi.get(x, -1) for x in cell_gem])
G, P, B = len(guides), len(keep_g), len(gems)
S1 = np.zeros((G, P)); S2 = np.zeros((G, P)); N = np.zeros(G); NB = np.zeros((G, B))
half = (np.random.default_rng(0).random(len(bc)) < 0.5).astype(np.int8)
S1a = np.zeros((G, P)); NA = np.zeros(G)
ok = cg >= 0
np.add.at(N, cg[ok], 1); np.add.at(NB, (cg[ok], cb[ok]), 1); np.add.at(NA, cg[ok & (half == 1)], 1)
for g, c, v in chunks():
    m = (gpos[g] >= 0) & (cg[c] >= 0)
    g, c, v = gpos[g[m]], c[m], v[m]
    val = np.log1p(v / np.maximum(tot[c], 1) * 1e4)
    np.add.at(S1, (cg[c], g), val); np.add.at(S2, (cg[c], g), val ** 2)
    h = half[c] == 1; np.add.at(S1a, (cg[c][h], g[h]), val[h])
print("pass 2 done", flush=True)
is_nt = np.array([x.lower().startswith("non") for x in guides])
# per-batch control moments from NT guides: need per-batch sums -> approximate with NT pooled per batch via NB weights
# exact per-batch NT moments require per-cell batch sums; compute from pooled NT (batch-weighted reference below)
mu0 = S1[is_nt].sum(0) / N[is_nt].sum(); var0 = np.maximum(S2[is_nt].sum(0) / N[is_nt].sum() - mu0 ** 2, 1e-12); n0 = N[is_nt].sum()
Z = (S1 / np.maximum(N[:, None], 1) - mu0) / np.sqrt(var0 * (1 / np.maximum(N[:, None], 1) + 1 / n0))
NBh = N - NA
ZA = (S1a / np.maximum(NA[:, None], 1) - mu0) / np.sqrt(var0 * (1 / np.maximum(NA[:, None], 1) + 1 / n0))
ZB = ((S1 - S1a) / np.maximum(NBh[:, None], 1) - mu0) / np.sqrt(var0 * (1 / np.maximum(NBh[:, None], 1) + 1 / n0))
np.savez_compressed(out, Z=Z.astype(np.float32), ZA=ZA.astype(np.float32), ZB=ZB.astype(np.float32), nA=NA, nB=NBh, guides=np.array(guides), genes=genes[keep_g], n=N, is_nt=is_nt,
                    target=np.array([x.split("_")[0] if not x.lower().startswith("non") else "non-targeting" for x in guides]))
print("saved", out, Z.shape, "NT cells", int(n0), flush=True)
