import os
WRAPPERS = os.environ.get("PERTURBAUDIT_WRAPPERS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wrappers"))
import sys, os, json, subprocess, tempfile, shutil, numpy as np, pandas as pd

P = int(sys.argv[1]) if len(sys.argv) > 1 else 300
COND = sys.argv[2] if len(sys.argv) > 2 else "Stim8hr"
OUT = f"derived/cd4_inspre_v2_p{P}_{COND}"
os.makedirs(OUT, exist_ok=True)
DG = "derived/cd4_by_guide"; DD = "derived/cd4_by_donors"
var = pd.read_parquet(f"{DG}/var.parquet"); vid = np.asarray(var.gene_id.values, dtype=object)
vpos = {g: i for i, g in enumerate(vid)}
o1 = pd.read_parquet(f"{DG}/guide_1_obs.parquet"); o2 = pd.read_parquet(f"{DG}/guide_2_obs.parquet")
Z = {"g1": np.load(f"{DG}/guide_1_zscore.npy", mmap_mode="r"), "g2": np.load(f"{DG}/guide_2_zscore.npy", mmap_mode="r")}
F = {"g1": np.load(f"{DG}/guide_1_log_fc.npy", mmap_mode="r"), "g2": np.load(f"{DG}/guide_2_log_fc.npy", mmap_mode="r")}
r1 = {t: i for i, t in enumerate(o1.target_condition)}; r2 = {t: i for i, t in enumerate(o2.target_condition)}
splits = [("CE0006864_CE0008162", "CE0008678_CE0010866")]
oa = pd.read_parquet(f"{DD}/{splits[0][0]}_obs.parquet"); ob = pd.read_parquet(f"{DD}/{splits[0][1]}_obs.parquet")
ra = {t: i for i, t in enumerate(oa.target_condition)}; rb = {t: i for i, t in enumerate(ob.target_condition)}
vard = pd.read_parquet(f"{DD}/var.parquet").gene_id.values; vdpos = {g: i for i, g in enumerate(vard)}

cand = []
for t, i in r1.items():
    if not t.endswith("_" + COND) or t not in r2: continue
    j = r2[t]
    if not (o1.ontarget_significant.iat[i] and o2.ontarget_significant.iat[j]): continue
    g = o1.target_contrast.iat[i]
    if g not in vpos or t not in ra or t not in rb or g not in vdpos: continue
    cand.append((t, g, i, j, float(o1.n_total_de_genes.iat[i] + o2.n_total_de_genes.iat[j])))
cand.sort(key=lambda x: -x[4])
sel = cand[:P]
genes = [c[1] for c in sel]; cols = np.array([vpos[g] for g in genes]); colsd = np.array([vdpos[g] for g in genes])
print("panel genes:", len(genes))

def tce_from(zrows, frows):
    zz = np.asarray(zrows, dtype=np.float64); ff = np.asarray(frows, dtype=np.float64)
    diag = np.diag(ff).copy()
    diag[np.abs(diag) < 1e-3] = np.nan
    R = ff / diag[:, None]
    se = np.abs(ff / np.where(np.abs(zz) > 1e-6, zz, np.nan)) / np.abs(diag)[:, None]
    se = np.where(np.isfinite(se), se, np.nanmedian(se) * 10)
    R = np.where(np.isfinite(R), R, 0.0)
    np.fill_diagonal(R, 1.0); np.fill_diagonal(se, np.nanmedian(se))
    return R, se

mats = {}
for k, rows, src in [("guide1", [c[2] for c in sel], "g1"), ("guide2", [c[3] for c in sel], "g2")]:
    zz = np.stack([np.asarray(Z[src][r], np.float32)[cols] for r in rows])
    ff = np.stack([np.asarray(F[src][r], np.float32)[cols] for r in rows])
    mats[k] = (zz, ff)
for k, oo, rr, nm in [("donorsA", oa, ra, splits[0][0]), ("donorsB", ob, rb, splits[0][1])]:
    Zd = np.load(f"{DD}/{nm}_zscore.npy", mmap_mode="r"); Fd = np.load(f"{DD}/{nm}_log_fc.npy", mmap_mode="r")
    zz = np.stack([np.asarray(Zd[rr[c[0]]], np.float32)[colsd] for c in sel])
    ff = np.stack([np.asarray(Fd[rr[c[0]]], np.float32)[colsd] for c in sel])
    mats[k] = (zz, ff)

R0, _ = tce_from(*mats["guide1"])
off = np.abs(R0 - np.diag(np.diag(R0)))
lam = np.exp(np.linspace(np.log(off.max()), np.log(off.max() * 1e-2), 20))
open(f"{OUT}/lambda.txt", "w").write("\n".join(f"{x:.8g}" for x in lam))
res = {}
for k in ["guide1", "guide2", "donorsA", "donorsB"]:
    zz, ff = mats[k]
    if os.path.exists(f"{OUT}/W_{k}.npy"):
        res[k] = np.load(f"{OUT}/W_{k}.npy"); print(k, "cached", flush=True); continue
    R, se = tce_from(zz, ff)
    tmp = tempfile.mkdtemp()
    pd.DataFrame(R, index=genes, columns=genes).to_csv(f"{tmp}/R.tsv", sep="\t")
    pd.DataFrame(se, index=genes, columns=genes).to_csv(f"{tmp}/SE.tsv", sep="\t")
    subprocess.run(["Rscript", os.path.join(WRAPPERS, "run_inspre_from_R.R"), f"{tmp}/R.tsv", f"{tmp}/SE.tsv", f"{tmp}/S.tsv", f"{OUT}/lambda.txt"], check=True, capture_output=True)
    S = pd.read_csv(f"{tmp}/S.tsv", sep="\t", index_col=0).reindex(index=genes, columns=genes).fillna(0).values
    shutil.rmtree(tmp)
    np.fill_diagonal(S, 0)
    np.save(f"{OUT}/W_{k}.npy", S); np.save(f"{OUT}/Z_{k}.npy", zz.astype(np.float32))
    res[k] = S
    print(k, "nonzero edges", int((S > 0).sum()), flush=True)
json.dump({"genes": genes, "condition": COND}, open(f"{OUT}/meta.json", "w"))

def top(S, k):
    v = np.sort(S[S > 0])[::-1]
    if len(v) == 0: return set()
    t = v[min(k, len(v)) - 1]
    return set(zip(*np.nonzero(S >= t)))
J = lambda a, b: len(a & b) / max(len(a | b), 1)
rows = []
for k in [50, 100, 200, 500, 1000]:
    rows.append(dict(top_k=k, guide1_vs_guide2=J(top(res["guide1"], k), top(res["guide2"], k)),
                     donorsA_vs_donorsB=J(top(res["donorsA"], k), top(res["donorsB"], k))))
df = pd.DataFrame(rows); df.to_csv(f"{OUT}/jaccard.csv", index=False); print(df.round(3).to_string(index=False))
