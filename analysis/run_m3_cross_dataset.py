import sys, gzip, re, json, numpy as np, pandas as pd, anndata as ad

from perturbaudit import decoy

tss_id, tss_nm = {}, {}
with gzip.open("data/annot/gencode.v43.basic.gtf.gz", "rt") as f:
    for line in f:
        if line.startswith("#"): continue
        c = line.split("\t", 9)
        if c[2] != "gene": continue
        gid = re.search(r'gene_id "([^".]+)', c[8]).group(1); gnm = re.search(r'gene_name "([^"]+)', c[8]).group(1)
        v = (c[0], int(c[3]) if c[6] == "+" else int(c[4]))
        tss_id[gid] = v; tss_nm.setdefault(gnm, v)

BINS = [(0, 1_000), (1_000, 10_000), (10_000, 100_000), (100_000, 1_000_000)]

def analyse(name, Z, pert_keys, read_keys, lookup, thr=5.0, modality="CRISPRi"):
    pk = np.array([lookup.get(k, (None, None)) for k in pert_keys], dtype=object)
    rk = np.array([lookup.get(k, (None, None)) for k in read_keys], dtype=object)
    okp = np.array([x[0] is not None for x in pk]); okr = np.array([x[0] is not None for x in rk])
    Z = np.clip(np.asarray(Z, dtype=np.float64), -1e4, 1e4).astype(np.float32)[okp][:, okr]; pk = pk[okp]; rk = rk[okr]
    pkeys = np.asarray(pert_keys, dtype=object)[okp]; rkeys = np.asarray(read_keys, dtype=object)[okr]
    pc = np.array([x[0] for x in pk]); pp = np.array([x[1] for x in pk], dtype=np.int64)
    rc = np.array([x[0] for x in rk]); rp = np.array([x[1] for x in rk], dtype=np.int64)
    rows = []
    sig_counts = {b: 0 for b in BINS}; all_counts = {b: 0 for b in BINS}; n_sig = 0; n_all = 0
    for i in range(Z.shape[0]):
        same = rc == pc[i]
        d = np.where(same, np.abs(rp - pp[i]), 10**12)
        selfm = (rkeys == pkeys[i]) | (d == 0)
        sig = (np.abs(Z[i]) >= thr) & ~selfm
        valid = ~selfm
        n_sig += sig.sum(); n_all += valid.sum()
        for b in BINS:
            inb = (d >= b[0]) & (d < b[1])
            sig_counts[b] += (sig & inb).sum(); all_counts[b] += (valid & inb).sum()
    for b in BINS:
        o = sig_counts[b] / max(n_sig, 1); e = all_counts[b] / max(n_all, 1)
        rows.append(dict(dataset=name, modality=modality, window=f"{b[0]//1000}-{b[1]//1000} kb", n_perturbations=int(Z.shape[0]),
                         sig_edges=int(n_sig), sig_in_window=int(sig_counts[b]), observed=o, expected=e, enrichment=o / e if e > 0 else np.nan))
    return rows

out = []
D = "derived/k562_essential"
st = np.load(f"{D}/element_stats.npz"); nm = np.load(f"{D}/names.npz", allow_pickle=True); cm = np.load(f"{D}/cell_meta.npz")
nt = np.load(f"{D}/nt_cells_lognorm.npy", mmap_mode="r")
is_nt = np.array(["non-targeting" in s for s in nm["element"]]); real = np.flatnonzero(~is_nt & (st["N"] >= 20))
Xr = np.asarray(nt, dtype=np.float64); mu, var = Xr.mean(0), Xr.var(0, ddof=1); del Xr
Z = decoy.null_var_z(st["N"][real], st["S1"][real], mu, var, int(is_nt.sum() and len(cm["nt_idx"])))
Z[:, mu < np.log1p(0.1)] = 0
out += analyse("Replogle K562 essential", Z, list(nm["element_gene"][real]), list(nm["var_gene_name"]), tss_nm)
del Z
for nmx, f in [("Replogle K562 genome-wide", "K562_gwps_normalized_bulk_01.h5ad"), ("Replogle RPE1 essential", "rpe1_normalized_bulk_01.h5ad")]:
    a = ad.read_h5ad(f"data/replogle/{f}")
    n = a.obs["num_cells_filtered"].values; ok = np.nan_to_num(n) >= 25
    Z = np.nan_to_num(np.asarray(a.X)[ok] * np.sqrt(n[ok])[:, None])
    pg = [s.split("_")[-1] for s in a.obs_names[ok]]
    out += analyse(nmx, Z, pg, list(a.var_names), tss_id)
    del Z, a
DG = "derived/cd4_by_guide"
o1 = pd.read_parquet(f"{DG}/guide_1_obs.parquet"); var = pd.read_parquet(f"{DG}/var.parquet")
Z1 = np.load(f"{DG}/guide_1_zscore.npy", mmap_mode="r")
sel = np.flatnonzero((o1.culture_condition.values == "Stim8hr") & o1.ontarget_significant.values.astype(bool))
out += analyse("Zhu/Marson CD4 T cells (Stim8hr, guide 1)", np.asarray(Z1[sel]), list(o1.target_contrast.values[sel]), list(var.gene_id.values), tss_id)
p = ad.read_h5ad("data/scperturb/PapalexiSatija2021_eccite_RNA.h5ad")
import scipy.sparse as sp
Xs = sp.csr_matrix(p.X).astype(np.float32)
lib = np.asarray(Xs.sum(1)).ravel()
L = sp.diags(1e4 / lib) @ Xs
L.data = np.log1p(L.data)
L = L.tocsr()
gene = np.asarray(p.obs["perturbation"].astype(str).str.replace(r"g\d+$", "", regex=True).values, dtype=str)
ctrl = (gene == "control")
def mom(m):
    A = L[m]; mu_ = np.asarray(A.mean(0)).ravel(); m2 = np.asarray(A.multiply(A).mean(0)).ravel()
    return mu_, (m2 - mu_ ** 2) * m.sum() / max(m.sum() - 1, 1)
mu, v = mom(ctrl); n0 = ctrl.sum()
tg = [g for g in np.unique(gene[~ctrl]) if g in tss_nm]
Zp = np.stack([(mom(gene == g)[0] - mu) / np.sqrt(v * (1 / (gene == g).sum() + 1 / n0) + 1e-12) for g in tg])
Zp[:, mu < np.log1p(0.1)] = 0
out += analyse("Papalexi THP-1 (Cas9 KO)", Zp, tg, list(p.var_names), tss_nm, modality="CRISPR KO")
df = pd.DataFrame(out)
df.to_csv("derived/m3_cross_dataset.csv", index=False)
pd.set_option("display.width", 200)
print(df.pivot_table(index=["dataset", "modality", "n_perturbations", "sig_edges"], columns="window", values="enrichment").round(1).to_string())
