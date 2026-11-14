import gzip, re, json, sys, os, numpy as np, pandas as pd

D = os.environ.get("PANEL_SRC", "derived/k562_essential")
OUT = sys.argv[1] if len(sys.argv) > 1 else "derived/panels/k562_p100"
P_TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 100
K_DECOY = int(sys.argv[3]) if len(sys.argv) > 3 else 20
import os; os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, 'perturbaudit/src')
from perturbaudit import sim
K_KNOCK = int(sys.argv[4]) if len(sys.argv) > 4 else 0
rng = np.random.default_rng(int(os.environ.get('PANEL_SEED', '7')))
CELL_FRAC = float(os.environ.get('CELL_FRAC', '1.0'))

nm = np.load(f"{D}/names.npz", allow_pickle=True); st = np.load(f"{D}/element_stats.npz"); cm = np.load(f"{D}/cell_meta.npz")
el, el_gene, vname, panel_cols = nm["element"], nm["element_gene"], nm["var_gene_name"], nm["panel_cols"]
sub = np.load(f"{D}/subsample_panel_lognorm.npy", mmap_mode="r")
panel_names = vname[panel_cols]
S1, N = st["S1"], st["N"]
is_nt = np.array(["non-targeting" in s for s in el])
mu_nt = S1[is_nt].sum(0) / N[is_nt].sum()
gene_el = {}
for e in np.flatnonzero(~is_nt & (N >= 50)):
    g = el_gene[e]
    if g in set(panel_names):
        gene_el.setdefault(g, []).append(e)
kd = {}
for g, es in gene_el.items():
    j = np.flatnonzero(vname == g)[0]
    best = max(es, key=lambda e: N[e])
    kd[g] = (S1[best, j] / N[best] - mu_nt[j], best)
EXPR_FROM = os.environ.get('EXPR_FROM')
if EXPR_FROM:
    import anndata as _ad
    _b = _ad.read_h5ad(EXPR_FROM, backed='r'); _cm = dict(zip(_b.var['gene_name'].astype(str), _b.var['mean'].astype(float)))
    expr = {g: _cm.get(g, 0.0) for g in gene_el}
else:
    expr = {g: mu_nt[np.flatnonzero(vname == g)[0]] for g in gene_el}
KD_THR = float(os.environ.get('KD_THR', '-0.3')); EXPR_THR = float(os.environ.get('EXPR_THR', '0.5'))
eligible = [g for g in gene_el if kd[g][0] < KD_THR and expr[g] > EXPR_THR]

lib = pd.read_csv("repos/perturb_seed/guide_libraries/sgrna_replogle.csv")
lib["guide_id"] = lib["sgID_A"] + "|" + lib["sgID_B"]
h = pd.read_csv("data/hartman/perturb_seed_reproduce/data/replogle_k562_2kb_filt_no_lfc_cutoff_off_target_candidates.csv.gz", low_memory=False)
hc = h[(h.seed_match_len >= 12) & (h.offtarget_log2fc < -0.5)]
ot_pairs = [] if os.environ.get('NO_HARTMAN') else [(r.gene, r.offtarget_gene, bool(r.seed_match_len == 20 and r.guide_hamming_dist == 0)) for r in hc.itertuples() if r.gene in eligible and r.offtarget_gene in eligible]

tss = {}
with gzip.open("data/annot/gencode.v43.basic.gtf.gz", "rt") as f:
    for line in f:
        if line.startswith("#"): continue
        c = line.split("\t", 9)
        if c[2] != "gene": continue
        nmx = re.search(r'gene_name "([^"]+)', c[8]).group(1)
        tss.setdefault(nmx, (c[0], int(c[3]) if c[6] == "+" else int(c[4])))
el_set = [g for g in eligible if g in tss]
near_pairs = []
for i, a in enumerate(el_set):
    for b in el_set[i + 1:]:
        if tss[a][0] == tss[b][0] and abs(tss[a][1] - tss[b][1]) < 10000:
            near_pairs.append((a, b, abs(tss[a][1] - tss[b][1])))

chosen = []
for a, b, _ in ot_pairs: chosen += [a, b]
for a, b, _ in near_pairs: chosen += [a, b]
chosen = list(dict.fromkeys(chosen))[: P_TARGET // 2]
rest = sorted([g for g in eligible if g not in chosen], key=lambda g: -expr[g])
genes = chosen + rest[: P_TARGET - len(chosen)]
p = len(genes)
cols = np.array([np.flatnonzero(panel_names == g)[0] for g in genes])

sub_el = cm["sub_el"]
X_parts, labels, umi_parts = [], [], []
sub_umi = cm["sub_umi"]
for g in genes:
    e = kd[g][1]
    idx = np.sort(np.flatnonzero(sub_el == e))
    if CELL_FRAC < 1: idx = np.sort(rng.choice(idx, max(5, int(len(idx) * CELL_FRAC)), replace=False))
    X_parts.append(np.asarray(sub[idx][:, cols], dtype=np.float32)); labels += [g] * len(idx); umi_parts.append(sub_umi[idx])
nt_idx = np.flatnonzero(is_nt[sub_el]); nt_idx = rng.permutation(nt_idx)
n_dec_cells = 150
dec_pool = nt_idx[: (K_DECOY + K_KNOCK) * n_dec_cells]; obs_idx = nt_idx[(K_DECOY + K_KNOCK) * n_dec_cells:]
if CELL_FRAC < 1: obs_idx = rng.choice(obs_idx, int(len(obs_idx) * CELL_FRAC), replace=False)
X_obs = np.asarray(sub[np.sort(obs_idx)][:, cols], dtype=np.float32)
X_dec = np.asarray(sub[np.sort(dec_pool)][:, cols], dtype=np.float32)
X_real = np.vstack(X_parts + [X_obs, X_dec])
umi = np.concatenate(umi_parts + [sub_umi[np.sort(obs_idx)], sub_umi[np.sort(dec_pool)]])
lab = labels + ["non-targeting"] * len(X_obs)
dec_lab = []
KT = K_DECOY + K_KNOCK
dec_names = [f"DECOYP{k:02d}" for k in range(K_DECOY)] + [f"DECOYK{k:02d}" for k in range(K_KNOCK)]
for k in range(KT): dec_lab += [dec_names[k]] * n_dec_cells
lab += dec_lab
n = len(X_real)
templates = rng.choice(p, KT, replace=True)
kd_vals = np.array([kd[g][0] for g in genes])
Xd = np.empty((n, KT), dtype=np.float32)
dec_start = n - len(X_dec)
obs_rows = np.arange(n - len(X_dec) - len(X_obs), n - len(X_dec))
Xk_all = sim.gaussian_knockoffs(X_real[obs_rows].astype(np.float64), X_real.astype(np.float64), rng) if K_KNOCK else None
lab_arr = np.array(lab[:n], dtype=object)
for k in range(KT):
    v = X_real[rng.permutation(n), templates[k]].copy() if k < K_DECOY else Xk_all[:, templates[k]].astype(np.float32).copy()
    rows = np.arange(dec_start + k * n_dec_cells, dec_start + (k + 1) * n_dec_cells)
    if os.environ.get('DECOY_KD', 'empirical') == 'shift':
        shift = rng.choice(kd_vals)
        v[rows] = np.maximum(v[rows] + shift, 0)
    else:  # empirical: resample the template gene's own knocked-down expression (matched marginal, incl. residual variance)
        own = X_real[lab_arr == genes[templates[k]], templates[k]]
        v[rows] = rng.choice(own, len(rows), replace=True)
    Xd[:, k] = v
X = np.hstack([X_real, Xd])
names = genes + dec_names
np.savez_compressed(f"{OUT}/panel.npz", X=X, labels=np.array(lab, dtype=object), names=np.array(names, dtype=object), umi=umi)
meta = dict(p_real=p, k_decoy=KT, k_perm=K_DECOY, k_knock=K_KNOCK, n_cells=int(n), n_obs=int(len(X_obs)), decoy_cells_each=n_dec_cells,
            offtarget_pairs=[(a, b, fm) for a, b, fm in ot_pairs if a in genes and b in genes],
            neighbour_pairs=[(a, b, d) for a, b, d in near_pairs if a in genes and b in genes],
            decoy_design="permuted template gene column, NT cells relabelled, knockdown: " + ("additive shift drawn from real panel knockdowns" if os.environ.get("DECOY_KD","empirical")=="shift" else "resampled from the template gene's own knocked-down cells"))
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in meta.items()}))
