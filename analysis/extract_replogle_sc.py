import argparse, json, os, time
import h5py
import numpy as np

def decode_cats(f, key):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in f[f"obs/__categories/{key}"][:]], dtype=object)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5ad")
    ap.add_argument("out")
    ap.add_argument("--cells-per-element", type=int, default=150)
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prenormalized", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    f = h5py.File(a.h5ad, "r")
    X = f["X"]
    n, p = X.shape
    el_codes = f["obs/gene_transcript"][:].astype(np.int64)
    el_names = decode_cats(f, "gene_transcript")
    gene_codes = f["obs/gene"][:].astype(np.int64)
    gene_names = decode_cats(f, "gene")
    gem = f["obs/gem_group"][:].astype(np.int64)
    umi = f["obs/UMI_count"][:].astype(np.float64)
    var_gene_id = np.array([x.decode() if isinstance(x, bytes) else x for x in f["var/gene_id"][:]], dtype=object)
    vn_cats = np.array([x.decode() if isinstance(x, bytes) else x for x in f["var/__categories/gene_name"][:]], dtype=object)
    var_gene_name = vn_cats[f["var/gene_name"][:].astype(np.int64)]
    n_el = len(el_names)
    is_nt = np.array(["non-targeting" in str(s) for s in el_names])[el_codes]
    el_gene = np.empty(n_el, dtype=object)
    for e in range(n_el):
        idx = np.flatnonzero(el_codes == e)
        el_gene[e] = gene_names[gene_codes[idx[0]]] if len(idx) else ""
    pert_genes = set(g for g in gene_names if g and "non-targeting" not in g)
    panel_cols = np.array([j for j in range(p) if var_gene_name[j] in pert_genes])
    rng = np.random.default_rng(a.seed)
    keep_cell = np.zeros(n, dtype=bool)
    for e in range(n_el):
        idx = np.flatnonzero(el_codes == e)
        if len(idx) == 0:
            continue
        if "non-targeting" in str(el_names[e]):
            keep_cell[idx] = True
        else:
            keep_cell[rng.choice(idx, min(len(idx), a.cells_per_element), replace=False)] = True
    nt_idx = np.flatnonzero(is_nt)
    sub_idx = np.flatnonzero(keep_cell)
    S1 = np.zeros((n_el, p)); S2 = np.zeros((n_el, p)); N = np.zeros(n_el, dtype=np.int64)
    gem_levels = np.unique(gem)
    NG = np.zeros((n_el, len(gem_levels)), dtype=np.int64)
    gem_pos = {g: i for i, g in enumerate(gem_levels)}
    nt_out = np.lib.format.open_memmap(os.path.join(a.out, "nt_cells_lognorm.npy"), mode="w+", dtype=np.float32, shape=(len(nt_idx), p))
    sub_out = np.lib.format.open_memmap(os.path.join(a.out, "subsample_panel_lognorm.npy"), mode="w+", dtype=np.float16, shape=(len(sub_idx), len(panel_cols)))
    nt_ptr = 0; sub_ptr = 0
    for s in range(0, n, a.chunk):
        e_ = min(s + a.chunk, n)
        C = X[s:e_].astype(np.float64)
        L = C if a.prenormalized else np.log1p(C / umi[s:e_, None] * 1e4)
        codes = el_codes[s:e_]
        np.add.at(N, codes, 1)
        order = np.argsort(codes, kind="stable")
        sc = codes[order]
        bounds = np.flatnonzero(np.diff(sc)) + 1
        starts = np.r_[0, bounds]; ends = np.r_[bounds, len(sc)]
        for st, en in zip(starts, ends):
            rows = order[st:en]
            S1[sc[st]] += L[rows].sum(0)
            S2[sc[st]] += (L[rows] ** 2).sum(0)
        for g, c in zip(gem[s:e_], codes):
            NG[c, gem_pos[g]] += 1
        m = is_nt[s:e_]
        k = m.sum()
        if k:
            nt_out[nt_ptr:nt_ptr + k] = L[m].astype(np.float32); nt_ptr += k
        m2 = keep_cell[s:e_]
        k2 = m2.sum()
        if k2:
            sub_out[sub_ptr:sub_ptr + k2] = L[m2][:, panel_cols].astype(np.float16); sub_ptr += k2
        if (s // a.chunk) % 10 == 0:
            print(f"{e_}/{n} rows  {time.time()-t0:.0f}s", flush=True)
    nt_out.flush(); sub_out.flush()
    np.savez_compressed(os.path.join(a.out, "element_stats.npz"), S1=S1, S2=S2, N=N, NG=NG, gem_levels=gem_levels)
    np.savez_compressed(os.path.join(a.out, "cell_meta.npz"),
                        nt_idx=nt_idx, nt_el=el_codes[nt_idx], nt_gem=gem[nt_idx], nt_umi=umi[nt_idx],
                        sub_idx=sub_idx, sub_el=el_codes[sub_idx], sub_gem=gem[sub_idx], sub_umi=umi[sub_idx])
    meta = dict(source=os.path.basename(a.h5ad), n_cells=int(n), n_genes=int(p), n_elements=int(n_el),
                n_nt_cells=int(len(nt_idx)), n_subsample_cells=int(len(sub_idx)), n_panel_genes=int(len(panel_cols)),
                normalization="as provided (Replogle z-normalized)" if a.prenormalized else "log1p(counts / UMI_count * 1e4)", cells_per_element=a.cells_per_element, seed=a.seed,
                seconds=round(time.time() - t0))
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    np.savez(os.path.join(a.out, "names.npz"), element=el_names, element_gene=el_gene, var_gene_id=var_gene_id,
             var_gene_name=var_gene_name, panel_cols=panel_cols)
    print(json.dumps(meta))

if __name__ == "__main__":
    main()
