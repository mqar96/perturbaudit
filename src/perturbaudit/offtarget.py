import numpy as np
import pandas as pd


def load_hartman(path, min_seed=12, max_lfc=-0.5):
    h = pd.read_csv(path, low_memory=False)
    hc = h[(h.seed_match_len >= min_seed) & (h.offtarget_log2fc < max_lfc)].copy()
    hc["full_match"] = (hc.seed_match_len == 20) & (hc.guide_hamming_dist == 0)
    return hc


def offtarget_map(hc, gene_col="gene", ot_col="offtarget_gene"):
    m = {}
    for g, o in zip(hc[gene_col], hc[ot_col]):
        m.setdefault(g, set()).add(o)
    return m


def flag_edges(edges, ot_map, downstream=None, parent="parent", child="child"):
    direct, via = [], []
    for a, b in zip(edges[parent], edges[child]):
        ots = ot_map.get(a, set())
        direct.append(b in ots)
        via.append(bool(downstream) and any(b in downstream.get(o, set()) for o in ots))
    out = edges.copy()
    out["offtarget_direct"] = direct
    out["offtarget_downstream"] = via
    out["offtarget_flag"] = out["offtarget_direct"] | out["offtarget_downstream"]
    return out


def profile_similarity(Z, row_parent, row_offtarget, exclude_cols=()):
    mask = np.ones(Z.shape[1], dtype=bool)
    mask[list(exclude_cols)] = False
    return float(np.corrcoef(Z[row_parent, mask], Z[row_offtarget, mask])[0, 1])
