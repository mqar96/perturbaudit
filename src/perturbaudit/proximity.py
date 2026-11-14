import gzip, re
import numpy as np
import pandas as pd


def load_tss_gtf(path, key="gene_name"):
    tss = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.split("\t", 9)
            if c[2] != "gene":
                continue
            m = re.search(rf'{key} "([^".]+)', c[8])
            if not m:
                continue
            tss.setdefault(m.group(1), (c[0], int(c[3]) if c[6] == "+" else int(c[4]), c[6]))
    return tss


def tss_distance(a, b, tss):
    if a not in tss or b not in tss or tss[a][0] != tss[b][0]:
        return np.inf
    return abs(tss[a][1] - tss[b][1])


def flag_edges(edges, tss, window=10_000, parent="parent", child="child"):
    d = np.array([tss_distance(a, b, tss) for a, b in zip(edges[parent], edges[child])], dtype=float)
    out = edges.copy()
    out["tss_distance"] = d
    out["neighbour_flag"] = d < window
    return out


def enrichment(edge_pairs, all_pairs, tss, bins=((0, 1_000), (1_000, 10_000), (10_000, 100_000), (100_000, 1_000_000))):
    de = np.array([tss_distance(a, b, tss) for a, b in edge_pairs], dtype=float)
    da = np.array([tss_distance(a, b, tss) for a, b in all_pairs], dtype=float)
    rows = []
    for lo, hi in bins:
        o = ((de >= lo) & (de < hi)).mean() if len(de) else np.nan
        e = ((da >= lo) & (da < hi)).mean() if len(da) else np.nan
        rows.append(dict(window_kb=f"{lo // 1000}-{hi // 1000}", observed=o, expected=e,
                         enrichment=o / e if e and e > 0 else np.nan, n_edges=int(((de >= lo) & (de < hi)).sum())))
    return pd.DataFrame(rows)
