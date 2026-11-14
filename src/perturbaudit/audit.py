import numpy as np
import pandas as pd
from . import heldout, proximity, offtarget


def edge_table(W, names, top_k=None, min_score=0.0):
    A = np.nan_to_num(np.abs(np.asarray(W, dtype=np.float64)))
    np.fill_diagonal(A, 0)
    nz = np.argwhere(A > min_score)
    sc = A[nz[:, 0], nz[:, 1]]
    order = np.argsort(-sc)
    if top_k:
        order = order[:top_k]
    i, j = nz[order, 0], nz[order, 1]
    return pd.DataFrame(dict(parent=np.asarray(names, dtype=object)[i], child=np.asarray(names, dtype=object)[j],
                             i=i, j=j, score=sc[order], rank=np.arange(1, len(order) + 1)))


def run_audit(W, names, Ztce=None, tce_q=None, support_q=0.05, support_z=3.0, tss=None, ot_map=None, ot_downstream=None,
              W_reps=None, W_guide_a=None, W_guide_b=None, top_k=None, stability_min=0.5):
    E = edge_table(W, names, top_k)
    if Ztce is not None:
        Z = np.abs(np.asarray(Ztce, dtype=np.float64))
        fwd = Z[E.i, E.j]; rev = Z[E.j, E.i]
        if tce_q is not None:
            Q = np.asarray(tce_q)
            supported = Q[E.i, E.j] <= support_q
            rev_sig = Q[E.j, E.i] <= support_q
        else:
            supported = fwd >= support_z
            rev_sig = rev >= support_z
        E["z_parent_to_child"] = fwd
        E["z_child_to_parent"] = rev
        E["unsupported"] = ~supported
        E["reversed"] = (~supported) & rev_sig
    if tss is not None:
        E = proximity.flag_edges(E, tss)
    if ot_map is not None:
        E = offtarget.flag_edges(E, ot_map, ot_downstream)
    if W_reps:
        k = len(E)
        hits = np.zeros(len(E))
        for Wr in W_reps:
            Er = edge_table(Wr, names, k)
            s = set(zip(Er.i, Er.j))
            hits += np.array([(a, b) in s for a, b in zip(E.i, E.j)])
        E["stability"] = hits / len(W_reps)
        E["unstable"] = E["stability"] < stability_min
    if W_guide_a is not None and W_guide_b is not None:
        k = len(E)
        sa = set(zip(*edge_table(W_guide_a, names, k)[["i", "j"]].values.T))
        sb = set(zip(*edge_table(W_guide_b, names, k)[["i", "j"]].values.T))
        E["in_guide_a"] = [(a, b) in sa for a, b in zip(E.i, E.j)]
        E["in_guide_b"] = [(a, b) in sb for a, b in zip(E.i, E.j)]
        E["guide_specific"] = E["in_guide_a"] != E["in_guide_b"]
    flags = [c for c in ["unsupported", "neighbour_flag", "offtarget_flag", "unstable", "guide_specific"] if c in E]
    E["n_flags"] = E[flags].sum(1) if flags else 0
    E["audited"] = E["n_flags"] == 0
    return E


def summary(E, ks=(25, 50, 100, 200)):
    rows = []
    for k in ks:
        if k > len(E):
            continue
        s = E.iloc[:k]
        r = dict(top_k=k, audited_fraction=float(s["audited"].mean()))
        for c in ["unsupported", "reversed", "neighbour_flag", "offtarget_flag", "unstable", "guide_specific"]:
            if c in s:
                r[c] = int(s[c].sum())
        rows.append(r)
    return pd.DataFrame(rows)
