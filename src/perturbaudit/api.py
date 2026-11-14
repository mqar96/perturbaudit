"""One-call audit of an inferred network against a Perturb-seq screen's internal controls."""
import numpy as np
import pandas as pd
from . import decoy, proximity, offtarget, guidesplit

VERDICT_ORDER = ["untestable_parent_not_perturbed", "untestable_child_not_measured", "untestable_ineffective_knockdown",
                 "reversed", "unsupported", "offtarget", "neighbour", "guide_specific", "unstable", "audited"]


def network_to_edges(network, names=None, top_k=None):
    """Accept a DataFrame with parent/child[/score] columns or a square score matrix with `names`."""
    if isinstance(network, pd.DataFrame) and {"parent", "child"} <= set(network.columns):
        E = network.copy()
        if "score" not in E:
            E["score"] = np.arange(len(E), 0, -1, dtype=float)
    else:
        A = np.nan_to_num(np.abs(np.asarray(network, dtype=np.float64)))
        np.fill_diagonal(A, 0)
        i, j = np.nonzero(A)
        E = pd.DataFrame(dict(parent=np.asarray(names, dtype=object)[i], child=np.asarray(names, dtype=object)[j], score=A[i, j]))
    E = E[E.parent.astype(str) != E.child.astype(str)]
    E = E.sort_values("score", ascending=False).reset_index(drop=True)
    if top_k:
        E = E.iloc[:top_k].copy()
    E["rank"] = np.arange(1, len(E) + 1)
    return E


def audit_network(screen, network, names=None, top_k=None, support_z=3.0, support_q=0.05, reverse_z=5.0,
                  kd_min_abs_z=2.0, n_decoys=200, seed=0, tss=None, neighbour_window=10_000, offtarget_map=None,
                  replicate_networks=None, stability_min=0.5, guide_disc_z=3.0, guide_het_q=0.05):
    """Return (edges, summary). Each edge gets the evidence columns and one `verdict` (first failing check in
    VERDICT_ORDER). Support = |knockdown z| >= support_z OR decoy-calibrated q <= support_q (both directions use
    the same batch-matched non-targeting reference)."""
    E = network_to_edges(network, names, top_k)
    Zr, Zd = screen.decoys(n_decoys=n_decoys, seed=seed)  # half-reference z: used only for decoy-calibrated q
    Zfull = screen.effects()  # full-reference z: used for the |z| support/reversal thresholds (more power)
    kd = screen.knockdown()
    meas = set(screen.genes)
    perturbed = set(Zr.index)
    q = decoy.qvalues(np.abs(Zr.values).ravel(), np.abs(Zd.values).ravel()).reshape(Zr.shape)
    Q = pd.DataFrame(q, index=Zr.index, columns=Zr.columns)
    gi = screen.gene_index
    z_fwd, z_rev, q_fwd, q_rev, kd_p = [], [], [], [], []
    for a, b in zip(E.parent.astype(str), E.child.astype(str)):
        z_fwd.append(Zfull.at[a, b] if a in perturbed and b in meas else np.nan)
        q_fwd.append(Q.at[a, b] if a in perturbed and b in meas else np.nan)
        z_rev.append(Zfull.at[b, a] if b in perturbed and a in meas else np.nan)
        q_rev.append(Q.at[b, a] if b in perturbed and a in meas else np.nan)
        kd_p.append(kd.get(a, np.nan))
    E["z_parent_kd_on_child"] = z_fwd; E["q_parent_kd_on_child"] = q_fwd
    E["z_child_kd_on_parent"] = z_rev; E["q_child_kd_on_parent"] = q_rev
    E["parent_knockdown_z"] = kd_p
    E["parent_perturbed"] = E.parent.astype(str).isin(perturbed)
    E["child_measured"] = E.child.astype(str).isin(meas)
    E["kd_effective"] = ~(np.abs(E.parent_knockdown_z) < kd_min_abs_z)  # direction-agnostic (CRISPRi/a/KO); NaN = parent not measured, treated as testable
    sup = (np.abs(E.z_parent_kd_on_child) >= support_z) | (E.q_parent_kd_on_child <= support_q)
    E["supported"] = sup.fillna(False)
    rev = (np.abs(E.z_child_kd_on_parent) >= reverse_z) & (E.q_child_kd_on_parent <= support_q)
    E["reverse_evidence"] = rev.fillna(False)
    if tss is not None:
        E = proximity.flag_edges(E, tss, neighbour_window)
    if offtarget_map is not None:
        # downstream: children moved (|z| >= support_z) by knocking down the off-target gene itself, when it is in the screen
        ots = {o for v in offtarget_map.values() for o in v}
        downstream = {o: set(Zfull.columns[np.abs(Zfull.loc[o].values) >= support_z]) for o in ots if o in Zfull.index}
        E = offtarget.flag_edges(E, offtarget_map, downstream)
    if "guide" in screen.groups:
        Zg = screen.effects("guide")
        ng = pd.Series(screen.groups["guide"]["n"], index=screen.groups["guide"]["names"])
        by_t = {}
        for gname in Zg.index:
            by_t.setdefault(screen.guide_target.get(gname), []).append(gname)
        spec, ztxt = [], []
        for a, b in zip(E.parent.astype(str), E.child.astype(str)):
            gs = sorted(by_t.get(a, []), key=lambda g: -ng.get(g, 0))[:2]
            if len(gs) < 2 or b not in meas:
                spec.append(np.nan); ztxt.append(""); continue
            z1, z2 = Zg.at[gs[0], b], Zg.at[gs[1], b]
            _, p = guidesplit.heterogeneity(np.array([z1]), np.array([z2]), np.array([ng[gs[0]]]), np.array([ng[gs[1]]]))
            p = float(np.ravel(p)[0])
            spec.append(bool(max(abs(z1), abs(z2)) >= guide_disc_z and min(abs(z1), abs(z2)) < 1.0 and p < guide_het_q))
            ztxt.append(f"{z1:.1f}|{z2:.1f}")
        E["guide_z"] = ztxt
        E["guide_specific"] = spec
    if replicate_networks:
        hits = np.zeros(len(E))
        for R in replicate_networks:
            Er = network_to_edges(R, names, len(E))
            s = set(zip(Er.parent.astype(str), Er.child.astype(str)))
            hits += np.array([(a, b) in s for a, b in zip(E.parent.astype(str), E.child.astype(str))])
        E["stability"] = hits / len(replicate_networks)
    E["verdict"] = pd.Series([_verdict(r, stability_min) for r in E.to_dict("records")], index=E.index, dtype=object)
    return E, summarize(E)


def _verdict(r, stability_min):
    if not r["parent_perturbed"]:
        return "untestable_parent_not_perturbed"
    if not r["child_measured"]:
        return "untestable_child_not_measured"
    if not r["supported"]:
        if not r["kd_effective"]:
            return "untestable_ineffective_knockdown"
        return "reversed" if r["reverse_evidence"] else "unsupported"
    if r.get("offtarget_flag"):
        return "offtarget"
    if r.get("neighbour_flag"):
        return "neighbour"
    if r.get("guide_specific") is True:
        return "guide_specific"
    if "stability" in r and r["stability"] == r["stability"] and r["stability"] < stability_min:
        return "unstable"
    return "audited"


def summarize(E, ks=(25, 50, 100, 200, 500, 1000)):
    rows = []
    for k in [k for k in ks if k <= len(E)] + [len(E)]:
        s = E.iloc[:k]
        testable = s[~s.verdict.astype(str).str.startswith("untestable")]
        r = dict(top_k=k, n_testable=len(testable),
                 audited_fraction_of_testable=round((testable.verdict == "audited").mean(), 3) if len(testable) else np.nan)
        for v in VERDICT_ORDER:
            r[v] = int((s.verdict == v).sum())
        rows.append(r)
    return pd.DataFrame(rows).drop_duplicates("top_k")
