"""C9: do edges that pass the audit in K562 replicate in RPE1 more often than flagged edges?
RPE1 evidence for edge s->t: z of target t in RPE1 cells with s knocked down (best element), vs pooled NT cells.
Replicated = |z_RPE1| >= 3 and sign equal to the K562 knockdown effect sign.
Usage: run_transfer_k562_rpe1.py OUTDIR"""
import sys, os, glob, json
import numpy as np, pandas as pd
from scipy.stats import fisher_exact

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
R = "derived/rpe1_essential"
z = np.load(f"{R}/element_stats.npz", allow_pickle=True); nm = np.load(f"{R}/names.npz", allow_pickle=True)
S1, S2, N = z["S1"].astype(np.float64), z["S2"].astype(np.float64), z["N"].astype(np.float64)
eg = np.array([str(x) for x in nm["element_gene"]]); vg = np.array([str(x) for x in nm["var_gene_name"]])
is_nt = np.array(["non-targeting" in g or g.startswith("non") for g in eg])
n0 = N[is_nt].sum(); mu0 = S1[is_nt].sum(0) / n0; var0 = S2[is_nt].sum(0) / n0 - mu0 ** 2
best = {}
for e in np.flatnonzero(~is_nt):
    if N[e] >= 20 and (eg[e] not in best or N[e] > N[best[eg[e]]]): best[eg[e]] = e
col = {g: i for i, g in enumerate(vg)}

def z_rpe1(s, t):
    if s not in best or t not in col: return np.nan
    e, j = best[s], col[t]
    if var0[j] <= 0: return np.nan
    return (S1[e, j] / N[e] - mu0[j]) / np.sqrt(var0[j] * (1 / N[e] + 1 / n0))

def signed_tce(panel_dir):
    p = np.load(f"{panel_dir}/panel.npz", allow_pickle=True)
    X, lab, names = p["X"].astype(np.float64), p["labels"].astype(str), [str(x) for x in p["names"]]
    obs = X[lab == "non-targeting"]; mu, var, k0 = obs.mean(0), obs.var(0, ddof=1), len(obs)
    Z = np.zeros((len(names), len(names)))
    for i, g in enumerate(names):
        m = lab == g
        if m.sum() >= 5: Z[i] = (X[m].mean(0) - mu) / np.sqrt(var * (1 / m.sum() + 1 / k0))
    np.fill_diagonal(Z, 0)
    return Z, names, json.load(open(f"{panel_dir}/meta.json"))

def edges(panel_dir, method, k, freq=None):
    Z, names, meta = signed_tce(panel_dir)
    real = [i for i, n in enumerate(names) if not n.startswith("DECOY")]
    W = np.nan_to_num(np.abs(np.load(f"{panel_dir}/W_{method}.npy")))[np.ix_(real, real)]
    np.fill_diagonal(W, 0)
    Zr = Z[np.ix_(real, real)]; rn = [names[i] for i in real]
    near = {frozenset((a, b)) for a, b, _ in meta["neighbour_pairs"]}
    idx = [i for i in np.argsort(W.ravel())[::-1][:k] if W.ravel()[i] > 0]
    rows = []
    for f in idx:
        a, b = divmod(f, len(rn)); s, t = rn[a], rn[b]
        zk, zr = Zr[a, b], z_rpe1(s, t)
        uns = abs(zk) < 2; rev = uns and abs(Zr[b, a]) >= 5
        nb = frozenset((s, t)) in near
        fr = freq.get((s, t), 0) if freq is not None else np.nan
        rows.append(dict(method=method, source=s, target=t, rank=len(rows) + 1, z_k562=zk, z_rpe1=zr,
                         unsupported=uns, reversed=rev, neighbour=nb, stability=fr,
                         testable=not np.isnan(zr),
                         replicated=bool(abs(zr) >= 3 and np.sign(zr) == np.sign(zk)) if not np.isnan(zr) else np.nan,
                         replicated_any_sign=bool(abs(zr) >= 3) if not np.isnan(zr) else np.nan))
    return pd.DataFrame(rows)

def compare(df, flag):
    d = df[df.testable]
    a = d[~d[flag]]; b = d[d[flag]]
    ra, rb = a.replicated.astype(float), b.replicated.astype(float)
    if len(a) and len(b):
        tab = [[int(ra.sum()), int(len(ra) - ra.sum())], [int(rb.sum()), int(len(rb) - rb.sum())]]
        orr, pv = fisher_exact(tab)
    else:
        orr, pv = np.nan, np.nan
    return dict(flag=flag, n_pass=len(a), rep_pass=round(ra.mean(), 3) if len(a) else np.nan,
                n_flag=len(b), rep_flag=round(rb.mean(), 3) if len(b) else np.nan, odds_ratio=round(orr, 2) if orr == orr else orr, p=pv)

res, all_edges = [], []
# A. p=300 panel, top-100 and top-200 per method
for k in [100, 200]:
    for m in ["tce", "gbm_grn", "inspre", "dotears"]:
        df = edges("derived/panels/k562_p300e", m, k); df["panel"] = "k562_p300e"; df["top_k"] = k
        all_edges.append(df)
        d = df[df.testable]
        res.append(dict(panel="k562_p300e", method=m, top_k=k, n=len(df), n_testable=len(d),
                        replication=round(d.replicated.astype(float).mean(), 3), **{f"{c['flag']}_{x}": c[x] for c in [compare(df, "unsupported")] for x in ["rep_pass", "n_flag", "rep_flag", "p"]}))
# B. p=100 multi-draw: stability as an extra flag (draw d11 edges, stability from 5 draws)
fq = pd.read_csv("derived/multidraw_k562_p100e/multidraw_edge_freq_top100.csv")
for m in ["gbm_grn", "inspre", "dotears"]:
    freq = {(r.source, r.target): r.freq for r in fq[fq.method == m].itertuples()}
    df = edges("derived/panels/k562_p100e_d11", m, 100, freq); df["panel"] = "k562_p100e_d11"; df["top_k"] = 100
    df["unstable"] = df.stability < 0.6
    df["any_flag"] = df.unsupported | df.unstable | df.neighbour
    all_edges.append(df)
    d = df[df.testable]
    row = dict(panel="k562_p100e_d11", method=m, top_k=100, n=len(df), n_testable=len(d), replication=round(d.replicated.astype(float).mean(), 3))
    for fl in ["unsupported", "unstable", "any_flag"]:
        c = compare(df, fl); row.update({f"{fl}_{x}": c[x] for x in ["rep_pass", "n_flag", "rep_flag", "p"]})
    sup = df[~df.unsupported]; c = compare(sup, "unstable")
    row.update({f"unstable_among_supported_{x}": c[x] for x in ["rep_pass", "n_flag", "rep_flag", "p"]})
    res.append(row)
# baseline: random real-real pairs on p300 panel
Z, names, meta = signed_tce("derived/panels/k562_p300e")
rn = [n for n in names if not n.startswith("DECOY")]
rng = np.random.default_rng(0); hits, tot = 0, 0
for _ in range(5000):
    a, b = rng.choice(len(rn), 2, replace=False); zr = z_rpe1(rn[a], rn[b])
    if not np.isnan(zr): tot += 1; hits += abs(zr) >= 3
base = hits / max(tot, 1)
R_ = pd.DataFrame(res); R_["random_pair_rep_any_sign"] = round(base, 3)
R_.to_csv(f"{OUT}/transfer_summary.csv", index=False)
pd.concat(all_edges).to_csv(f"{OUT}/transfer_edges.csv", index=False)
pd.set_option("display.width", 250)
print(R_.T.to_string())
