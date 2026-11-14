"""Transfer test: edges audited in a source screen (api_edges_<method>.csv) checked in a target cell line's
knockdown-effect matrix (npz with Z, perts, genes). Replicated = |z_target| >= 3, same sign as the source effect.
Usage: run_transfer_generic.py TARGET_NPZ TARGET_NAME OUT_CSV PANEL_DIR:METHOD:TOPK [...]"""
import sys, numpy as np, pandas as pd
from scipy.stats import fisher_exact
T = np.load(sys.argv[1], allow_pickle=True); tname, out = sys.argv[2], sys.argv[3]
Z = T["Z"].astype(np.float32); perts = [str(x) for x in T["perts"]]; genes = [str(x) for x in T["genes"]]
pi = {p: i for i, p in enumerate(perts)}; gi = {g: i for i, g in enumerate(genes)}
def zt(s, t): return float(Z[pi[s], gi[t]]) if s in pi and t in gi else np.nan
rows = []
rng = np.random.default_rng(0)
for spec in sys.argv[4:]:
    D, M, K = spec.split(":"); K = int(K)
    E = pd.read_csv(f"{D}/api_edges_{M}.csv").head(K)
    E["z_target"] = [zt(a, b) for a, b in zip(E.parent, E.child)]
    E["z_target_rev"] = [zt(b, a) for a, b in zip(E.parent, E.child)]
    E = E[~E.z_target.isna()]
    E["rep"] = (np.abs(E.z_target) >= 3) & (np.sign(E.z_target) == np.sign(E.z_parent_kd_on_child))
    E["rev_rep"] = np.abs(E.z_target_rev) >= 3
    genes_p = sorted(set(E.parent) | set(E.child))
    hits = tot = 0
    for _ in range(4000):
        a, b = rng.choice(genes_p, 2, replace=False); v = zt(a, b)
        if not np.isnan(v): tot += 1; hits += abs(v) >= 3
    for v, g in E.groupby("verdict"):
        rows.append(dict(target=tname, panel=D.rstrip("/").split("/")[-1], method=M, top_k=K, verdict=v, n=len(g),
                         replicated=round(g.rep.mean(), 3), child_kd_moves_parent=round(g.rev_rep.mean(), 3),
                         random_pair_any_sign=round(hits / max(tot, 1), 3)))
    a = E[E.verdict == "audited"]; u = E[E.verdict.isin(["unsupported", "reversed"])]
    if len(a) and len(u):
        _, p = fisher_exact([[a.rep.sum(), (~a.rep).sum()], [u.rep.sum(), (~u.rep).sum()]])
        rows.append(dict(target=tname, panel=D.rstrip("/").split("/")[-1], method=M, top_k=K, verdict="audited_vs_flagged_p", n=len(a) + len(u), replicated=p))
R = pd.DataFrame(rows); R.to_csv(out, index=False); print(R.to_string(index=False))
