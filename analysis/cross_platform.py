"""How well do K562 GWPS knockdown effects carry over to other lines and platforms, and what predicts it?
Strong K562 pairs (|z| >= 5, parent != child) in the GWPS 788-gene panel; replication = same sign and |z| >= 3 in the target.
Stratified by K562 effect size, by the parent's own knock-down strength in the target line, and by child expression."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "perturbaudit/src")
from perturbaudit.screen import Screen
z = np.load("derived/panels/gwps_p788/panel.npz", allow_pickle=True)
X, lab, names = z["X"].astype(np.float64), z["labels"].astype(str), [str(n) for n in z["names"]]
real = [i for i, n in enumerate(names) if not n.startswith("DECOY")]; keep = ~np.char.startswith(lab, "DECOY")
rn = [names[i] for i in real]; E = Screen(X[keep][:, real], lab[keep], rn, "non-targeting").effects()
own_k = {p: E.at[p, p] for p in E.index if p in E.columns}
rows = []
for path, name in [("data/nadig/rpe1_effects.npz", "RPE1"), ("data/nadig/nadig_hepg2_effects.npz", "HepG2"), ("data/nadig/nadig_jurkat_effects.npz", "Jurkat"), ("data/orion/orion_hct116_compact.npz", "HCT116")]:
    T = np.load(path, allow_pickle=True); Z = T["Z"].astype(np.float32); P = [str(p) for p in T["perts"]]; G = [str(g) for g in T["genes"]]
    pi = {p: i for i, p in enumerate(P)}; gi = {}
    for i, g in enumerate(G): gi.setdefault(g, i)
    mu = T["mu"] if "mu" in T.files else None
    for p in E.index:
        if p not in pi: continue
        ownT = float(Z[pi[p], gi[p]]) if p in gi else np.nan
        for g in E.columns:
            if g == p or g not in gi: continue
            v = E.at[p, g]
            if abs(v) < 5: continue
            b = float(Z[pi[p], gi[g]])
            if np.isnan(b): continue
            rows.append((name, p, g, v, b, ownT, own_k.get(p, np.nan), float(mu[gi[g]]) if mu is not None else np.nan))
D = pd.DataFrame(rows, columns=["line", "parent", "child", "zK562", "zT", "own_target", "own_K562", "child_mu"])
D["rep"] = (np.sign(D.zK562) == np.sign(D.zT)) & (D.zT.abs() >= 3); D["sign"] = np.sign(D.zK562) == np.sign(D.zT)
D["effect_bin"] = pd.cut(D.zK562.abs(), [5, 10, 20, 1e9], labels=["5-10", "10-20", ">20"])
D["kd_bin"] = pd.cut(D.own_target, [-1e9, -10, -5, -3, 1e9], labels=["<= -10", "-10..-5", "-5..-3", "> -3 (weak)"])
D.to_csv("derived/cross_platform_pairs.csv.gz", index=False)
o = []
o.append(D.groupby("line")[["rep", "sign"]].mean().assign(n=D.groupby("line").size()))
o.append(D.groupby(["line", "effect_bin"], observed=True).rep.mean().unstack().round(3))
o.append(D.groupby(["line", "kd_bin"], observed=True).rep.agg(["mean", "size"]).unstack().round(3))
H = D[D.line == "HCT116"].copy(); H["mu_bin"] = pd.qcut(H.child_mu, 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
o.append(H.groupby(["kd_bin", "mu_bin"], observed=True).rep.mean().unstack().round(3))
# best-case subset: strong K562 effect, strong KD in target
o.append(D[(D.zK562.abs() > 20) & (D.own_target <= -10)].groupby("line").rep.agg(["mean", "size"]).round(3))
for t in o: print(t.to_string(), "\n")
