"""Independent-construct replication in Replogle 2022 eLife (dual-sgRNA library vs Dolcetto single sgRNAs, K562 CRISPRi).
Strong effect = |z| >= 5 in the discovery set (target gene excluded); replicated = |z| >= 2 with the same sign.
Comparisons per target gene (effective constructs only, own-gene z <= -3):
  split-half   : two random halves of the same construct's cells (technical noise baseline, sizes halved)
  same library : two different constructs for the same gene (half-size matched by using construct halves)
  cross library: dual construct vs Dolcetto sgRNA for the same gene
Writes derived/elife_replication.csv and prints a summary."""
import numpy as np, pandas as pd, itertools
from scipy.stats import fisher_exact
D = np.load("derived/elife_dual_effects.npz", allow_pickle=True); O = np.load("derived/elife_dolcetto_effects.npz", allow_pickle=True)
common = sorted(set(D["genes"]) & set(O["genes"]))
def prep(z):
    gi = {g: i for i, g in enumerate(z["genes"])}; cols = np.array([gi[g] for g in common])
    return dict(Z=z["Z"][:, cols], ZA=z["ZA"][:, cols], ZB=z["ZB"][:, cols], target=z["target"], is_nt=z["is_nt"], guides=z["guides"])
D, O = prep(D), prep(O); ci = {g: i for i, g in enumerate(common)}
def eff(L):
    out = {}
    for r, t in enumerate(L["target"]):
        if L["is_nt"][r] or t not in ci: continue
        if L["Z"][r, ci[t]] <= -3: out.setdefault(t, []).append(r)
    return out
eD, eO = eff(D), eff(O)
def rep(z1, z2, mask):
    s = (np.abs(z1) >= 5) & mask
    r = s & (np.abs(z2) >= 2) & (np.sign(z1) == np.sign(z2))
    one = s & (np.abs(z2) < 1)
    return int(s.sum()), int(r.sum()), int(one.sum())
rows = []
for name, L, E in [("dual", D, eD), ("dolcetto", O, eO)]:
    for t, rs in E.items():
        m = np.ones(len(common), bool); m[ci[t]] = False
        for r in rs:  # split-half baseline (half vs half)
            n, k, one = rep(L["ZA"][r], L["ZB"][r], m); rows.append(dict(library=name, comparison="split-half (same construct)", gene=t, strong=n, replicated=k, one_only=one))
        for a, b in itertools.combinations(rs, 2):  # different constructs, half vs half (size matched to split-half)
            n, k, one = rep(L["ZA"][a], L["ZB"][b], m); rows.append(dict(library=name, comparison="different constructs, same library", gene=t, strong=n, replicated=k, one_only=one))
for t in set(eD) & set(eO):
    m = np.ones(len(common), bool); m[ci[t]] = False
    for a in eD[t]:
        for b in eO[t]:
            n, k, one = rep(D["ZA"][a], O["ZB"][b], m); rows.append(dict(library="cross", comparison="dual vs Dolcetto (different libraries)", gene=t, strong=n, replicated=k, one_only=one))
R = pd.DataFrame(rows); R.to_csv("derived/elife_replication.csv", index=False)
S = R.groupby("comparison")[["strong", "replicated", "one_only"]].sum()
S["replication"] = (S.replicated / S.strong).round(3); S["one_guide_only"] = (S.one_only / S.strong).round(3)
S["pairs"] = R.groupby("comparison").size(); S["genes"] = R.groupby("comparison").gene.nunique()
print(S.to_string())
b = S.loc["split-half (same construct)"]
for c in S.index:
    if c.startswith("split"): continue
    x = S.loc[c]
    _, p = fisher_exact([[x.replicated, x.strong - x.replicated], [b.replicated, b.strong - b.replicated]])
    print(f"{c}: replication {x.replication} vs split-half {b.replication}, Fisher P = {p:.2e}")
