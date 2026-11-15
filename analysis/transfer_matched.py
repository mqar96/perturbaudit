"""Cross-cell-line transfer of K562 edges with like-for-like baselines.

For each source panel x method (top-K edges audited with the corrected package) and each target line:
  - replication: |z_target| >= 3 with the same sign as the K562 effect (z of parent knockdown on child)
  - audited vs flagged (unsupported or reversed): edge-level Fisher P and a 95% CI for the difference from a
    cluster bootstrap that resamples parent genes (edges sharing a parent are not independent)
  - random_same_sign: random ordered panel pairs scored with the same same-sign rule
  - matched_null: for each audited edge, a random panel pair whose K562 |z| falls in the same effect-size bin
    (3-5, 5-10, 10-20, >20); asks whether the method's audited edges replicate better than random pairs with
    equal source effects
  - effect_rank: the same number of panel pairs taken purely by K562 |z| (equal budget, no network method)

Usage: python analysis/transfer_matched.py PANELS_DIR GTF OUT_CSV TARGET_NAME=TARGET_NPZ [...] -- PANEL:METHOD:K [...]
Target npz: Z (perturbations x genes knockdown z), perts, genes.
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from perturbaudit.screen import Screen
from perturbaudit.api import audit_network
from perturbaudit import proximity

argv = sys.argv[1:]
PANELS, GTF, OUT = argv[:3]
cut = argv.index("--")
targets = dict(a.split("=", 1) for a in argv[3:cut])
specs = [s.split(":") for s in argv[cut + 1:]]
tss = proximity.load_tss_gtf(GTF)
BINS = [3, 5, 10, 20, np.inf]
rng = np.random.default_rng(0)

T = {}
for name, path in targets.items():
    z = np.load(path, allow_pickle=True)
    T[name] = (z["Z"].astype(np.float32), {str(p): i for i, p in enumerate(z["perts"])}, {str(g): i for i, g in enumerate(z["genes"])})


def zt(name, a, b):
    Z, pi, gi = T[name]
    return float(Z[pi[a], gi[b]]) if a in pi and b in gi else np.nan


def rep(name, a, b, zs):
    v = zt(name, a, b)
    return np.nan if np.isnan(v) or np.isnan(zs) else float(abs(v) >= 3 and np.sign(v) == np.sign(zs))


def boot_diff(A, F, n=2000):
    """A, F: DataFrames with parent, rep. Resample parents (shared across both groups)."""
    parents = np.array(sorted(set(A.parent) | set(F.parent)))
    ga = A.groupby("parent").rep.agg(["sum", "count"]).reindex(parents, fill_value=0)
    gf = F.groupby("parent").rep.agg(["sum", "count"]).reindex(parents, fill_value=0)
    d = []
    for _ in range(n):
        idx = rng.integers(len(parents), size=len(parents))
        sa, ca = ga["sum"].values[idx].sum(), ga["count"].values[idx].sum()
        sf, cf = gf["sum"].values[idx].sum(), gf["count"].values[idx].sum()
        if ca and cf:
            d.append(sa / ca - sf / cf)
    return np.percentile(d, [2.5, 97.5]) if d else (np.nan, np.nan)


rows = []
done = set()
if os.path.exists(OUT):
    prev = pd.read_csv(OUT); rows = prev.to_dict("records")
    done = set(zip(prev.source, prev.method, prev.top_k.astype(int)))
cache = {}
for panel, method, K in specs:
    K = int(K)
    if (panel, method, K) in done:
        continue
    if panel not in cache:
        cache.clear()
        import gc; gc.collect()
        z = np.load(os.path.join(PANELS, panel, "panel.npz"), allow_pickle=True)
        X, lab, names = z["X"].astype(np.float64), z["labels"].astype(str), [str(n) for n in z["names"]]
        real = [i for i, n in enumerate(names) if not n.startswith("DECOY")]
        keep = ~np.char.startswith(lab, "DECOY")
        batch = z["batch"][keep] if "batch" in z.files and len(z["batch"]) else None
        guides = z["guides"][keep] if ("guides" in z.files and len(z["guides"]) and not panel.startswith("gwps")) else None
        rn = [names[i] for i in real]
        sc = Screen(X[keep][:, real], lab[keep], rn, "non-targeting", batch, guides)
        Zs = sc.effects()
        cache = {panel: (sc, real, rn, Zs)}
    sc, real, rn, Zs = cache[panel]
    W = np.load(os.path.join(PANELS, panel, f"W_{method}.npy"))[np.ix_(real, real)]
    E, _ = audit_network(sc, W, rn, top_k=K, tss=tss, n_decoys=200, seed=0)
    # candidate pairs for baselines: perturbed parent, measured child, parent != child
    P = [p for p in Zs.index if p in set(rn)]
    zmat = Zs.loc[P, rn].values
    pp, cc = np.meshgrid(np.arange(len(P)), np.arange(len(rn)), indexing="ij")
    pp, cc = pp.ravel(), cc.ravel()
    pidx = np.array([rn.index(p) for p in P])
    ok = (pidx[pp] != cc) & np.isfinite(zmat.ravel())
    pp, cc, zz = pp[ok], cc[ok], zmat.ravel()[ok]
    absz = np.abs(zz)
    binid = np.digitize(absz, BINS) - 1
    order = np.argsort(-absz)
    for tname in T:
        E["rep"] = [rep(tname, a, b, s) for a, b, s in zip(E.parent.astype(str), E.child.astype(str), E.z_parent_kd_on_child)]
        Et = E.dropna(subset=["rep"])
        A = Et[Et.verdict == "audited"]; F = Et[Et.verdict.isin(["unsupported", "reversed"])]
        r = dict(target=tname, source=panel, method=method, top_k=K, n_audited=len(A), n_flagged=len(F),
                 audited=round(A.rep.mean(), 3) if len(A) else np.nan, flagged=round(F.rep.mean(), 3) if len(F) else np.nan)
        if len(A) and len(F):
            r["fisher_p"] = fisher_exact([[A.rep.sum(), len(A) - A.rep.sum()], [F.rep.sum(), len(F) - F.rep.sum()]])[1]
            lo, hi = boot_diff(A, F)
            r["diff_ci_lo"], r["diff_ci_hi"] = round(lo, 3), round(hi, 3)
        # random same-sign baseline
        pick = rng.choice(len(pp), min(4000, len(pp)), replace=False)
        rr = [rep(tname, P[pp[i]], rn[cc[i]], zz[i]) for i in pick]
        r["random_same_sign"] = round(np.nanmean(rr), 3)
        # effect-size-matched null for audited edges
        if len(A):
            mm = []
            for s in np.abs(A.z_parent_kd_on_child.values):
                b = np.digitize([s], BINS)[0] - 1
                cand = np.flatnonzero(binid == b)
                if len(cand) == 0:
                    continue
                for i in rng.choice(cand, min(20, len(cand)), replace=False):
                    v = rep(tname, P[pp[i]], rn[cc[i]], zz[i])
                    if not np.isnan(v):
                        mm.append(v)
            r["matched_null"] = round(np.mean(mm), 3) if mm else np.nan
            top, got = [], 0
            for i in order:
                v = rep(tname, P[pp[i]], rn[cc[i]], zz[i])
                if not np.isnan(v):
                    top.append(v)
                    if len(top) >= len(A):
                        break
            r["effect_rank"] = round(np.mean(top), 3) if top else np.nan
        rows.append(r)
        print(r, flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    del E
pd.DataFrame(rows).to_csv(OUT, index=False)
