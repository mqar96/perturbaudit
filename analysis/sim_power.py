"""H3 power curves (simulation). Off-target effects injected for 30% of perturbations, only in guide-1 cells.
Sensitivity = share of supported artefact edges (i -> off-target gene or its descendants, not a true descendant
of i) that the audit flags (M4 with the seed map; M2 guide-specific without it). Specificity side: share of
supported TRUE-descendant edges wrongly flagged. Grid: off-target strength x cells per perturbation, REPS each."""
import sys, os, zlib, numpy as np, pandas as pd

from perturbaudit.sim import random_dag
from perturbaudit.screen import Screen
from perturbaudit.api import audit_network
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
OUT = os.environ.get("SIM_POWER_OUT", "derived/sim_power.csv"); P = 40
def screen(p, n_cells, strength, seed, off_frac=0.3, n_obs=4000, n_lat=4, lat_scale=0.6):
    rng = np.random.default_rng(seed); W = random_dag(p, 0.06, rng); L = rng.normal(0, lat_scale, (n_lat, p))
    T = np.linalg.inv(np.eye(p) - W)
    off = {i: int(rng.choice([k for k in range(p) if k != i])) for i in range(p) if rng.random() < off_frac}
    def draw(n, kd=None, g1=False):
        U = rng.normal(size=(n, n_lat)) @ L + rng.standard_t(5, size=(n, p)) * 0.7; s = np.zeros(p)
        if kd is not None:
            s[kd] = -2.0 * rng.uniform(0.5, 1.0)
            if g1 and kd in off: s[off[kd]] = -strength
        return (U + s) @ T
    X = [draw(n_obs)]; lab = ["non-targeting"] * n_obs; gd = ["NT"] * n_obs
    for i in range(p):
        h = n_cells // 2
        X += [draw(h, i, True), draw(n_cells - h, i, False)]; lab += [f"G{i}"] * n_cells; gd += [f"G{i}_g1"] * h + [f"G{i}_g2"] * (n_cells - h)
    desc = (np.abs(T) > 1e-12) & ~np.eye(p, dtype=bool)
    return np.vstack(X), np.array(lab), np.array(gd), desc, off
rows = []
for strength in [0.4, 0.8, 1.2, 1.6, 2.4]:
    for nc in [50, 150, 400]:
        for rep in range(REPS):
            X, lab, gd, desc, off = screen(P, nc, strength, seed=rep * 7919 + int(strength * 100) + nc)
            names = [f"G{i}" for i in range(P)]
            art = np.zeros((P, P), bool)
            for i, o in off.items():
                art[i, o] = True; art[i] |= desc[o]
            art &= ~desc; np.fill_diagonal(art, False)
            A = np.ones((P, P)); np.fill_diagonal(A, 0)
            otm = {f"G{a}": {f"G{b}"} for a, b in off.items()}
            for mode in ["M4_seedmap", "M2_guides"]:
                sc = Screen(X, lab, names, "non-targeting", None, gd if mode == "M2_guides" else None)
                E, _ = audit_network(sc, A, names, n_decoys=50, seed=rep, offtarget_map=otm if mode == "M4_seedmap" else None)
                i = E.parent.str[1:].astype(int).values; j = E.child.str[1:].astype(int).values
                isart = art[i, j]; istrue = desc[i, j]
                sup = ~E.verdict.isin(["unsupported", "reversed"]) & ~E.verdict.str.startswith("untestable")
                flag = E.verdict.isin(["offtarget"] if mode == "M4_seedmap" else ["guide_specific"])
                rows.append(dict(strength=strength, cells=nc, rep=rep, mode=mode, n_art_supported=int((isart & sup).sum()),
                                 sensitivity=float(flag[isart & sup].mean()) if (isart & sup).any() else np.nan,
                                 false_flag_true=float(flag[istrue & sup].mean()) if (istrue & sup).any() else np.nan))
            pd.DataFrame(rows).to_csv(OUT, index=False)
        print(strength, nc, "done", flush=True)
D = pd.DataFrame(rows)
print(D.groupby(["mode", "strength", "cells"])[["n_art_supported", "sensitivity", "false_flag_true"]].mean().round(3).to_string())
