"""Bootstrap (B=200, resampling parent genes = perturbations) 95% CIs for headline numbers:
(a) audited / unsupported / reversed fractions of top-50 edges per method x panel;
(b) transfer replication of audited vs flagged edges per target line."""
import glob, os, numpy as np, pandas as pd
rng = np.random.default_rng(0); B = 200
def boot(E, stat):
    parents = E.parent.unique(); groups = {p: E[E.parent == p] for p in parents}; vals = []
    for _ in range(B):
        s = pd.concat([groups[p] for p in rng.choice(parents, len(parents), replace=True)])
        vals.append(stat(s))
    return np.nanpercentile(vals, [2.5, 97.5])
rows = []
for f in sorted(glob.glob("derived/panels/*/api_edges_*.csv")):
    pn = f.split("/")[2]; m = os.path.basename(f)[10:-4]
    E = pd.read_csv(f).head(50)
    E = E[~E.verdict.str.startswith("untestable")]
    if len(E) < 10: continue
    for v in ["audited", "unsupported", "reversed"]:
        st = lambda s, v=v: (s.verdict == v).mean()
        lo, hi = boot(E, st)
        rows.append(dict(kind="top50_fraction", panel=pn, method=m, quantity=v, estimate=round(st(E), 3), ci_low=round(lo, 3), ci_high=round(hi, 3), n=len(E)))
pd.DataFrame(rows).to_csv("derived/bootstrap_ci_top50.csv", index=False)
# transfer: rebuild per-edge tables
import sys
T = {}
for name, f in [("RPE1", "data/nadig/rpe1_effects.npz"), ("HepG2", "data/nadig/nadig_hepg2_effects.npz"), ("Jurkat", "data/nadig/nadig_jurkat_effects.npz")]:
    z = np.load(f, allow_pickle=True); T[name] = (z["Z"].astype(np.float32), {p: i for i, p in enumerate(map(str, z["perts"]))}, {g: i for i, g in enumerate(map(str, z["genes"]))})
rows2 = []
for pn, m, k in [("k562_p300e", "gbm_grn", 200), ("k562_p300e", "dotears", 200), ("k562_p300e", "inspre", 200), ("k562_p300e", "adapre", 200), ("k562_p300e", "sdcd", 200), ("gwps_p788", "gbm_grn", 500), ("gwps_p788", "inspre", 500), ("gwps_p788", "adapre", 500)]:
    E0 = pd.read_csv(f"derived/panels/{pn}/api_edges_{m}.csv").head(k)
    for tn, (Z, pi, gi) in T.items():
        E = E0.copy()
        E["zt"] = [Z[pi[a], gi[b]] if a in pi and b in gi else np.nan for a, b in zip(E.parent, E.child)]
        E = E[~E.zt.isna() & E.verdict.isin(["audited", "unsupported", "reversed"])]
        E["rep"] = (np.abs(E.zt) >= 3) & (np.sign(E.zt) == np.sign(E.z_parent_kd_on_child))
        diff = lambda s: s[s.verdict == "audited"].rep.mean() - s[s.verdict != "audited"].rep.mean()
        lo, hi = boot(E, diff)
        rows2.append(dict(kind="transfer_diff", panel=pn, method=m, target=tn, audited=round(E[E.verdict == "audited"].rep.mean(), 3),
                          flagged=round(E[E.verdict != "audited"].rep.mean(), 3), diff=round(diff(E), 3), ci_low=round(lo, 3), ci_high=round(hi, 3)))
R2 = pd.DataFrame(rows2); R2.to_csv("derived/bootstrap_ci_transfer.csv", index=False); print(R2.to_string(index=False))
R1 = pd.DataFrame(rows); print(R1[(R1.quantity == "audited") & R1.panel.isin(["gwps_p788", "k562_p300e", "rpe1_p100e"])].to_string(index=False))
