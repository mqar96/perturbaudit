"""Simulation benchmark of the end-to-end audit (perturbaudit.api) with known truth.
For each scenario x replicate x method: precision of all top-k edges, of audited edges and of flagged edges,
and the fraction of true edges in the top-k that the audit keeps.
Usage: sim_audit_benchmark.py REPS [methods]"""
import sys, os, json, time, numpy as np, pandas as pd
sys.path.insert(0, "perturbaudit/src"); sys.path.insert(0, "perturbaudit/scripts")
from perturbaudit import sim
from perturbaudit.screen import Screen
from perturbaudit.api import audit_network
import run_method as RM

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
METHODS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["tce", "inspre", "gbm_grn", "dotears"]
OUT = "derived/sim_audit"; os.makedirs(OUT, exist_ok=True)
SETTINGS = {"clean": {}, "confounded": {"lat_scale": 1.2}, "offtarget15": {"off_frac": 0.15},
            "inactive30": {"inactive_frac": 0.3}, "misassign5": {"misassign": 0.05}, "offtarget15_withM4": {"off_frac": 0.15}}
ONLY = os.environ.get("ONLY_SETTINGS")
if ONLY: SETTINGS = {k: v for k, v in SETTINGS.items() if k in ONLY.split(",")}
P, K = 40, 40
res_f = f"{OUT}/sim_audit_results.csv"
done = set()
rows = pd.read_csv(res_f).to_dict("records") if os.path.exists(res_f) else []
for r in rows: done.add((r["setting"], r["rep"], r["method"]))
for sname, kw in SETTINGS.items():
    for rep in range(REPS):
        if all((sname, rep, m) in done for m in METHODS): continue
        seed_name = "offtarget15" if sname == "offtarget15_withM4" else sname
        s = sim.simulate_screen(p=P, seed=7919 * rep + sum(map(ord, seed_name)), **kw)
        otm = {f"G{a}": {f"G{b}"} for a, b in s["off"].items()} if sname.endswith("withM4") else None
        X = np.asarray(s["X"], dtype=np.float64); lab = np.asarray(s["labels"]).astype(str)
        names = [f"G{i}" for i in range(P)]; lab = np.array([l if l == "non-targeting" else f"G{l}" for l in lab])
        direct = np.asarray(s["W"]) != 0; desc = np.asarray(s["desc"]).astype(bool)
        sc = Screen(X, lab, names)
        for m in METHODS:
            if (sname, rep, m) in done: continue
            t = time.time()
            try:
                W = {"tce": RM.run_tce, "inspre": RM.run_inspre, "gbm_grn": RM.run_gbm_grn, "dotears": RM.run_dotears}[m](X, lab.astype(object), names)
            except Exception as e:
                print("fail", sname, rep, m, str(e)[:80], flush=True); continue
            W = np.nan_to_num(np.abs(np.asarray(W, dtype=float)))
            E, _ = audit_network(sc, W, names, top_k=K, n_decoys=100, seed=rep, offtarget_map=otm)
            truth = desc if m == "tce" else direct
            E["true"] = [bool(truth[int(a[1:]), int(b[1:])]) for a, b in zip(E.parent, E.child)]
            E["true_desc"] = [bool(desc[int(a[1:]), int(b[1:])]) for a, b in zip(E.parent, E.child)]
            aud = E.verdict == "audited"; fl = ~aud & ~E.verdict.str.startswith("untestable")
            rows.append(dict(setting=sname, rep=rep, method=m, n=len(E), precision_all=E.true.mean() if len(E) else np.nan,
                             precision_audited=E[aud].true.mean() if aud.any() else np.nan,
                             precision_flagged=E[fl].true.mean() if fl.any() else np.nan,
                             desc_precision_audited=E[aud].true_desc.mean() if aud.any() else np.nan,
                             n_audited=int(aud.sum()), n_flagged=int(fl.sum()),
                             true_kept=(E[aud].true.sum() / E.true.sum()) if E.true.sum() else np.nan,
                             reversed_true_frac=E[E.verdict == "reversed"].true.mean() if (E.verdict == "reversed").any() else np.nan,
                             seconds=round(time.time() - t, 1)))
            pd.DataFrame(rows).to_csv(res_f, index=False)
        print(sname, rep, "done", flush=True)
D = pd.DataFrame(rows)
summ = D.groupby(["setting", "method"])[["precision_all", "precision_audited", "precision_flagged", "true_kept", "n_audited"]].agg(["mean", "std"]).round(3)
summ.to_csv(f"{OUT}/sim_audit_summary.csv")
print(D.groupby(["setting", "method"])[["precision_all", "precision_audited", "precision_flagged", "true_kept"]].mean().round(2).to_string())
