"""Canonical scoring of GEARS and baselines from gears_preds.npz (written by gears_run.py on the owner's PC).
generic precision@20: children ranked by |predicted delta vs control|; supported if real |z| >= 3 with the predicted sign.
specific precision@20: children ranked by |predicted delta - mean training delta|; supported if real |z| >= 3 and the real
deviation from the mean response has the predicted sign. Chance computed with random children and random signs."""
import numpy as np, pandas as pd
z = np.load("derived/vcm/gears_preds.npz", allow_pickle=True)
test = list(z["test"]); genes = list(z["genes"]); gi = {g: i for i, g in enumerate(genes)}
Z = z["real_z"]; Dl = z["real_delta"]; models = {k[5:]: z[k] for k in z.files if k.startswith("pred_")}
mu = models["mean_of_training_perturbations"][0]; gene_of = lambda c: [g for g in c.split("+") if g != "ctrl"][0]
rows = []
for name, P in models.items():
    gen, spec = [], []
    for r, c in enumerate(test):
        d = P[r].copy(); t = gi.get(gene_of(c))
        if t is not None: d[t] = 0
        top = np.argsort(-np.abs(d))[:20]; gen.append(((np.sign(Z[r, top]) == np.sign(d[top])) & (np.abs(Z[r, top]) >= 3)).mean())
        if name != "mean_of_training_perturbations":
            ds = d - mu
            if t is not None: ds[t] = 0
            ts = np.argsort(-np.abs(ds))[:20]; spec.append(((np.sign(Dl[r, ts] - mu[ts]) == np.sign(ds[ts])) & (np.abs(Z[r, ts]) >= 3)).mean())
    rows.append(dict(model=name, edge_precision_at20=np.mean(gen), specific_edge_precision_at20=np.mean(spec) if spec else np.nan))
rng = np.random.default_rng(0); sc = []
for r in range(len(test)):
    g = rng.choice(len(genes), 20, replace=False); s = rng.choice([-1, 1], 20); sc.append(((np.sign(Dl[r, g] - mu[g]) == s) & (np.abs(Z[r, g]) >= 3)).mean())
R = pd.DataFrame(rows); R["chance_generic"] = (np.abs(Z) >= 3).mean() / 2; R["chance_specific"] = np.mean(sc)
R.to_csv("derived/vcm/gears_audit_final.csv", index=False); print(R.round(3).to_string(index=False))
