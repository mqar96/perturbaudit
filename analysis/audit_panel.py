import sys, json, glob, os, numpy as np, pandas as pd

from perturbaudit import heldout

panel_dir = sys.argv[1]
meta = json.load(open(f"{panel_dir}/meta.json"))
d = np.load(f"{panel_dir}/panel.npz", allow_pickle=True)
names = list(d["names"]); p, K = meta["p_real"], meta["k_decoy"]
idx = {n: i for i, n in enumerate(names)}
near = set()
for a, b, _ in meta["neighbour_pairs"]:
    near.add((idx[a], idx[b])); near.add((idx[b], idx[a]))
Ztce = np.load(f"{panel_dir}/W_tce.npy") if os.path.exists(f"{panel_dir}/W_tce.npy") else None
rows = []
for f in sorted(glob.glob(f"{panel_dir}/W_*.npy")):
    m = os.path.basename(f)[2:-4]
    W = np.nan_to_num(np.abs(np.load(f))); np.fill_diagonal(W, 0)
    real_real = W[:p, :p].copy(); np.fill_diagonal(real_real, -np.inf)
    KP = meta.get("k_perm", K); KK = meta.get("k_knock", 0)
    out_dec = W[p:p + KP, :p]; into_dec = W[:p, p:p + KP]; out_knock = W[p + KP:p + KP + KK, :p]
    allscores = np.concatenate([real_real[np.isfinite(real_real)], out_dec.ravel(), into_dec.ravel()])
    n_rr = p * (p - 1); n_out = KP * p; n_in = p * KP
    for k in [50, 100, 200, 400, 800]:
        if k > n_rr: continue
        t = np.sort(real_real[np.isfinite(real_real)])[::-1][k - 1]
        if t <= 0: continue
        r = (real_real >= t).sum(); do = (out_dec >= t).sum(); di = (into_dec >= t).sum()
        fdr_out = min(1.0, (do / n_out) * n_rr / max(r, 1))
        fdr_in = min(1.0, (di / n_in) * n_rr / max(r, 1))
        dk = int((out_knock >= t).sum()) if KK else 0
        fdr_k = min(1.0, (dk / max(KK * p, 1)) * n_rr / max(r, 1)) if KK else float("nan")
        sel = set(zip(*np.nonzero(real_real >= t)))
        n_near = len(sel & near); exp_near = r * len(near) / n_rr
        uns = rev = np.zeros(1)
        if Ztce is not None:
            uns, rev = heldout.intervention_support(real_real >= t, Ztce)
        rows.append(dict(method=m, top_k=k, unsupported=int(uns.sum()), reversed=int(rev.sum()), real_edges=int(r), decoy_out_edges=int(do), decoy_in_edges=int(di),
                         est_FDR_source_decoys=round(fdr_out, 4), est_FDR_knockoff_decoys=round(fdr_k, 4), est_FDR_target_decoys=round(fdr_in, 4),
                         neighbour_edges=n_near, neighbour_expected=round(exp_near, 2)))
df = pd.DataFrame(rows)
df.to_csv(f"{panel_dir}/audit_summary.csv", index=False)
print(df.to_string(index=False))
