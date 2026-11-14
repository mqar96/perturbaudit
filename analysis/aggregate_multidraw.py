"""Aggregate repeated decoy draws: mean/sd of audit metrics and per-edge stability across draws.
Usage: aggregate_multidraw.py TAG OUTDIR"""
import sys, glob, os, json, itertools
import numpy as np, pandas as pd

tag, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
dirs = sorted(glob.glob(f"derived/panels/{tag}_d*"))
dirs = [d for d in dirs if os.path.exists(f"{d}/audit_summary.csv")]
print("draws:", [os.path.basename(d) for d in dirs])

# 1. metric mean/sd
A = pd.concat([pd.read_csv(f"{d}/audit_summary.csv").assign(draw=os.path.basename(d)) for d in dirs])
for c in ["unsupported", "reversed", "neighbour_edges"]:
    A[c + "_frac"] = A[c] / A["top_k"]
cols = ["unsupported_frac", "reversed_frac", "est_FDR_source_decoys", "est_FDR_knockoff_decoys",
        "est_FDR_target_decoys", "neighbour_edges"]
agg = A.groupby(["method", "top_k"])[cols].agg(["mean", "std"]).round(3)
agg.columns = ["_".join(c) for c in agg.columns]
agg = agg.reset_index()
agg["n_draws"] = len(dirs)
agg.to_csv(f"{out}/multidraw_metrics.csv", index=False)

# 2. per-edge stability among real genes
def real_names(d):
    z = np.load(f"{d}/panel.npz", allow_pickle=True)
    names = [str(x) for x in z["names"]]
    return names

methods = sorted({os.path.basename(f)[2:-4] for d in dirs for f in glob.glob(f"{d}/W_*.npy")})
rows, edge_rows = [], []
for m in methods:
    for k in [50, 100, 200]:
        sets = []
        for d in dirs:
            f = f"{d}/W_{m}.npy"
            if not os.path.exists(f):
                continue
            names = real_names(d)
            W = np.nan_to_num(np.abs(np.load(f)))
            real = [i for i, n in enumerate(names) if not n.startswith("DECOY")]
            Wr = W[np.ix_(real, real)].copy()
            np.fill_diagonal(Wr, 0)
            idx = np.argsort(Wr.ravel())[::-1][:k]
            idx = [i for i in idx if Wr.ravel()[i] > 0]
            rn = [names[i] for i in real]
            sets.append({(rn[i // len(rn)], rn[i % len(rn)]) for i in idx})
        if len(sets) < 2:
            continue
        jac = [len(a & b) / max(len(a | b), 1) for a, b in itertools.combinations(sets, 2)]
        cnt = {}
        for s in sets:
            for e in s:
                cnt[e] = cnt.get(e, 0) + 1
        freq = np.array(list(cnt.values())) / len(sets)
        rows.append(dict(method=m, top_k=k, n_draws=len(sets), jaccard_mean=round(np.mean(jac), 3),
                         jaccard_sd=round(np.std(jac), 3),
                         frac_edges_in_all_draws=round(np.mean([cnt.get(e, 0) == len(sets) for s in sets for e in s]), 3),
                         frac_edges_in_one_draw=round(np.mean([cnt.get(e, 0) == 1 for s in sets for e in s]), 3),
                         n_distinct_edges=len(cnt)))
        if k == 100:
            for (a, b), c in cnt.items():
                edge_rows.append(dict(method=m, source=a, target=b, draws=c, freq=c / len(sets)))
S = pd.DataFrame(rows)
S.to_csv(f"{out}/multidraw_stability.csv", index=False)
pd.DataFrame(edge_rows).sort_values(["method", "freq"], ascending=[True, False]).to_csv(
    f"{out}/multidraw_edge_freq_top100.csv", index=False)
print(agg[agg.top_k.isin([50, 100])].to_string())
print(S.to_string())
