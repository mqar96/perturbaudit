"""Exposure check of the target-line effect matrices (built with v0.2.0) for the two reference bugs fixed in v0.2.1.
A perturbation with cells in a batch without control cells would get z ~ 1e5 on almost every gene (variance floor
1e-12): look for non-finite values and rows with many extreme values. A gene constant in a batch's controls gives an
extreme z for that gene: list off-diagonal |z| >= 50 among pairs of genes in the K562 panels used for transfer.
Usage: python analysis/check_target_effects.py PANELS_DIR OUT_CSV NAME=NPZ ..."""
import sys, numpy as np, pandas as pd
PANELS, OUT = sys.argv[1], sys.argv[2]
targets = dict(a.split("=", 1) for a in sys.argv[3:])
panels = {}
for p in ["gwps_p788", "gwps_p1434", "k562_p300e"]:
    z = np.load(f"{PANELS}/{p}/panel.npz", allow_pickle=True)
    panels[p] = {str(n) for n in z["names"] if not str(n).startswith("DECOY")}
rows = []
for name, f in targets.items():
    z = np.load(f, allow_pickle=True)
    Z = z["Z"].astype(np.float32); P = z["perts"].astype(str); G = z["genes"].astype(str)
    A = np.abs(np.nan_to_num(Z))
    row_extreme = (A > 30).sum(1)
    I, J = np.where(A >= 50)
    for p, S in panels.items():
        pi = np.isin(P, list(S)); gj = np.isin(G, list(S))
        hits = [(P[i], G[j], round(float(Z[i, j]), 1)) for i, j in zip(I, J) if P[i] != G[j] and pi[i] and gj[j]]
        rows.append(dict(target=name, panel=p, perturbations=int(Z.shape[0]), genes=int(Z.shape[1]),
                         nonfinite=int((~np.isfinite(Z)).sum() - np.isnan(Z).sum()), nan_fraction=round(float(np.isnan(Z).mean()), 5),
                         max_row_count_abs_z_gt30=int(row_extreme.max()), median_abs_z=round(float(np.median(A)), 3),
                         panel_pairs=int(pi.sum() * gj.sum()), panel_offdiag_abs_z_ge50=len(hits),
                         examples="; ".join(f"{a}>{b} {v}" for a, b, v in sorted(hits, key=lambda h: -abs(h[2]))[:5])))
pd.DataFrame(rows).to_csv(OUT, index=False)
print(pd.DataFrame(rows).drop(columns="examples").to_string(index=False))
