"""Subset the X-Atlas/Orion HCT116 effect matrix (orion_hct116_compact.npz, from orion_hct116.py and orion_post.py) to
the genes of the K562 transfer panels, for transfer_matched.py.
Usage: python analysis/make_hct116_panel_effects.py ORION_COMPACT_NPZ PANELS_DIR OUT_NPZ"""
import sys, numpy as np
src, panels, out = sys.argv[1:4]
genes = set()
for p in ["gwps_p788", "gwps_p1434", "k562_p300e"]:
    z = np.load(f"{panels}/{p}/panel.npz", allow_pickle=True)
    genes |= {str(n) for n in z["names"] if not str(n).startswith("DECOY")}
O = np.load(src, allow_pickle=True)
P = np.array([str(x) for x in O["perts"]]); G = np.array([str(x) for x in O["genes"]])
pi = np.flatnonzero(np.isin(P, list(genes)))
seen, gi = set(), []
for i, g in enumerate(G):
    if g in genes and g not in seen:
        seen.add(g); gi.append(i)
gi = np.array(gi)
Z = O["Z"][np.ix_(pi, gi)]
np.savez_compressed(out, Z=Z, perts=P[pi], genes=G[gi], n=O["n"][pi])
print(Z.shape)
