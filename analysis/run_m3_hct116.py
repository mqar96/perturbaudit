"""M3 neighbour enrichment in X-Atlas/Orion HCT116 (CRISPRi), same analyse() as run_m3_cross_dataset.py."""
import numpy as np, pandas as pd
src = open("analysis/run_m3_cross_dataset.py").read().split("\nout = []")[0]
exec(src)
T = np.load("data/orion/orion_hct116_compact.npz", allow_pickle=True)
Z = np.nan_to_num(T["Z"].astype(np.float32)); perts = list(T["perts"]); genes = list(T["genes"])
rows = analyse("X-Atlas/Orion HCT116 (CRISPRi)", Z, perts, genes, tss_nm)
R = pd.DataFrame(rows); R.to_csv("derived/m3_hct116.csv", index=False); print(R.to_string(index=False))
