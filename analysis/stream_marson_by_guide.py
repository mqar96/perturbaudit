import io, os, sys, subprocess, json
import h5py, numpy as np, pandas as pd
sys.path.insert(0, "research/scripts")
from httpfile import HTTPRangeFile

URL = "https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/GWCD4i.DE_stats.by_guide.h5mu"
SIZE = 29424424894
OUT = "derived/cd4_by_guide"
os.makedirs(OUT, exist_ok=True)
f = h5py.File(io.BufferedReader(HTTPRangeFile(URL, SIZE), buffer_size=1 << 20), "r")

def dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a], dtype=object)

for mod in ["guide_1", "guide_2"]:
    g = f[f"mod/{mod}"]
    obs = {}
    for k in g["obs"].keys():
        o = g["obs"][k]
        if isinstance(o, h5py.Group) and "codes" in o:
            obs[k] = dec(o["categories"][:])[o["codes"][:]]
        elif isinstance(o, h5py.Dataset) and o.ndim == 1:
            v = o[:]
            obs[k] = dec(v) if v.dtype.kind in "OS" else v
    pd.DataFrame(obs).to_parquet(f"{OUT}/{mod}_obs.parquet")
    if mod == "guide_1":
        pd.DataFrame({"gene_id": dec(g["var/gene_ids"][:]), "gene_name": dec(g["var/gene_name"][:])}).to_parquet(f"{OUT}/var.parquet")
    for layer in ["zscore", "log_fc"]:
        ds = g[f"layers/{layer}"]
        off = ds.id.get_offset(); n, p = ds.shape
        out = np.lib.format.open_memmap(f"{OUT}/{mod}_{layer}.npy", mode="w+", dtype=np.float16, shape=(n, p))
        rows_per = max(1, (256 << 20) // (p * 8))
        for s in range(0, n, rows_per):
            e = min(n, s + rows_per)
            b0 = off + s * p * 8; b1 = off + e * p * 8 - 1
            raw = subprocess.run(["curl", "-sSL", "--fail", "--retry", "5", "-r", f"{b0}-{b1}", URL], capture_output=True, check=True).stdout
            arr = np.frombuffer(raw, dtype="<f8").reshape(e - s, p)
            out[s:e] = np.clip(np.nan_to_num(arr, nan=0.0), -60000, 60000).astype(np.float16)
            print(mod, layer, e, "/", n, flush=True)
        out.flush(); del out
json.dump({"source": URL, "layers": ["zscore", "log_fc"], "dtype": "float16", "nan_to_zero": True}, open(f"{OUT}/meta.json", "w"))
print("DONE")
