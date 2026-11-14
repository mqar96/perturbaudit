import io, os, sys, subprocess
import h5py, numpy as np, pandas as pd
sys.path.insert(0, "research/scripts")
from httpfile import HTTPRangeFile
URL = "https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/GWCD4i.DE_stats.by_donors.h5mu"
OUT = "derived/cd4_by_donors"; os.makedirs(OUT, exist_ok=True)
f = h5py.File(io.BufferedReader(HTTPRangeFile(URL, 16866278447), buffer_size=1 << 20), "r")
dec = lambda a: np.array([x.decode() if isinstance(x, bytes) else x for x in a], dtype=object)
for mod in f["mod"]:
    g = f["mod"][mod]
    obs = {}
    for k in ["target_condition", "target_contrast", "culture_condition", "ontarget_significant", "distal_offtarget_flag"]:
        o = g["obs"][k]
        obs[k] = dec(o["categories"][:])[o["codes"][:]] if isinstance(o, h5py.Group) else (dec(o[:]) if o.dtype.kind in "OS" else o[:])
    pd.DataFrame(obs).to_parquet(f"{OUT}/{mod}_obs.parquet")
    if not os.path.exists(f"{OUT}/var.parquet"):
        pd.DataFrame({"gene_id": dec(g["var/gene_ids"][:])}).to_parquet(f"{OUT}/var.parquet")
    ds = g["layers/zscore"]; off = ds.id.get_offset(); n, p = ds.shape
    out = np.lib.format.open_memmap(f"{OUT}/{mod}_zscore.npy", mode="w+", dtype=np.float16, shape=(n, p))
    rp = max(1, (256 << 20) // (p * 8))
    for s in range(0, n, rp):
        e = min(n, s + rp)
        raw = subprocess.run(["curl", "-sSL", "--fail", "--retry", "5", "-r", f"{off+s*p*8}-{off+e*p*8-1}", URL], capture_output=True, check=True).stdout
        out[s:e] = np.clip(np.nan_to_num(np.frombuffer(raw, "<f8").reshape(e - s, p)), -60000, 60000).astype(np.float16)
    out.flush(); del out; print(mod, "done", flush=True)
print("DONE")
