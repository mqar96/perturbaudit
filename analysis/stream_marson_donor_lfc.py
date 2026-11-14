import io, os, sys, subprocess, h5py, numpy as np
sys.path.insert(0, "research/scripts")
from httpfile import HTTPRangeFile
URL = "https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/GWCD4i.DE_stats.by_donors.h5mu"
OUT = "derived/cd4_by_donors"
f = h5py.File(io.BufferedReader(HTTPRangeFile(URL, 16866278447), buffer_size=1 << 20), "r")
for mod in ["CE0006864_CE0008162", "CE0008678_CE0010866"]:
    ds = f[f"mod/{mod}/layers/log_fc"]; off = ds.id.get_offset(); n, p = ds.shape
    out = np.lib.format.open_memmap(f"{OUT}/{mod}_log_fc.npy", mode="w+", dtype=np.float16, shape=(n, p))
    rp = max(1, (256 << 20) // (p * 8))
    for s in range(0, n, rp):
        e = min(n, s + rp)
        raw = subprocess.run(["curl", "-sSL", "--fail", "--retry", "5", "-r", f"{off+s*p*8}-{off+e*p*8-1}", URL], capture_output=True, check=True).stdout
        out[s:e] = np.clip(np.nan_to_num(np.frombuffer(raw, "<f8").reshape(e - s, p)), -60000, 60000).astype(np.float16)
    out.flush(); del out; print(mod, "done", flush=True)
