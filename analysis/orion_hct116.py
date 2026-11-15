"""X-Atlas/Orion HCT116 (CC BY-NC-SA 4.0; analysis only, nothing redistributed) on the owner's PC.
Waits for the Bradu parse to finish, then streams the 109 HCT116 parquet batches one at a time (download ->
accumulate -> delete): per-target sums of log1p-CP10k expression for all genes, per-sample NT moments.
Output: orion_hct116_effects.npz (batch-matched null-variance z, perturbation x gene, float16) + per-guide-pair sums
for the Hartman off-target analysis."""
import os, sys, time, json, urllib.request, subprocess
R = os.environ.get("ORION_DIR", ".")
TEST = os.environ.get("ORION_TEST")
log = open(os.path.join(R, "orion.log"), "a", buffering=1)
while not TEST and "STEP1 DONE" not in (open(os.path.join(R, "bradu_step1.log"), errors="ignore").read() if os.path.exists(os.path.join(R, "bradu_step1.log")) else ""):
    time.sleep(60)
if not TEST: subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyarrow"])
import numpy as np, pyarrow.parquet as pq, pyarrow.compute as pc, scipy.sparse as sp
API = "https://huggingface.co/api/datasets/Xaira-Therapeutics/X-Atlas-Orion/tree/main/data"
files = sorted(f["path"] for f in json.load(urllib.request.urlopen(API, timeout=120)) if "HCT116" in f["path"]) if not TEST else sorted(os.listdir(TEST))
gm = os.path.join(R, "orion_gene_metadata.parquet")
(urllib.request.urlretrieve("https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion/resolve/main/metadata/gene_metadata.parquet", gm) if not TEST else None)
meta = pq.read_table(gm).to_pandas(); ng = int(meta.gene_token_id.max()) + 1
log.write(f"{len(files)} files, {ng} gene tokens\n")
targets, tidx = [], {}; samples, sidx = [], {}; pairs, pidx = [], {}
S1 = {}; N = {}; NB = {}; P1 = {}; PN = {}
c_s1 = {}; c_s2 = {}; c_n = {}
def tget(d, k, n): 
    if k not in d: d[k] = np.zeros(n, np.float64)
    return d[k]
import threading, concurrent.futures as cf
BASE = "https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion/resolve/main/"
def pdl(f, dest, nconn=16):
    if TEST: __import__("shutil").copy(os.path.join(TEST, f), dest); return dest
    url = BASE + f
    for att in range(20):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"}), timeout=60) as h:
                size = int(h.headers["Content-Range"].split("/")[-1]); break
        except Exception as e: log.write(f"size retry {f} {e}\n"); time.sleep(10)
    with open(dest, "wb") as fo: fo.truncate(size)
    step = -(-size // nconn); errs = []
    def part(a, b):
        pos = a
        for att in range(30):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={pos}-{b}"}), timeout=60) as h, open(dest, "r+b") as fo:
                    fo.seek(pos)
                    while True:
                        c = h.read(1 << 20)
                        if not c: break
                        fo.write(c); pos += len(c)
                if pos > b: return
            except Exception as e: time.sleep(5)
        errs.append((a, b))
    th = [threading.Thread(target=part, args=(i * step, min(size, (i + 1) * step) - 1)) for i in range(nconn) if i * step < size]
    [t.start() for t in th]; [t.join() for t in th]
    if errs or os.path.getsize(dest) != size: raise RuntimeError(f"download failed {f} {errs}")
    return dest
state_f = os.path.join(R, "orion_state.npz"); done_f = os.path.join(R, "orion_done.txt")
done = set(open(done_f).read().split()) if (os.path.exists(done_f) and os.path.exists(state_f)) else set()
acc = None
if os.path.exists(state_f) and done:
    acc = dict(np.load(state_f, allow_pickle=True))
    targets = list(acc["targets"]); samples = list(acc["samples"]); pairs = list(acc["pairs"])
    tidx = {t: i for i, t in enumerate(targets)}; sidx = {s: i for i, s in enumerate(samples)}; pidx = {p: i for i, p in enumerate(pairs)}
cap = 25000
S1m = acc["S1"] if acc is not None else np.zeros((cap, ng), np.float32); Nm = acc["N"] if acc is not None else np.zeros(cap)
NBm = acc["NB"] if acc is not None else np.zeros((cap, 256)); CS1 = acc["CS1"] if acc is not None else np.zeros((256, ng)); CS2 = acc["CS2"] if acc is not None else np.zeros((256, ng)); CN = acc["CN"] if acc is not None else np.zeros(256)
todo = [(k, f) for k, f in enumerate(files) if f not in done]
POOL = cf.ThreadPoolExecutor(1)
def fetch(i):
    k, f = todo[i]; d = os.path.join(R, f"orion_dl_{k}.parquet")
    for att in range(5):
        try: return pdl(f, d)
        except Exception as e: log.write(f"retry {f} {e}\n"); time.sleep(30)
    raise RuntimeError(f)
fut = POOL.submit(fetch, 0) if todo else None
for ii, (k, f) in enumerate(todo):
    t0 = time.time(); local = fut.result()
    fut = POOL.submit(fetch, ii + 1) if ii + 1 < len(todo) else None
    log.write(f"  got {f} (waited {time.time()-t0:.0f}s)\n")
    fh = open(local, "rb"); pf = pq.ParquetFile(fh)
    for rg in range(pf.num_row_groups):
        T = pf.read_row_group(rg, columns=["gene_token_id", "gene_expression", "total_counts", "sample", "gene_target", "pass_guide_filter"])
        keep = np.asarray(T.column("pass_guide_filter").to_numpy(zero_copy_only=False), bool)
        tg = np.asarray(T.column("gene_target").to_pylist(), dtype=object);         sm = np.asarray(T.column("sample").to_pylist(), dtype=object); tot = T.column("total_counts").to_numpy().astype(np.float64)
        ids = T.column("gene_token_id"); ex = T.column("gene_expression")
        lens = pc.list_value_length(ids).to_numpy(); indptr = np.concatenate([[0], np.cumsum(lens)])
        col = pc.list_flatten(ids).to_numpy().astype(np.int64); val = pc.list_flatten(ex).to_numpy().astype(np.float64)
        rows = np.repeat(np.arange(len(lens)), lens); val = np.log1p(val / np.maximum(tot[rows], 1) * 1e4)
        X = sp.csr_matrix((val, col, indptr), shape=(len(lens), ng))
        for name in set(tg[keep]):
            if name not in tidx: tidx[name] = len(targets); targets.append(name)
        for s in set(sm):
            if s not in sidx: sidx[s] = len(samples); samples.append(s)
        ti = np.array([tidx.get(x, -1) for x in tg]); si = np.array([sidx[x] for x in sm]); ok = keep & (ti >= 0)
        I = sp.csr_matrix((np.ones(ok.sum()), (ti[ok], np.flatnonzero(ok))), shape=(cap, len(lens)))
        def addrows(M, A):
            A = A.tocsr(); r = np.flatnonzero(np.diff(A.indptr))
            if len(r): M[r] += A[r].toarray().astype(M.dtype)
        addrows(S1m, I @ X); np.add.at(Nm, ti[ok], 1); np.add.at(NBm, (ti[ok], si[ok]), 1)
        nt = ok & np.array([str(x).lower() in ("non-targeting", "nontargeting", "ntc", "safe-targeting") for x in tg])
        if nt.any():
            Ib = sp.csr_matrix((np.ones(nt.sum()), (si[nt], np.flatnonzero(nt))), shape=(256, len(lens)))
            addrows(CS1, Ib @ X); addrows(CS2, Ib @ X.multiply(X)); np.add.at(CN, si[nt], 1)
    try: pf.close()
    except Exception: pass
    fh.close(); del pf; T = None; import gc; gc.collect()
    for _ in range(30):
        try: os.remove(local); break
        except PermissionError: time.sleep(2)
    log.write(f"  NT cells so far {int(CN.sum())}\n")
    done.add(f)
    log.write(f"{k+1}/{len(files)} {f} {time.time()-t0:.0f}s targets {len(targets)} samples {len(samples)}\n")
    if (ii + 1) % 5 == 0 or ii + 1 == len(todo):
        np.savez(state_f + ".tmp.npz", S1=S1m, N=Nm, NB=NBm, CS1=CS1, CS2=CS2, CN=CN, targets=np.array(targets, dtype=object), samples=np.array(samples, dtype=object), pairs=np.array(pairs, dtype=object))
        os.replace(state_f + ".tmp.npz", state_f); open(done_f, "w").write("\n".join(sorted(done)) + "\n"); log.write(f"  checkpoint {len(done)} files\n")
T_ = len(targets); B = len(samples)
mu = CS1[:B] / np.maximum(CN[:B, None], 1); var = np.maximum(CS2[:B] / np.maximum(CN[:B, None], 1) - mu ** 2, 1e-12)
n = Nm[:T_]; w = NBm[:T_, :B] / np.maximum(n[:, None], 1)
Z = (S1m[:T_] / np.maximum(n[:, None], 1) - w @ mu) / np.sqrt((w @ var) / np.maximum(n[:, None], 1) + (w ** 2) @ (var / np.maximum(CN[:B, None], 1)))
gn = meta.set_index("gene_token_id").reindex(range(ng)).gene_name.astype(str).values
np.savez_compressed(os.path.join(R, "orion_hct116_effects.npz"), Z=Z.astype(np.float16), perts=np.array(targets, dtype=str), genes=gn, n=n)
log.write(f"DONE effects {Z.shape}\n")
