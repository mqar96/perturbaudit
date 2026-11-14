"""Write PROVENANCE.csv: for every result file, sha256 of the file, of the input panel/data it came from, and of the
code that produced it (scripts + package source), plus run parameters where recorded."""
import hashlib, os, glob, json, csv, time
B = "."
def sha(p, n=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(n)
            if not b: break
            h.update(b)
    return h.hexdigest()[:16]
code_files = sorted(glob.glob(f"{B}/perturbaudit/src/perturbaudit/*.py") + glob.glob(f"{B}/perturbaudit/scripts/*.py") + glob.glob(f"{B}/perturbaudit/wrappers/*.R"))
code_hash = hashlib.sha256("".join(sha(f) for f in code_files).encode()).hexdigest()[:16]
rows = []
for D in sorted(glob.glob(f"{B}/derived/panels/*/")):
    pn = os.path.join(D, "panel.npz")
    if not os.path.exists(pn): continue
    ph = sha(pn)
    for f in sorted(glob.glob(D + "W_*.npy") + glob.glob(D + "api_*.csv") + glob.glob(D + "m5_*.csv") + glob.glob(D + "m6_*.csv") + glob.glob(D + "audit_summary.csv")):
        run = f.replace("W_", "run_").replace(".npy", ".json")
        params = json.load(open(run)) if os.path.exists(run) and f.endswith(".npy") else {}
        rows.append(dict(result=os.path.relpath(f, B), result_sha256=sha(f), input=os.path.relpath(pn, B), input_sha256=ph,
                         code_sha256=code_hash, params=json.dumps(params), mtime=time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(f)))))
for f in sorted(glob.glob(f"{B}/derived/*.csv") + glob.glob(f"{B}/derived/*/*.csv")):
    if "/panels/" in f: continue
    rows.append(dict(result=os.path.relpath(f, B), result_sha256=sha(f), input="see LOG.md", input_sha256="", code_sha256=code_hash, params="",
                     mtime=time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(f)))))
for f in sorted(glob.glob(f"{B}/data/nadig/*.npz") + glob.glob(f"{B}/data/replogle/*.h5ad") + glob.glob(f"{B}/data/annot/*.gz")):
    rows.append(dict(result=os.path.relpath(f, B), result_sha256=sha(f), input="public download", input_sha256="", code_sha256="", params="", mtime=""))
out = f"{B}/project/02_PerturbSeq_Falsification_Toolkit/05_Development_Validation/PROVENANCE.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
json.dump({"code_sha256": code_hash, "n_code_files": len(code_files), "files": [os.path.relpath(f, B) for f in code_files]},
          open(out.replace(".csv", "_code.json"), "w"), indent=1)
print(len(rows), "records; code hash", code_hash)
