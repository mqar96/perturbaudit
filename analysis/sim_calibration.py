import sys, os, json, time, subprocess, numpy as np, pandas as pd

from perturbaudit import sim

OUT = os.environ.get("SIM_OUT", "derived/sim_calib2")
os.makedirs(OUT, exist_ok=True)
settings = {
    "clean": {},
    "confounded": {"lat_scale": 1.2},
    "offtarget15": {"off_frac": 0.15},
    "inactive30": {"inactive_frac": 0.3},
    "misassign5": {"misassign": 0.05},
}
if os.environ.get("DOSE"):
    settings = {"off0": {"off_frac": 0.0}, "off05": {"off_frac": 0.05}, "off10": {"off_frac": 0.10}, "off15": {"off_frac": 0.15}, "off25": {"off_frac": 0.25},
                "clean": {}, "confounded": {"lat_scale": 1.2}, "inactive30": {"inactive_frac": 0.3}, "misassign5": {"misassign": 0.05}}
methods = sys.argv[1].split(",") if len(sys.argv) > 1 else ["tce", "dotears", "sdcd"]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
import zlib
p = int(os.environ.get('SIM_P', '40')); KP = KK = min(10, max(5, p // 4))
K = KP + KK
rows = []
for sname, kw in settings.items():
    for rep in range(reps):
        d = f"{OUT}/{sname}_r{rep}"
        os.makedirs(d, exist_ok=True)
        s = sim.simulate_screen(p=p, seed=1000 * rep + zlib.crc32(sname.encode()) % 997 + p, **kw)
        rng = np.random.default_rng(rep)
        X, lab, dnames = sim.add_decoys(s["X"], s["labels"], KP, KK, 150, rng)
        names = [str(i) for i in range(p)] + dnames
        lab = np.array([str(l) for l in lab], dtype=object)
        np.savez_compressed(f"{d}/panel.npz", X=X.astype(np.float32), labels=lab, names=np.array(names, dtype=object))
        np.save(f"{d}/truth_direct.npy", s["W"] != 0); np.save(f"{d}/truth_desc.npy", s["desc"])
        json.dump({"p_real": p, "k_decoy": K, "neighbour_pairs": [], "offtarget": {str(k): v for k, v in s["off"].items()}}, open(f"{d}/meta.json", "w"))
        for m in methods:
            if os.path.exists(f"{d}/W_{m}.npy"): continue
            py = os.environ.get("SDCD_PYTHON", sys.executable) if m == "sdcd" else sys.executable
            here = os.path.dirname(os.path.abspath(__file__))
            res = subprocess.run([py, os.path.join(here, "run_method.py"), m, d], capture_output=True, text=True)
            if res.returncode != 0 or not os.path.exists(f"{d}/W_{m}.npy"):
                raise RuntimeError(f"run_method.py {m} failed on {d}:\n{res.stderr[-2000:]}")
        for m in methods:
            if not os.path.exists(f"{d}/W_{m}.npy"): continue
            Wm = np.abs(np.load(f"{d}/W_{m}.npy")); np.fill_diagonal(Wm, 0)
            truth = np.load(f"{d}/truth_desc.npy") if m == "tce" else np.load(f"{d}/truth_direct.npy")
            truth_desc = np.load(f"{d}/truth_desc.npy")
            rr = Wm[:p, :p].copy(); np.fill_diagonal(rr, -np.inf); odp = Wm[p:p + KP, :p]; odk = Wm[p + KP:, :p]
            vals = np.sort(rr[np.isfinite(rr)])[::-1]
            for k in [20, 40, 80, 160]:
                t = vals[k - 1]
                if t <= 0: continue
                sel = rr >= t
                R = sel.sum()
                true_fdp = (sel & ~truth).sum() / R
                true_fdp_desc = (sel & ~truth_desc).sum() / R
                estp = min(1.0, ((odp >= t).sum() / (KP * p)) * (p * (p - 1)) / R)
                estk = min(1.0, ((odk >= t).sum() / (KK * p)) * (p * (p - 1)) / R)
                rows.append(dict(p=p, setting=sname, rep=rep, method=m, top_k=k, edges=int(R), true_FDP=round(true_fdp, 3),
                                 true_FDP_vs_descendants=round(true_fdp_desc, 3), est_perm=round(estp, 3), est_knockoff=round(estk, 3)))
        pd.DataFrame(rows).to_csv(f"{OUT}/calibration_results.csv", index=False)
        print(sname, rep, "done", flush=True)
df = pd.DataFrame(rows)
print(df.groupby(["setting", "method", "top_k"])[["true_FDP", "true_FDP_vs_descendants", "est_perm", "est_knockoff"]].mean().round(3).to_string())
