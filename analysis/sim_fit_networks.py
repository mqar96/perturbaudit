"""Fit the simulation-benchmark networks for a subset of settings and replicates, in parallel with or instead of
sim_audit_benchmark.py. Writes OUT/W/<setting>_r<rep>_<method>.npy exactly as sim_audit_benchmark.py caches them, so
the benchmark then only runs the audit.

Usage: python analysis/sim_fit_networks.py OUT SETTING REP_FROM REP_TO METHODS"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perturbaudit import sim
import run_method as RM

OUT, S, A, B, METHODS = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5].split(",")
SETTINGS = {"clean": {}, "confounded": {"lat_scale": 1.2}, "offtarget15": {"off_frac": 0.15},
            "inactive30": {"inactive_frac": 0.3}, "misassign5": {"misassign": 0.05}}
P = 40
FN = {"tce": RM.run_tce, "inspre": RM.run_inspre, "gbm_grn": RM.run_gbm_grn, "dotears": RM.run_dotears}
os.makedirs(os.path.join(OUT, "W"), exist_ok=True)
for rep in range(A, B):
    s = sim.simulate_screen(p=P, seed=7919 * rep + sum(map(ord, S)), **SETTINGS[S])
    X = np.asarray(s["X"], dtype=np.float64); lab = np.asarray(s["labels"]).astype(str)
    names = [f"G{i}" for i in range(P)]; lab = np.array([l if l == "non-targeting" else f"G{l}" for l in lab])
    for m in METHODS:
        wf = os.path.join(OUT, "W", f"{S}_r{rep}_{m}.npy")
        if os.path.exists(wf):
            continue
        W = FN[m](X, lab.astype(object), names)
        np.save(wf, np.asarray(W, dtype=float))
    print(S, rep, "done", flush=True)
