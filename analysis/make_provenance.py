"""Write PROVENANCE.csv and ENVIRONMENT.json for a data deposit.

For every file in the deposit: full sha256, size, the script that produced it, the command, its inputs and the code
version (v0.2.1 = re-run with the corrected audit; v0.2.0 = carried over, with the reason it is unaffected).

Usage: python analysis/make_provenance.py DEPOSIT_DIR REPO_DIR
"""
import csv, fnmatch, hashlib, json, os, platform, subprocess, sys, time

DEP, REPO = sys.argv[1], sys.argv[2]

# pattern -> (script, command / note, inputs, code version)
P = "panels/<panel>/panel.npz, panels/<panel>/W_<method>.npy"
GTF = "gencode.v43.basic.annotation.gtf.gz"
RULES = [
    ("results_v0.2.1/verdicts_top50.csv", "analysis/freeze_audits.py", "freeze_audits.py panels GTF out", P + "; " + GTF, "v0.2.1"),
    ("results_v0.2.1/audited_vs_chance.csv", "analysis/freeze_audits.py", "freeze_audits.py panels GTF out", P + "; " + GTF, "v0.2.1"),
    ("results_v0.2.1/chance_all_pairs.csv", "analysis/freeze_audits.py", "freeze_audits.py panels GTF out", P + "; " + GTF, "v0.2.1"),
    ("results_v0.2.1/reference_coverage.csv", "analysis/freeze_audits.py", "freeze_audits.py panels GTF out", P, "v0.2.1"),
    ("results_v0.2.1/edges_top50/*", "analysis/freeze_audits.py", "freeze_audits.py panels GTF out", P + "; " + GTF, "v0.2.1"),
    ("results_v0.2.1/transfer_matched_*", "analysis/transfer_matched.py", "see REPRODUCE.md",
     "panels/gwps_*, panels/k562_p300e; target_effects/*.npz; " + GTF, "v0.2.1"),
    ("results_v0.2.1/sim_audit_results.csv", "analysis/sim_audit_benchmark.py", "sim_audit_benchmark.py 20 tce,inspre,gbm_grn,dotears (124 of 400 networks fitted with sim_fit_networks.py on a second machine)",
     "simulated (perturbaudit.sim)", "v0.2.1"),
    ("results_v0.2.1/sim_power.csv", "analysis/sim_power.py", "sim_power.py 10", "simulated", "v0.2.1"),
    ("results_v0.2.1/m6/*", "analysis/m6_reaudit.py", "m6_reaudit.py panels/k562_p300e out m6_edges_*.csv",
     "panels/k562_p300e/panel.npz; results_carried_over/m6/m6_edges_*.csv", "v0.2.1"),
    ("results_v0.2.1/lpo/*", "analysis/run_m5_lpo.py", "run_m5_lpo.py panels/k562_p300e METHOD 300 10", "panels/k562_p300e/panel.npz", "v0.2.1"),
    ("results_v0.2.1/target_effects_check*", "analysis/check_target_effects.py", "check_target_effects.py panels out NAME=NPZ ...",
     "target_effects/*.npz; panels/gwps_*, panels/k562_p300e", "v0.2.1"),
    ("results_carried_over/sim_calib_dotears.csv", "analysis/sim_calibration.py", "sim_calibration.py dotears 10",
     "simulated (perturbaudit.sim)", "v0.2.0 (does not use Screen or decoys)"),
    ("results_v0.2.1/gwps_inspre_case/*", "analysis/gwps_inspre_case.py", "gwps_inspre_case.py panels GTF out NAME=NPZ ...",
     "panels/gwps_p788; target_effects/*.npz; " + GTF, "v0.2.1"),
    ("results_v0.2.1/sim_calibration/*", "analysis/sim_calibration.py", "SIM_P=20|40|80 DOSE=1 sim_calibration.py tce 50; sim_calibration.py dotears 10",
     "simulated (perturbaudit.sim)", "v0.2.1"),
    ("results_v0.2.1/exposure_check/*", "comparison of v0.2.0 and v0.2.1 top-50 verdicts", "see CHANGELOG 0.2.1", P, "v0.2.0 vs v0.2.1"),
    ("results_carried_over/argen*", "analysis/audit_published_edges_gwps.py", "rule-based audit of published edges (|z| thresholds, no decoys)",
     "ARGEN Zenodo 22052266; K562 GWPS pseudo-bulk", "v0.2.0 (does not use Screen or decoys)"),
    ("results_carried_over/m2_*", "analysis/run_m2_cd4.py", "guide/donor split statistics", "Zhu, Marson et al. CD4 screen",
     "v0.2.0 (does not use Screen or decoys)"),
    ("results_carried_over/m3_*", "analysis/run_m3_cross_dataset.py", "neighbour enrichment by TSS distance", "screens; " + GTF,
     "v0.2.0 (does not use Screen or decoys)"),
    ("results_carried_over/m4_*", "analysis/m4_genomewide_gwps.py", "genome-wide seed-match ranks", "K562 GWPS; seed maps",
     "v0.2.0 (does not use Screen or decoys)"),
    ("results_carried_over/elife_*", "analysis/elife_replication.py", "split-half and guide-pair replication", "Replogle 2023 eLife",
     "v0.2.0 (does not use Screen or decoys)"),
    ("results_carried_over/bradu/*", "analysis/elife_replication.py (Bradu variant)", "split-half replication", "Bradu et al. VIPerturb-seq",
     "v0.2.0 (does not use Screen or decoys)"),
    ("results_carried_over/vcm/*", "analysis/vcm_audit.py, analysis/vcm_gears_score.py", "prediction-model edge precision",
     "model predictions; K562 GWPS", "v0.2.0 (does not use Screen or decoys)"),
    ("results_carried_over/m6/*", "analysis/run_m6_stress.py", "stress retention (refitted networks vs clean network)",
     "panels/k562_p300e", "v0.2.0 (retention does not use the audit; verdicts re-audited in results_v0.2.1/m6)"),
    ("results_carried_over/gwps_*", "analysis/extract_gwps_panel.py", "panel gene lists", "K562 GWPS", "v0.2.0 (gene selection only)"),
    ("panels/*/panel.npz", "analysis/build_panel_*.py, analysis/extract_gwps_panel.py", "pseudo-cell panels", "public screens (DATA.md)", "v0.2.0 (input data)"),
    ("panels/*/W_*.npy", "analysis/run_method.py", "network inference; parameters in run_<method>.json", "panels/<panel>/panel.npz",
     "method outputs (independent of the audit)"),
    ("panels/*", "analysis/build_panel_*.py, analysis/run_method.py", "panel metadata, run settings, preflight checks", "", "v0.2.0 (metadata)"),
    ("offtarget_maps/*", "analysis/build_offtarget_maps.py", "seed-match maps", "guide libraries; hg38; " + GTF, "v0.2.0 (does not use Screen)"),
    ("target_effects/*", "analysis/rpe1_effects.py (RPE1); analysis/nadig_effects.py (HepG2, Jurkat); analysis/orion_hct116.py, orion_post.py, make_hct116_panel_effects.py (HCT116)", "per-line knockdown z matrices. RPE1 regenerated with v0.2.1 (equal within float16 rounding); HCT116 panel subset regenerated exactly from the compact matrix; HepG2, Jurkat and the HCT116 compact matrix were built with v0.2.0 from raw data deleted after processing (exposure check: results_v0.2.1/target_effects_check*)", "source screens listed in DATA.md",
     "v0.2.0 (input data)"),
    ("cross_platform_pairs*", "analysis/cross_platform.py", "cross_platform.py", "panels/gwps_p788; target_effects",
     "v0.2.0 (pooled-control z; verified identical under v0.2.1)"),
    ("high_confidence_edges*", "analysis/hc_edges.py", "hc_edges.py", "cross_platform_pairs; Replogle dual seed map; " + GTF,
     "v0.2.0 (inputs verified identical under v0.2.1)"),
    ("*.md", "documentation", "", "", ""),
]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rule(rel):
    for pat, *rest in RULES:
        if fnmatch.fnmatch(rel, pat):
            return rest
    return ["", "", "", "unknown"]


def git(*a):
    try:
        return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


rows = []
for root, _, files in os.walk(DEP, followlinks=True):
    for f in sorted(files):
        p = os.path.join(root, f)
        rel = os.path.relpath(p, DEP).replace(os.sep, "/")
        if rel in ("PROVENANCE.csv", "ENVIRONMENT.json"):
            continue
        script, cmd, inputs, ver = rule(rel)
        rows.append(dict(file=rel, bytes=os.path.getsize(p), sha256=sha256(p), script=script, command=cmd, inputs=inputs, code_version=ver))
rows.sort(key=lambda r: r["file"])
unknown = [r["file"] for r in rows if r["code_version"] == "unknown"]
if unknown:
    raise SystemExit("no provenance rule for: " + ", ".join(unknown[:20]))
with open(os.path.join(DEP, "PROVENANCE.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)

env = dict(created=time.strftime("%Y-%m-%d %H:%M:%S %Z"), code_commit=git("rev-parse", "HEAD"),
           code_tag=git("describe", "--tags", "--always"), code_dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
           python=sys.version.split()[0], platform=platform.platform())
for m in ["numpy", "scipy", "pandas", "anndata", "sklearn", "matplotlib"]:
    try:
        env[m] = __import__(m).__version__
    except Exception:
        env[m] = None
env["analysis_script_sha256"] = {os.path.relpath(p, REPO): sha256(p) for p in
                                 sorted(os.path.join(REPO, d, f) for d in ("analysis", "src/perturbaudit", "wrappers")
                                        for f in os.listdir(os.path.join(REPO, d)) if f.endswith((".py", ".R", ".sh")))}
json.dump(env, open(os.path.join(DEP, "ENVIRONMENT.json"), "w"), indent=1)
print(len(rows), "files;", "commit", env["code_commit"][:12], "dirty" if env["code_dirty"] else "clean")
