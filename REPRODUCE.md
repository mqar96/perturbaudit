# Reproducing the paper's results

Every number in the paper comes from a file in one of the two Zenodo data records. `PROVENANCE.csv` inside each
record lists, for every file, its full SHA-256, the script that produced it, the command, its inputs and the code
version; `ENVIRONMENT.json` records the code commit and package versions. The commands below regenerate the files
that were re-run with version 0.2.1.

## 1. Install the package and check it

```bash
python -m venv pa && . pa/bin/activate          # Windows: pa\Scripts\activate
pip install "perturbaudit @ git+https://github.com/mqar96/perturbaudit@v0.2.1"
git clone --branch v0.2.1 https://github.com/mqar96/perturbaudit && cd perturbaudit
pip install pytest && pytest -q                   # 25 tests
```

The analysis scripts import the installed package; run them from the repository root.

## 2. Get the data

- Zenodo record 1 (CC BY 4.0): unzip into `deposit_ccby/`. It holds `panels/` (the ten analysis panels with their
  network matrices `W_<method>.npy`), `target_effects/` (RPE1, HepG2 and Jurkat knockdown z-matrices), seed off-target
  maps and all frozen outputs.
- Zenodo record 2 (CC BY-NC-SA 4.0, derived from Xaira X-Atlas/Orion): unzip into `deposit_ncsa/` for the HCT116
  analyses. Respect its licence terms.
- GENCODE v43 basic annotation: `gencode.v43.basic.annotation.gtf.gz` from https://www.gencodegenes.org.

Raw screens are not redistributed; `DATA.md` lists their sources and the scripts that build panels from them.

## 3. Frozen top-50 audits and the all-pairs chance baseline (Fig. 3, Tables S2–S3)

```bash
python analysis/freeze_audits.py deposit_ccby/panels gencode.v43.basic.annotation.gtf.gz out/frozen
```

Writes `verdicts_top50.csv`, `chance_all_pairs.csv`, `audited_vs_chance.csv`, `reference_coverage.csv` and
`edges_<panel>_<method>.csv`. Name panels after the GTF argument to run a subset, for example `adamson_p100`
(about a minute). The genome-wide panels need about 6 GB of memory each.

## 4. Matched transfer (Fig. 4, Table S5)

```bash
python analysis/transfer_matched.py deposit_ccby/panels gencode.v43.basic.annotation.gtf.gz out/transfer_3lines.csv \
  RPE1=deposit_ccby/target_effects/rpe1_effects.npz \
  HepG2=deposit_ccby/target_effects/nadig_hepg2_effects.npz \
  Jurkat=deposit_ccby/target_effects/nadig_jurkat_effects.npz \
  -- k562_p300e:gbm_grn:200 k562_p300e:dotears:200 k562_p300e:inspre:200 k562_p300e:adapre:200 \
     k562_p300e:sdcd:200 k562_p300e:tce:200 gwps_p788:inspre:500 gwps_p788:adapre:500 gwps_p788:gbm_grn:500 \
     gwps_p788:sdcd:24 gwps_p788:dotears:500 gwps_p788:tce:500 gwps_p1434:sdcd:35 gwps_p1434:gbm_grn:500 \
     gwps_p1434:tce:500
```

For HCT116, replace the three targets with `HCT116=deposit_ncsa/target_effects/hct116_panel_effects.npz`. The script
is resumable: it appends one row per test and skips finished tests if rerun. The exposure check of the target
matrices is `analysis/check_target_effects.py`.

## 5. Simulations (Supplementary Figs S1–S3)

```bash
export DOTEARS_DIR=/path/to/dotears             # clone of the dotears repository (needed for dotears runs)
SIM_P=40 SIM_OUT=out/sim_calib_p40 python analysis/sim_calibration.py tce 1        # smoke test, about 1 minute
for p in 20 40 80; do SIM_P=$p DOSE=1 SIM_OUT=out/sim_calib_p$p python analysis/sim_calibration.py tce 50; done
SIM_OUT=out/sim_calib_dotears python analysis/sim_calibration.py dotears 10
SIM_AUDIT_OUT=out/sim_audit python analysis/sim_audit_benchmark.py 20 tce,inspre,gbm_grn,dotears
SIM_POWER_OUT=out/sim_power.csv python analysis/sim_power.py 10
```

inspre needs R with the `inspre` package (wrappers in `wrappers/`). A calibration replicate run on Linux and on Windows
gave identical tables. `sim_calibration.py` stops with an error if a
method subprocess fails. `sim_audit_benchmark.py` caches each fitted network under `out/sim_audit/W/`, so the audit
can be re-run without refitting; `sim_fit_networks.py` fits a subset of settings and replicates into the same cache, so
the fitting can be spread over several machines.

## 6. Robustness checks (Supplementary Notes 4–5)

```bash
python analysis/m6_reaudit.py deposit_ccby/panels/k562_p300e out/m6_reaudit_summary.csv \
  deposit_ccby/results_carried_over/m6/m6_edges_*.csv
N_JOBS=16 GBM_PER_GROUP=100 GBM_OBS=5000 python analysis/run_m5_lpo.py deposit_ccby/panels/k562_p300e gbm_grn 300 10
SDCD_EPOCHS=300 SDCD_BATCH=1024 python analysis/run_m5_lpo.py deposit_ccby/panels/k562_p300e sdcd 300 10
```

## 7. Figures and checksums

```bash
python analysis/make_fig34.py out/frozen figures                 # Fig. 3 and Fig. 4 (needs the transfer tables in out/frozen)
python analysis/fig1_schematic.py figures/Fig1_overview.png
python analysis/make_provenance.py deposit_ccby .                # rewrites PROVENANCE.csv and ENVIRONMENT.json
```

## Results carried over from version 0.2.0

Files under `results_carried_over/` were produced before the 0.2.1 fix and were not re-run, because they do not use
the batch-matched reference or the decoys (for example the seed scans, guide-replication tables, published-edge
audits and prediction-model scores), or because their inputs were verified to be identical under 0.2.1 (the
cross-platform pairs and knockdown-edge resource). `PROVENANCE.csv` gives the reason for each file.
