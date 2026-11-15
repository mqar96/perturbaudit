# Changelog
## 0.2.1
- Fix: perturbed cells in batches with fewer than `min_ref_cells` controls were compared against an empty reference
  (zero mean, near-zero variance), which could produce extreme z-scores and false `audited` verdicts. Such cells are now
  excluded and counted; genes constant in a batch's controls give NaN; missing batch labels raise an error.
  New verdict `untestable_no_reference`; `Screen.reference_coverage()`; CLI writes `<out>_reference_coverage.csv`.
- Fix: public decoys are now batch-matched (within-batch reference/pool split; each decoy copies a real perturbation's
  batch composition), as described in the paper. Previously they were drawn from a pooled half.
- Fix: q-values ignore non-finite scores (NaN could previously propagate a spuriously small q).
- `kd_status` and `checks_run` columns; `guide_het_p` (unadjusted P; `guide_het_q` kept as an alias).
- CI also tests the installed wheel outside the checkout.
- Analysis scripts run from the release layout (`analysis/`), fail loudly on subprocess errors and record graph size;
  `analysis/freeze_audits.py`, `analysis/transfer_matched.py` (effect-size-matched null, parent-clustered bootstrap,
  equal-budget effect ranking), `m6_reaudit.py`, `gwps_inspre_case.py`, `check_target_effects.py`, `make_fig34.py`;
  simulation and LPO drivers cache fitted networks; `REPRODUCE.md`; `make_provenance.py` writes full SHA-256, script,
  inputs and code version for every deposited file.
- All paper results that depend on the reference or decoys were frozen or re-run with 0.2.1 (67 of 2,544 top-50
  verdicts changed, mostly on Norman and Tian CRISPRa).
## 0.2.0 (not released; superseded by 0.2.1)
- `audit_anndata` one-call API; genome-wide seed scanner (`seedscan`) with `perturbaudit seedmap` CLI; `--offtarget-map` option; docs site; CI config.
## 0.1.0 (unreleased)
- Decoys, verdicts, neighbour and off-target checks, guide replication, held-out tests, stress tests, HTML report, CLI.
