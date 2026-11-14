# Data sources and how to rebuild every result

Raw screens are public and are not redistributed here. Analysis scripts expect to run from a working directory with
`data/` (raw downloads) and `derived/` (intermediate files) next to this repository's `analysis/` folder.

| Screen | Source | Licence |
| --- | --- | --- |
| Replogle et al. 2022 K562 essential, K562 genome-wide, RPE1 | figshare / scPerturb (Zenodo 7041849) | CC BY 4.0 |
| Nadig et al. 2025 HepG2, Jurkat | GEO GSE264667 | public |
| Zhu, Marson et al. 2026 primary CD4 T cells | authors' data release | see source |
| Bradu et al. VIPerturb-seq K562 | authors' release | see source |
| Replogle et al. 2023 eLife (dual-guide) | eLife / GEO GSE205310 | CC BY 4.0 |
| Xaira X-Atlas/Orion HCT116 | Hugging Face Xaira-Therapeutics/X-Atlas-Orion | CC BY-NC-SA 4.0 (analysed, not redistributed) |
| Frangieh 2021, Papalexi 2021, Adamson 2016, Norman 2019, Tian 2021 | scPerturb | per source |
| Hartman et al. 2026 off-target tables | authors' repository | see source |
| ARGEN published edges | Zenodo 22052266 | see source |
| Guide libraries | Weissman lab ScreenProcessing (GitHub); eLife 81856 Supplementary file 1 | per source |
| GENCODE v43 basic GTF; hg38 genome | GENCODE; UCSC | public |

Scripts by result (in `analysis/`): panels `build_panel_*.py`, `extract_gwps_panel.py`; methods `run_method.py`; audits
`audit_panel_api.py`; chance baseline `pair_baseline.py`; transfer `run_transfer_generic.py`; robustness `run_m5_lpo.py`,
`run_m6_stress.py`; simulations `sim_*.py`; guide replication `elife_*.py`; HCT116 `run_m3_hct116.py`,
`m4_genomewide_hct116.py`; genome-wide seed test `m4_genomewide_gwps.py`; maps `build_promoter_windows.py`,
`build_offtarget_maps.py`; cross-platform `cross_platform.py`; high-confidence edges `hc_edges.py`; prediction-model audit
`vcm_audit.py`; figures `fig_*.py`. Every result file's sha256, inputs and code version are listed in `PROVENANCE.csv` on Zenodo.
