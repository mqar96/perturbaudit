# perturbaudit

Internal-control falsification of causal gene networks inferred from CRISPR Perturb-seq screens.

perturbaudit checks any network (an edge list or score matrix, from any method) against the controls a Perturb-seq screen
already contains — non-targeting cells, independent guides, guide sequences, genomic positions and the interventions
themselves — and gives every edge a verdict: audited, unsupported, reversed, off-target, neighbour, guide-specific,
unstable or untestable. Verdicts are total-effect evidence; see `docs/verdicts.md` for exact definitions. Compare
audited fractions with the all-pairs chance baseline (`analysis/freeze_audits.py`), which the paper reports for every screen.

```bash
pip install git+https://github.com/mqar96/perturbaudit@v0.2.1
perturbaudit audit --h5ad screen.h5ad --network edges.csv --perturbation-col gene --out results/run1 --html
perturbaudit seedmap --guides my_library.csv --fasta hg38.fa.gz --gtf gencode.v43.basic.annotation.gtf.gz --out my_map.csv.gz
```

```python
import perturbaudit as pa
edges, summary = pa.audit_anndata(adata, network, perturbation_key="gene", control="non-targeting", top_k=200)
```

## What is in this repository

| Folder | Contents |
| --- | --- |
| `src/perturbaudit/` | The package: screen statistics, decoys, verdicts, neighbour and seed off-target checks, HTML report, CLI |
| `tests/` | Unit tests (`pytest`) |
| `docs/` | Documentation site (`mkdocs serve`) |
| `tutorials/` | Executed quick-start notebook (Papalexi 2021 screen) |
| `analysis/` | Every script used for the paper's analyses and figures |
| `results/` | Small result tables behind the paper's numbers |
| `figures/` | Paper figures |
| `wrappers/` | R wrappers for inspre, ADAPRE, CausalGRN/GIES |

Large derived data are on Zenodo in two records: CC BY 4.0 (seed off-target maps for four public CRISPRi/a libraries,
same-platform high-confidence edges, analysis panels, frozen v0.2.1 audit outputs, checksums) and CC BY-NC-SA 4.0
(everything derived from the Xaira X-Atlas/Orion HCT116 screen). DOIs: software 10.5281/zenodo.24221718; data 10.5281/zenodo.24220327 (CC BY) and 10.5281/zenodo.24221716 (CC BY-NC-SA). Raw screens are not
redistributed; see `DATA.md`. `REPRODUCE.md` gives the commands that regenerate the frozen outputs.

## Key results (preprint in preparation; numbers from the frozen v0.2.1 outputs)

- Across 38 networks from seven methods on ten screen panels (those with at least 20 testable top-ranked edges), 0–98%
  of top-50 testable edges are not supported by the screen's own knockdown of the claimed parent; seven method–panel
  combinations score at or below the all-pairs chance rate.
- Within one laboratory protocol, K562 edges that pass the audit replicate in RPE1, HepG2 and Jurkat more often than
  flagged edges (36/36 comparisons; parent-clustered bootstrap CI excludes 0 in 33). They also beat random pairs with the same K562 effect
  sizes in 31/36, but the same number of pairs ranked by K562 effect size alone replicates best (32/36): the verdicts
  mark which edges carry intervention evidence, and effect size is the stronger simple predictor of transfer.
- Distant promoter seed matches of 11–12 nt are associated with knockdown in 6–15% of cases vs about 0.5% expected,
  in the two genome-wide CRISPRi Perturb-seq screens we examined.
## Citation

Rao MQA. perturbaudit (preprint in preparation). Please cite the software DOI in `CITATION.cff`.

## License

MIT (code). Derived data on Zenodo: CC BY 4.0, except material derived from CC BY-NC-SA sources, which keeps that licence.
