# perturbaudit

Internal-control falsification of causal gene networks inferred from CRISPR Perturb-seq screens.

perturbaudit checks any network (an edge list or score matrix, from any method) against the controls a Perturb-seq screen
already contains — non-targeting cells, independent guides, guide sequences, genomic positions and the interventions
themselves — and gives every edge a verdict: audited, unsupported, reversed, off-target, neighbour, guide-specific,
unstable or untestable. Results are always reported against a chance baseline.

```bash
pip install perturbaudit            # after release; for now: pip install .
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

Large derived data (seed off-target maps for five CRISPRi/a libraries, high-confidence edges, analysis panels, full result
tables and checksums) are on Zenodo: DOI to be added at release. Raw screens are not redistributed; see `DATA.md`.

## Key results (manuscript in preparation)

- 6–98% of top-ranked edges from current methods are not supported by the screen's own knockdown of the claimed parent;
  several methods score at or below random gene pairs.
- Audited K562 edges replicate in RPE1, HepG2 and Jurkat 1.7–7.7× more often than flagged edges (36/36 tests, 32 significant);
  transfer to an independent platform (Xaira HCT116) is weak, and we characterise why.
- CRISPRi seed off-targets act genome-wide in Perturb-seq: 11–12-nt seed matches at distant promoters are knocked down
  6–15% of the time vs about 0.5% expected, in two genome-wide screens.
- Edge-level audit of perturbation-response models: GEARS's strongest predicted effects are no better than the average
  response (precision@20 0.48 vs 0.50), but its perturbation-specific predictions are 5.4× chance and 2.5× simple baselines.

## Citation

Rao MQA. perturbaudit (in preparation). Please cite the Zenodo DOI once available. See `CITATION.cff`.

## License

MIT (code). Derived data on Zenodo: CC BY 4.0, except material derived from CC BY-NC-SA sources, which keeps that licence.
