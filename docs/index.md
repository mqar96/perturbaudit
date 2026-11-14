# perturbaudit

perturbaudit checks a gene network inferred from a CRISPR Perturb-seq screen against the controls the screen already
contains: non-targeting cells, independent guides, guide sequences and genomic positions, and the interventions themselves.
Every edge gets a verdict — audited, unsupported, reversed, off-target, neighbour, guide-specific, unstable or untestable —
and every result is reported against a chance baseline.

It works with networks from any method (an edge list or a score matrix) and with screens stored as AnnData.

```bash
pip install perturbaudit
perturbaudit audit --h5ad screen.h5ad --network edges.csv --perturbation-col gene --out results/run1 --html
```
