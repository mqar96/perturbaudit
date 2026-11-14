# Quick start

```python
import anndata as ad, pandas as pd
import perturbaudit as pa

adata = ad.read_h5ad("screen.h5ad")                 # cells x genes, normalized; perturbed gene in obs
net = pd.read_csv("my_network.csv")                 # parent, child[, score]
edges, summary = pa.audit_anndata(adata, net, perturbation_key="gene", control="non-targeting",
                                  batch_key="gem_group", guide_key="guide", top_k=200)
print(summary)
edges.to_csv("verdicts.csv", index=False)
```

Add the genomic-neighbour check with a GENCODE GTF, and the off-target check with a seed map:

```python
tss = pa.proximity.load_tss_gtf("gencode.v43.basic.annotation.gtf.gz")
otm = pa.seedscan.load_offtarget_map("hCRISPRi_dual_Replogle_2022_seed_offtargets_ge10.csv.gz", min_len=11)
edges, summary = pa.audit_anndata(adata, net, "gene", tss=tss, offtarget_map=otm)
```

A worked notebook on the Papalexi 2021 screen is in `tutorials/quickstart_papalexi.ipynb`.
