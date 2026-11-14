# Python API

| Object | Purpose |
| --- | --- |
| `Screen(X, perturbation, genes, control_label, batch, guide)` / `Screen.from_anndata(...)` | Holds the screen; `.effects()`, `.knockdown()`, `.decoys()` |
| `audit_anndata(adata, network, perturbation_key, ...)` | One-call audit for AnnData screens |
| `audit_network(screen, network, names, top_k, support_z, support_q, n_decoys, tss, offtarget_map, replicate_networks)` | Per-edge verdicts and a summary table |
| `seedscan.promoter_windows(fasta, gtf, out_tsv)` → `seedscan.build_sites(tsv)` → `seedscan.scan(guides, sites)` | Genome-wide seed matches |
| `seedscan.load_offtarget_map(path, min_len)` | Read a published map for `offtarget_map=` |
| `report.html_report(edges, summary, title, params)` | Standalone HTML report |
| `preflight.own_intervention_variance(...)` | Detects degenerate inputs before a method runs |
