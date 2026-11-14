"""Command line: perturbaudit audit --h5ad screen.h5ad --network edges.csv --perturbation-col gene --out results/run1"""
import argparse, json, os, sys
import numpy as np
import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser(prog="perturbaudit", description="Audit a causal gene network against a Perturb-seq screen's internal controls.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit", help="per-edge verdicts for a network")
    a.add_argument("--h5ad", required=True, help="single-cell screen (cells x genes), normalized expression in X or --layer")
    a.add_argument("--network", required=True, help="CSV with parent,child[,score] columns, or .npy square matrix with --names")
    a.add_argument("--names", help="text file with one gene name per row/column of the .npy matrix")
    a.add_argument("--perturbation-col", required=True, help="obs column with the targeted gene per cell")
    a.add_argument("--control-label", default="non-targeting")
    a.add_argument("--batch-col", help="obs column with batch / GEM group (controls are batch-matched)")
    a.add_argument("--guide-col", help="obs column with guide identity (enables the independent-guide check)")
    a.add_argument("--gene-col", help="var column with gene symbols (default: var_names)")
    a.add_argument("--layer")
    a.add_argument("--gtf", help="GENCODE GTF(.gz) for the genomic-neighbour check")
    a.add_argument("--offtarget", help="CSV with gene,offtarget_gene columns (e.g. Hartman et al. seed-match table)")
    a.add_argument("--offtarget-map", help="perturbaudit seed off-target map (.csv.gz from resources/ or `perturbaudit seedmap`)")
    a.add_argument("--min-seed", type=int, default=11, help="seed length (nt) that counts as an off-target in --offtarget-map")
    a.add_argument("--replicates", nargs="*", help="networks inferred on resampled data (same format) for stability")
    a.add_argument("--top-k", type=int)
    a.add_argument("--support-z", type=float, default=3.0)
    a.add_argument("--support-q", type=float, default=0.05)
    a.add_argument("--n-decoys", type=int, default=200)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--html", action="store_true", help="also write <out>_report.html")
    a.add_argument("--out", required=True, help="output prefix; writes <out>_edges.csv, <out>_summary.csv, <out>_params.json")
    m = sub.add_parser("seedmap", help="genome-wide seed off-target map for a guide library")
    m.add_argument("--guides", required=True, help="CSV with guide,target,protospacer columns")
    m.add_argument("--fasta", required=True, help="genome FASTA(.gz), e.g. hg38")
    m.add_argument("--gtf", required=True, help="GENCODE GTF(.gz)")
    m.add_argument("--window", type=int, default=1000)
    m.add_argument("--min-seed", type=int, default=10)
    m.add_argument("--out", required=True, help="output .csv.gz")
    args = ap.parse_args(argv)
    if args.cmd == "seedmap":
        import tempfile
        from . import seedscan
        tmp = tempfile.NamedTemporaryFile(suffix=".tsv", delete=False).name
        seedscan.promoter_windows(args.fasta, args.gtf, tmp, args.window)
        R = seedscan.scan(pd.read_csv(args.guides), seedscan.build_sites(tmp), min_len=args.min_seed)
        R[R.target != R.gene].to_csv(args.out, index=False); os.remove(tmp)
        print(f"wrote {args.out}: {len(R)} guide-gene pairs >= {args.min_seed} nt"); return

    import anndata as ad
    from .screen import Screen
    from .api import audit_network
    from . import proximity

    def load_net(path):
        if path.endswith(".npy"):
            if not args.names:
                sys.exit("--names is required with a .npy network")
            return np.load(path)
        return pd.read_csv(path)

    names = [l.strip() for l in open(args.names)] if args.names else None
    adata = ad.read_h5ad(args.h5ad)
    sc = Screen.from_anndata(adata, args.perturbation_col, args.control_label, args.batch_col, args.guide_col, args.layer, args.gene_col)
    tss = proximity.load_tss_gtf(args.gtf) if args.gtf else None
    otm = None
    if args.offtarget:
        ot = pd.read_csv(args.offtarget)
        otm = {}
        for g, o in zip(ot.iloc[:, 0], ot.iloc[:, 1]):
            otm.setdefault(str(g), set()).add(str(o))
    if args.offtarget_map:
        from .seedscan import load_offtarget_map
        for t, gs in load_offtarget_map(args.offtarget_map, args.min_seed).items():
            otm = otm or {}; otm.setdefault(t, set()).update(gs)
    reps = [load_net(p) for p in args.replicates] if args.replicates else None
    E, S = audit_network(sc, load_net(args.network), names, top_k=args.top_k, support_z=args.support_z, support_q=args.support_q,
                         n_decoys=args.n_decoys, seed=args.seed, tss=tss, offtarget_map=otm, replicate_networks=reps)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    E.to_csv(f"{args.out}_edges.csv", index=False)
    S.to_csv(f"{args.out}_summary.csv", index=False)
    json.dump({k: v for k, v in vars(args).items()}, open(f"{args.out}_params.json", "w"), indent=1)
    if args.html:
        from .report import html_report
        open(f"{args.out}_report.html", "w", encoding="utf-8").write(html_report(E, S, title=f"perturbaudit: {os.path.basename(args.network)}", params=vars(args)))
    print(S.to_string(index=False))


if __name__ == "__main__":
    main()
