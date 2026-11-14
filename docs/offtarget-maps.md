# Seed off-target maps

Ready-made maps for hCRISPRi-v2, hCRISPRa-v2, hCRISPRi-v1, the Replogle 2022 dual-guide library and Xaira guides list every
guide × gene pair with a ≥ 10-nt PAM-proximal seed match at a promoter (±1 kb of any GENCODE v43 transcript TSS).
About two thirds of guides in the standard CRISPRi libraries have at least one ≥ 12-nt match far from their target; in two
genome-wide Perturb-seq screens such matched genes were knocked down 10–15% of the time vs 0.5% expected.

Build a map for your own library (genome streamed, no aligner needed):

```bash
perturbaudit seedmap --guides my_library.csv --fasta hg38.fa.gz --gtf gencode.v43.basic.annotation.gtf.gz --out my_map.csv.gz
```

`my_library.csv` needs `guide,target,protospacer` columns (20-nt protospacers; longer ones are trimmed to the 3' 20 nt).
