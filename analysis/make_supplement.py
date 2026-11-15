"""Supplementary Information v2, generated from the frozen v0.2.1 outputs and the carried-over tables.
Usage: python make_supplement_v2.py OUT_MD"""
import os, sys
import numpy as np, pandas as pd

out = sys.argv[1]
P = os.environ.get("PA_WORKDIR", "/home/claude/p02/")
FZ, D = P + "derived/frozen/", P + "derived/"
SIM_AUDIT = P + "simrerun/out/sim_audit_results.csv"
SIM_POWER = P + "powrerun/sim_power.csv"
LPO = P + "derived/lpo_v021/"
md = lambda df: df.astype(object).where(df.notna(), "–").to_markdown(index=False, disable_numparse=True)
NAMES = {"adapre": "ADAPRE", "causalgrn": "CausalGRN", "dotears": "dotears", "gbm_grn": "GBM-GRN", "gies": "GIES",
         "inspre": "inspre (ρ = 10)", "inspre_default": "inspre (ρ = 100)", "inspre_dag": "inspre (DAG)", "sdcd": "SDCD",
         "sdcd_full": "SDCD (all cells)", "tce": "TCE (control)"}
PANELS = {"gwps_p788": "K562 GW, 788", "gwps_p1434": "K562 GW, 1,434", "k562_p300e": "K562 ess., 300",
          "k562_p100cg": "K562 ess., 100 (CausalBench)", "k562_p50": "K562 ess., 50", "rpe1_p100e": "RPE1, 100",
          "adamson_p100": "Adamson", "norman_p100": "Norman", "tian21i_p100": "Tian CRISPRi", "tian21a_p100": "Tian CRISPRa"}
ORDER = list(PANELS)
L = []

# ---------- simulation summaries ----------
sa = pd.read_csv(SIM_AUDIT)
net = sa[sa.method.isin(["dotears", "gbm_grn", "inspre"])]
g = net.groupby(["setting", "method"])[["precision_audited", "precision_flagged", "true_kept"]].mean()
tce = sa[sa.method == "tce"].groupby("setting")[["precision_audited", "precision_flagged", "desc_precision_audited", "n_flagged"]].mean()
sim_txt = dict(runs=len(sa), aud_lo=g.precision_audited.min(), aud_hi=g.precision_audited.max(),
               fl_lo=g.precision_flagged.min(), fl_hi=g.precision_flagged.max(), kept_lo=g.true_kept.min(), kept_hi=g.true_kept.max())
pw = pd.read_csv(SIM_POWER).groupby(["mode", "strength", "cells"])[["sensitivity", "false_flag_true"]].mean().reset_index()
m4, m2 = pw[pw["mode"] == "M4_seedmap"], pw[pw["mode"] == "M2_guides"]
m4_strong = m4[m4.strength >= 0.8]
m2_rest = m2[~((m2.strength == 0.4) & (m2.cells == 50))]

L.append("""---
title: "Supplementary Information — Auditing causal gene networks inferred from Perturb-seq with the screens' own controls"
author: "Muhammad Qudduse Ahmad Rao"
---

Contents: Supplementary Notes 1–7; Supplementary Tables S1–S7; Supplementary Figures S1–S7. Tables are generated from the released result files by `make_supplement_v2.py`. Unless stated otherwise, audits use perturbaudit 0.2.1.

# Supplementary Note 1. Knockdown statistic and decoy behaviour

In the K562 essential screen (2,134 perturbations × 7,015 expressed genes, 15.0 M pairs), a Welch z produced a heavy decoy tail: P(|z| > 5) was 2.4 × 10^−4^, against 5.7 × 10^−7^ for a Gaussian. About 495 decoy values exceeded |z| = 20, all caused by groups with zero variance in genes with low expression. The null-variance z gave a Gaussian-like decoy tail (2.1 × 10^−6^); the target–decoy estimate was 0.28%% at |z| > 4 and about 0 at |z| > 5. Batch-matched and unmatched decoys behaved alike in this screen, where GEM groups are balanced. This analysis builds its own decoy groups from per-element statistics (`run_m1_k562_essential.py`) and does not use the code changed in version 0.2.1.

**Batch references (version 0.2.1).** Version 0.2.0 treated a batch with no NT cells as a reference with zero mean and a variance floor of 10^−12^, which could turn noise into an extreme z, and it split NT cells into reference and decoy halves across the whole screen rather than within each batch. Version 0.2.1 requires at least 20 NT cells per reference batch (10 for the half-reference used by decoys), excludes and counts perturbed cells in batches without a valid reference, returns no statistic for genes that are constant in a batch's controls, and splits NT cells within each batch, with each decoy copying a real perturbation's batch composition. Table S2b gives the coverage of every panel; the change affected 67 of 2,544 top-50 verdicts.

**Decoy construction for network methods.** An early design created decoy variables by an additive shift with clipping at zero, which left some decoys with zero variance in their own cells and broke variance-weighted methods such as dotears. The released design resamples the template gene's own knocked-down expression. A preflight check (`preflight.own_intervention_variance`) flags genes whose own-intervention variance is below 5%% of the NT variance.

# Supplementary Note 2. Simulation design and results

**Model.** Linear structural equation models on random DAGs (edge density 0.06) with t-distributed noise, optional latent confounders and CRISPRi-like knockdowns.

**Injected violations.** Seed off-target guides (0–25%%), inactive guides (30%%) and ambient misassignment (5%%).

**Experiments.**

- Calibration of decoy-variable FDR: TCE network, p = 20, 40 and 80, 9 settings, 50 replicates each; dotears, p = 40, 5 settings, 10 replicates.
- End-to-end audit: p = 40, 6 settings × 20 replicates × 4 methods (%(runs)d runs), top 40 edges.
- Power of the off-target and guide checks: off-target strength 0.4–2.4 × 50, 150 or 400 cells per perturbation, off-target effects in guide-1 cells only, 10 replicates per cell.

**Truth.** Direct edges for network methods; descendants for TCE, which estimates total effects.

**Seeds and machines.** Seeds are derived with `zlib.crc32` or fixed integers, so every replicate is reproducible. A calibration replicate gave identical tables on Linux and Windows. Of the 400 benchmark networks, 276 were fitted in a Linux cloud workspace and 124 on the author's Windows laptop (`sim_fit_networks.py`); the two networks fitted on both machines differed by at most 1.2 × 10^−4^. All audits of the benchmark ran on one machine from the cached networks.

**Calibration (Fig. S1).** For TCE without systematic violations, mean permutation and knockoff estimates were within 0.11 of the mean true FDP for every graph size and list length. With 25%% off-target guides, the true FDP of the top 20 edges was 0.44–0.51 while both estimates were 0.00–0.07. For dotears, permutation decoys reported 0 in every setting while the true FDP of the top 20 edges was 0.44–0.72; knockoff estimates were 0.20–0.45, below the truth in all five settings.

**End-to-end audit (Fig. S2).** For the three network methods (dotears, GBM-GRN, inspre), mean precision across settings was %(aud_lo).2f–%(aud_hi).2f for audited edges and %(fl_lo).2f–%(fl_hi).2f for flagged edges, and the audit kept %(kept_lo).2f–%(kept_hi).2f of the true edges among the top 40. TCE flags few edges (mean 0–4 per run), so its flagged precision rests on small numbers; its audited precision against descendants is shown in Fig. S2.
""" % sim_txt)
L.append(f"""**Power (Fig. S3).** With a seed map, the off-target check flagged {m4_strong.sensitivity.min():.2f}–{m4_strong.sensitivity.max():.2f} of supported artefact edges at off-target strength ≥ 0.8, and {m4[(m4.strength == 0.4)].sensitivity.min():.2f}–{m4[(m4.strength == 0.4)].sensitivity.max():.2f} at strength 0.4 (lowest with 50 cells per perturbation); false flags on true edges were {m4.false_flag_true.min():.2f}–{m4.false_flag_true.max():.2f}. Without a map, the guide-specific check flagged {m2[(m2.strength == 0.4) & (m2.cells == 50)].sensitivity.iloc[0]:.2f} at strength 0.4 with 50 cells and {m2_rest.sensitivity.min():.2f}–{m2_rest.sensitivity.max():.2f} otherwise, with {m2.false_flag_true.min():.2f}–{m2.false_flag_true.max():.2f} false flags.
""")

L.append("""# Supplementary Note 3. Method settings and deviations

- **TCE:** the knockdown z network, a positive control rather than a competitor.
- **inspre:** R package; ρ = 10, the package default ρ = 100, and DAG = TRUE; 20 lambdas. The edge score is the number of lambdas at which the edge is non-zero, plus its relative magnitude.
- **ADAPRE:** authors' code with adaptive_lambda = TRUE. In the public `fit_inspre_from_X`, the unfiltered `beta_obs` is passed to the solver. The solver applies the adaptive penalty only if the length of `beta_obs` equals the number of genes left after TCE filtering, so whenever filtering drops genes the public code returns exactly inspre's result (we verified identical matrices). Our wrapper passes `beta_obs[rownames(R_hat)]`; setting ADAPRE_FIX = 0 reproduces the public behaviour.
- **SDCD:** sdcd 0.1.4. Two set-ups: 300 epochs on CPU with 60 cells per perturbation plus 3,000 controls, or 600 epochs on GPU with all panel cells. On RPE1 both were run (the GPU run is labelled "all cells"); the K562 and genome-wide networks are GPU runs.
- **dotears:** authors' code, lambda1 = 0.1, z-scored input.
- **GBM-GRN:** gradient-boosting regression of each gene on all others, scored by feature importance. It replaces GRNBoost2, whose arboreto/dask stack no longer installs.
- **CausalGRN and GIES:** run at p = 50 and p = 100 with 60 cells per perturbation plus 3,000 NT cells.
- **ARGEN:** published edges from the authors' Zenodo release (22052266), BH < 0.05, judged against GWPS pseudo-bulk z (z = normalised bulk × √cells) with |z| ≥ 3 for support and |z| ≥ 5 for reversal, without decoys. Because ARGEN estimates direct effects within chromosome panels, "unsupported" means no total-effect evidence, not proof of absence.

**Wrapper orientation.** Checked on a simulated DAG with p = 40: inspre top-k precision was 0.80 as oriented vs 0.07 transposed; ADAPRE gave the same.

**Panels.** K562 essential and RPE1: 50–300 genes chosen as the strongest effective knockdowns (the 100-gene CausalBench panel follows CausalBench's gene set). GWPS: 788 and 1,434 genes, chosen by inspre's selection rule. Other screens: 64–96 genes.

**Deviations from the authors' pipelines.** Our inspre runs used a cell subsample (at most 100 cells per gene) and our wrapper's defaults rather than the authors' full pipeline with cross-validation. Claims concern the edges each run produced, checked against the screen's interventions.
""")

# ---------- Note 4: M6 ----------
st = pd.concat([pd.read_csv(D + "panels/k562_p300e/m6_stress_inspre.csv")] + [pd.read_csv(D + f"m6/m6_stress_{m}.csv") for m in ["sdcd", "dotears", "gbm_grn"]])
ra = pd.read_csv(D + "m6_v021/m6_reaudit_summary.csv")
ret = st.groupby(["method", "kind", "rate"]).retained.mean().unstack(["kind", "rate"]).round(2)
ret.columns = [f"{k} {int(r * 1000) / 10:g}%" for k, r in ret.columns]
ret = ret.reset_index().replace({"method": NAMES}).rename(columns={"method": "Method"})
fr = ra[["method", "n_edges", "verdicts_changed", "fragile_all", "fragile_audited", "fragile_unsupported", "fragile_reversed"]].replace({"method": NAMES})
fr.columns = ["Method", "Edges", "Verdicts changed", "Fragile (all)", "Audited", "Unsupported", "Reversed"]
L.append("""# Supplementary Note 4. Robustness to misassignment and doublets (M6)

On the K562 essential 300-gene panel we reassigned 1.5–9% of cells to random labels, or added 5–10% simulated doublets, refitted each method and measured how many of its top edges (top 200; SDCD returned 60) were retained. inspre has three replicates per setting; the other methods one. A noise-free reseeding control kept 0.87–0.895 of GBM-GRN's edges, so resampling explains 10–13% of its loss.

Retention by setting:

""" + md(ret) + """

Fragile edges (retained in fewer than half of the stress runs) by verdict of the clean network, re-audited with version 0.2.1 (no verdict changed):

""" + md(fr) + """

Fragility was not concentrated among flagged edges, so robustness and verdict are reported as separate axes.
""")

# ---------- Note 5: LPO ----------
lpo_txt = "⟦LPO pending⟧"
lpo_tab = ""
if os.path.isdir(LPO) and os.listdir(LPO):
    from scipy.stats import wilcoxon
    rows, parts = [], []
    for m in ["gbm_grn", "sdcd"]:
        f = os.path.join(LPO, f"m5_lpo_{m}.csv")
        if not os.path.exists(f):
            continue
        R = pd.read_csv(f)
        pv = R.pivot_table(index="fold", columns="network", values="precision")
        a, u, r = pv["audited"], pv["unaudited_matched"], pv["random_matched"]
        p = wilcoxon(a, u).pvalue
        rows.append({"Method": NAMES[m], "Folds": len(pv), "Audited": round(a.mean(), 3), "Unaudited": round(u.mean(), 3),
                     "Random": round(r.mean(), 3), "Audited higher": int((a > u).sum()), "Wilcoxon P": round(p, 3),
                     "Audited edges": round(R[R.network == "audited"].n_edges.mean(), 1)})
    lpo_tab = md(pd.DataFrame(rows))
    lpo_txt = "Precision of the responders implied for held-out knockdowns, audited vs unaudited networks of matched size (mean over folds):"
L.append(f"""# Supplementary Note 5. Leave-perturbation-out (LPO)

We used 10 folds over the perturbed genes of the K562 essential 300-gene panel. For each fold the method was fitted without the held-out genes' cells, audited on the training screen only (top 300 edges), and asked which genes respond when a held-out gene is knocked down (its descendants). Responders were genes with |z| ≥ 3 in the held-out cells; non-responders |z| < 2. The unaudited comparison network takes the same number of edges from the top of the method's own ranking; the random network has the same size. {lpo_txt}

{lpo_tab}

LPO cannot be applied to TCE, inspre or ADAPRE, which give a held-out gene no outgoing edges by construction, and dotears was too slow to refit ten times on this panel.
""")

L.append("""# Supplementary Note 6. Scoring perturbation-response models

**GWPS baselines.** 559 held-out perturbations (20% of the 2,798 GWPS perturbations with a transcriptional effect, energy test p < 0.001) were predicted from the other perturbations. The edges of a model are its 20 most extreme predicted children per held-out perturbation; an edge is correct if the measured child lies in the top or bottom 1% across perturbations with the predicted sign (chance about 1%).

**GEARS.** We used cell-gears (MIT licence) on the K562 essential data provided with GEARS (5,000 genes), its standard 'simulation' split, seed 1, hidden size 64 and 20 epochs. There were 733 training and 272 unseen single perturbations. Two scores were used:

- **Generic precision:** children ranked by their predicted change from control; an edge is correct if the measured |z| ≥ 3 with the predicted sign.
- **Perturbation-specific precision:** children ranked by their deviation from the mean training response; an edge is correct if the measured |z| ≥ 3 and the measured deviation from the mean response has the predicted sign.

Chance rates come from random children with random signs. GEARS was run in a separate environment; five compatibility issues in its code on Windows were fixed there without changing the model. A first in-run summary used an older scoring function (0.350); it is superseded by `vcm_gears_score.py`. These analyses do not use the code changed in version 0.2.1.

# Supplementary Note 7. Reproducibility and provenance

The paper's network audits, chance baselines, transfer tests, simulations and robustness re-audits were frozen or re-run with perturbaudit 0.2.1. Results that do not use the batch-matched reference or decoys (seed scans, guide-replication tables, neighbour enrichment, published-edge audits and prediction-model scores) were carried over from version 0.2.0. The target-line effect matrices were built with version 0.2.0. The RPE1 matrix uses pooled controls and was regenerated with 0.2.1 (identical within float16 rounding); the HepG2, Jurkat and HCT116 matrices were built from raw data that were deleted after processing, so we checked all four for the two failure modes fixed in 0.2.1. No matrix had non-finite values or rows with widespread extreme values (the signature of a batch without controls), and among the 0.08–1.9 million gene pairs of the K562 panels used for transfer, at most 53 per target had off-diagonal |z| ≥ 50, mostly highly expressed stress or ribosomal genes (`target_effects_check.csv`). The cross-platform pair table uses pooled controls and was verified to be identical under 0.2.1.

Each data record contains `PROVENANCE.csv` (for every file: full SHA-256, producing script, command, inputs and code version, with the reason when a file was carried over) and `ENVIRONMENT.json` (code commit and package versions). `REPRODUCE.md` in the repository gives the commands. The GWPS 788-gene panel was rebuilt on a separate machine (free Google Colab) from the public file and gave an identical checksum.

Compute used: a 2-CPU cloud workspace, the author's laptop (RTX 5070 Laptop GPU, 24 threads, 34 GB RAM) and free Google Colab.
""")

S1 = pd.DataFrame([
    ["Replogle 2022 K562 essential", "CRISPRi, K562", "310,385", "figshare / scPerturb", "CC BY 4.0"],
    ["Replogle 2022 K562 genome-wide (GWPS)", "CRISPRi, K562", "1,989,578", "scPerturb (Zenodo 7041849); figshare", "CC BY 4.0"],
    ["Replogle 2022 RPE1 essential", "CRISPRi, RPE1", "247,914", "figshare", "CC BY 4.0"],
    ["Nadig 2025 HepG2 / Jurkat", "CRISPRi", "145,473 / 262,956", "GEO GSE264667", "public (GEO)"],
    ["Zhu 2026 primary CD4+ T cells", "CRISPRi, 2 guides/gene, 4 donors", "~22 M (guide-level statistics)", "authors' data release", "see source"],
    ["Frangieh 2021", "Cas9 KO, melanoma", "218,331", "scPerturb", "per source"],
    ["Papalexi 2021", "Cas9 KO, THP-1", "20,729", "scPerturb", "per source"],
    ["Adamson 2016", "CRISPRi, K562 (UPR)", "65,337", "scPerturb", "per source"],
    ["Norman 2019", "CRISPRa, K562", "111,445", "scPerturb", "per source"],
    ["Tian 2021", "CRISPRi / CRISPRa, iPSC neurons", "32,300 / 21,193", "scPerturb", "per source"],
    ["Replogle 2022 eLife", "CRISPRi, K562, dual and single sgRNA", "60,049 + 64,914", "GEO GSE205310", "CC BY 4.0"],
    ["Bradu 2026 (VIPerturb-seq)", "CRISPRi, K562, 3 guides/gene", "326,247", "Zenodo 18460279", "CC BY 4.0"],
    ["Huang 2025 X-Atlas/Orion", "CRISPRi, HCT116, genome-wide", "~3.2 M (18,294 perturbations)", "Hugging Face Xaira-Therapeutics/X-Atlas-Orion", "CC BY-NC-SA 4.0"],
], columns=["Screen", "Modality / cells", "Cells used", "Source", "Licence"])
L.append("# Supplementary Table S1. Screens analysed\n\n" + md(S1) + "\n\nAnnotations: GENCODE v43 basic transcripts; hg38 (UCSC); Hartman *et al.* (2026) candidate seed off-target tables.\n")

# ---------- Table S2: verdicts and coverage ----------
a = pd.read_csv(FZ + "audited_vs_chance.csv")
a["pi"] = a.panel.map({p: i for i, p in enumerate(ORDER)})
a = a.sort_values(["pi", "method"])
s2 = pd.DataFrame(dict(Panel=a.panel.map(PANELS), Network=a.method.map(NAMES), Edges=a.top_k, Test=a.n_testable,
                       Aud=(a.audited * a.n_testable).round().astype("Int64"), Unsup=a.unsupported, Rev=a.reversed,
                       Neigh=a.neighbour, Guide=a.guide_specific, WeakKD=a.untestable_ineffective_knockdown,
                       NoRef=a.untestable_no_reference))
s2.loc[a.n_testable.values == 0, "Aud"] = pd.NA
cov = pd.read_csv(FZ + "reference_coverage.csv")
cov["pi"] = cov.panel.map({p: i for i, p in enumerate(ORDER)})
cov = cov.sort_values("pi").drop(columns="pi")
cov.insert(0, "Panel", cov.pop("panel").map(PANELS))
cov.columns = ["Panel", "Batches", "Batches < 20 NT", "Min NT per batch", "Perturbed cells", "Cells excluded", "Groups untestable"]
L.append("# Supplementary Table S2. Verdicts of the top-50 edges of every network\n\n(a) Counts among each network's top 50 edges (all edges if fewer). Test = testable; Aud = audited; Unsup = unsupported; Rev = reversed; Neigh = neighbour; Guide = guide-specific; WeakKD = untestable (weak knockdown); NoRef = untestable (no reference). The neighbour check was applied everywhere and the guide check where the panel records independent guides (Norman, Tian CRISPRi and CRISPRa); the off-target check, which needs a seed map, was used in the separate off-target analyses, not here. inspre returned no edges on Norman and Tian CRISPRa.\n\n" + md(s2) + "\n\n(b) Reference coverage per panel (version 0.2.1).\n\n" + md(cov) + "\n")

# ---------- Table S3: enrichment ----------
s3 = pd.DataFrame(dict(Panel=a.panel.map(PANELS), Network=a.method.map(NAMES), **{"Audited fraction": a.audited.round(3),
                       "Chance (all pairs)": a.chance_audited.round(4), "Testable pairs": a.testable_pairs, "Enrichment": a.enrichment}))
L.append("# Supplementary Table S3. Enrichment of audited edges over the all-pairs chance rate\n\nChance = audited fraction of all ordered gene pairs in the panel, computed with the same screen object, decoys and TSS annotation as the networks.\n\n" + md(s3) + "\n")

# ---------- Table S4: GWPS networks ----------
gw = a[a.panel.str.startswith("gwps")]
s4 = pd.DataFrame(dict(Network=gw.method.map(NAMES), Panel=gw.panel.map(PANELS), Edges=gw.top_k,
                       Audited=[f"{x:.0%}" for x in gw.audited], Unsupported=gw.unsupported, Reversed=gw.reversed,
                       Neighbour=gw.neighbour, Enrichment=gw.enrichment))
argen = pd.DataFrame([["ARGEN published (BH < 0.05)", "GWPS (pseudo-bulk)", 1983, "36.3%", 1070, 175, "", ""],
                      ["ARGEN random set", "GWPS (pseudo-bulk)", 20796, "8.9%", 17951, 650, "", ""]], columns=s4.columns)
L.append("# Supplementary Table S4. Genome-wide (GWPS) networks\n\nTop-50 edges (SDCD returned 24 and 35). ARGEN rows use the rule-based pseudo-bulk audit of Supplementary Note 3 and are not directly comparable.\n\n" + md(pd.concat([s4, argen])) + "\n")

# ---------- Table S5: transfer ----------
t = pd.concat([pd.read_csv(FZ + "transfer_matched_3lines.csv"), pd.read_csv(FZ + "transfer_matched_hct116.csv")])
t5 = pd.DataFrame(dict(Target=t.target, Source=t.source.map(PANELS), Network=t.method.map(NAMES), K=t.top_k,
                       **{"nA": t.n_audited, "nF": t.n_flagged, "Aud": t.audited.round(2), "Flag": t.flagged.round(2),
                          "P": t.fisher_p.map(lambda x: "" if pd.isna(x) else (f"{x:.0e}".replace("e-0", "e-") if x < 0.001 else f"{x:.3f}")),
                          "CI": [("" if pd.isna(l) else f"{l:.2f}/{h:.2f}") for l, h in zip(t.diff_ci_lo, t.diff_ci_hi)],
                          "Rand": t.random_same_sign.round(2), "Null": t.matched_null.round(2), "Rank": t.effect_rank.round(2)}))
L.append("# Supplementary Table S5. Cross-cell-line transfer of K562 edges\n\nReplicated = |z| ≥ 3 with the K562 sign in the target line. K = number of top-ranked edges tested; nA, nF = audited and flagged (unsupported or reversed) edges testable in the target; Aud, Flag = their replication rates. P: Fisher's exact test. CI: lower/upper 95% bound of the difference Aud − Flag, bootstrap over parents. Rand: random same-sign panel pairs. Null: random pairs matched to the audited edges' K562 |z| (bins 3–5, 5–10, 10–20, > 20). Rank: the same number of pairs chosen by K562 |z|. TCE rows have no flagged edges.\n\n" + md(t5) + "\n")

# ---------- Table S6: seed maps ----------
s = pd.read_csv(D + "offtarget_maps/SUMMARY.csv").drop(columns=["file"]).rename(columns={
    "pairs_ge10": "pairs ≥10 nt", "distant_ge11": "distant ≥11", "distant_ge12": "distant ≥12", "guides_with_distant_ge12": "guides w/ distant ≥12",
    "expected_distant_knockdowns_K562": "exp. distant KD (K562)", "frac_guides_with_distant_ge12": "frac. guides"})
L.append("# Supplementary Table S6. Seed off-target maps\n\nPairs with a ≥ 10-nt PAM-proximal seed match within ±1 kb of any GENCODE v43 transcript TSS; 'distant' = more than 1 Mb from the target. Expected distant knockdowns use the K562 GWPS rates per seed length. The hCRISPRa-v2 entries are binding candidates; CRISPRa effects were not calibrated. The X-Atlas/Orion guide map is in the CC BY-NC-SA record.\n\n" + md(s) + "\n")

v = pd.read_csv(D + "vcm/vcm_audit_summary.csv").rename(columns={"test_perturbations": "n test", "edge_precision_at20": "prec@20", "edge_precision_sd": "sd",
    "pearson_delta": "Pearson δ", "distinct_children_predicted": "distinct children", "top_child_share": "top child share"})
ge = pd.read_csv(D + "vcm/gears_audit_final.csv").round(3).rename(columns={"edge_precision_at20": "generic prec@20", "specific_edge_precision_at20": "specific prec@20",
    "chance_generic": "chance (generic)", "chance_specific": "chance (specific)"})
L.append("# Supplementary Table S7. Perturbation-response model audit\n\n(a) GWPS held-out perturbations\n\n" + md(v.round(3)) + "\n\n(b) GEARS, K562 essential, standard simulation split (Pearson δ on top-20 DE genes: GEARS 0.539, mean baseline 0.561, linear 0.558, kNN 0.566)\n\n" + md(ge) + "\n")

figs = [("fig_h1h2_calibration", "Decoy-variable FDR estimates in simulation (TCE network, 50 replicates per point). (A) Settings without systematic artefacts, three graph sizes. (B) True FDP and permutation-decoy estimate of the top 20 edges as the share of off-target guides rises."),
        ("fig_sim_audit", "End-to-end audit benchmark (p = 40; 6 settings × 20 replicates × 4 methods): precision of all top-40, audited and flagged edges; 'keeps' = share of true top-40 edges that pass the audit. Truth is direct edges for network methods and descendants for TCE."),
        ("fig_h3_power", "Power of the off-target check (with seed map) and the guide-specific check (no map) by off-target strength and cells per perturbation (10 replicates)."),
        ("fig_m3_neighbour_enrichment", "Neighbour enrichment of strong effects (|z| ≥ 5) by TSS distance in four CRISPRi and two Cas9 knockout screens. Three further CRISPRi screens were analysed after this panel was drawn (0–1 kb enrichment): HepG2 12.3× (84 of 486 pairs), Jurkat 35.7× (166/492) and HCT116 375× (450 of 112,915 strong effects); together seven CRISPRi screens, 9–894×. Knockout screens show no excess."),
        ("fig_m2_guide_vs_donor", "CD4+ T cells: replication between independent guides vs between donor halves."),
        ("fig_gwps_inspre_case", "inspre on the K562 genome-wide 788-gene panel (version 0.2.1). (A) Verdicts of testable edges by list length, without (N) and with (D) the DAG constraint. (B) inspre's top hub RPL3: forward (RPL3 knockdown) and reverse (child knockdown) z. (C) Share of inspre's top-500 edges, by K562 verdict, for which the child's knockdown moves the parent with the K562 sign (|z| ≥ 3) in RPE1, HepG2 and Jurkat."),
        ("fig_concordance_4lines", "Raw concordance of strong K562 GWPS effects (|z| ≥ 5) with each target line.")]
for i, (f, cap) in enumerate(figs, 1):
    L.append(f"# Supplementary Figure S{i}\n\n![](figures/{f}.png){{width=95%}}\n\n**Supplementary Fig. S{i}.** {cap}\n")
open(out, "w").write("\n".join(L))
print("ok")
