# Verdicts

All evidence is **total-effect** evidence from the screen's own knockdowns. A verdict says what the screen's
controls show about an edge; it does not by itself establish a direct mechanism or the true causal direction.

An edge receives the first check it fails, in this order:

1. **untestable_no_reference** — the parent's cells sit only in batches with fewer than `min_ref_cells` (default 20)
   non-targeting cells, or the child is constant in the matched controls. Perturbed cells in batches without a valid
   reference are excluded from the statistics and counted in the reference-coverage report.
2. **untestable_parent_not_perturbed / untestable_child_not_measured.**
3. **untestable_ineffective_knockdown** — the edge is unsupported *and* the parent's own-gene |z| < 2. A supported edge
   stays supported even with a weak own-gene z, because its child moved. `kd_status` records `effective`,
   `ineffective` or `not_measured` (the parent's own transcript is not in the matrix).
4. **reversed** — unsupported, and knocking down the child moves the parent (|z| ≥ 5 and decoy q ≤ 0.05). This flags
   the edge for scrutiny; it is not proof that the true direction is reversed.
5. **unsupported** — knocking down the parent does not move the child (|z| < 3 and decoy q > 0.05).
6. **off-target** — the child is a seed-match off-target of the parent's guide, or is moved by knocking down that
   off-target gene (needs an off-target map).
7. **neighbour** — parent and child TSSs are within 10 kb (needs a TSS annotation). CRISPRi spreads locally.
8. **guide-specific** — of the parent's two best-covered guides, one moves the child (|z| ≥ 3) and the other does not
   (|z| < 1), with an unadjusted heterogeneity P < 0.05 (`guide_het_p`; needs independent guides).
9. **unstable** — the edge appears in fewer than half of resampled networks (needs replicate networks).

Otherwise the edge is **audited**. The `checks_run` column lists which optional checks (6–9) could be applied to that
edge; an audited edge has passed exactly those checks. Report it alongside the verdict.

**Statistics.** Effects are batch-matched z-scores that use control-cell variance. **Decoys** are batch-matched: the
non-targeting cells of each batch are split into a reference half and a decoy pool, and each decoy copies the cell
number and batch composition of a real perturbation. The decoy q-value is a diagnostic of how often control-only
groups reach a given |z|; support is |z| ≥ 3 *or* q ≤ 0.05, so it is not a formal error-rate guarantee for selected edges.

**Chance baseline.** The audited fraction depends on how densely a panel's genes affect each other. Compare it with
the audited fraction of all ordered gene pairs in the same panel (`analysis/freeze_audits.py` computes both with
identical settings); the CLI reports verdicts only.
