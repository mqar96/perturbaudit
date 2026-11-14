# Verdicts

An edge receives the first check it fails, in this order:

1. **untestable** — the parent was not perturbed, the child not measured, or the parent's knockdown is ineffective (own-gene |z| < 2).
2. **reversed** — unsupported, and knocking down the child moves the parent (|z| ≥ 5, q ≤ 0.05).
3. **unsupported** — knocking down the parent does not move the child (|z| < 3 and decoy q > 0.05).
4. **off-target** — the child is a seed-match off-target of the parent's guide, or is moved by knocking down that off-target gene.
5. **neighbour** — parent and child TSSs are within 10 kb (CRISPRi spreads locally).
6. **guide-specific** — one guide moves the child (|z| ≥ 3), the other does not (|z| < 1), heterogeneity P < 0.05.
7. **unstable** — the edge appears in fewer than half of resampled networks.

Otherwise the edge is **audited**. Effects are batch-matched z-scores using control-cell variance; q-values come from
target–decoy competition with held-out non-targeting cells.
