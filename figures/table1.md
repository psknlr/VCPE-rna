Table 1 | Held-out performance under two gene representations

| Method | ESM2 only (640-d) | + STRING neighbourhood (1,280-d) | Δ |
|---|---|---|---|
| Conditioned head | +0.2927 ± 0.0055 | +0.3346 ± 0.0014 | +0.0419 |
| Ridge regression (closed form) | +0.2653 ± 0.0203 | +0.3370 ± 0.0238 | +0.0717 |
| k-NN retrieval | +0.2115 ± 0.0211 | +0.2637 ± 0.0272 | +0.0523 |
| Training mean | +0.0121 ± 0.0024 | +0.0121 ± 0.0024 | +0.0000 |
| No change | +0.0000 | +0.0000 | +0.0000 |

Values are the mean Pearson *r* between predicted and observed residual response over held-out target genes, averaged across three splits; ± is the sample s.d. across splits. *r* is computed within each perturbation and then averaged, never pooled across perturbations. Both columns use the identical split, gene panel, measured-gene mask and estimator; only the gene representation differs, and it is shared by every method in the column. For the head, Δ compares 80 epochs on the enriched table with 30 epochs on ESM2 only; at the same 30-epoch budget it is +0.0364. The training mean predicts zero up to rounding in these runs, so it is a floor. n = 340 held-out target genes of 2,269 in each split (347-352 held-out perturbations); 3 splits.
