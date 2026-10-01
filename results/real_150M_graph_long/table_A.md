# pearson_dev on a shared protocol

Every row below was produced with the same `data_dirs`, `n_hvg`, `hvg_from`, `split_by`, `test_frac`, `min_cells`, `seed`, `use_mask`, `esm_table`.

- `data_dirs` = `('/home/user/data_real/gears_fmt/replogle_rpe1.h5ad', '/home/user/data_real/gears_fmt/tian2021_crispri.h5ad')`
- `n_hvg` = `500`
- `hvg_from` = `train`
- `split_by` = `target_gene`
- `test_frac` = `0.15`
- `min_cells` = `1`
- `seed` = `(0, 1, 2)`
- `use_mask` = `True`
- `esm_table` = `/home/user/data_real/esm2_150M_graph.pt`

| model | pearson_dev | runs | sd | range | kind |
|---|---|---|---|---|---|
| ridge_esm2 | +0.3370 | 3 | 0.0238 | +0.310..+0.353 | multi-seed mean |
| P3 head | +0.3346 | 3 | 0.0014 | +0.334..+0.336 | multi-seed mean |
| knn_esm2 | +0.2637 | 3 | 0.0272 | +0.238..+0.292 | multi-seed mean |
| train_mean | +0.0121 | 3 | 0.0024 | +0.010..+0.015 | multi-seed mean |
| zero | +0.0000 | 3 | 0.0000 | +0.000..+0.000 | multi-seed mean |

## Paired per-seed comparison against the P3 head

Pairing removes split-to-split variance, which is larger than the differences being compared.

| baseline | mean paired difference | verdict |
|---|---|---|
| ridge_esm2 | -0.0024 | mixed: model wins on 1/3 seeds |
| knn_esm2 | +0.0709 | model beats this baseline on every seed |
| train_mean | +0.3225 | model beats this baseline on every seed |
| zero | +0.3346 | model beats this baseline on every seed |
