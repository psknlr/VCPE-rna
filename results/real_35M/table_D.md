# pearson_dev on a shared protocol

Every row below was produced with the same `data_dirs`, `n_hvg`, `hvg_from`, `split_by`, `test_frac`, `min_cells`, `seed`, `use_mask`, `esm_table`.

- `data_dirs` = `('/home/user/data_real/gears_fmt/tian2021_crispri.h5ad',)`
- `n_hvg` = `500`
- `hvg_from` = `train`
- `split_by` = `target_gene`
- `test_frac` = `0.15`
- `min_cells` = `1`
- `seed` = `(0,)`
- `use_mask` = `True`
- `esm_table` = `/home/user/data_real/esm2_35M_human.pt`

| model | pearson_dev | runs | sd | range | kind |
|---|---|---|---|---|---|
| CPA | +0.0180 | 1 | — | — | single run |
| P3 head | +0.0026 | 1 | — | — | single run |
| zero | +0.0000 | 1 | — | — | deterministic control |
| train_mean | -0.0023 | 1 | — | — | deterministic control |
| ridge_esm2 | -0.0031 | 1 | — | — | deterministic control |
| GEARS | -0.0049 | 3 | 0.0364 | -0.047..+0.017 | mean over runs |
| knn_esm2 | -0.0093 | 1 | — | — | deterministic control |

## What is not controlled

These come from each adapter's own `fairness` block. A number from this table quoted without them is misleading.

- D_gears.json: Tuning effort is not equalised: the external model runs at or near its defaults, while VCPE's hyperparameters were chosen against this data over many runs. That asymmetry favours VCPE and no protocol alignment removes it; the honest fix is an equal tuning budget for both.
- D_gears.json: Capacity is not equalised: the external models are generally far larger than the 5.7M-parameter VCPE head. If VCPE wins it wins as the smaller model; if it loses, the size difference is an explanation rather than an excuse.
- D_gears.json: Each external model keeps its own preprocessing (its own graph, its own highly-variable-gene or DE selection). Disabling that would not be a comparison against the published method.
- D_gears.json: GEARS's perturbation graph is GO-derived, so a target gene can be in the training split and still be unpredictable; the count of held-out genes present in the graph is reported above.
- D_gears.json: GEARS is broken on a current scipy/pandas and required a dense-X workaround scoped to its constructor; see the note in _dense_X. A GEARS comparison is not reproducible today without that or an older scipy, and which was used belongs next to this number.
- D_gears.json: GEARS does not reproduce its own result at a fixed seed, so its number here is a mean over --runs repetitions with the spread reported.
- D_gears.json: Input granularity differs on purpose. GEARS trained on individual cells (tian2021_crispri_cells20.h5ad) while the P3 head trains on the pseudobulk mean of the same cells; the split, the gene panel, the predicted quantity and the metric are identical. GEARS cannot run on pseudobulk at all -- its differential-expression step has no statistics for a group of one -- so the alternative was not a fairer comparison but no comparison.
- D_cpa.json: Tuning effort is not equalised: the external model runs at or near its defaults, while VCPE's hyperparameters were chosen against this data over many runs. That asymmetry favours VCPE and no protocol alignment removes it; the honest fix is an equal tuning budget for both.
- D_cpa.json: Capacity is not equalised: the external models are generally far larger than the 5.7M-parameter VCPE head. If VCPE wins it wins as the smaller model; if it loses, the size difference is an explanation rather than an excuse.
- D_cpa.json: Each external model keeps its own preprocessing (its own graph, its own highly-variable-gene or DE selection). Disabling that would not be a comparison against the published method.
- D_cpa.json: Input granularity differs on purpose: CPA trained on individual cells while the P3 head trains on the pseudobulk mean of the same cells. The split, panel, predicted quantity and metric are identical. CPA models variation within a condition, so a pseudobulk input would not be a fairer comparison, only a meaningless one.
