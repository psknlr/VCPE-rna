# Errata

This file records defects found in the evaluation code of releases up to and
including v3, the published numbers each one invalidates, and what was changed.
It exists because several headline figures in earlier versions of the README
cannot be reproduced or do not measure what their labels claim.

**Summary: every performance number published before v4 should be treated as
withdrawn pending a re-run with the corrected code.** The corrected numbers are
expected to be lower. Nothing in this file is a statement about the *direction*
of the underlying biology; it is a statement about what the metrics measured.

Each entry gives the defect, the file and line where it lived, how it was
verified, and its status.

---

## E1 — Per-screen Spearman was computed against a feature, not the label

**Severity: invalidates the ASO head-to-head comparison.**

`train_efficacy_v2.py` built a tuple of (features..., label), extracted the
label, then removed it:

```python
out.append(torch.tensor(part["inhibition"].values, ...))   # label appended last
y_te = sb[-1]                                              # label extracted
sb = sb[:-1]                                               # label removed
...
screen_rhos.append(spearmanr(sb[-1].numpy()[sel], preds[sel]).statistic)
```

After the removal, `sb[-1]` is the last **feature** tensor — with region
features enabled, `occ_log = log1p(n_occurrences)`. The per-screen statistic
therefore measured the correlation between an input feature and the model's own
predictions, which is unrelated to accuracy.

This was the only per-screen code path in the repository, and its output fed
`per_screen_median` in `results/efficacy_v2/cv_results*.json`, reported in the
README as **"per-screen median 0.261"** and compared against OligoAI's 0.419.
OligoAI reports a per-screen median, so that comparison rested entirely on this
number.

*Verified by:* `tests/test_metrics.py::test_per_screen_uses_labels_not_features`,
which constructs a model that tracks the feature while being anti-correlated
with the truth; the buggy estimator reads >0.9, the correct one <0.4.

*Status:* fixed — `metrics.per_screen_spearman` takes labels explicitly. **The
published per-screen numbers are withdrawn.**

---

## E2 — Top-5% enrichment was scaled by 5 instead of 20

**Severity: 4x understatement; hides a negative result.**

`train_efficacy.py:110` (and two sibling copies):

```python
enrich = len(top_true & top_pred) / max(1, len(top_true)) * 5  # 1.0 = random
```

With `k = n/20` selected on each side, the expected intersection fraction under
a random ranking is `k/n = 0.05`. Multiplying by 5 puts chance at **0.25**, not
at the 1.0 the comment asserts. The correct factor is 20.

Consequences:

* The docstring's target bar ("5.72x random") was on a different scale from the
  numbers being compared to it.
* Re-reading the committed `results/efficacy_v1/train_log.jsonl` on the correct
  scale:

  | slice | Spearman | `enrich_top5` (as logged) | on the corrected scale |
  |---|---|---|---|
  | `val_patent_group` (seen genes) | 0.499 | 1.169 | **≈4.7x random** |
  | `gene_holdout` (15 unseen genes, n=31,147) | 0.292 | 0.2441 | **≈0.98x random** |

  On unseen target genes, top-5% selection performed **indistinguishably from
  chance**. This result was present in the committed log and was not mentioned
  anywhere in the README, while `predict_service.py` described the same model as
  extrapolating to new targets via the ESM2 axis.

*Verified by:* `tests/test_metrics.py::test_random_ranking_gives_enrichment_of_one`
and `::test_legacy_factor_five_understates_by_four` (300-replicate Monte Carlo).

*Status:* fixed in `metrics.enrichment_top5`; the gene-holdout result is now
reported in the README.

---

## E3 — Reported epochs were selected on the slice being reported

**Severity: all pre-v4 headline numbers are optimistically biased.**

Four scripts — including the main response line — selected the reported epoch
using the very data they then reported:

| file | line | what it did |
|---|---|---|
| `train_p3.py` | 351 | saved the checkpoint on the best **test-split** `pearson_dev`, and re-scored the test split every epoch |
| `train_efficacy_v2.py` | 203 | `best = max(best, m["spearman"])` where `m` is the held-out fold |
| `train_sirna_v1.py` | 105–106 | kept the best test-fold epoch **and its predictions**, then pooled those |
| `train_efficacy.py` | 195 | saved the checkpoint on `val_group`, then reported `val_group` |

None of the four carved out an inner validation split. For `train_p3.py` this
means every published `pearson_dev` — including the 0.3079 that cleared the
G2''' gate "by a hair" — is a maximum over epochs rather than an estimate;
`results/p3_v21e/train_log.jsonl` shows the per-epoch value oscillating between
0.153 and 0.276 across 40 epochs, so the gap is not negligible. The per-epoch
conditioning ablation also read the test split and has been moved to the inner
split.

Arithmetic confirmation for the ASO line (recomputed from the committed JSON):

```
v2.5 reported pooled_mean      : 0.2670
     mean(pooled_spearman)     : 0.2552
     mean(best_over_epochs)    : 0.2670   <-- exact match
```

So the published v2.5 figure is the mean of per-fold **best-over-epochs on the
test fold**, not the mean of the per-fold pooled values. The same holds for the
siRNA line, where `cv_results.json` reports `internal_cv_reference: 0.6387`
(the mean of per-fold bests) alongside a pooled 0.607 that is itself assembled
from best-epoch predictions.

*Status:* fixed — every script now selects on a grouped inner split of the
training data and touches the held-out set exactly once. **Published v1 (0.50),
v2.5 (0.267), siRNA CV (0.607 / 0.6387) and all P3 `pearson_dev` figures are
withdrawn.**

---

## E4 — `pearson_dev` named two different estimators, and the two were subtracted

**Severity: invalidates the headline "0.31 → 0.71" improvement.**

| file | line | estimator |
|---|---|---|
| `train_p3.py` | 242–246 | mean over perturbations of the within-perturbation r |
| `eval_fair.py` | 107–110 | one r over the **flattened** `[n_pert, n_hvg]` matrix |

Both printed under the bare name `pearson_dev`. The published improvement
subtracts the first (0.3079, from the P2.3-B report) from the second (0.7126,
from the data-expansion report).

The pooled estimator does not centre per perturbation, so a correctly predicted
between-perturbation **level** contributes to it while contributing nothing to
the per-perturbation estimator. A model with no within-profile signal at all can
score high on one and ~0 on the other.

This is also the explanation for the discrepancy recorded as an unresolved
backlog item in `data_expansion_report.md` ("in-training evaluate 0.2076 vs
eval_fair 0.3141 on the same ckpt — under investigation"). The cause is the
change of estimator, not a bug in either number.

A second contributor: `eval_fair.py` did not pass `pert_ctrl_expr`, so the
v2-1a self-response gate was **off** there and **on** during training and
in-training evaluation — the two scripts were scoring different model
behaviours.

*Verified by:* `tests/test_metrics.py::test_pooled_exceeds_per_pert_when_between_item_levels_differ`.

*Status:* fixed — the estimators are named `per_item_correlation` and
`pooled_correlation`, both are reported side by side, and `eval_fair.py` now
passes `pert_ctrl_expr`. **The "×2.3 / 0.31 → 0.71" claim is withdrawn.**

---

## E5 — The P3 split did not hold out target genes

**Severity: the expanded-data result is not an unseen-perturbation estimate.**

`train_p3.py:96-100` split on `(dataset_index, condition)` pairs with no
grouping by target gene. A perturbation reaches the model only as
`(pert_row, rna_emb)`, i.e. by target-gene identity.

The expanded training set includes Replogle **gwps**, a genome-wide K562 screen
covering essentially every target gene in the legacy adamson / norman /
replogle_ess datasets. A "held-out" legacy perturbation for gene *X* therefore
had gene *X* in training via gwps — and adamson, norman and gwps are all K562.

*Status:* fixed — `--split_by target_gene` is now the default and holds out
whole genes; the old behaviour remains as `--split_by pert` and prints a
warning naming the number of genes present on both sides.

---

## E6 — Unmeasured HVG columns entered the loss and every metric unmasked

**Severity: suspected inflation of `pearson_dev`; magnitude not yet quantified.**

`train_p3.py` built targets and control means with `np.zeros(...)` and filled
only the columns present in each dataset's panel. For a gene not measured in a
given panel, `fc == 0` and therefore `dev == -common_fc` — a value identical for
**every perturbation of that dataset** and containing no perturbation-specific
information. The training loss (`:310`) and all metrics (`:239-247`) covered all
2000 columns with no mask.

Across seven heterogeneous panels (gwps ~8k genes, adamson ~5k) this block can
be a large fraction of the matrix, and a model can score on it from `ds_emb` and
`ctrl_feat` alone.

Mechanistically this is the **same shared-component inflation the P2 erratum
documented**, one level down — which is the main reason this repository's own
methodological lesson needs restating rather than treating as resolved.

*Verified by:* `tests/test_metrics.py::test_unmeasured_columns_inflate_unmasked_correlation`
(unmasked ≈0.8+ where masked ≈0).

*Status:* fixed — a `measured` mask is propagated through the common core, the
loss and all metrics. The magnitude of the effect on published numbers can only
be established by re-running with data.

---

## E7 — The conditioning ablation changed two factors at once

`train_p3.py:336-341` passed `rna_emb` to the ablated branch but not to the real
one:

```python
kw = dict(rna_emb=..., pert_esm_override=ov_all[lo:hi])
reals.append(model(..., pert_ctrl_expr=pe_t[lo:hi]))        # no rna_emb
zeros.append(model(..., pert_ctrl_expr=pe_t[lo:hi], **kw))  # rna_emb + zeroed ESM2
```

So `ablation_r` compared *"no RNA axis + true ESM2"* against *"RNA axis + zeroed
ESM2"*, and neither branch matched the configuration used by `evaluate()`. Any
run with the RNA axis enabled — including the one that produced
`results/p3_v21e/train_log.jsonl` — has unusable `ablation_r` values.

A caveat remains even after the fix, and was already acknowledged in the P2.3-B
report: zeroing the ESM2 vector does not remove perturbation identity, because
`is_target` and `is_neighbor` still encode it. A clean test requires shuffling
those too.

*Status:* fixed (single-factor); the per-axis ablations remain outstanding.

---

## E8 — Exported caches always used dataset-context index 0

`export_vcpe_cache_v2.py:213` hard-coded `ds0 = torch.zeros(...)`, while
`--context_name` only ever reached a metadata string. A cache built with
`--context_name hepg2` or `jurkat` — the command the README documents — was
generated with the adamson/K562 dataset embedding and labelled otherwise.

*Status:* fixed — an explicit `CONTEXT_DS_INDEX` mapping plus a `--ds_idx`
override, and the run now fails loudly on an unknown context instead of
silently defaulting.

---

## E9 — The ASO sequence encoder had no positional encoding

**Severity: the lead architectural claim did not hold.**

Neither `model_efficacy.py` nor `EfficacyV2` added any positional signal before
`nn.TransformerEncoder`. Self-attention is permutation-equivariant and
mean/max pooling is permutation-invariant, so the sequence branch was a **bag of
(base, sugar, backbone) triples**: it could not distinguish a 5-10-5 MOE gapmer
from the same nucleotides in any other order, and could only read off per-type
counts.

The README claimed that "modification × sequence-context interactions are
captured directly by attention". Without positions there is no context, only
composition. The showcase example (same sequence, ~11% as plain DNA vs ~38% with
LNA/cEt wings) is driven by the *number* of modified sugars; relocating the same
modifications to the middle of the molecule would have produced an identical
prediction.

Related: `padding_idx=0` in the v1 sugar and backbone embeddings pinned the rows
for **DNA** and **PO** — the gap chemistry of every gapmer, and the unmodified
backbone — to zero with no gradient.

*Status:* fixed — learned positional embeddings (`--no_pos_emb` reproduces the
old behaviour so the ablation can be reported), and a dedicated PAD index.
Checkpoints now record `arch_config` so a pre-v4 checkpoint is rebuilt with its
own architecture instead of being silently reinterpreted.

---

## E10 — External-comparison claims were not reproduced

The repository contains **no code that runs GEARS, scGPT, CPA, AIDO.RNA-Pert,
OligoAI, ASOptimizer, OligoWalk or RNAGenesis**. Every such value existed only
as a hard-coded comparator string, e.g.:

```python
comparator = "OligoAI 0.419 median [0.290-0.533]; ASOptimizer 0.076; OligoWalk 0.147"
comparator = "RNAGenesis Huesken benchmark (paper, Fig 3h)"   # no number at all
```

The only computed baselines on the response line are the trivial ones (predict
no change; the perturbation mean — and the latter is itself broken, see E12).

The README nonetheless stated "comparable to OligoAI baseline" and "matches
RNAGenesis benchmark". Beyond the absence of a shared split or preprocessing,
the ASO comparison also mixed estimators (pooled 0.283 against a per-screen
median 0.419) and rested on the per-screen number invalidated by E1. The siRNA
comparison used a 419-row OligoGym subset against RNAGenesis's 702-row original.

*Status:* claims removed from the README and from the result JSONs, which now
record explicitly that the comparators are literature context and support no
head-to-head statement. **Running real baselines remains the single largest
outstanding item.**

---

## E11 — Internal inconsistencies in the committed result files

Recomputed from `results/efficacy_v2/cv_results*.json`:

| claim | recomputed | note |
|---|---|---|
| `lineage: "v3 0.3007 (pooled)"` | `pooled_mean = 0.2830` | inconsistent within one file |
| `pooled_std = 0.0319` | population sd of the **best-over-epochs** values = 0.0319 | derived from the biased quantity (E3); sample sd is 0.0357 |
| README "region features +0.034" | **+0.0278** on a like-for-like estimator | v3 fold spread is 0.184–0.370, sample sd 0.080 |

A ±0.03 difference sits well inside a ±0.08 fold-to-fold spread. No bootstrap
CI, repeated CV or significance test existed anywhere in the repository.

Separately, the committed `cv_results*.json` files do **not** match the schema
written by the committed `train_efficacy_v2.py`, so the script that produced
them was never committed. The repository is also a single squashed commit, so
no code-to-number provenance can be recovered.

*Status:* CIs and multi-seed support added; result files now carry the estimator
name, the config and an `errata` block. The stale JSONs are retained for the
record and must not be cited.

---

## E12 — Smaller correctness defects

| # | file:line | defect |
|---|---|---|
| a | `train_efficacy.py:256-266` | `pert_mean_global` used `=` instead of accumulation, so the "perturbation-mean baseline" was the last training perturbation's profile, not a mean |
| b | `train_efficacy.py`, `train_efficacy_v2.py` | model moved to the device while every tensor stayed on CPU — **any CUDA run raised a device mismatch**, so only the CPU path was ever exercised despite a "GPU-optional" docstring |
| c | `sirna_seed_search.py:66` | used `os.path` without importing `os`; every run crashed with `NameError` while writing its results |
| d | `eval_sirna_external.py:91` | `astype(str).str.len() > 0` turns `NaN` into the 3-character string `"nan"`, so empty-target rows all survived. The shipped `external_ichihara2.json` shows both symptoms: `dropped_empty_target: 0` and a `"nan"` key in `per_target_spearman` |
| e | several | unmappable gene symbols silently fell back to **ESM row 0**, substituting an arbitrary real gene's embedding |
| f | `model_dev.py:59` | the frozen ESM2 table was a *persistent* buffer, so ~405 MB of a 428 MB checkpoint was a redundant copy of a public lookup table (`maprna_p1/model_kd.py` already did this correctly) |
| g | `predict_service.py:113` | `kd` was clamped to a 0.20 floor while `inhibition_pct` was not, so one call could return `inhibition_pct=11.0` with `kd=0.200`, and all sub-20% predictions collapsed to one value |
| h | `predict_service.py` | the forward pass was re-implemented inline rather than calling the module, so any layer added to the model was skipped at inference |
| i | `download_oligogym.py:19` | a developer-local proxy (`127.0.0.1:7892`) was hard-coded into the retry path |
| j | `sirnamod_model.py` | no code path saved a model, yet `data/drive_weights/sirnamod_xgb_v1.joblib` is shipped and led the README; random (ungrouped) KFold over 907 rows × 538 features, with no early stopping and no nested CV |
| k | `diag_fc.py:200-225` | printed `(real vs zeroPert r)` while `pred_fc` held the **swapPert** output; the conclusion direction is unaffected but the label is wrong |
| l | `gate_cache_self_fc.py:39-41` | HPA TSV path hard-coded to a directory outside the repository, with no override and no existence check |

*Status:* all fixed. (j: grouped nested CV, early stopping and a producer script
for the artifact. k: both `r(real, zeroPert)` and `r(real, swapPert)` are now
computed and printed separately, and the default probe count was raised from
8 test + 4 train. l: resolved via `--hpa_tsv` / `$VCPE_HPA_TSV` / an in-repo
default, with an actionable error and a quote-tolerant header parser.)

A thirteenth item, `eval_headtohead.py`, was repaired rather than deleted but
**its numbers remain uninterpretable by construction**: the XGBoost encodes the
target gene as a one-hot over genes with ≥100 training rows, so on the
gene-holdout slice it has zero gene information while the transformer has an
ESM2 embedding — and on the patent-group slice the transformer checkpoint was
selected on that very slice (E3). The two biases run in opposite directions.
The script now says so at the top, and the dose-duplication bug (the missing
flag carried a copy of the dose) and the stale `build_split` signature were
fixed. It still requires a private package via `$RNA_ROBOT_HOME` and no result
file for it has ever been committed.

---

## What is not affected

* The **siRNA external evaluation** (`eval_sirna_external.py`) is mechanically
  sound: it trains for a fixed epoch budget on one dataset and scores the
  external set once, with no test-based early stopping. Its caveats are that the
  hyperparameters were inherited from a protocol that did use test-fold epoch
  selection (E3), that the two files come from one literature family and target
  overlap was asserted but never checked, and that it was a single unseeded run.
  An explicit overlap check and CI have been added.
* The **P2 erratum itself** — the finding that conditioning was short-circuited
  and that mse / pearson / cosine / top-k can all be inflated by the shared
  response — stands. E4 and E6 are further instances of exactly that failure
  mode, which strengthens rather than undermines it.
* The documented **negative results** stand: the LOCO cross-cell-line gate
  (`results/loco_results.json`, zero-shot 0.081 against a 0.15 threshold,
  recorded as FAIL) and the virtual-liver G0 adjudication (44% direction
  agreement, negative Spearman, recorded as NO-GO). Both were pre-registered and
  reported as failures.

---

## Outstanding

Ordered by what most constrains any claim the project can make:

1. Run real external baselines. The **cheap controls are now implemented**
   (`src/maprna_p3/baselines.py`: predict-no-change, train-mean,
   k-nearest-neighbour over frozen ESM2, and closed-form ridge) and
   `train_p3.py` reports them beside the model on the same split, the same mask
   and the same metrics, printing a warning when the learned head fails to beat
   one. They have **not yet been run on real data**, so it remains unestablished
   that a 5.7M-parameter conditioned head beats nearest-neighbour retrieval over
   the embeddings it is conditioned on. Running GEARS, and
   OligoAI/ASOptimizer/OligoWalk for the ASO head, is still outstanding.
2. Re-run every reported number under the corrected protocol and replace the
   withdrawn figures.
3. Quantify E6 on real data (masked vs unmasked, same checkpoint).
4. Per-axis ablations: shuffle `is_target` / `is_neighbor`, and a `ds`-shuffle
   control for dataset-level residual commonality (acknowledged as future work
   in the P2.3-B report and still outstanding).
5. Report the positional-embedding ablation (E9) rather than assuming position
   matters.
6. Near-duplicate sequence analysis for ASO Atlas: patent families republish
   identical sequences and tile targets with single-nucleotide shifts, so
   exact-match de-duplication is a lower bound.
7. Multi-seed runs everywhere; single-run point estimates should not be quoted.
