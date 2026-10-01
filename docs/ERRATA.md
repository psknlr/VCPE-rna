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
loss and all metrics. `--no_mask` reproduces the pre-v4 behaviour so the size of
the effect can be measured with one flag.

**Read the real-data A/B with its unmeasured fraction in hand, and note this
before the numbers rather than after them.** The effect scales with how much of
the panel is unmeasured, which is a property of the dataset combination and not
of the defect. The two CRISPRi screens now available here overlap heavily —
Replogle RPE1's 8749 genes are nearly a subset of Tian 2021's 33538 — so the
measured fraction is **0.987**, and only about 1.3% of the matrix is the constant
block. A small A/B difference on this pair is therefore expected and is **not**
evidence that the defect was minor in the withdrawn runs, which combined panels
as disjoint as a genome-wide screen with a ~5k-gene one. Quantifying it at the
magnitude that mattered needs those panels, which are on the host this
environment cannot reach.

The mechanism itself was demonstrated directly, on synthetic data where the
unmeasured fraction can be set: unmasking lifted `train_mean` — a baseline that
carries **no** perturbation information at all and must therefore score zero —
from **-0.083 to +0.108**. A baseline that cannot possibly know which gene was
perturbed scoring positively is the defect, stated as plainly as it can be.

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

## How the fixes were verified

No real data or released checkpoint was available while making these changes, so
verification is by construction rather than by reproducing a published number:

* **50 tests**, up from zero. `tests/test_metrics.py` and
  `tests/test_eval_metrics.py` pin E1, E2, E4 and E6 numerically (e.g. a
  300-replicate Monte Carlo confirming that the corrected enrichment centres on
  1.0 under random ranking). `tests/test_model_forward.py` exercises both heads
  on synthetic tensors and asserts that the ASO encoder is now order-sensitive
  while the legacy configuration is provably invariant to reversal — the
  mechanical statement of E9. `tests/test_baselines.py` pins the self-retrieval
  exclusion.
* **An end-to-end smoke test** (`tests/test_pipeline_smoke.py`) builds two
  synthetic GEARS-format datasets with different gene panels and overlapping
  target genes, then runs training and the ablation as subprocesses. It asserts
  the grouped split holds genes out, the mask is partial and reported, the test
  split is scored exactly once, the baselines appear beside the model, and the
  ablation distinguishes a live conditioning channel from an inert one. Every
  defect in this file was of a kind that only shows up when the pieces run
  together.
* **A static name-resolution check** in CI, because `py_compile` does not catch a
  missing import — it was added after exactly that class of bug was found twice
  (E12c, and once in this work's own first draft).
* CI runs the metric tests plus a byte-compile on every push, and the
  forward-pass and pipeline tests in a job with CPU torch installed.

What this does **not** establish: any statement about performance. The smoke test
uses 60 synthetic genes and asserts plumbing, not accuracy.

One incidental finding from the synthetic run is worth recording, because it
illustrates why the protocol changes matter: with the gene-grouped split, the
inner-validation `pearson_dev` climbed to ~0.57 while the held-out-gene value sat
at ~0.00 and lost to the k-nearest-neighbour control. On synthetic data that is
not a result about biology — but under the pre-v4 protocol the same run would
have reported the 0.57.

## E13 — Defects found while translating the comments to English

Translating a comment requires understanding what the code does, which is why
this pass found things the audit had not. Each was verified independently before
being recorded.

**a. The `_proc` ingest file was never a subset (`ingest_gse293987.py`).**
The mask was `(condition != "ctrl") | (control == 1)`. Both terms derive from
`aso_class == "CONTROL"`, so `control == 1` holds for exactly the rows whose
`condition == "ctrl"` and the two terms cover every row: the mask is identically
True. `gse293987_proc.h5ad` has therefore always been byte-identical to
`gse293987_full.h5ad`, `utc_ref` rows (UTC, OTHER and sub-threshold ACTN1)
included, despite its name and docstring describing a filtered file. Confirmed by
enumerating every `aso_class` x concentration class.
*Status:* the row counts are now printed and a no-op subset warns explicitly;
`--proc_drop_utc_ref` applies the intended filter. The default is left as the
no-op so that a re-run reproduces the file downstream work already consumed —
changing it silently would alter what an existing artifact name means.

**b. `nearest_sim` is a Pearson correlation, reported as a cosine.**
`train_cell_embedding.py` computes it with `np.corrcoef` between dataset
embeddings, printed it as `emb cos=`, and stored it as `nearest_sim` in
`results/loco_results.json` (0.9639). Cosine does not centre; Pearson does, and
for these embeddings the two differ. The value is correct for what it measures —
only the name was wrong, and that name is in a published result file.
*Status:* relabelled in the code and named explicitly in the result file
(`nearest_sim_metric`). The value is unchanged.

**c. Two ingest docstrings described superseded behaviour.**
`ingest_gse289964.py` claimed `condition = "Scarb1@{modification}{timepoint}"`
over 12 conditions; the code writes `condition = "SCARB1"` for every ASO sample
and puts the detailed label in `perturbation_raw`, and the distinct
modification x timepoint combinations number 3 x 3 = 9, not 12 (48 = 12 x 4
replicates is the total sample count, PBS included).
`ingest_gse293987.py` claimed `condition = "ctrl"` covers UTC and CONTROL at all
concentrations; the code assigns `"ctrl"` to CONTROL only, routing UTC, OTHER and
sub-threshold ACTN1 to `"utc_ref"` — deliberately, per the comment above the
assignment, so that transfection stress does not leak into the baseline.
*Status:* both docstrings corrected to match the code, with the reason recorded.

**d. The pre-registered LOCO gate criterion was written in Chinese.**
`results/loco_results.json`'s `gate_rule` documents the threshold for one of the
three negative results this project can stand behind. Translated; criterion and
values unchanged.

## E14 — What running an external baseline actually costs

Two findings from making E10's comparisons real, both of which change how a
baseline number should be read.

**a. GEARS does not reproduce its own result at a fixed seed.**
Seven identical invocations of `baseline_gears.py --seed 0` on the same data and
split gave `pearson_dev` from **-0.023 to +0.051**, sample sd **0.022** -- larger
than the mean (+0.020) itself. A four-run set at a different epoch budget spanned
+0.033 to +0.163. A single GEARS run is therefore not a usable baseline number,
and any table quoting one is quoting noise.

This was found the right way round, and the method is worth recording: after a
refactor changed a GEARS number, the refactor was first cleared by checking the
shared helpers against the originals on synthetic input (byte-identical), and only
then was the variance measured over repeated runs. Attributing the change to
either cause without that check would have been a guess. `--runs` (default 5) now
repeats and reports mean, sd and range.

**b. CPA cannot share an environment with this repository, and needs four
separate workarounds.**
Each was hit in turn:

1. `cpa-tools` pins `torch<2.0.0`; VCPE requires `torch>=2.1`. Installing CPA into
   the VCPE environment silently downgraded torch from 2.14 to 1.13 and broke the
   model code -- **95 passing tests became 8 failed and 6 errors**. pip states the
   conflict outright. This is why `baseline_cpa.py` is split into three stages
   (`export` / `run` / `score`) with the interface on disk instead of being a
   single script like the GEARS adapter.
2. torch 1.13 does not import on Python 3.11+ (`ValueError: mutable default ...
   use default_factory`, a dataclass rule that tightened in 3.11), so the venv is
   built on Python 3.10.
3. `cpa/_model.py` line 4 is `from tkinter import N` -- a stray IDE auto-import.
   `N` is never used anywhere in the package (verified: it is the only tkinter
   reference, and the `N` occurrences in `_plotting.py` are local parameter
   names), but it makes CPA unimportable on any Python without tkinter, which
   includes most container images. `--prepare_venv` writes a minimal stub
   providing only that one name, so a genuine tkinter dependency would still fail
   loudly.
4. `scvi-tools 0.20.3` imports `anndata._core.sparse_dataset.SparseDataset`, which
   newer anndata removed, so anndata and scanpy have to be pinned back as well.

None of this is CPA's fault as science, but all of it is the real cost of the
comparison, and it belongs in a methods section rather than being discovered by
the next person. A reproduction must record **both** environments.

Two further walls appeared after those four, and are also handled:

5. `scvi-tools 0.20.3` imports `jaxlib.xla_extension.Device`. That module is gone,
   and jaxlib older than 0.4.14 is no longer distributed for Python 3.10, so
   pinning back is not possible. The class was **renamed**, not removed
   (`jaxlib.xla_client.Device`), and scvi uses it only in type annotations, so the
   real class is re-exported. Nothing is faked, and if the attribute cannot be
   found the original ImportError is left in place rather than masked.
6. The stack still uses `np.float_`, removed in NumPy 2.0, so numpy is pinned back.

And one bug inside CPA itself:

7. `setup_anndata` builds a perturbation-to-SMILES map **unconditionally** -- the
   block runs whenever the class attribute is None, regardless of whether
   `smiles_key` was passed -- so a gene-perturbation dataset fails with
   `KeyError: None`. Pre-setting the map to empty skips it. That is the honest
   workaround rather than inventing a SMILES column: `_model.py:111` shows the map
   is consumed only under `use_rdkit_embeddings` (default False), so CPA uses its
   learned per-perturbation embeddings, which is the right representation for gene
   knockdowns. A placeholder SMILES string would make every perturbation
   chemically identical if that path were ever switched on.

Separately, a labelling mismatch that is a modelling question rather than an
installation one: GEARS-format data writes a single-gene perturbation as
`"GENE+ctrl"`, and CPA splits its perturbation key on `+` and counts every part.
Passed through raw, CPA reads `"MYC+ctrl"` as a two-way combination of MYC with
the control -- which produces ragged combination lengths and crashes in
`np.vstack`, and would also misrepresent the experiment. The adapter strips the
suffix so CPA sees one perturbed gene, and translates the split sets identically
so the two cannot disagree.

*Status:* **CPA runs.** All three stages verified end to end on synthetic data with
real gene symbols: export in the VCPE environment, training and prediction in the
isolated one, scoring back in the VCPE environment. Stage 3 shares
`baseline_common` with every other baseline, so no baseline can drift into being
scored differently -- which is E1/E2 one level up. Not yet run on real
Perturb-seq data.

## E15 — The scored gene panel was chosen with the test split visible

**Defect.** `make_hvg_list` ranked genes by their variance across **every**
perturbation, and `build_dev_data` called it **before** the train/test split. The
2000 genes the model is trained and scored on were therefore selected using the
held-out perturbations' expression.

`src/maprna_p1/ds_knockdown.py:138` (the ranking), `src/maprna_p3/train_p3.py:93`
(the call, before the split).

This is feature selection on the full data, not a per-item label leak, so it is
the mildest defect in this document — but it is the kind a reviewer checks, and
"the held-out genes are genuinely unseen" (E5's whole point) is not true of the
panel if it is true of the split.

**Fix.** The split now runs first and the panel is ranked on training
perturbations only (`--hvg_from train`, the default). The split itself is
untouched: it needs only the perturbation labels and draws from its own
generator, so moving it earlier leaves it bit-identical. `--hvg_from all`
reproduces the old behaviour and prints a warning, so the size of the effect can
be measured rather than asserted.

**Verification.** Three levels:

1. With `--hvg_from all`, every one of the 18 arrays `build_dev_data` returns is
   byte-identical to the pre-restructure code on real data — so the
   restructuring changed nothing by itself.
2. With `--hvg_from train`, `train_items`, `test_items`, `rows_tr`, `rows_te` and
   `is_inner_val` are unchanged and only `hvg_rows` and what derives from it
   (`dev`, `fc`, `mask`, `common_fc`, `ctrl_feat_all`) move — i.e. the panel
   moved and the split did not.
3. Footprint on Replogle RPE1 + Tian 2021 CRISPRi:

   | `n_hvg` | seed | genes shared | only in the test-visible panel |
   |---|---|---|---|
   | 500 | 0 | 491/500 | 9 (1.8%) |
   | 500 | 1 | 491/500 | 9 (1.8%) |
   | 2000 | 0 | 1967/2000 | 33 (1.7%) |
   | 2000 | 1 | 1971/2000 | 29 (1.5%) |

   About 1.7% of the panel differs. It is small because the training split is 85%
   of the items, so a variance ranking over it barely moves — which is an
   argument about this dataset pair, not about the practice.

**Status.** Fixed; default changed. The effect on the reported score is measured
with the real table, not with the placeholder used for the wiring check above.

## E16 — Top-k overlap broke ties by column position

**Defect.** `top_k_overlap` took `np.argsort(-|x|)[:k]` for the truth and the
prediction alike. When the prediction has ties -- and "predict no change" is
nothing *but* ties, every value 0 -- `argsort` returns the tied columns in index
order, so the constant prediction "selects" the first k columns of the panel.
`make_hvg_list` sorts the panel by variance, so the first k columns are exactly
the genes most likely to be among the true top k.

**Size, on committed numbers.** On the main benchmark (ESM2-150M + STRING graph,
80 epochs, split 0) the no-change predictor scored **top50_dev = 0.387**, above
the conditioned head's **0.379** and nearly four times the chance level
k/n = 50/500 = 0.10. On the cell-level screen alone it scored 0.149 against a
head at 0.096. No headline in this repository rests on top-50 overlap, which is
why this went unnoticed; it is recorded because the metric is still reported
beside every run and a reader comparing those columns would have been misled.

A second, separate observation from the same files is not a tie artefact and is
not fixed by this change: on split 0 of the plain-ESM2-150M model, setting the
target vector to zero *raises* top-50 overlap from 0.321 to 0.340
(`results/real_150M/C_ablation.json`). The ablated predictions are continuous, so
there are no ties; top-50 overlap simply rewards predicting which genes respond
to perturbations in general, which a model with its conditioning removed still
does. It is not a measure of perturbation-specific prediction here, and it should
not be read as one.

**Fix.** Ties are now scored by their expectation under uniformly random
tie-breaking (`eval_metrics._topk_inclusion`): values strictly above the k-th
largest |value| count fully, values tied with it share the remaining slots
equally. Without ties this is exactly the old definition (tested); a constant
prediction now scores exactly k/n (tested). `train_p3.py`'s private copy of the
metric was replaced by the shared function. The old form is kept as
`top_k_overlap_index_ties` only so the defect can be reproduced.

**Not changed:** the legacy P1/P2 training scripts and the unused copy in
`src/efficacy/metrics.py` keep the old form. Nothing reported uses them.

**Status.** Fixed for every future run. Committed `top50_dev` values were
computed with the old form; for the trained models and the ridge and k-NN
controls they are unaffected (continuous predictions, no ties), for the
no-change predictor they are the artefact described above.

## E17 — The inner-validation items were also trained on

**Defect.** `build_dev_data` carves an inner-validation slice out of the training
items (15% of training target genes) and the epoch is selected on it. But the
training loop in `train_p3.py` iterated over **all** training items: `perm =
torch.randperm(n)` with `n = len(rows_tr)`. The inner-validation perturbations
were in every epoch's batches. Epoch selection therefore measured fit to items
the model had been trained on, not generalisation.

**How it showed.** Two symptoms, both visible in the committed logs and only
read correctly once the figures put them side by side (Extended Data Fig. 3):

* selection drifts to the end of whatever budget is set -- epochs 78, 77 and 78
  of 80, and 150 of 150 in the long probe -- because training fit keeps rising;
* the inner-validation score at the selected epoch is far above the held-out
  score, and the gap grows with training: 0.48 against 0.29 at 30 epochs,
  0.69 against 0.34 at 80, 0.81 against 0.35 at 150 (split 0).

**What it does not affect.** The held-out split was never touched by selection,
so every held-out number remains a single-shot estimate for the checkpoint that
was chosen, and the head and the controls were fit on the same items, so the
comparisons between them in Table 1 and Fig. 2 stand. What the defect removes is
the *meaning* of the selected epoch. In particular the statement in
RESULTS_REAL.md, in the Fig. 4b annotation and in the IEEE Extended Data Fig. 4
caption (`make_ieee.py`) -- that selection "never leaves the budget, so the head
has not converged" -- inferred convergence from a criterion that tracks training
fit and would have said the same at any budget. All three are corrected. The
held-out values on split 0 (0.294, 0.336, 0.350 at 30, 80 and 150 epochs) do
still rise, so the conclusion that the head is still improving at 80 epochs is
supported -- by those values, on one split, and not by the selection rule.

**Fix.** The inner-validation items are now held out of the training batches by
default (`training_indices`), and the ridge, k-NN and train-mean controls are fit
on exactly the items the head trains on, so a difference between them is still
not a difference in training data. `--inner_val_in_training` restores the old
behaviour and is required to reproduce any committed run. The report records
`inner_val_in_training` and `n_items_fit`, and `run_seeds.py` refuses to pool
seeds that disagree on it.

The same defect made the external comparison asymmetric. The GEARS and CPA
adapters train on the training items *minus* the inner split
(`baseline_common.inner_split_items`, documented as "matching what train_p3
trains"), while the head and its controls trained on all of them -- about 15% more
training data on VCPE's side of Table D. Every method there sits at the floor, so
no conclusion moves, but the asymmetry favoured VCPE and is recorded as such. With
the fix the docstring is true.

Two milder couplings remain and are stated rather than fixed: the response
panel and the train-only common core are computed over all training items,
inner-validation ones included. Neither carries a label of an inner-validation
item into the head's parameters, but both make the inner-validation score
somewhat optimistic.

**Status.** Fixed in code; **the reported numbers predate the fix and have not
been regenerated.** They need re-running before they are quoted as the result of
the protocol drawn in the figures.

## E18 — The response head was described as 5.7M parameters

README, the docstrings of `baselines.py` and `baseline_gears.py`, and the
fairness caveat that `baseline_common.py` writes into every external-comparison
report described "the 5.7M-parameter VCPE head". Instantiated with the
configuration of the reported runs (`d_model` 256, no RNA branch), the head has
**1,039,889** parameters on the 480-d ESM2-35M table, **1,121,809** on the 640-d
ESM2-150M table and **1,449,489** on the 1,280-d graph table. 5.7M belongs to an
earlier configuration with a 5,120-d embedding and the RNA encoder. The caveat
text was corrected at its source; `results/real_35M/D_*.json` and `table_D.md`
were written before the correction and still carry the old sentence. The
architecture figure reads the count from an instantiated model, so it cannot
drift again.

## E19 — Three-split confidence intervals used the normal quantile

**Defect.** `run_seeds.py` reported every 95% interval as mean ± 1.96 × s.e.
With three splits the mean has 2 degrees of freedom and the 95% multiplier is
t(0.975, 2) = 4.30, so every interval in the committed seed summaries -- and every
claim resting on one -- was 2.2 times too narrow.

**What changes.** Recomputed from the per-split differences the summaries store:

| configuration | head vs | mean paired diff | 95% t interval | paired t P | splits won |
|---|---|---|---|---|---|
| ESM2-35M | ridge | +0.0051 | [-0.0407, +0.0509] | 0.678 | 2/3 |
| ESM2-35M | k-NN | +0.0272 | [-0.0513, +0.1058] | 0.275 | 2/3 |
| ESM2-150M | ridge | +0.0274 | [-0.0108, +0.0657] | 0.091 | 3/3 |
| ESM2-150M | k-NN | +0.0812 | [+0.0361, +0.1264] | 0.016 | 3/3 |
| + STRING indicator | ridge | +0.0279 | [-0.0105, +0.0663] | 0.089 | 3/3 |
| + STRING indicator | k-NN | +0.0817 | [+0.0360, +0.1275] | 0.017 | 3/3 |
| + STRING graph, 30 ep | ridge | -0.0079 | [-0.0277, +0.0119] | 0.230 | 1/3 |
| + STRING graph, 30 ep | k-NN | +0.0654 | [+0.0049, +0.1258] | 0.043 | 3/3 |
| + STRING graph, 80 ep | ridge | -0.0024 | [-0.0648, +0.0600] | 0.885 | 1/3 |
| + STRING graph, 80 ep | k-NN | +0.0709 | [+0.0002, +0.1416] | 0.050 | 3/3 |

The claim that changes is the one this project had leaned on most: **on plain
ESM2-150M the head led ridge on all three splits by +0.027, but the interval is
[-0.011, +0.066] and the paired t-test gives P = 0.09.** "Beats ridge, CI
excluding zero" was not supported; "led on every split, not significant at three
splits" is what the data say. Three wins out of three is itself weak evidence: a
two-sided sign test on 3/3 gives P = 0.25. The margins over k-NN on the 150M
representations do exclude zero (P = 0.016-0.05), the graph-table 80-epoch one
only just. Nothing about the trivial baselines changes.

Read with E17, the overall statement is simpler than the one it replaces: no
configuration gives the head a significant advantage over a ridge fit on the same
features, and the gain the project can defend is the representational one, which
every method shares.

**Fix.** `run_seeds.py` now uses the t quantile for every interval, records the
method (`ci_method`) and adds a two-sided paired-t P value (`p_paired_t`); a test
pins the 4.30 multiplier. The figures do not read the stored intervals at all:
`figdata.paired` recomputes them from the per-split differences and checks the
stored mean, so the committed summaries' old intervals cannot reach a figure.

**Status.** Fixed in code and in every figure. The committed `seed_summary.json`
files keep the old intervals as written; RESULTS_REAL.md and README now cite the
recomputed ones.

## E20 — The held-out gene count was misreported

**Defect.** RESULTS_REAL.md, this file, Table 1's footnote, three figure legends,
the IEEE captions and a box inside Fig. 1 all said the combined task has "2,328
distinct target genes, 349 held out". Neither number is a gene count. Re-running
the repository's own `split_items` on the committed data gives 2,329 perturbation
items covering **2,269 distinct target genes** (60 genes are targeted in both
screens) and **340 held-out target genes in every split** (round(0.15 × 2,269)).
349 is the held-out *perturbation* count of split 2 alone; the three splits score
352 / 347 / 349 held-out perturbations, which is what `n_scored_perturbations`
records and what the remaining legends already said.

**What changes.** Only the stated n. No score, interval or P value used either
number, and the per-split perturbation counts were always correct.

**Fix.** `fig1_design.py`, the legend and footnote text in `make_all.py` and
`make_ieee.py`, RESULTS_REAL.md and the real-data summary further down this file
now give 2,269 genes and 340 held out per split; the figures and legends were
regenerated.

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

See [REPRODUCE.md](REPRODUCE.md) for how to regenerate every number under the corrected protocol, including how to read the control
baselines and the per-axis ablation table.

## Outstanding

**Update: the first real-data runs are done.** The two items that most
constrained every claim -- "no number regenerated" and "no external model run"
-- are addressed. Two real CRISPRi screens now go through the whole corrected
pipeline, and the head, four controls, GEARS and CPA are all scored on one
footing. The result is in [RESULTS_REAL.md](RESULTS_REAL.md), and the honest
headline is that on real data the model clears the trivial floors decisively and
its edge over a ridge regression on the same embeddings is embedding-dependent:
a tie on ESM2-35M (paired CI [-0.016, +0.026]) that becomes a small consistent
win on ESM2-150M (+0.027, CI [+0.010, +0.045], every seed). [Corrected by E19:
those intervals used 1.96 with three splits; with the t quantile the 150M margin
over ridge is [-0.011, +0.066], P = 0.09 -- consistent, not significant.] A confident claim I
had written here -- that a stronger embedding "could only weaken the model's
case" -- was falsified by the 150M run, which helped the non-linear head more
than the linear controls; the correction is in RESULTS_REAL.md. The per-axis
ablation has also now run on real data and the conditioning is genuinely alive --
permuting the target-gene vectors costs -0.172 of +0.288 -- which is the opposite
of the P2 failure that started this document. The remaining items below are about
widening the measurement, not whether one exists. They are ordered by
what most constrains any claim the project can make:

0. **Real data.** Until now nothing in this repository had been run on real
   Perturb-seq data, and nothing could be: the ESM2 conditioning table existed
   only on the machine that produced the withdrawn results, with no script to
   rebuild it, and `dataverse.harvard.edu` — which hosts GEARS' own
   `perturb_processed.h5ad` files — is refused outright by this environment's
   egress policy.

   Both are now addressed. `tools/build_esm2_gene_table.py` derives the table
   from the UniProt REST API and an ESM2 checkpoint by a recorded procedure, and
   `src/maprna_p3/ingest_scperturb.py` converts the scPerturb copies on Zenodo
   into the layout the loader reads. Two real CRISPRi screens now go through the
   whole pipeline: **ReplogleWeissman2022 RPE1** (2204 perturbations surviving a
   20-cell gate, 3000 control cells, 8749 genes) and **TianKampmann2021 CRISPRi**
   (184 perturbations, iPSC-derived neurons, 33538 genes). Together they give
   2269 distinct target genes (2329 perturbations) and a grouped split that
   holds out 340 of them in each split (352 / 347 / 349 held-out perturbations).

   The one caveat that must travel with every real number: the ESM2 variant.
   650M — what the withdrawn results used — needs roughly a day for the human
   proteome on 4 CPUs, so the tables are built with smaller variants and the
   variant is recorded in each table's sidecar. It does not affect the comparison
   against the ESM2 retrieval and ridge controls, which read the same table, but
   it does affect the absolute scores. Note the direction: the controls are pure
   functions of the embedding while the head also sees the control profile and
   the dataset embedding, so a weaker table handicaps the controls more than the
   head. That is why results are reported under more than one table rather than
   the cheapest one.

1. Run real external baselines. **DONE on the Tian screen** -- GEARS
   (-0.005, mean of 3 runs) and CPA (+0.018) are reported in
   [RESULTS_REAL.md](RESULTS_REAL.md) Table D on the head's exact split, panel,
   residual target and estimator, each trained on the cell-level input it is
   published on. On that 27-gene task every method including the head is at the
   floor, so the table proves the machinery, not a ranking. A larger screen at
   cell level (Replogle did not fit this host's memory for GEARS's per-cell
   graphs) is the remaining widening. Below is how the adapters were built.
   **GEARS runs for real**
   (`src/maprna_p3/baseline_gears.py`), on VCPE's exact split (via GEARS's
   `split='custom'` mechanism), VCPE's HVG panel and mask, VCPE's residual
   target, and VCPE's estimators. The cheap controls
   (`src/maprna_p3/baselines.py`) report alongside. Verified end to end on
   synthetic data with real gene symbols: GEARS trains, predicts, and is scored
   against the VCPE head on identical footing.

   Two things matter for anyone using it. First, **GEARS's expression prediction
   is converted to the same residual VCPE predicts** (fc against the same control
   mean, minus the same train-only common core); scoring raw expression against a
   residual would flatter GEARS enormously, because the shared response dominates
   raw expression -- the P2 finding applied to a baseline comparison. Second,
   what is *not* controlled is stated in the script and in its output:
   GEARS runs at defaults while VCPE's hyperparameters were tuned on this data
   over many runs, and GEARS at defaults is far larger than the 5.7M-parameter
   head.

   Practical obstacle worth recording: **GEARS is broken on a current
   scipy/pandas**, and inconsistently so. `get_dropout_non_zero_genes` and
   `get_coexpression_network_from_train` call `X.toarray()` (X must be sparse),
   while `GEARS.__init__` indexes X with a pandas boolean Series (which current
   scipy rejects for a sparse matrix). The adapter densifies for the constructor
   only and restores the sparse matrix; the alternative is an older scipy. A
   GEARS comparison is not reproducible today without one of the two, and which
   was used belongs next to the number.

   Still outstanding: **it has not been run on real Perturb-seq data**, and
   scGPT, CPA, AIDO.RNA-Pert, OligoAI, ASOptimizer, OligoWalk and RNAGenesis
   remain unrun.
2. **A fresh measurement now exists** ([RESULTS_REAL.md](RESULTS_REAL.md)); the
   withdrawn numbers themselves are not "re-run". They were measured on
   adamson / norman / replogle_rpe1_essential, from a host this environment
   cannot reach, under the broken protocol -- so there is nothing to reproduce,
   only a corrected measurement to put in their place, on the datasets that are
   reachable. Presenting the new numbers as a withdrawn figure "corrected" to a
   new value would itself be a misrepresentation; they are a different dataset
   pair, measured properly. The original intent, retained for the record:
   re-run every reported number under the corrected protocol and replace the
   withdrawn figures.
3. ~~Quantify E6 on real data~~ — **DONE**
   ([RESULTS_REAL.md](RESULTS_REAL.md)). Same config, one flag apart, on the 150M
   seed-0 run: unmasking moves the model by **-0.0004** (nothing) while lifting
   `train_mean`, which carries no perturbation information, by **+0.0046**. The
   mechanism is confirmed in the right direction and the magnitude is negligible
   here for exactly the reason predicted in advance: 99.86% of the panel is
   measured, so the constant block is 0.14% of the matrix. It says nothing about
   the magnitude in the withdrawn runs, which combined a genome-wide screen with
   a ~5k-gene panel. Superseded detail, retained: the A/B was **expected to be
   small and was initially deferred**: the two reachable CRISPRi
   screens overlap so heavily that the measured fraction is 0.987, so only ~1.3%
   of the matrix is the constant block the mask removes. The run was started but
   competed with the 150M table build for this 4-CPU host badly enough to be
   impractical (each epoch slowed ~40x under the contention), and on a 0.987-
   measured corpus the result would only confirm the prediction of a negligible
   effect. It remains a quick check to run alone; it is not expected to change
   any conclusion, because the masked `train_mean` on real data already scores
   +0.012 -- i.e. the mask is working, a no-information baseline is at the floor.
4. ~~Per-axis ablations~~ — **implemented** (`src/maprna_p3/ablate_axes.py`).
   Perturbation identity reaches the model through four channels (target ESM2
   vector, RNA embedding, `is_target`, `is_neighbor`) plus the `ds` embedding;
   the old single `ablation_r` zeroed only the first. The script now runs each on
   its own plus an all-off floor, reporting both `r_vs_real` and the held-out
   `pearson_dev` under each ablation, and states the only defensible
   conditioning claim: the gap between the intact model and the no-conditioning
   floor. Model hooks (`is_tgt_override`, `is_nb_override`,
   `indicator_features`) were added for this; the v2-1a expression gate stays
   keyed to the true `is_target` so that ablating the feature does not also move
   the gate (which would repeat E7).

   **Now run on real data** ([RESULTS_REAL.md](RESULTS_REAL.md)), with two
   findings. First, the conditioning is real: permuting the target-gene vectors
   costs **-0.172 of +0.288** and drops the prediction's correlation with the
   intact one to 0.16, while `ds_shuffle` costs only -0.064. This is the opposite
   of the P2 failure, measured the same way that exposed it. Second, the
   script's own `conditioning_gain` (+0.282, against the *zeroed* floor)
   **overstates** the effect, because zeroing a vector is off-distribution and
   flatters the floor; the shuffle ablation preserves input statistics and is the
   number to quote. Third, three of the four identity channels leave the
   prediction bit-identical, so in this configuration the four-channel
   architecture is effectively **one** channel. The RNA axis is inert because no
   encoder was loaded. `is_target` is inert because at `--n_hvg 500` the
   perturbed gene is in the response panel for only ~6% of perturbations.

   `is_neighbor` was inert because no graph had been built -- so one was
   ([RESULTS_REAL.md](RESULTS_REAL.md)): STRING v12.0, 443,966 edges at
   combined_score >= 700, 77% gene coverage. It made almost no difference
   (**+0.0005**, an order of magnitude below the seed sd), and the reason is
   structural rather than biological: a sparse binary partner flag over the output
   panel fires on **0.27%** of scored (item, gene) cells, and that fraction is
   about degree/gene-universe **whatever the panel size**, so no amount of data
   rescues it. Folding the same graph densely into the embedding instead --
   `[own || mean(partners)]`, handed identically to the head and to both ESM2
   controls -- produced the largest gain measured anywhere in this project
   (ridge +0.072, head +0.036). The channel was the wrong place to put the
   information, not the wrong information.
5. Report the positional-embedding ablation (E9) rather than assuming position
   matters.
6. ~~Near-duplicate sequence analysis for ASO Atlas~~ — **implemented**
   (`src/efficacy/seq_dedup.py`, plus `--near_dup_check` in
   `train_efficacy.py`). Patent families republish identical sequences and tile
   targets one nucleotide at a time, so exact-match de-duplication is a lower
   bound. Two sequences are treated as near-duplicates when the shorter is
   contained in the longer up to a length threshold — the relation that catches
   a shift of any size and a trim at either end, which whole-string edit
   distance does not (a 1-nt shift has edit distance 2 but is the same binding
   site). Candidate pairs come from a k-mer inverted index and comparisons are
   confined to a shared target gene. The script reports, per candidate grouping
   column, how many near-duplicate pairs **cross** a group boundary — i.e. how
   many would survive a grouped split. A grouping column only controls leakage
   if that count is small. **Not yet run on ASO Atlas**, which the repository
   does not ship.
7. Multi-seed runs everywhere. **DONE for the response line** -- Table A is 3
   seeds, each varying both initialisation and the split, with paired
   per-seed comparison and CIs (RESULTS_REAL.md). Still single-seed on the
   efficacy line. Single-run point estimates should not be quoted.
   (`--seeds` exists on the ASO CV; the response line still runs one seed at a
   time.)
8. ~~Result provenance~~ — **implemented** (`src/provenance.py`). Every result
   file now carries the git revision and whether the tree was dirty, the exact
   command line, all parsed arguments (so the split, the seeds and every
   protocol switch travel with the number), the resolved versions of the
   packages that affect numerics, and the platform. None of the pre-v4 result
   files can be tied to a revision or an environment: the history is one
   squashed commit, nothing but torch was pinned, and E11 records a committed
   JSON whose schema does not match the committed script said to have produced
   it.
