# Reproduction guide

This describes how to regenerate every number this repository reports, under the
corrected evaluation protocol. It will **not** reproduce the figures in tags
before v4 — those are withdrawn, and why is recorded in
[ERRATA.md](ERRATA.md).

Read the two rules first, because most of the withdrawn numbers came from
breaking one of them:

1. **Never report a slice you selected on.** Every script here selects on an
   inner split carved out of training data and touches the held-out set once.
2. **Never compare two estimators.** `per_item` (mean of within-perturbation
   correlations) and `pooled` (one correlation over the flattened matrix) both
   used to be called `pearson_dev` and differ by roughly 1.5× on the same
   checkpoint. Compare like with like.

---

## 0. What runs without any data

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
```

85 tests, no data or checkpoint assets. This includes an end-to-end smoke test
that builds synthetic Perturb-seq data and runs training plus the per-axis
ablation as subprocesses, asserting that the grouped split holds genes out, the
measured-column mask is applied, the test split is scored exactly once, and the
ablation separates live conditioning channels from inert ones.

If you change anything about evaluation, this suite is the first thing to run.
Several of the defects in ERRATA.md are now pinned by a test specifically so they
cannot return quietly.

---

## 1. Data you need to obtain

Nothing is redistributed here — see [../data/README.md](../data/README.md) for
sources and licences. Minimum for the response line:

| Asset | For |
|---|---|
| `adamson` / `norman` / `replogle_rpe1_essential` `perturb_processed.h5ad` | training and evaluation |
| `Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt` | conditioning lookup |

Optional: scPerturb datasets (data-expansion track), STRING edges
(`is_neighbor` feature), GENCODE transcript FASTA (RNA sequence axis).

For the efficacy line you additionally need ASO Atlas (patent-derived — read the
licence note) and/or the OligoGym siRNA sets.

⚠️ **No checkpoints are downloadable.** This repository publishes no releases, and
the release URL used before v4 pointed at a different account. Everything below
trains from scratch.

---

## 2. Response line

### 2.1 Single run

```bash
python src/maprna_p3/train_p3.py \
  --data_dirs data/adamson/perturb_processed.h5ad \
              data/norman/perturb_processed.h5ad \
              data/replogle_rpe1_essential/perturb_processed.h5ad \
  --esm_table data/drive_weights/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt \
  --out_dir runs/p3_seed0 --epochs 60 --split_by target_gene
```

`--split_by target_gene` is the default and holds out whole target genes. The
alternative, `--split_by pert`, reproduces the historical split and prints a
warning naming how many target genes appear on both sides; with a genome-wide
screen in the training set that number is large, and a "held-out perturbation"
whose gene was trained on is not held out.

`runs/p3_seed0/final_report.json` contains the held-out metrics, the control
baselines, the selected epoch, and a `provenance` block (git revision, dirty
flag, full command line, all arguments, resolved package versions).

### 2.2 Read the baselines before the model

The report prints the model next to four controls:

| Baseline | What it tests |
|---|---|
| `zero` | predict no residual — the floor conditioning must clear |
| `train_mean` | the mean training residual, ignoring which gene was perturbed |
| `knn_esm2` | average the residuals of the k nearest training perturbations in ESM2 space — **retrieval with no learning** |
| `ridge_esm2` | closed-form linear map from the ESM2 vector to the residual profile |

`knn_esm2` is the one that matters. It uses the same conditioning information the
model receives. If the learned head does not beat it, the head is an expensive
nearest-neighbour lookup, and the run prints a warning saying so. In this field
cheap baselines have repeatedly proved hard to beat, so this is not a formality.

### 2.3 Multiple seeds — required before quoting anything

```bash
python src/maprna_p3/run_seeds.py --seeds 0 1 2 3 4 --out_dir runs/p3_seeds -- \
  --data_dirs ... --esm_table ... --epochs 60 --split_by target_gene
```

The seed changes **both** the initialisation and the grouped split, so the
resulting interval answers "does this survive a different choice of held-out
genes", which is the claim being made. The summary reports mean ± sample sd, a CI
on the mean, and the **paired** per-seed difference against each baseline —
paired because model and baselines share a split, so pairing removes the
split-to-split variance that dominates the marginal spread.

Report the paired verdict verbatim. "Model wins on 3/5 seeds" is a result; the
mean difference alone is not.

### 2.4 Per-axis conditioning ablation

```bash
python src/maprna_p3/ablate_axes.py --ckpt runs/p3_seed0/ckpt_p3_best_dev.pt \
  --data_dirs ... --esm_table ... --split_by target_gene \
  --out_json runs/p3_seed0/ablation.json
```

`--split_by` and `--data_dirs` must match the training run, or the reconstructed
test split is not the model's test split.

Perturbation identity reaches the model through four channels (target ESM2
vector, RNA embedding, `is_target`, `is_neighbor`), plus `ds_emb` as a fifth
source of apparent skill. The single `ablation_r` reported before v4 zeroed only
the first. The table here separates them, and the line that matters is the last
one: **intact minus the all-off floor**. That gap is the conditioning claim.
The intact value on its own is not, because a model can score without using any
perturbation information at all.

Watch `ds_shuffle` in particular: if shuffling dataset identity collapses the
metric, much of the score is dataset-level commonality rather than
perturbation-specific biology — the same class of artefact as the P2
shared-response shortcut.

### 2.5 Stratified evaluation

```bash
python src/maprna_p3/eval_fair.py --ckpt runs/p3_seed0/ckpt_p3_best_dev.pt \
  --data_dirs ... --esm_table ... --split_by target_gene --out_json strat.json
```

Reports both estimators side by side, per source dataset. Compare `per_pert` with
`per_pert` and `pooled` with `pooled`, never across.

---

## 3. Efficacy line

### 3.1 ASO, official patent-grouped folds

```bash
# with GENCODE region features (v3 configuration)
python src/efficacy/train_efficacy_v2.py --seeds 0 1 2 3 4
# ablations, each a separate run: report them, do not assume them
python src/efficacy/train_efficacy_v2.py --seeds 0 1 2 3 4 --no_region
python src/efficacy/train_efficacy_v2.py --seeds 0 1 2 3 4 --no_pos_emb
```

The `--no_pos_emb` arm reproduces the pre-v4 encoder, which had no positional
signal and was therefore a permutation-invariant bag of (base, sugar, backbone)
triples. Any claim that the model captures modification × sequence-context
interaction is the gap between those two arms. Until that is measured, the claim
is untested — which is why it was removed from the README.

Output carries both estimators: `pooled` over all held-out rows, and
`per_screen_median`, which is the estimator OligoAI reports. Do not place one
next to the other's literature value.

### 3.2 ASO, gene-holdout slice

```bash
python src/efficacy/train_efficacy.py --near_dup_check
```

Reports two held-out slices at the selected epoch: rows from held-out groups of
`--group_col`, and all rows of ~15 unseen target genes.

Two things to note. `--group_col` defaults to `custom_id`, which ASO Atlas
defines as the source patent **table**, not the patent — a single patent
contributes several tables, so this is not a patent-level split; pass a true
patent column if the table has one. And `--near_dup_check` counts held-out rows
sharing a long exact substring with a training row on the same target, because
patent families republish sequences and tile targets one nucleotide at a time, so
exact-match overlap is only a lower bound.

The gene-holdout slice is the informative one. Under the corrected enrichment
normalisation (1.0 = random), the previously logged value there was ≈0.98×
random: top-5% selection on unseen target genes was indistinguishable from
chance. Expect that to be the headline finding of this line, not the pooled
Spearman.

### 3.3 Cross-group leakage audit

```bash
python src/efficacy/seq_dedup.py \
  --parquet data/aso_atlas/aso_atlas_clean.parquet \
  --group_cols custom_id patent_number --threshold 0.9 \
  --out_json dedup_report.json
```

For each candidate grouping column, reports how many near-duplicate pairs cross a
group boundary. A grouping column controls leakage only if that count is small.
Run this **before** believing any grouped-split number. Use `--sample` for a
quick pass over the 188k-row set.

### 3.4 siRNA

```bash
python src/efficacy/train_sirna_v1.py        # target-grouped CV, inner-val selection
python src/efficacy/eval_sirna_external.py   # external Ichihara set, scored once
python src/efficacy/sirnamod_model.py --also_random_cv \
  --save_model data/drive_weights/sirnamod_xgb_v2.joblib
```

The external evaluation is the most defensible number in the repository: fixed
epoch budget, no test-based stopping, one scoring pass. It now also prints an
explicit target/sequence overlap check — previously the absence of overlap was
asserted but never verified, and the two files come from one literature family.
Its remaining caveat is that the hyperparameters were inherited from a protocol
that did use test-fold epoch selection.

`--also_random_cv` on the siRNAmod model shows what the previous ungrouped random
KFold reported. The gap to the grouped result is the part of the old number
attributable to seeing the same parent duplex on both sides of the split, not to
predicting modification effects.

---

## 4. Reporting checklist

Before a number leaves this repository:

- [ ] Selected on an inner split, not the reported slice
- [ ] ≥5 seeds; mean ± sample sd and a CI, never a bare point estimate
- [ ] Compared against `knn_esm2` / `ridge_esm2` (response) or the literature
      comparators **with the estimator named** (efficacy), paired where possible
- [ ] Estimator stated explicitly (`per_pert` vs `pooled`; `pooled` vs
      `per_screen_median`)
- [ ] Split grouping stated, and its leakage audited (`seq_dedup.py`, or the
      gene-overlap warning from `train_p3.py`)
- [ ] `provenance` block present, and `git.dirty` is `false`
- [ ] Ablations reported, not assumed — positional embedding, region features,
      and the per-axis conditioning table

A number that fails any of these is not ready, regardless of its value.

---

## 5. Known gaps

These are open and listed in ERRATA.md's Outstanding section:

* **No external model has been run.** GEARS, scGPT, CPA, AIDO.RNA-Pert, OligoAI,
  ASOptimizer, OligoWalk and RNAGenesis exist in this repository only as
  hard-coded comparator strings quoted from their papers, under different splits
  and preprocessing. The cheap controls are implemented; the published models are
  not. This is the single largest gap.
* **No number has been regenerated** under the corrected protocol. The code is
  fixed; the results are not.
* **The size of the masking effect (ERRATA E6) is unquantified** on real data —
  run with and without the mask on one checkpoint to measure it.
* **The L2 RNA-encoder track has no results**, only a launch command.
