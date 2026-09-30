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

86 tests, no data or checkpoint assets. This includes an end-to-end smoke test
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
sources and licences. Two assets are needed for the response line, and **neither
can simply be downloaded any more**, so both are now built by a script in this
repository.

### 1.1 The conditioning table (build it; it is not an asset)

Every response number is conditioned on a `gene_symbol -> ESM2 vector` table.
The table used before v4 existed only on the machine that produced the withdrawn
results: it was never committed and there was no script to rebuild it, so no
result was reproducible even in principle. Build it from public inputs:

```bash
# 1. human reviewed proteome with primary gene symbols (~20k entries, ~1 min)
python - <<'EOF'
import urllib.request, urllib.parse, gzip, re
base, rows, hdr = "https://rest.uniprot.org/uniprotkb/search", [], None
url = base + "?" + urllib.parse.urlencode(dict(
    query="(organism_id:9606) AND (reviewed:true)", format="tsv",
    fields="accession,gene_primary,sequence,length,protein_existence", size="500"))
while url:
    r = urllib.request.urlopen(urllib.request.Request(
        url, headers={"Accept-Encoding": "gzip"}), timeout=180)
    raw = r.read()
    while raw[:2] == b"\x1f\x8b":        # sniff: a proxy may re-gzip
        raw = gzip.decompress(raw)
    lines = raw.decode().splitlines()
    if hdr is None: hdr = lines[0]
    rows += lines[1:]
    m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", ""))
    url = m.group(1) if m else None
open("uniprot_human_reviewed.tsv", "w").write("\n".join([hdr] + rows) + "\n")
EOF

# 2. embed them (resumable; writes shards, so an interruption costs minutes not hours)
python tools/build_esm2_gene_table.py \
  --uniprot_tsv uniprot_human_reviewed.tsv \
  --out esm2_150M_human.pt \
  --model facebook/esm2_t30_150M_UR50D --threads 4
```

**The ESM2 variant is part of the result, not a detail.** On 4 CPUs the 650M
variant used before v4 needs roughly a day, 150M about 5–6 hours and 35M about
90 minutes. The variant is written into the `.provenance.json` sidecar. It does
not affect the *comparison* against the ESM2 retrieval and ridge controls — all
three read the same table — but it does affect the absolute scores, so a score
and its table must be quoted together. Note the direction of the bias if you
economise here: the controls are pure functions of the embedding, while the model
also sees the control profile and the dataset embedding, so a weaker table
handicaps the controls more than the model.

### 1.2 The Perturb-seq data

GEARS' own `perturb_processed.h5ad` files are hosted on
`dataverse.harvard.edu`, which some environments refuse outright (this one
returns HTTP 403 for every path). The same experiments are on Zenodo in
[scPerturb](https://doi.org/10.5281/zenodo.13350497), which packages them
differently — raw counts, `obs['perturbation']`, no `condition` column — so they
go through an adapter:

```bash
# CRISPRi screens, which is what the response head claims to predict
for f in ReplogleWeissman2022_rpe1 TianKampmann2021_CRISPRi; do
  curl -L -o $f.h5ad \
    "https://zenodo.org/api/records/13350497/files/$f.h5ad/content"
done

# pseudobulk: what the P3 head predicts
python src/maprna_p3/ingest_scperturb.py --h5ad ReplogleWeissman2022_rpe1.h5ad \
  --out gears_fmt/replogle_rpe1.h5ad --min_cells 20 --n_ctrl 3000 --block 8192

# cells: what GEARS and CPA need (see 2.4). Leave the cap off, or the cell file's
# per-condition mean stops equalling the pseudobulk file and the panel diverges.
python src/maprna_p3/ingest_scperturb.py --h5ad TianKampmann2021_CRISPRi.h5ad \
  --out gears_fmt/tian2021_crispri_cells.h5ad --min_cells 20 --mode cells
```

Runs on the pseudobulk output must pass `--min_cells 1`: the cell-count gate
already ran inside the adapter, and after aggregation every perturbation is one
row, so the loader's own gate would otherwise drop everything.

Two datasets scPerturb ships are **not** usable here and it is worth knowing why
before reaching for them. `AdamsonWeissman2016_GSM2406681_10X010` has no
`control` label at all — its `nperts` column reads 2 for single guides, and the
only candidates are `nan` and `*` — and guessing which guide is the control
would silently corrupt every fold change, so the adapter refuses. Any dataset
with genuine multi-gene perturbations is also refused, because the separator
differs per dataset and mislabelling a combination as a single gene is worse
than not running.

Optional: STRING edges (`is_neighbor` feature), GENCODE transcript FASTA (RNA
sequence axis).

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
  --data_dirs gears_fmt/replogle_rpe1.h5ad gears_fmt/tian2021_crispri.h5ad \
  --esm_table esm2_150M_human.pt \
  --out_dir runs/p3_seed0 --epochs 60 --split_by target_gene \
  --min_cells 1 --n_hvg 2000
```

(`--min_cells 1` because the ingest adapter already applied the cell-count gate;
see 1.2. With GEARS' own `perturb_processed.h5ad` files, drop it.)

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

### 2.4b External baselines on the same footing

The two external models are cell-level; the P3 head is a pseudobulk model. This
is not a detail that can be papered over: GEARS' differential-expression step
calls scanpy's `rank_genes_groups` per condition and **fails outright** on a
pseudobulk file, because a group of one sample has no statistics. So each model
gets the input it is published on, from the same cells, and everything that
defines the comparison is taken from the file the P3 head reads:

```bash
python src/maprna_p3/baseline_gears.py \
  --data_dirs gears_fmt/tian2021_crispri.h5ad          `# split, panel, target` \
  --train_h5ad gears_fmt/tian2021_crispri_cells.h5ad   `# what GEARS trains on` \
  --esm_table esm2_150M_human.pt \
  --out_json runs/gears.json --n_hvg 2000 --min_cells 1 \
  --split_by target_gene --hvg_from train --seed 0 --runs 5
```

The adapter refuses to run if the two files disagree about which conditions
exist, because that would move the split without saying so. It also checks the
cell file up front and tells you to build one if you passed pseudobulk, rather
than dying inside scanpy with a ValueError that names every perturbation at once
and reads like a data problem.

Quote GEARS as a mean over `--runs` repetitions. A single GEARS run is noise:
seven identical `--seed 0` invocations spanned `pearson_dev` −0.023 to +0.051
(ERRATA E14a).

CPA needs its own environment — it pins `torch<2.0.0` against this
repository's `torch>=2.1`, and installing it in place turned 95 passing tests
into 8 failures — so it runs in three stages with the interface on disk:

```bash
python src/maprna_p3/baseline_cpa.py export --work runs/cpa \
  --data_dirs gears_fmt/tian2021_crispri.h5ad \
  --train_h5ad gears_fmt/tian2021_crispri_cells.h5ad \
  --esm_table esm2_150M_human.pt --n_hvg 2000 --min_cells 1 \
  --split_by target_gene --hvg_from train
python src/maprna_p3/baseline_cpa.py run   --work runs/cpa --prepare_venv
python src/maprna_p3/baseline_cpa.py score --work runs/cpa --out_json runs/cpa.json
```

**`--hvg_from` and `--n_hvg` must match the P3 run exactly.** The panel is part
of the protocol, not a detail: it decides which genes are scored. Every script
that builds the comparison data exposes the same flags, and
`tests/test_protocol_flags_agree.py` fails if a new one does not — a flag added
to `train_p3.py` alone once left both external adapters unable to be told which
panel they were supposed to share.

Every report carries a `fairness` block naming what is **not** controlled:
tuning effort, model capacity, each model's own preprocessing, and — when
`--train_h5ad` is used — the deliberate difference in input granularity. Read it
before quoting any number from it.

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
* **The size of the masking effect (ERRATA E6) is unquantified** on real data.
  The harness is a flag: run the same configuration twice, once with `--no_mask`,
  and compare. Watch the `train_mean` baseline as well as the model — it carries
  no perturbation information, so any gain it shows when unmasked is pure
  artefact. On synthetic data unmasking lifted it from −0.08 to +0.11.
* **The L2 RNA-encoder track has no results**, only a launch command.
