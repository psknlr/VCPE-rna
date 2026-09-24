<div align="center">

# VCPE-rna

**Virtual-Cell Perturbation-response Engine — RNA branch**

A license-clean (MIT / Apache-2.0 all the way down) **knockdown perturbation-response prediction engine**.
Given a knockdown perturbation (CRISPRi / siRNA / ASO target gene), it predicts the transcriptome-wide
response direction and magnitude — a commercially usable replacement for Non-Commercial weights.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)]()
[![Single GPU](https://img.shields.io/badge/Hardware-1%20x%2024GB%20GPU%20or%20CPU-orange.svg)]()

</div>

---

> ## ⚠️ Performance numbers are withdrawn pending re-run
>
> An audit of the evaluation code (v4) found defects that invalidate most
> previously published figures in this README. Among them: the per-screen
> Spearman was computed against an input **feature** rather than the label; the
> top-5% enrichment factor was scaled so that chance read 0.25 rather than 1.0;
> the reported epoch was selected on the very slice being reported; the headline
> `0.31 → 0.71` improvement subtracts **two different estimators** from one
> another; and the response-head split did not hold out target genes.
>
> Every defect, the numbers it invalidates, and how it was verified are in
> **[docs/ERRATA.md](docs/ERRATA.md)**. The code in `src/` is fixed; the numbers
> have not yet been regenerated, and the corrected ones are expected to be
> **lower**. Please do not cite figures from earlier tags.
>
> What survives unchanged: the P2 shared-response finding (strengthened — the
> same failure mode recurred twice more in our own pipeline), and the three
> documented negative results.

## Highlights

- **A methodological finding that kept reproducing.** The P2 gate "pass" was
  retracted by our own diagnostics: conditioning had been short-circuited and
  the model had learned only the dataset-level shared knockdown signature
  (`ablation_r = 1.000`). The per-gene deviation head restored conditioning.
  The v4 audit then found the *same class of error* twice more, in code written
  after that lesson — unmeasured panel columns carrying a per-dataset constant
  into an unmasked loss, and a silent switch of correlation estimator between
  two scripts. The transferable result is that shared and constant components
  inflate perturbation-prediction metrics **persistently**, and that catching
  one instance does not inoculate a pipeline against the next.
- **Chemistry is trained jointly with sequence — no bolted-on modification
  sub-model.** The ASO efficacy head embeds per-position **sugar** (2′-F /
  2′-MOE / cEt / unmodified) and **backbone** (PS/PO) features at the same level
  as the bases (16-d each → 48-d per position) into the *same* 2-layer
  transformer (`model_efficacy.py`). ⚠️ Releases up to v3 had **no positional
  encoding**, which made that encoder a permutation-invariant *bag* of
  (base, sugar, backbone) triples — it could not represent where a modification
  sat. Positional embeddings were added in v4 and the with/without ablation is
  now runnable (`--no_pos_emb`); until it is reported, treat "captures
  modification × sequence-context interactions" as untested. See ERRATA E9.
- **Pre-registered negative results, reported as failures.** Cross-cell-line
  zero-shot transfer: gate FAIL (`results/loco_results.json`, 0.081 against a
  0.15 threshold). Cell-line → in-vivo liver: NO-GO (44% direction agreement,
  negative Spearman). ASO potency on **unseen target genes**: top-5% enrichment
  ≈ 0.98× random — present in the committed logs since v1 and, until now, not
  mentioned here.
- **Cheap and reproducible core.** The response head is 5.7M parameters and
  trains in ~80 s on one GPU with no dependency on the MAP/SE backbone.

## Status of each result

Nothing below is a validated performance claim. The table records what was
reported, and why it is or is not currently usable.

| Stage | Previously reported | Status |
|---|---|---|
| **P1** — MAP base + knockdown conditioning | mse_DE 0.0215 · fm_cosine 0.9604 | ⚠️ epoch selected on the reported slice; the perturbation-mean baseline was miscomputed (E3, E12a) |
| **P2** — dual-axis conditioning | fm_cosine 0.9710 | ❌ retracted by our own erratum: conditioning bypassed, `ablation_r = 1.000` |
| **P2.3-B** — per-gene deviation head | ablation_r 1.000 → 0.277 · pearson_dev 0.3079 | ⚠️ conclusion (conditioning restored) stands; the ablation compared two factors at once when the RNA axis was on (E7) |
| **Data expansion** (+6 scPerturb datasets) | legacy pearson_dev 0.31 → 0.71 (×2.3) | ❌ withdrawn — estimator switch (E4), no gene-level holdout (E5), unmasked constant columns (E6) |
| **ASO efficacy head** (ASO Atlas) | pooled Spearman 0.283 · per-screen median 0.261 | ❌ per-screen figure void (E1); pooled figure selected on test (E3) |
| **siRNA efficacy head** (Huesken) | CV Spearman 0.607 | ❌ withdrawn — epoch and predictions taken from the best test-fold epoch (E3) |
| **siRNA external** (Ichihara_2007_2) | Spearman 0.588 | ✅ mechanically clean — single scoring pass, no test-based stopping. Caveats: hyperparameters inherited from a test-peeking protocol, single unseeded run, target/sequence overlap now checked explicitly |
| **siRNAmod XGBoost** (Martinelli 2023) | led the Highlights | ❌ no metrics existed anywhere in the repo; CV was ungrouped over parent duplexes (E12j) |

**No external baseline has ever been run in this repository.** GEARS, scGPT,
CPA, AIDO.RNA-Pert, OligoAI, ASOptimizer, OligoWalk and RNAGenesis appear only
as hard-coded comparator *strings* quoted from their papers, under different
splits and preprocessing. Earlier claims of being "comparable to OligoAI" and
of "matching the RNAGenesis benchmark" were not supported and have been removed
(E10). Running real baselines — including cheap ridge and
k-nearest-neighbour-on-ESM2 controls — is the largest outstanding item.

> **The methodological lesson, restated.** Conventional metrics (MSE / Pearson /
> cosine / top-k overlap) can all be inflated by a component shared across
> perturbations, so a perturbation-response model **must** be evaluated with
> shared-component-free discrimination metrics plus a conditioning ablation.
> This project discovered that in P2, wrote it down as its headline lesson, and
> then reproduced the same failure mode twice in later code. See
> [the P2 erratum](docs/reports/p2_result.md),
> [the P2.3-B report](docs/reports/p2_3b_result.md) and
> [docs/ERRATA.md](docs/ERRATA.md).

## Parallel Tracks

| Module | Path | Status |
|---|---|---|
| L2 RNA encoder | `src/l2/` | ⚠️ **incomplete** — scripts for RNAcentral filtering, MLM pretraining and InfoNCE alignment exist, but none of its three pre-registered gates (L2-a/b/c) has a recorded result. `docs/reports/l2_pretraining_log.md` contains only a launch command |
| Efficacy heads | `src/efficacy/` | code runs; all reported numbers withdrawn except the siRNA external evaluation (see ERRATA) |
| Data expansion | `src/data_expansion/`, `src/maprna_p3/ingest_*.py` | scPerturb (Zenodo) download → pseudobulk processing; ingestion itself is sound, the downstream evaluation was not (ERRATA E4–E6) |

## Evidence Chain

Reports written before the v4 audit are kept unedited as a record. Where they
state a performance number, read [docs/ERRATA.md](docs/ERRATA.md) first.

| Report | Contents |
|---|---|
| [docs/ERRATA.md](docs/ERRATA.md) | **Start here.** Every evaluation defect found, the numbers it invalidates, how each was verified, and what remains outstanding |
| [docs/PLAN.md](docs/PLAN.md) | Project plan v0.3: decisions D1–D3, staged go/no-go gates, benchmark protocol, license checklist |
| [docs/reports/p1_result.md](docs/reports/p1_result.md) | P1 gate report: 13-epoch trajectory, AMP spike diagnosis |
| [docs/reports/p2_result.md](docs/reports/p2_result.md) | P2 verdict + **erratum**: complete evidence chain of the shared-response shortcut |
| [docs/reports/p2_3b_result.md](docs/reports/p2_3b_result.md) | Why conditioning came back to life (per-gene direct head vs additive token) |
| [docs/reports/data_expansion_report.md](docs/reports/data_expansion_report.md) | ×8.8 expansion: per-domain comparison table and domain-gradient interpretation |
| [docs/reports/p3_platform_integration.md](docs/reports/p3_platform_integration.md) | Platform integration design: offline cache mode, zero new dependencies |
| [docs/reports/aso_sirna_data_landscape.md](docs/reports/aso_sirna_data_landscape.md) | Public ASO/siRNA data landscape (ASO Atlas and friends) |
| [docs/reports/l2_pretraining_log.md](docs/reports/l2_pretraining_log.md) | L2 encoder pretraining log |
| [docs/virtual_liver_p0_report.md](docs/virtual_liver_p0_report.md) | Virtual-liver P0: organ-level exam design, donor-context repair (P0.5), G0 adjudication |


## How to start

## Use a pretrained checkpoint (inference path)

> **⚠️ Checkpoint availability is currently broken.** This repository publishes
> **no releases**, and the release URL used in earlier versions of this README
> pointed at a different account (`elenalulu/VCPE-rna`) that is not the remote
> of this repository. Until a release is attached here, the files below cannot
> be obtained from this repository and the checksums in
> `results/MANIFEST_sha256.txt` cannot be checked against anything. Treat the
> inference path as unavailable and use the training path below.
>
> Two further caveats apply to any pre-v4 checkpoint that does surface:
> * the ASO efficacy checkpoint has **no positional encoding** (ERRATA E9);
>   `predict_service` detects this and labels its output `position_aware: false`
> * the P3 checkpoint embeds a redundant ~405 MB copy of the public ESM2 table
>   (ERRATA E12f); loaders now ignore it, so a v4 checkpoint is ~23 MB

Were a release attached, the layout would be:

```
ckpt_p3_best_dev.pt        →  results/p3_v21e/ckpt_p3_best_dev.pt   (or any path, pass via --ckpt)
ckpt_efficacy_v1.pt        →  results/efficacy_v1/ckpt_efficacy_v1.pt
sirna_v1_full_train.pt     →  results/sirna_v1/sirna_v1_full_train.pt
vcpe_cache_v6_*.json.gz    →  your virtual-cell engine's cache directory (drop-in)
```

**What you still need:** the ESM2 gene-embedding table, one Perturb-seq h5ad for the
control context, and `gene_transcripts.fa` (public/third-party data, ~3 GB total — see
[data/README.md](data/README.md)). These are method dependencies (conditioning lookup +
control-expression baseline + RNA sequences), not a training cost.

**1) Build a platform cache from the released P3 ckpt** (~30 min CPU, no MAP/SE deps;
the RNA sequence encoder embedded in the ckpt is loaded automatically):

```bash
python src/maprna_p3/export_vcpe_cache_v2.py \
  --ckpt results/p3_v21e/ckpt_p3_best_dev.pt \
  --adamson_h5ad data/adamson/perturb_processed.h5ad \
  --esm_table data/drive_weights/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt \
  --fasta data/gene_transcripts.fa \
  --out vcpe_cache_v6_k562a.json.gz --context_name k562a
```

Output: a cache in the platform's schema (`genes[sym] = {top_up,
top_down}` ranked by the residual after common-core subtraction, plus `common_response_core`
and MIT/Apache-only metadata) — swapping the file swaps the engine, zero new dependencies.
Other contexts: `--context_name hepg2` / `jurkat` (+ `--ctrl_h5ad`). ⚠️ Before v4 this
flag reached only the metadata string while the model always received dataset index 0,
so a cache labelled `hepg2` was generated with the adamson/K562 context embedding
(ERRATA E8). Any cache built with an earlier version should be regenerated.

**2) ASO / siRNA efficacy scoring** (Python API; loads
`results/efficacy_v1/ckpt_efficacy_v1.pt` from that default path). Pass the chemistry
explicitly — the default is unmodified DNA with a PS backbone, which is **not a gapmer**
and sits outside the training distribution (~53% 5-10-5 MOE, ~30% 3-10-3 cEt):

```python
import sys; sys.path.insert(0, "src/efficacy")
from predict_service import predict_inhibition
predict_inhibition("TGCATCGTACGTAGCTGATC", "APOC3", cell_line="hepg2", wing_mod="moe")
# → inhibition_pct, kd (= inhibition_pct/100), position_aware, heldout_metrics
# wing_mod: 'dna' (default, NOT a gapmer) / 'moe' / 'lna' (→ cEt) / 'f' / 'mix'
```

`predict_kd_hybrid` also exists, but for external users it is a thin wrapper around
`predict_inhibition`: its alternative branch requires `$RNA_ROBOT_HOME` pointing at a
**private** repository, and the `xgboost_v5` model it would route to is not distributed
here — no weights, no training script, no feature extractor, no evaluation. The
comparison that motivated the routing (PCC 0.58 vs 0.51) is not reproducible from this
repository and should not be cited.

**Sensitivity to chemistry — claim withdrawn.** Earlier versions of this README
advertised "~11% as plain DNA vs ~38% with LNA/cEt wings" as evidence that the model
captures modification × sequence-context interactions. Because the pre-v4 encoder had no
positional encoding (ERRATA E9), that difference could only reflect the *number* of
modified sugars: moving the same modifications elsewhere in the molecule yields an
identical prediction. The claim is reinstated only once the `--no_pos_emb` ablation is
reported.

**Model scope — read before over-interpreting.** Inputs are gapmer/guide **sequence +
wing chemistry + cell line**. Dose is a trained input but is not exposed in the public
API — every call is made at the training-mean dose with the missing flag set.
**Delivery is NOT a model input**: GalNAc / LNP / dosing route / tissue exposure are out
of scope (training data is in-vitro, delivery-normalised potency). Treat outputs as
*intrinsic in-vitro potency*, not tissue-level efficacy. Note also that top-5% selection
on **unseen target genes** performed at ≈0.98× random (ERRATA E2), so do not rely on
this model to triage a target it was not trained on.

**3) Full virtual-cell page integration** (the rna_robot platform pattern): the cache file
alone drives post-knockdown response display — see
[docs/reports/p3_platform_integration.md](docs/reports/p3_platform_integration.md).

## Reproduce from scratch (training path)

The P3 response head has **no dependency** on the SE backbone, the MAP repo, or
flash-attention — it trains in minutes on a single GPU (~80 s for 60 epochs) or ~2 h on
CPU. Note that this regenerates the numbers under the **corrected** protocol, so results
will not match the withdrawn figures in earlier tags.

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # metric-definition regression tests, no data needed

# 1) Prepare data (see data/README.md):
#    - Perturb-seq h5ad in GEARS format (adamson / norman / replogle_rpe1_essential)
#    - ESM2 gene embedding table Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt

# 2) Train the P3 per-gene deviation head (60 epochs ≈ 80 s on GPU, ~2 h on CPU).
#    --split_by target_gene (the default) holds out whole target genes; use
#    --split_by pert only to reproduce historical numbers, and read ERRATA E5 first.
python src/maprna_p3/train_p3.py \
  --data_dirs data/adamson/perturb_processed.h5ad \
              data/norman/perturb_processed.h5ad \
              data/replogle_rpe1_essential/perturb_processed.h5ad \
  --esm_table data/drive_weights/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt \
  --out_dir ./p3_out --epochs 60 --split_by target_gene

# 3) Stratified evaluation. Reports BOTH estimators side by side; --split_by must
#    match the value used for training or the reconstructed split is not the
#    model's split.
python src/maprna_p3/eval_fair.py --ckpt ./p3_out/ckpt_p3_best_dev.pt \
  --data_dirs ... --esm_table ... --split_by target_gene --out_json eval.json

# 4) ASO efficacy CV, with inner-validation epoch selection and bootstrap CIs.
#    Run both arms of the positional ablation and report both.
python src/efficacy/train_efficacy_v2.py --seeds 0 1 2 3 4
python src/efficacy/train_efficacy_v2.py --seeds 0 1 2 3 4 --no_pos_emb

# 5) siRNAmod, grouped by parent duplex; --also_random_cv shows how much of the
#    previously reported number was parent-duplex recall.
python src/efficacy/sirnamod_model.py --also_random_cv \
  --save_model data/drive_weights/sirnamod_xgb_v2.joblib

# 6) Full chain (P1/P2 need the MAP base + SE weights, 1 × 24G GPU)
#    Command templates live in each script's docstring and in docs/PLAN.md
```

**Pretrained artifacts.** `results/p3_v21e/` ships the aligned RNA sequence encoder
(`rna_enc_from_ckpt.pt`, 9 MB) and an example full-gene vector cache. The larger
checkpoints and platform caches are **not currently downloadable** — see the note in the
inference section above. `results/MANIFEST_sha256.txt` records their SHA-256 digests, but
with no release attached there is nothing to verify them against.

`data/drive_weights/sirnamod_xgb_v1.joblib` is shipped, but **no committed script
produced it** and no metrics for it exist anywhere in the repository. `sirnamod_model.py`
now contains a producer (`--save_model`), so regenerate the artifact rather than relying
on the shipped one.

## License & Attribution

- Code in this repository: **MIT License** (see [LICENSE](LICENSE))
- Architecture upstream: [MAP](https://github.com/MAGIC-AI4Med/MAP) (MIT) + MAP-KG (Apache-2.0); the P2.3-B deviation head is an independent implementation
- Per-dataset licenses and acquisition paths: [data/README.md](data/README.md)
- ⚠️ **ASO Atlas is derived from USPTO patents.** This repository ships training/eval code only, never the data; any commercial use requires independent legal review


## Methodology Notes

1. **Shared and constant components inflate perturbation-response metrics, and
   they keep coming back.** MSE / Pearson / cosine / top-k can all be dominated
   by a component shared across perturbations, so evaluation must include
   shared-component-free discrimination metrics *and* a conditioning ablation.
   This project established that in P2 — and then hit the same failure mode
   twice more in code written afterwards: unmeasured panel columns carrying a
   per-dataset constant into an unmasked loss (E6), and a silent change of
   correlation estimator between two scripts (E4). Knowing about the trap is
   not sufficient; the checks have to be mechanical.
2. **Name your estimator.** `per_item_correlation` and `pooled_correlation`
   answer different questions — the second also rewards getting between-item
   *levels* right — and they differ by ~1.5× on the same checkpoint here.
   Reporting both under one name produced a headline improvement that was
   partly an artefact of switching between them.
3. **Grouped splits, or no generalisation claim.** A genome-wide screen in the
   training set covers essentially every target gene elsewhere in the corpus, so
   a random split over (dataset, condition) pairs does not produce unseen
   perturbations (E5). The same applies to ASO Atlas patent tables and to
   siRNAmod parent duplexes.
4. **Select on data you will not report.** Three independent scripts here chose
   the reported epoch using the reported slice (E3). An inner validation split
   costs one line and is the difference between an estimate and a maximum.
5. **A permutation-invariant encoder cannot model position.** Self-attention
   plus mean/max pooling with no positional signal is a bag of tokens; it cannot
   express where a chemical modification sits, however many modification
   features are fed in at the same level as the bases (E9).
6. **Additive tokens get structurally drowned** in a large backbone's residual
   stream — in data-limited regimes a per-gene direct residual head is the right
   architecture (the P2.3-B conclusion, unaffected by the audit).
7. **Re-calibrate LR after data scaling**: ×8.8 data with lr 1e-3 collapses into
   the zero-residual basin; lr 3e-4 trains 40 epochs with no instability.
8. **Chemistry is in the model; delivery is deliberately out.** GalNAc / LNP /
   route are not modelled at all (in-vitro, delivery-normalised training data).
   Delivery-aware extrapolation belongs in a PK/PD layer, not the efficacy head
   — do not read `kd` as tissue exposure. The magnitude of the chemistry effect
   is not currently quantified; see the withdrawn claim above.
