# Data Acquisition Guide

VCPE-rna does **not redistribute any dataset** (size and licensing reasons). This file lists, module by
module, what each component needs, where to get it, and under which license. Download scripts:
`src/l2/download_rnacentral.py`, `src/data_expansion/download_scperturb.py`, `src/efficacy/download_oligogym.py`.

## 1. Main chain: Perturb-seq (GEARS-format h5ad)

| Dataset | Used for | Source | License |
|---|---|---|---|
| adamson / norman / replogle_rpe1_essential `perturb_processed.h5ad` | P1 / P2 / P3 training | [GEARS](https://github.com/snap-stanford/GEARS) data release (~0.5–2.7 GB each) | see GEARS repo |
| ReplogleWeissman2022 rpe1, TianKampmann2021 CRISPRi, K562 gwps, NadigOConner2024 hepg2/jurkat, et al. | P3 training and evaluation; data-expansion track | [scPerturb, Zenodo record 13350497](https://zenodo.org/records/13350497) (`src/data_expansion/download_scperturb.py`, supports `--all`) | CC BY 4.0 |
| ESM2 gene embedding table | conditioning lookup | **built locally** by `tools/build_esm2_gene_table.py` from the UniProt REST API | UniProt: CC BY 4.0; ESM2 weights: MIT (facebook/esm2_*) |
| `gene_transcripts.fa` | RNA sequence axis | extracted from GENCODE v44 transcript FASTA | GENCODE license |

**Two practical notes on the Perturb-seq data, both learned the hard way:**

`dataverse.harvard.edu`, which hosts GEARS' own `perturb_processed.h5ad` files, is
refused outright by some environments' egress policy (HTTP 403 on every path,
including this project's). The scPerturb copies on Zenodo are reachable and cover
the same experiments, but they are packaged differently — raw counts,
`obs['perturbation']`, no `condition` column — so they need
`src/maprna_p3/ingest_scperturb.py`. See
[../docs/REPRODUCE.md](../docs/REPRODUCE.md) §1.2.

Not every scPerturb file is usable, and the adapter refuses rather than guesses:
`AdamsonWeissman2016_GSM2406681_10X010` has no `control` label at all (its
`nperts` column reads 2 for single guides, and the only candidates are `nan` and
`*`), and picking a control guide by name would silently corrupt every fold change
in the run. Files with genuine multi-gene perturbations are refused for the same
reason — the separator differs per dataset.

**The ESM2 table is no longer treated as a downloadable asset.** The 392 MB file
listed below was never committed and no script rebuilt it, so every number
conditioned on it was unreproducible even in principle
([../docs/ERRATA.md](../docs/ERRATA.md)). It is now derived from public inputs by
a recorded procedure, and the ESM2 variant, the truncation length and the pooling
are written into a `.provenance.json` sidecar — the variant matters, because on
4 CPUs the 650M model needs roughly a day for the human proteome while 35M needs
about 90 minutes, and the absolute scores differ between them even though the
comparison against the ESM2 retrieval and ridge controls does not.

## 2. MAP base weights (only P1/P2 need them; the P2.3-B head does not)

From the MAP / MAP-KG releases ([MAGIC-AI4Med/MAP](https://github.com/MAGIC-AI4Med/MAP), MIT;
HF `RainGate/MAP-KG`, Apache-2.0):

| File | Size |
|---|---|
| `epoch_3.pt` (perturbation-predictor base) | 4.2 GB |
| `se600m.safetensors` (SE cell encoder) | 2.4 GB |
| `Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt` | 392 MB |
| `mapkg_encoder_v3.pt` (KG encoder) | 1.1 GB |

P1/P2 training additionally needs the patched MAP repo (10 engineering patches by this team; patch
notes in `docs/gpu_pack/`).

## 3. Efficacy heads (`src/efficacy/`)

| Dataset | Used for | Source | License | ⚠️ |
|---|---|---|---|---|
| ASO Atlas v1/v2 (188k gapmers, chemistry + potency) | ASO efficacy head | HF `barneyhill/aso-atlas` / `aso-atlas-2` (OligoAI paper companion) | see HF dataset page | **derived from USPTO patents**; independent legal review required before any commercial use; not included in this repo |
| OligoGym, 12 datasets (ichihara / alharbi / huesken, etc.) | siRNA/ASO external eval | HF `CollageBio/oligo-datasets` (direct download via `download_oligogym.py`) | CC BY 4.0 | — |
| GSE183535 (MYC ASO RNA-seq), GSE289964 (in vivo whole liver), GSE293987 | domain-transfer eval | GEO | see each series | — |

## 4. L2 RNA encoder pretraining (`src/l2/`)

| Dataset | Source | License |
|---|---|---|
| RNAcentral active fasta (~11 GB, 45M seqs) | [RNAcentral FTP](https://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/) (`download_rnacentral.py`, resumable) | CC0 |
| STRING v12 human edges | [string-db.org](https://string-db.org) | CC BY 4.0 |

## 5. Miscellaneous

| Dataset | Used for | Source |
|---|---|---|
| GTEx v10 gene median TPM | primary-hepatocyte donor context (`p05_donor_context.py`) | [gtexportal.org](https://www.gtexportal.org) |
| MAP-KG knowledge-graph CSVs | KG alignment | HF `RainGate/MAP-KG` (Apache-2.0) |

## Conventions

- Default data root: `data/` inside the repository (git-ignored); some scripts honor `VCPE_DATA_DIR`
- Why not committed: ~35 GB total + several licenses that do not allow redistribution
