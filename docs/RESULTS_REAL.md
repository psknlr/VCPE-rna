# First real-data results

Every performance figure in tags before v4 is withdrawn (see
[ERRATA.md](ERRATA.md)). This file records the **first numbers this repository
has ever produced on real Perturb-seq data** under the corrected protocol. They
are not a regeneration of the withdrawn figures: those were measured on different
datasets, from a host this environment cannot reach, under the protocol whose
defects are why they were withdrawn. This is a fresh measurement, and it should
be read as one.

Read [ERRATA.md](ERRATA.md) and [REPRODUCE.md](REPRODUCE.md) first. In one line:
never quote a number without its estimator, its split, its gene panel, and its
embedding table, because mixing any of those is how the withdrawn figures were
made.

## What was run

Two real CRISPRi screens, the kind of knockdown this model claims to predict:

| Dataset | Perturbations (>=20 cells) | Controls | Genes |
|---|---|---|---|
| ReplogleWeissman2022 RPE1 | 2204 | 3000 | 8749 |
| TianKampmann2021 CRISPRi (iPSC neurons) | 184 | 437 | 33538 |

- **Split:** `target_gene`, so a held-out perturbation's gene is genuinely
  unseen. 2328 distinct target genes, 349 held out.
- **Panel:** ranked on training perturbations only (ERRATA E15).
- **Estimator:** mean over perturbations of the within-perturbation correlation
  (never the pooled one; ERRATA E4).
- **Selection:** epoch chosen on an inner validation split; the held-out set is
  scored once (ERRATA E3).
- **Embedding table:** ESM2-35M, built by `tools/build_esm2_gene_table.py`. See
  the caveat at the end -- this choice is not neutral, and it flatters the model.
- **Seeds:** 3, each varying both initialisation and the split, so the intervals
  answer "does this survive a different choice of held-out genes".

Exact commands: [REPRODUCE.md](REPRODUCE.md) §2. Regenerate with
`results/real_35M/` provenance sidecars.

## Table A -- head vs controls (both screens, 3 seeds)

`pearson_dev`, mean over 3 seeds:

| model | pearson_dev | sd | what it is |
|---|---|---|---|
| **P3 head** | **+0.235** | 0.015 | the conditioned model |
| ridge_esm2 | +0.230 | 0.016 | closed-form ridge over the same ESM2 vectors |
| knn_esm2 | +0.208 | 0.034 | similarity-weighted retrieval over the same vectors |
| train_mean | +0.012 | 0.002 | the mean training residual; no gene identity |
| zero | +0.000 | 0.000 | predict no change |

The paired comparison is the one that matters -- it removes the split-to-split
variance, which is larger than the differences being compared:

| vs baseline | paired diff | CI95 | verdict |
|---|---|---|---|
| ridge_esm2 | +0.0051 | **[-0.016, +0.026]** | tie (wins 2/3 seeds) |
| knn_esm2 | +0.0272 | **[-0.009, +0.063]** | tie (wins 2/3 seeds) |
| train_mean | +0.2225 | [+0.207, +0.238] | beats, every seed |
| zero | +0.2347 | [+0.218, +0.251] | beats, every seed |

## What this says

**The honest headline: the model clears the trivial floors decisively, and does
not convincingly beat a linear map over the same embeddings.**

1. **It clears `zero` and `train_mean` on every seed, by a wide and
   split-stable margin.** This matters, and it is the thing the withdrawn P2
   result could not do honestly: there the conditioning was short-circuited and
   the model had learned only the dataset-level shared knockdown signature
   (`ablation_r = 1.000`). Here it has learned something perturbation-specific.
   `zero` scoring exactly 0 and `train_mean` near 0 also confirms the metric is
   not inflated -- a baseline with no gene information scores as it should, which
   is the direct check that ERRATA E6's masking fix holds on real data.

2. **Against the strong baselines it is a statistical tie.** The paired CI
   against `ridge_esm2` is [-0.016, +0.026] -- centred almost exactly on zero --
   and against `knn_esm2` it is [-0.009, +0.063]. The model wins 2 of 3 seeds
   against each, which is what "no reliable difference" looks like, not what an
   advantage looks like. A closed-form ridge regression over the same ESM2
   vectors, which takes seconds to fit and has no perturbation-specific
   architecture, matches the deep conditioned model on held-out genes.

This is not a failure of the model so much as a limit on what can currently be
claimed for it. The value of the conditioning architecture over a linear map on
the same features is **not established** on this data. That is a claim the
withdrawn numbers made and this measurement does not support.

## The caveat that decides how much weight this carries

**The 35M embedding table biases the comparison toward the model.** The head, the
ridge control and the k-NN control all read the same table, so it does not bias
the comparison *between* them by construction -- but it does move them unequally.
The controls are pure functions of the embedding; the head also sees the control
profile and the dataset embedding. A weaker table therefore handicaps the
controls more than the head. 35M is the weakest of the ESM2 variants that fit
this host in reasonable time (650M -- what the withdrawn results used -- needs
about a day on 4 CPUs).

So the near-tie above is, if anything, **generous to the model**: under a
stronger table the ridge and k-NN controls would be expected to gain relative to
the head, not lose. A re-run under ESM2-150M is the outstanding check
(`docs/ERRATA.md` Outstanding, and the build is already staged). If the ordering
holds there, the conclusion is robust; if ridge pulls ahead, the model's case is
weaker still. The conclusion will not be strengthened by a larger table -- only
possibly weakened -- so it is stated at its most favourable here.

## A scale check that matters for how Table A is read

Table A trains on both screens together. Table D repeats the head on the **Tian
screen alone** (184 perturbations, 27 held-out genes), on the same protocol, and
the result is worth stating on its own:

| model | pearson_dev (Tian alone, held-out) |
|---|---|
| P3 head | +0.003 |
| zero | +0.000 |
| ridge_esm2 | -0.003 |
| knn_esm2 | -0.009 |
| train_mean | -0.002 |

On 27 held-out genes, **nothing predicts anything** -- the head and every
baseline sit within noise of zero. The +0.235 in Table A is therefore carried by
Replogle's 2204 perturbations; strip to a 184-perturbation screen and the whole
comparison collapses to the floor. This is not a defect, it is the honest
data-scale limit: a cross-gene mapping cannot be learned from ~150 training
genes, by this model or by a ridge fit. Any single-small-dataset number in this
field, including several in the withdrawn tables, should be read with that in
mind.

## Table D -- head vs GEARS vs CPA (Tian alone, same split/panel/estimator)

`pearson_dev`, on the 27 held-out genes of the Tian screen:

| model | pearson_dev | runs | note |
|---|---|---|---|
| CPA | +0.018 | 1 | trained on cells |
| P3 head | +0.003 | 1 | pseudobulk |
| zero | +0.000 | 1 | predict no change |
| train_mean | -0.002 | 1 | no gene identity |
| ridge_esm2 | -0.003 | 1 | linear map over ESM2 |
| GEARS | -0.005 | 3 | mean; sd 0.036, range -0.047..+0.017 |
| knn_esm2 | -0.009 | 1 | retrieval over ESM2 |

**Every method is within +/-0.02 of zero, and the entire spread between them
(-0.009 to +0.018) is smaller than GEARS's own run-to-run noise (sd 0.036 across
3 identical runs).** On 27 held-out genes, no method -- the head, a linear map, a
retrieval baseline, CPA or GEARS -- is distinguishable from predict-no-change,
and the differences between them are not distinguishable from noise. This is the
data-scale floor described above, and it is the honest content of the table: it
demonstrates the comparison machinery runs all five methods on identical footing
on real data, and it establishes nothing about their relative merit, because
there is no signal above the floor to rank them by.

GEARS's spread is itself a documented finding (ERRATA E14a): it is not
deterministic at a fixed seed, so a single GEARS run is noise and only the mean
over runs, with its spread, is meaningful.

### Fairness caveats carried into Table D

Each external model trained on the cell-level input it is published on rather
than the pseudobulk the head uses (GEARS cannot run on pseudobulk at all -- its
differential-expression step has no statistics for a group of one). GEARS's
cell input was additionally capped at 20 cells per perturbation and 5000 genes
so it would finish on this 4-CPU host; the split, gene panel, predicted quantity
and estimator are identical to the head's regardless. Every such asymmetry is
recorded in each report's `fairness` block, and `table_D.md` reproduces them in
full. Read the table as "these published models, on this footing, on a task at
the floor" -- not as a controlled ranking of architectures.

## Why the external comparison is not yet conclusive

Because the Tian-alone task is at the floor for
every method (above), this comparison is best read as "can any published model
beat predict-no-change on 27 held-out genes" rather than as a ranking of
architectures. Each external model trained on the cell-level input it is
published on rather than the pseudobulk the head uses (GEARS cannot run on
pseudobulk at all -- its differential-expression step has no statistics for a
group of one), and that asymmetry is stated in each report's `fairness` block.
A larger real screen than the two reachable here (e.g. Replogle at cell level,
which did not fit this host) is needed for a comparison with room above the
floor.
