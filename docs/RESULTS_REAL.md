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
- **Embedding table:** the run is reported under both ESM2-35M and ESM2-150M
  (built by `tools/build_esm2_gene_table.py`). The choice is not neutral and the
  result depends on it -- see the two Table A sections and the correction near the
  end, where the direction of that dependence turned out opposite to my
  prediction.
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

## Table A under the stronger 150M embedding

The same run under ESM2-150M (640-dim, vs 35M's 480-dim), which is closer to the
650M table the withdrawn work used:

| model | pearson_dev | sd | 35M value |
|---|---|---|---|
| **P3 head** | **+0.293** | 0.006 | +0.235 |
| ridge_esm2 | +0.265 | — | +0.230 |
| knn_esm2 | +0.212 | — | +0.208 |
| train_mean | +0.012 | — | +0.012 |
| zero | +0.000 | — | +0.000 |

| vs baseline | paired diff | CI95 | verdict |
|---|---|---|---|
| ridge_esm2 | +0.0274 | **[+0.010, +0.045]** | beats, every seed |
| knn_esm2 | +0.0812 | **[+0.061, +0.102]** | beats, every seed |
| train_mean | +0.281 | [+0.273, +0.289] | beats, every seed |
| zero | +0.293 | [+0.287, +0.299] | beats, every seed |

## What this says

**The honest headline is embedding-dependent, and it moved in the model's favour
after I predicted it would not.** On the weaker 35M embedding the model ties the
linear and retrieval baselines (paired CIs include zero). On the stronger 150M
embedding it beats both on every seed, with paired CIs that exclude zero -- a
real, if modest, advantage (+0.027 over ridge, +0.081 over k-NN).

1. **It clears `zero` and `train_mean` on every seed, by a wide and
   split-stable margin.** This matters, and it is the thing the withdrawn P2
   result could not do honestly: there the conditioning was short-circuited and
   the model had learned only the dataset-level shared knockdown signature
   (`ablation_r = 1.000`). Here it has learned something perturbation-specific.
   `zero` scoring exactly 0 and `train_mean` near 0 also confirms the metric is
   not inflated -- a baseline with no gene information scores as it should, which
   is the direct check that ERRATA E6's masking fix holds on real data.

2. **Against the strong baselines the result depends on the embedding.** On 35M
   it is a tie: paired CI against `ridge_esm2` is [-0.016, +0.026], centred on
   zero, 2/3 seeds. On 150M the model beats `ridge_esm2` on every seed, paired CI
   [+0.010, +0.045] excluding zero, and `knn_esm2` by more, [+0.061, +0.102]. So
   the conditioning architecture does buy something over a linear map on the same
   features -- but the margin is small (+0.027 pearson over a ridge fit that takes
   seconds) and vanishes into noise on a weaker embedding. "Beats a ridge baseline
   by ~0.03 on a good embedding, ties it on a poor one" is the honest claim; it is
   not the large margin the withdrawn numbers asserted.

3. **What the model has is real conditioning, not a shared-response shortcut.**
   Permuting the target-gene vectors costs -0.172 of the +0.288 score and changes
   the prediction itself drastically; shuffling the dataset embedding costs only
   -0.064. The failure that started this whole errata -- a model scoring well
   while ignoring which gene was perturbed -- is absent here, checked with the
   same instrument that exposed it. Details and an important caveat about which
   floor to quote are in the ablation section below.

The embedding matters more than I expected, and in the opposite direction (next
section). The value of the architecture over a linear map is **modest and
established only on the stronger embedding** -- a narrower claim than the
withdrawn numbers made, but not the flat negative the 35M table alone suggested.
What is now also established, and was not before, is that the score rests on
gene-specific conditioning rather than on the artefact this project twice
mistook for signal.

## A prediction I got wrong, recorded because getting it wrong is the point

When only the 35M result existed, I argued in this file that the 35M table
*flattered* the model: the controls are pure functions of the embedding while the
head also sees the control profile, so -- I reasoned -- a weaker embedding should
handicap the controls more than the head, and a stronger table "can only weaken
the model's case." I stated the tie "at its most favourable" on that basis.

**The 150M re-run falsified this.** The stronger embedding helped the head *most*,
not least:

| going 35M -> 150M | gain in pearson_dev |
|---|---|
| P3 head | +0.058 |
| ridge_esm2 | +0.035 |
| knn_esm2 | +0.004 |

The head, a non-linear model, exploits the richer embedding geometry that a linear
ridge map and a nearest-neighbour lookup mostly cannot. My argument was plausible
and wrong, and the failure mode is exactly the one this audit is about: a
confident claim asserted from reasoning instead of measured. It is corrected here
rather than quietly replaced.

What this does **not** license is extrapolating the trend to the 650M table the
withdrawn work used. The head's advantage grew from tie (35M) to +0.027 (150M),
and it is tempting to say 650M would widen it further -- but that is precisely the
unmeasured inference that got the original numbers withdrawn. 650M needs about a
day on this 4-CPU host and has not been run. The claim stands where it was
measured: at 150M, a small and consistent advantage over a linear baseline, on a
mid-sized embedding, on two CRISPRi screens whose smaller member (Table D) is at
the floor for every method.

## Is the conditioning real? (per-axis ablation, ERRATA E7)

The score above only means something if it depends on *which gene was perturbed*.
The P2 erratum exists because once it did not: the prediction was unchanged when
conditioning was removed (`ablation_r = 1.000`). Run on real data for the first
time, on the 150M seed-0 checkpoint (352 held-out items):

| axis disabled | r vs intact prediction | pearson_dev | delta |
|---|---|---|---|
| intact | 1.0000 | +0.2878 | — |
| **esm_zero** (target vector -> 0) | 0.3163 | +0.0056 | **-0.2823** |
| **esm_shuffle** (target vectors permuted) | 0.1609 | +0.1161 | **-0.1717** |
| ds_shuffle (dataset embedding permuted) | 0.8347 | +0.2244 | -0.0635 |
| rna_off | 1.0000 | +0.2878 | 0.0000 |
| is_tgt_off / is_tgt_shuffle | 1.0000 | +0.2878 | 0.0000 |
| is_nb_off / is_nb_shuffle | 1.0000 | +0.2878 | 0.0000 |
| all_off | 0.3163 | +0.0056 | -0.2822 |

**The conditioning is real.** Destroying the target-gene identity collapses the
score to the floor, and the prediction itself changes drastically (r drops to
0.32). This is the precise opposite of the P2 failure mode, measured the same way
that exposed it. `ds_shuffle` costs only -0.064, so dataset-level commonality is
carrying a small part of the score rather than the bulk -- the P2 shared-response
shortcut is absent here.

**But "98% of the score is conditioning" would be an overstatement, and the
script's own `conditioning_gain` (+0.282) invites it.** Zeroing a vector pushes
the input off-distribution, so the model produces something degenerate and the
floor is flattered. `esm_shuffle` is the better-controlled ablation: it preserves
the input statistics exactly and destroys only the identity mapping. It leaves
+0.116. So the defensible claim is that **at least ~60% of the score
(-0.172 of +0.288) is gene-specific conditioning**, not 98%. Both numbers are in
the table; the shuffle one is the one to quote.

**Three of the four identity channels are inert**, to the bit: `rna_off`,
`is_tgt_*` and `is_nb_*` leave the prediction bit-identical (r = 1.0000,
delta = 0.0000), and `esm_zero` and `all_off` are the same number. All the
conditioning flows through the target gene's ESM2 vector alone. Two of those are
inert by configuration rather than by defect -- the RNA encoder and the STRING
neighbour graph were not loaded for this run. `is_target` is different: it marks
whether the perturbed gene is itself in the response panel, and at `--n_hvg 500`
out of ~8700 genes that is true for only ~6% of perturbations, so the channel is
almost always all-zero and its contribution cannot be measured here. The
architecture is described as four identity channels; in this configuration it is
**effectively a one-channel model**, and a wider panel would be needed to say
whether `is_target` carries anything at all.

## E6 on real data: the masking A/B, finally measured

Same configuration, same seed, one flag apart (ERRATA E6, previously deferred):

| | model | train_mean | knn_esm2 | ridge_esm2 | measured fraction |
|---|---|---|---|---|---|
| masked (default) | +0.2878 | +0.0120 | +0.1877 | +0.2428 | 0.9986 |
| `--no_mask` | +0.2874 | +0.0166 | +0.1881 | +0.2429 | 1.0000 |

**The prediction made in advance holds.** With 99.86% of the panel measured, the
constant block is 0.14% of the matrix and unmasking moves the model by -0.0004 --
nothing. The mechanism is still visible in the right place and the right
direction: `train_mean`, which carries no perturbation information at all and
must sit at the floor, gains **+0.0046** when unmasked. That is E6 in miniature,
and the reason it is miniature is the dataset pair, not the defect: these two
screens overlap almost completely. It says nothing about the magnitude in the
withdrawn runs, which combined a genome-wide screen with a ~5k-gene panel. Do not
quote one as an estimate of the other.

## Activating the dead network channel: a negative result, and why it was predictable

The ablation above found all conditioning flowing through one channel, so the
head's +0.027 edge over ridge is non-linearity on features the controls also
have. `is_neighbor` was the one channel that could change that: it encodes a
mechanistic prior -- knocking a gene down perturbs its interaction partners --
that ridge and k-NN structurally cannot express from a target vector. It had
simply never been fed. Built from STRING v12.0 (443,966 edges at
combined_score >= 700, 77% of genes covered, median degree 13) and re-run on the
same 3 seeds and 150M table:

| | pearson_dev |
|---|---|
| head without the network channel | +0.2927 +/- 0.0055 |
| head **with** the network channel | +0.2932 +/- 0.0056 |
| delta | **+0.0005** |

**It buys nothing.** +0.0005 is an order of magnitude below the seed-to-seed sd
(0.0055). The ablation agrees: `is_nb_off` costs -0.0001 and `is_nb_shuffle`
-0.0002, so the channel is now technically live (r = 0.9999 rather than exactly
1.0000) but carries no usable information. The graph-only control agrees too:
`neighbor_prior` scores **+0.0114** against `train_mean`'s +0.0121, i.e. the one
fitted graph parameter finds nothing -- the graph buys nothing on its own either.

**The reason is arithmetic, and it means the design cannot be rescued by more
data.** `is_neighbor` fires on **0.27%** of held-out (item, gene) cells: mean
1.37 flagged genes per 500-gene panel, and only **9 of 352** held-out items have
as many as 5 partner genes in the panel at all. A target's ~13-16 partners are a
tiny subset of the ~8,700-gene universe, so the fraction of any panel that is
"partner" is about degree/universe ~ 0.15-0.3% **whatever the panel size** --
widen the panel and numerator and denominator grow together. A perfect prior on
0.27% of cells cannot move a panel-wide correlation.

Scoring the two checkpoints on the partner cells alone confirms the metric is not
merely hiding an effect -- it confirms there is nothing measurable there:

| restricted to | without | with |
|---|---|---|
| STRING-partner cells (>=5 per item: **9 items**) | +0.0176 | -0.0169 |
| non-partner cells | +0.2880 | +0.2885 |

The partner-cell comparison rests on 9 items and is noise; it is reported because
omitting it would leave the panel-wide null looking like the only evidence.

**What this is and is not.** It is *not* evidence that protein-interaction priors
are useless for perturbation response -- this design cannot test that claim. It
is evidence that **a sparse binary partner indicator over the output panel is the
wrong way to inject network information**, because its support is a fixed ~0.2%
of the scored matrix. The measurement also says where network information would
have to enter instead: the ablation shows the target ESM2 vector is the *only*
live channel, so a graph signal should modulate that vector -- for example a
neighbourhood-averaged embedding, which is dense rather than 0.2%-sparse -- and
any such variant must hand the same augmented features to ridge and k-NN, or it
repeats the error of crediting extra input to architecture.

I expected this channel to help and said so before running it. It did not.

## Graph-augmented embeddings: the biggest gain so far, and it goes to the baseline

The redesign the previous section pointed to: fold the STRING neighbourhood into
the embedding itself, `[own || mean(partners)]`, 640-d -> 1280-d, with the
neighbourhood block norm-matched and zeroed for the 4605 isolated genes. It is a
**table**, not a model change, so the head, `ridge_esm2` and `knn_esm2` all read
identical features. Same 3 seeds, same split, same panel, 30 epochs:

| | plain 640-d | graph 1280-d | delta |
|---|---|---|---|
| P3 head | +0.2927 +/- 0.0055 | +0.3291 +/- 0.0309 | **+0.0364** |
| **ridge_esm2** | +0.2653 | **+0.3370** | **+0.0717** |
| knn_esm2 | +0.2115 | +0.2637 | +0.0523 |
| train_mean | +0.0121 | +0.0121 | 0.0000 |

**The network information is real and substantial.** Everything that reads the
embedding improves, `train_mean` (which does not read it) is unchanged to four
decimals as it must be, and **+0.337 is the best number this project has
produced** -- +0.044 above anything previously measured. The hypothesis from the
last section was right: the sparse indicator was the wrong injection point, and
the dense one works.

**And it dissolves the architecture's case.** Ridge gains twice what the head does
(+0.072 vs +0.036), so the head's advantage is erased and slightly reversed:

| on the graph table | paired diff | CI95 | verdict |
|---|---|---|---|
| head vs ridge_esm2 | **-0.0079** | [-0.0169, +0.0012] | mixed, model wins 1/3 seeds |
| head vs knn_esm2 | +0.0654 | [+0.0378, +0.0929] | beats, every seed |

For reference, on the plain table the head beat ridge +0.0274, CI [+0.010,
+0.045], every seed. So the head's edge over a linear map existed only while the
features were **impoverished**: its non-linearity was substituting for
information the representation lacked, and once that information is present in a
form a linear model can use, the linear model uses it better.

There is no leakage here to explain it away -- the augmented table is built from
the ESM2 vectors and the STRING graph alone, neither of which touches the
Perturb-seq expression data or the test labels, and the panel is still ranked on
training items only.

### The caveat was real: the head was undertrained, and the loss was mostly that

The head's hyperparameters were chosen against the 640-d table, and its
seed-to-seed sd tripled on the 1280-d one (0.0055 -> 0.0309) -- what an
undertrained model in a doubled feature space looks like. Ridge is closed-form
with no equivalent knob, so a longer head run is a test biased *towards* the head.
Run at 80 epochs instead of 30, same 3 seeds:

| head on graph features | pearson_dev | sd | epoch selected (of budget) |
|---|---|---|---|
| 30 epochs | +0.3291 | 0.0309 | 28, 26, 26 of 30 |
| 80 epochs | **+0.3346** | **0.0014** | 78, 77, 78 of 80 |
| ridge_esm2 (closed form) | +0.3370 | — | n/a |

| head(80ep) vs | paired diff | CI95 | verdict |
|---|---|---|---|
| ridge_esm2 | **-0.0024** | [-0.0308, +0.0260] | mixed, 1/3 seeds |
| knn_esm2 | +0.0709 | [+0.0387, +0.1031] | beats, every seed |

So the -0.0079 "ridge wins" at 30 epochs was largely a training-budget artefact.
At 80 epochs the head is level with ridge (-0.0024, CI straddling zero widely) and
its variance has collapsed by 20x. **The honest verdict on the graph features is a
tie, not a loss.**

**But the head has not converged, and that has to be said rather than left for a
reader to notice.** Inner-validation selection picks epochs 78, 77 and 78 out of
80 -- the model is still improving when the budget ends. Its trajectory is +0.3291
at 30 epochs, +0.3346 at 80, still climbing. So this comparison does **not**
establish that the head cannot beat ridge; it establishes that it has not yet,
within a budget it is still exhausting.

That invites an obvious abuse: extend the budget until the favoured model wins,
then stop. To avoid it, the follow-up budget was **declared before running and run
once** -- 150 epochs, 3 seeds -- and whatever it produced is reported below
without a further extension. Note also what "more epochs" means here: ridge is a
closed-form solve with no budget at all, so every epoch given to the head widens
an asymmetry already recorded in the fairness block. A head that needs 5x the
compute to match a linear solve has not made an efficiency case either.

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
