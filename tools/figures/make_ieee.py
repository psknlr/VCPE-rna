#!/usr/bin/env python3
"""Rebuild the IEEE-style architecture figure and Extended Data Figs. 1-4.

Run from the repository root:  python3 tools/figures/make_ieee.py

Unlike make_all.py (Nature set), everything here reads committed files only, so
it runs on a fresh clone. Exit status is non-zero if any item is off the IEEE
mechanical spec checked by ieee_style.save_ieee.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ieee_fig_architecture                                    # noqa: E402
import ieee_extended_data                                       # noqa: E402
import nature_style as ns                                       # noqa: E402

CAPTIONS = """Fig. 1. Architecture of the conditioned per-gene deviation head. (a) Data flow for one perturbation. The target gene's ESM2 embedding e_p and the frozen embeddings E of the N = 500 response-panel genes are projected to 256 dimensions; their element-wise product carries the gene-target interaction. For every panel gene i, the projected vectors are concatenated with three scalars (z-scored control expression c_i, a STRING-neighbour flag and an is-target flag) and an 8-d learned screen embedding u_k, giving a 1,035-d input to a two-hidden-layer MLP (512 units, GELU, dropout 0.1) whose weights are shared across all N genes. A fixed, non-learned sigmoid gate s = sigma(6(x_p - 0.7)), keyed to the raw control expression x_p of the target gene, scales only the target's own output row. The model is trained to predict the residual response (fold change minus its train-only mean over perturbations) with a mean-squared error restricted to genes measured in the screen. The RNA-sequence channel (dashed) exists in the code but no encoder was supplied in any reported run, so its 256-d slot is zero. D = 640 for ESM2-150M and 1,280 when the STRING neighbourhood embedding is concatenated. (b) Composition of the per-gene input vector x_i; the four 256-d blocks are drawn to scale and the 11 scalar dimensions are magnified.

Extended Data Fig. 1. Split-level results for every reported configuration. (a) Held-out Pearson r between predicted and observed residual response, computed within each perturbation and averaged over perturbations, for each of three splits (markers) and their mean (horizontal bar). Each split re-draws both the initialisation and the grouped held-out target genes. "flag" adds the binary STRING-neighbour indicator; "emb." concatenates the STRING neighbourhood embedding to ESM2. (b) Epoch selected on the inner validation split as a fraction of the training budget; points on the dashed line were selected at the last permitted epoch. In these runs the inner split was also in the training batches (ERRATA E17), so selection tracks training fit and says nothing about convergence. n = 349 held-out target genes per split.

Extended Data Fig. 2. Complete conditioning-channel ablation (split 0). (a) Change in held-out r when one channel is set to zero or permuted across perturbations, for the ESM2-150M model and the same model with the STRING-neighbour flag. (b) Pearson correlation between the ablated and intact predictions; a value of 1 means the channel does not influence the output. Only the target-gene ESM2 vector and, to a lesser extent, the screen embedding carry conditioning; the RNA channel is inactive in these runs and the indicator flags change r by less than 10^-3.

Extended Data Fig. 3. Search over graph-feature designs, ranked on inner validation only. (a) Ridge and (b) k-NN inner-validation r for eight ways of injecting the STRING neighbourhood into the gene representation (top-k neighbours; concatenation versus blending with weight a; with or without normalisation). Diamonds are means over three splits, circles individual splits; the filled diamond is the design carried forward. No held-out score was computed during the search. The dashed line marks the no-graph ridge reference.

Extended Data Fig. 4. Training budget and comparison with published models. (a) Held-out r on split 0 after training budgets of 30, 80 and 150 epochs; labels give the epoch chosen on the inner split, which in these runs was also trained on (ERRATA E17), so they track training fit and are not evidence about convergence; the held-out values themselves still rise between 80 and 150 epochs on this split. Ridge and k-NN are closed-form and budget-independent. Single split. (b) All methods on the smaller screen alone (Tian et al., 27 held-out genes) under the identical split, gene panel, target and metric; every method lies at the floor. GEARS is the mean of three runs with the range shown; other entries are single runs. Hatched bars are published models run at or near their defaults.
"""


def main():
    out = ROOT / "figures"
    items = [("Fig. 1 (architecture)", ieee_fig_architecture.main(str(out)))]
    for i, r in enumerate(ieee_extended_data.main(str(out)), 1):
        items.append((f"Extended Data Fig. {i}", r))
    bad = 0
    for name, r in items:
        print(f"{name:24s} {r['stem']:32s} {r['width_in']:.2f} x "
              f"{r['height_in']:.2f} in  {'OK' if not r['warnings'] else 'OFF-SPEC'}")
        for w in r["warnings"]:
            print(f"    ! {w}")
            bad += 1
    (out / "ieee_captions.md").write_text(CAPTIONS)
    (out / "ieee_manifest.json").write_text(json.dumps(dict(
        figures=[dict(name=n, **r) for n, r in items],
        font_note=("Times New Roman is not installed in this container. "
                   "Liberation Serif is metrically identical to it; math is set "
                   "in STIX, a Times-design face. Fonts are embedded as TrueType "
                   "(Type 42) and text remains editable."),
        data_note=("Every value is read from committed results/**/*.json; "
                   "architecture dimensions mirror src/maprna_p3/model_dev.py."),
        git=ns.git_rev()), indent=2, default=str))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
