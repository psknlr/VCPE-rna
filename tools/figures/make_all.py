#!/usr/bin/env python3
"""Rebuild every display item, audit each, and write the legends and Table 1.

Run from the repository root:  python3 tools/figures/make_all.py

Each figure is audited by `save_nature`; an empty `warnings` list is the pass
signal. A non-empty one is printed and the exit status is non-zero, so an
off-spec figure cannot be handed over by accident.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import figdata as F                                             # noqa: E402
import nature_style as ns                                       # noqa: E402
import fig1_design, fig2_main, fig3_mechanism                   # noqa: E402
import fig4_robustness, fig5_diagnostics                        # noqa: E402

FIGS = [("Fig. 1", fig1_design), ("Fig. 2", fig2_main),
        ("Fig. 3", fig3_mechanism), ("Fig. 4", fig4_robustness),
        ("Fig. 5", fig5_diagnostics)]


def table1_markdown(d):
    """Table 1 as editable text. A submitted table must not be a picture."""
    plain, graph = d["plain_150M"], d["graph_80ep"]

    def row(key, label):
        def g(s):
            if key == "P3 head":
                return s["model"]["mean"], s["model"].get("sd")
            b = s["baselines"].get(key, {})
            return b.get("mean"), b.get("sd")
        p, ps = g(plain)
        q, qs = g(graph)
        ptxt = f"{p:+.4f}" + (f" ± {ps:.4f}" if ps else "")
        qtxt = f"{q:+.4f}" + (f" ± {qs:.4f}" if qs else "")
        return f"| {label} | {ptxt} | {qtxt} | {q - p:+.4f} |"

    lines = [
        "Table 1 | Held-out performance under two gene representations",
        "",
        "| Method | ESM2 only (640-d) | + STRING neighbourhood (1,280-d) | Δ |",
        "|---|---|---|---|",
        row("P3 head", "P3 head (conditioned)"),
        row("ridge_esm2", "ESM2 ridge (closed form)"),
        row("knn_esm2", "ESM2 k-NN retrieval"),
        row("train_mean", "Train mean"),
        row("zero", "Predict no change"),
        "",
        "Values are the mean Pearson *r* between predicted and observed residual "
        "response over held-out target genes, averaged across three splits; ± is "
        "the sample s.d. across splits. *r* is computed within each perturbation "
        "and then averaged, never pooled across perturbations. Both columns use "
        "the identical split, gene panel, measured-gene mask and estimator; only "
        "the gene representation differs, and it is shared by every method in the "
        "column. n = 349 held-out target genes from 2,328; 3 splits.",
    ]
    return "\n".join(lines)


LEGENDS = """Fig. 1 | Study design and the three protocol choices that decide what the score means.
a, Scientific workflow from the two CRISPRi Perturb-seq screens to the reported number. Orange boxes mark the three choices that change the quantity being estimated: perturbations are held out by target gene so a held-out gene is unseen, the training epoch is chosen on an inner split so the held-out set is scored exactly once, and genes a screen never measured are masked because they carry a per-dataset constant rather than signal. b, Perturbations retained after a 20-cell gate. c, The same model evaluated on both screens and on the smaller screen alone. d, The same predictions scored against raw fold change and against the residual remaining after the response shared across perturbations is removed. n = 349 held-out target genes from 2,328; 3 splits.

Fig. 2 | Network-augmented gene embeddings improve every method that reads them, and remove the head's advantage over a linear fit.
a, Held-out Pearson r for the conditioned head and four controls under ESM2 embeddings alone and after each gene's STRING interaction neighbourhood is concatenated to its embedding. Bars are the mean of three splits; error bars are the sample s.d. across splits. b, Head minus ridge, paired within split; points are the mean paired difference, lines the 95% confidence interval, ticks the individual splits. c, Change in held-out r attributable to the richer representation. Train mean does not read the embedding and is unchanged to four decimals, which is the internal check that the gain is representational. n = 349 held-out target genes; 3 splits; two-sided paired comparison across splits.

Fig. 3 | The score rests on target-gene identity carried by a single channel.
a, Held-out r after disabling one conditioning channel at a time; the dashed line is the intact model. Grey channels leave the prediction bit-identical. b, Conditioning attributed to the model under two null floors: setting the target vector to zero takes the input off its distribution and flatters the floor, whereas permuting target vectors across perturbations preserves the input statistics and destroys only identity. The permuted floor is the defensible one. c, Gain over ESM2-only features from the same STRING graph injected as a sparse binary partner indicator over the output panel versus as a dense neighbourhood embedding. Single split (split 0) for a and b; 3 splits for c.

Fig. 4 | The head-versus-ridge comparison depends on the split, the budget and the representation.
a, Held-out r per split at a fixed 80-epoch budget on the network-augmented features. b, Effect of training budget on split 0 only; annotations give the epoch selected by inner validation, which never leaves the budget, so the head has not converged. c, Head minus ridge paired within split under three gene representations; points are mean paired differences with 95% confidence intervals. n = 349 held-out target genes; 3 splits except b, which is a single split and is labelled as such.

Fig. 5 | Three protocol choices each move the headline number, measured as A/B comparisons on real data.
a, The same predictions scored against raw fold change and against the residual. b, The same configuration scored with and without masking genes a screen never measured, on a logarithmic axis; the head is unchanged while the train-mean baseline, which carries no perturbation information and must sit at the floor, gains. Only 0.14% of this gene panel is unmeasured, so the effect is a lower bound. c, Eight gene-representation designs ranked by ridge on an inner-validation split; no held-out value was computed during the search, and the dashed line marks the held-out ridge score for reference only. n = 349 held-out target genes; 3 splits in c, single split in a and b.
"""


def main():
    outdir = ROOT / "figures"
    d = F.collect()
    results, failed = [], []
    for name, mod in FIGS:
        r = mod.main(outdir=str(outdir))
        results.append((name, r))
        status = "OK" if not r["warnings"] else "OFF-SPEC"
        print(f"{name:8s} {r['stem']:18s} {r['width_mm']:.0f} x "
              f"{r['height_mm']:.0f} mm  [{r['column_class']}]  {status}")
        for w in r["warnings"]:
            print(f"         ! {w}")
            failed.append((name, w))

    (outdir / "legends.md").write_text(LEGENDS)
    (outdir / "table1.md").write_text(table1_markdown(d) + "\n")

    manifest = dict(
        figures=[dict(name=n, **r) for n, r in results],
        font_note=("Arial/Helvetica are not installed in this container. "
                   "Liberation Sans is metrically identical to Arial, so the "
                   "layout is unchanged when a production system substitutes "
                   "real Arial. Text is embedded as Type 42 and remains "
                   "selectable and editable in the PDFs."),
        data_note=("Every value is read from results/**/*.json by "
                   "tools/figures/figdata.py. No measurement is typed into the "
                   "plotting code."),
        git=ns.git_rev(),
    )
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nwrote {outdir}/legends.md, table1.md, manifest.json")
    if failed:
        print(f"\n{len(failed)} off-spec warning(s); not ready to hand over")
        return 1
    print("all display items pass the Nature mechanical spec")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
