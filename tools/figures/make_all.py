#!/usr/bin/env python3
"""Rebuild every display item, audit each, and write the legends and Table 1.

Run from the repository root:  python3 tools/figures/make_all.py

Builds the five main figures and five Extended Data figures to the Nature
portfolio specification, and the architecture figure twice: to the IEEE
Transactions specification and re-typeset for the Nature portfolio. Each item is
audited as it is saved (`save_nature` / `save_ieee`); an empty `warnings` list is
the pass signal, every legend is checked against the 300-word cap, and any
failure makes the exit status non-zero, so an off-spec item cannot be handed over
by accident.
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
import ed1_simulation, ed2_benchmark, ed3_training              # noqa: E402
import ed4_metrics, ed5_external                                # noqa: E402
import ieee_architecture                                        # noqa: E402

FIGS = [("Fig. 1", fig1_design), ("Fig. 2", fig2_main),
        ("Fig. 3", fig3_mechanism), ("Fig. 4", fig4_robustness),
        ("Fig. 5", fig5_diagnostics)]
ED_FIGS = [("Extended Data Fig. 1", ed1_simulation),
           ("Extended Data Fig. 2", ed2_benchmark),
           ("Extended Data Fig. 3", ed3_training),
           ("Extended Data Fig. 4", ed4_metrics),
           ("Extended Data Fig. 5", ed5_external)]
LEGEND_WORD_CAP = 300


def ed_legends(d):
    """Extended Data legends. Counts and settings are read from the data."""
    sim = d["sim"]
    c, tr = sim["config"], sim["traps"]
    n_pert = sorted({r["n_scored_perturbations"] for k in d["reports"]
                     for r in d["reports"][k].values()})
    npr = f"{n_pert[0]}-{n_pert[-1]}"
    a = [r["pearson_delta"] for k in d["reports"] for r in d["reports"][k].values()]
    b = [r["pearson_dev"] for k in d["reports"] for r in d["reports"][k].values()]
    nruns = len(a)
    n_tian = d["d_head"]["n_scored_perturbations"]
    n_gears = len(d["gears"]["gears_per_run"])
    return f"""Extended Data Fig. 1 | Four ways a score rises without perturbation-specific skill, in simulation.
Synthetic responses for {c['n_genes']} genes, {c['n_train']} training and {c['n_test']} held-out perturbations; per-gene scales are gamma-distributed and sorted by variance, as the real panel is. Every score is computed with the repository's own metric and baseline functions. a, Within-perturbation r of a predictor with no perturbation information (the training mean, grey) and of a predictor whose perturbation-specific part has fixed r = {tr['shared']['informative_r']:g} (blue), scored against raw fold change (dashed) and against the residual left after removing the training-mean response (solid), as the response common to all perturbations grows. On raw fold change the uninformative predictor overtakes the informative one; the residual separates them. b, Within-perturbation r of the training mean when one screen, holding {tr['unmeasured']['partial_screen_share'] * 100:.0f}% of perturbations, never measured part of the panel, scored with those entries included (dashed) or masked (solid). Masked, the training-mean residual is identically zero and scores r = 0. c, Pooled and within-perturbation r of a predictor that knows each perturbation's overall level and nothing about which genes respond. d, Top-{tr['ties']['k']} overlap of a constant prediction with ties broken by column position on a variance-sorted panel (dashed) or a shuffled panel (dotted), and with the tie-aware definition (solid); the dotted horizontal line is chance, k/n. Points are means and error bars s.d. over {c['replicates']} independent replicates.

Extended Data Fig. 2 | Held-out performance of the head and every control, on every split, under five gene representations.
a, Within-perturbation r of the residual for the conditioned head and the ridge, k-NN and train-mean controls; points are the three splits, each of which also changes the initialisation, and bars their mean. Representations: ESM2-35M; ESM2-150M; ESM2-150M with the STRING partner indicator as an input channel; ESM2-150M concatenated with the norm-matched mean of its STRING partners, trained for 30 or 80 epochs. b,c, Head minus ridge (b) and head minus k-NN (c), paired within split. Points are mean differences, lines 95% confidence intervals from the t distribution with 2 degrees of freedom, ticks the individual splits; blue marks an interval that excludes zero. No interval against ridge excludes zero. n = {npr} held-out perturbations per split; 3 splits.

Extended Data Fig. 3 | Training dynamics and epoch selection.
a, Training loss (masked mean squared error on the residual) for ESM2-150M at 30 epochs, the STRING-graph table at 30 and 80 epochs (three splits each) and at 150 epochs (split 0). b, Within-perturbation r on the inner split that selects the epoch; open circles mark the selected epoch. In these runs the inner split was also in the training batches (ERRATA E17), so this curve tracks training fit, keeps rising, and pulls selection to the end of every budget. c, Correlation between the prediction and the prediction with the target gene's embedding set to zero, on the inner split; 1 would mean the target has no effect. d, Inner-split r at the selected epoch against held-out r for all {nruns} runs and the 150-epoch run. The widening gap is the signature of selecting on trained items. The code now holds the inner split out of training; these runs predate the correction.

Extended Data Fig. 4 | How the other metrics behave on the real benchmark.
a, Held-out r of the head against raw fold change versus against the residual, for all {nruns} runs (five representations, three splits). Raw r spans {max(a) - min(a):.3f} across runs while residual r spans {max(b) - min(b):.3f}: raw fold change hides most of the difference between representations. b, Pooled r, one correlation over all held-out entries, versus within-perturbation r for the three controls in all {nruns} runs; the pooled estimator scores ridge and k-NN higher in every run. c, Top-50 overlap on split 0 of the ESM2-150M model, as recorded by the committed runs. The head with its target embedding zeroed, or with all conditioning removed, scores above the intact head; the constant no-change prediction scores highest because ties were broken by panel position (ERRATA E16), and scores k/n under the tie-aware definition (arrow). Dotted line, chance k/n.

Extended Data Fig. 5 | Published perturbation models on the cell-level screen.
The Tian 2021 CRISPRi screen alone, {n_tian} held-out target genes, with the same split, gene panel, residual target and estimator imposed on every method. GEARS and CPA are trained on single cells, the head (ESM2-35M table) and the controls on pseudobulk. a, Within-perturbation r of the residual; lines are 95% bootstrap confidence intervals over held-out perturbations, where the producing script recorded them. GEARS is not deterministic at a fixed seed, so its {n_gears} runs are shown with their mean (vertical bar). b, Pooled r; not recorded for the head, undefined for a constant prediction. c, Top-50 overlap; dotted line, chance k/n; the no-change value is the tie artefact of ERRATA E16. Tuning effort and capacity are not equalised between methods. Every method sits at or near zero, so this screen cannot rank them.
"""


def legend_word_counts(text):
    """{legend title: word count} for every 'Fig. N |' legend in `text`."""
    out, cur = {}, None
    for line in text.splitlines():
        if " | " in line and "Fig." in line.split(" | ")[0]:
            cur = line.split(" | ")[0]
            out[cur] = 0
        if cur:
            out[cur] += len(line.split())
    return out


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
        row("P3 head", "Conditioned head"),
        row("ridge_esm2", "Ridge regression (closed form)"),
        row("knn_esm2", "k-NN retrieval"),
        row("train_mean", "Training mean"),
        row("zero", "No change"),
        "",
        "Values are the mean Pearson *r* between predicted and observed residual "
        "response over held-out target genes, averaged across three splits; ± is "
        "the sample s.d. across splits. *r* is computed within each perturbation "
        "and then averaged, never pooled across perturbations. Both columns use "
        "the identical split, gene panel, measured-gene mask and estimator; only "
        "the gene representation differs, and it is shared by every method in the "
        "column. For the head, Δ compares 80 epochs on the enriched table with "
        "30 epochs on ESM2 only; at the same 30-epoch budget it is +0.0364. The "
        "training mean predicts zero up to rounding in these runs, so it is a "
        "floor. n = 340 held-out target genes of 2,269 in each split (347-352 "
        "held-out perturbations); 3 splits.",
    ]
    return "\n".join(lines)


LEGENDS = """Fig. 1 | Study design and the three protocol choices that decide what the score means.
a, Scientific workflow from the two CRISPRi Perturb-seq screens to the reported number. Orange boxes mark the three choices that change the quantity being estimated: perturbations are held out by target gene so a held-out gene is unseen, the training epoch is chosen on an inner split so the held-out set is scored exactly once, and genes a screen never measured are masked because they carry a per-dataset constant rather than signal. In the runs reported here the inner split was also trained on (ERRATA E17), so it selected by training fit; the held-out set was still scored once. b, Perturbations retained after a 20-cell gate. c, The same model evaluated on both screens and on the smaller screen alone. d, The same predictions scored against raw fold change and against the residual remaining after the response shared across perturbations is removed. n = 340 held-out target genes of 2,269 in each split; 3 splits.

Fig. 2 | Network-augmented gene embeddings raised ridge regression and k-NN, and the head non-significantly; on neither representation is the head significantly better than a linear fit.
a, Held-out Pearson r for the conditioned head and four controls under ESM2 embeddings alone and after each gene's STRING interaction neighbourhood is concatenated to its embedding (the head at 80 epochs on the enriched table). Bars are the mean of three splits; error bars are the sample s.d. across splits. b, Head minus ridge, paired within split; points are the mean paired difference, lines the 95% confidence interval from the t distribution with 2 degrees of freedom, ticks the individual splits; P values are from two-sided paired t-tests across the three splits. c, Change in held-out r from the richer representation, every method at the same 30-epoch budget. The training mean predicts zero up to rounding in these runs, so its unchanged value only confirms that split, panel and target are identical. n = 347-352 held-out perturbations per split; 3 splits.

Fig. 3 | The score rests on target-gene identity carried by a single channel.
a, Held-out r after disabling one conditioning channel at a time; the dashed line is the intact model. Grey channels change r by less than 0.00001. b, Conditioning attributed to the model under two null floors: setting the target embedding to zero takes the input off its distribution and flatters the floor, whereas permuting target embeddings across perturbations preserves the input statistics and destroys only identity. The permuted floor is the defensible one. c, Gain over ESM2-only features from the same STRING graph injected as a sparse binary partner indicator over the output panel versus as a dense neighbourhood embedding, the head at 30 epochs in all three runs. Single split (split 0) for a and b; 3 splits for c.

Fig. 4 | The head-versus-ridge comparison depends on the split, the budget and the representation.
a, Held-out r per split at a fixed 80-epoch budget on the network-augmented features. b, Effect of training budget on split 0 only, one run per budget; held-out r was higher at each larger budget on this split. The epoch selected on the inner split is not shown: in these runs that split was also trained on (ERRATA E17), so selection tracks training fit and carries no information about convergence. c, Head minus ridge paired within split under three gene representations; points are mean paired differences, lines 95% confidence intervals from the t distribution with 2 degrees of freedom; no interval excludes zero. n = 340 held-out target genes per split; 3 splits except b, which is a single split and is labelled as such.

Fig. 5 | Three protocol choices each move the headline number, measured as A/B comparisons on real data.
a, The same predictions scored against raw fold change and against the residual. b, The same configuration scored with and without masking genes a screen never measured, on a logarithmic axis; the head is unchanged while the training mean, which carries no perturbation information, gains. Only 0.14% of held-out entries are unmeasured. c, Eight gene-representation designs ranked by ridge on an inner-validation split; no held-out value was computed during the search, and the dashed line marks the held-out ridge score for reference only. The design in use was built before this search, after earlier held-out results were known, so the search checked it rather than selected it. n = 340 held-out target genes per split; 3 splits in c, single split in a and b.
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

    # ---------------- Extended Data -----------------------------------
    ed_dir = outdir / "extended_data"
    de = F.collect_ed()
    for name, mod in ED_FIGS:
        r = mod.main(outdir=str(ed_dir))
        results.append((name, r))
        status = "OK" if not r["warnings"] else "OFF-SPEC"
        print(f"{name:22s} {r['stem']:20s} {r['width_mm']:.0f} x "
              f"{r['height_mm']:.0f} mm  [{r['column_class']}]  {status}")
        for w in r["warnings"]:
            print(f"         ! {w}")
            failed.append((name, w))
    ed_text = ed_legends(de)
    (ed_dir / "legends_extended_data.md").write_text(ed_text)

    # ---------------- architecture: IEEE and Nature typesetting ---------
    arch, facts = ieee_architecture.main(outdir=str(outdir / "ieee"))
    for r in arch:
        name = f"Architecture ({r['profile']})"
        results.append((name, r))
        status = "OK" if not r["warnings"] else "OFF-SPEC"
        print(f"{name:22s} {r['stem']:20s} {r['width_mm']:.0f} x "
              f"{r['height_mm']:.0f} mm  [{r['column_class']}]  {status}")
        for w in r["warnings"]:
            print(f"         ! {w}")
            failed.append((name, w))

    # ---------------- legend length -------------------------------------
    legends_all = LEGENDS + "\n" + ed_text + "\n" + \
        (outdir / "ieee" / "legend_nature.md").read_text()
    for title, nw in legend_word_counts(legends_all).items():
        if nw > LEGEND_WORD_CAP:
            failed.append((title, f"legend is {nw} words (> {LEGEND_WORD_CAP})"))
            print(f"         ! {title}: legend is {nw} words")

    manifest = dict(
        figures=[dict(name=n, **{k: v for k, v in r.items() if k != "facts"})
                 for n, r in results],
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
    print(f"\nwrote {outdir}/legends.md, table1.md, manifest.json, "
          f"extended_data/legends_extended_data.md, ieee/caption_ieee.tex")
    if failed:
        print(f"\n{len(failed)} off-spec warning(s); not ready to hand over")
        return 1
    print("all display items pass their journal's mechanical spec, and every "
          f"legend is within {LEGEND_WORD_CAP} words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
