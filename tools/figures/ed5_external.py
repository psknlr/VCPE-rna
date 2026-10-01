"""Extended Data Fig. 5 | Published models on the cell-level screen.

Supports Fig. 1c. GEARS and CPA need single-cell input, and only the smaller
screen (Tian 2021, 27 held-out target genes) could be run at cell level here, so
this is the only footing on which they meet the head. Every method is scored on
VCPE's split, panel, residual and estimator. Values from results/real_35M/
D_gears.json, D_cpa.json and D_head/final_report.json.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata as F                                             # noqa: E402
import nature_style as ns                                       # noqa: E402


def rows_of(d):
    """[(label, [values per run], [ci per run or None], pooled, top50, colour)]."""
    h = d["d_head"]
    out = [("P3 head (ESM2-35M)", [h["pearson_dev"]], [None], [None],
            [h["top50_dev"]], ns.C_MODEL)]
    for key, lab in (("ridge_esm2", "ESM2 ridge"), ("knn_esm2", "ESM2 k-NN"),
                     ("train_mean", "Train mean"), ("zero", "Predict no change")):
        b = h["baselines"][key]
        out.append((lab, [b["pearson_dev"]], [None], [b["pearson_dev_pooled"]],
                    [b["top50_dev"]], ns.C_BASE))
    g = d["gears"]["gears_per_run"]
    out.append((f"GEARS ({len(g)} runs)", [r["pearson_dev"] for r in g],
                [r["pearson_dev_ci95"] for r in g],
                [r["pearson_dev_pooled"] for r in g],
                [r["top50_dev"] for r in g], ns.C_ALT))
    c = d["cpa"]["cpa"]
    out.append(("CPA", [c["pearson_dev"]], [c["pearson_dev_ci95"]],
                [c["pearson_dev_pooled"]], [c["top50_dev"]],
                ns.OKABE_ITO["purple"]))
    return out


def dots(ax, yy, rows, which, ci=False):
    for y, r in zip(yy, rows):
        vals = r[{"r": 1, "pooled": 3, "top": 4}[which]]
        cis = r[2] if ci else [None] * len(vals)
        offs = np.linspace(-0.16, 0.16, len(vals)) if len(vals) > 1 else [0.0]
        for v, c, o in zip(vals, cis, offs):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            if c is not None:
                ax.plot(c, [y + o] * 2, color=r[5], lw=0.9,
                        solid_capstyle="butt")
            ax.plot([v], [y + o], "o", ms=3.0, color=r[5], zorder=3)
        ok = [v for v in vals if v is not None and not
              (isinstance(v, float) and np.isnan(v))]
        if not ok:
            # None: the producing script never wrote it. NaN: it was written and
            # is undefined -- a correlation with a constant prediction.
            why = "not recorded" if vals[0] is None else "undefined (constant)"
            ax.text(0.012, y, why, fontsize=5.2, ha="left", va="center")
        elif len(ok) > 1:
            m = float(np.mean(ok))
            ax.plot([m, m], [y - 0.30, y + 0.30], color=r[5], lw=1.0)


def main(outdir="figures/extended_data"):
    style = ns.apply_nature_style()
    d = F.collect_ed()
    rows = rows_of(d)
    n_te = d["gears"]["n_test_conditions"]
    assert n_te == d["d_head"]["n_scored_perturbations"] == \
        d["cpa"]["n_test_conditions"], "the three reports must score the same items"
    fig, w_mm = ns.nature_figure("double", height_mm=62)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0])
    yy = np.arange(len(rows))[::-1]

    ax = fig.add_subplot(gs[0, 0]); ns.tidy(ax); ns.panel_label(ax, "a", dx=-0.62)
    dots(ax, yy, rows, "r", ci=True)
    ax.axvline(0, color=ns.C_FLOOR, lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows], fontsize=5.6)
    ax.set_xlabel("Within-perturbation $r$ (residual)")
    ax.set_xlim(-0.11, 0.11)

    ax2 = fig.add_subplot(gs[0, 1], sharey=ax); ns.tidy(ax2)
    ns.panel_label(ax2, "b", dx=-0.10)
    dots(ax2, yy, rows, "pooled")
    ax2.axvline(0, color=ns.C_FLOOR, lw=0.6, ls=(0, (3, 2)))
    ax2.set_xlabel("$r$ pooled over all entries")
    ax2.tick_params(labelleft=False)
    ax2.set_xlim(-0.16, 0.16)

    ax3 = fig.add_subplot(gs[0, 2], sharey=ax); ns.tidy(ax3)
    ns.panel_label(ax3, "c", dx=-0.10)
    dots(ax3, yy, rows, "top")
    n_hvg = d["table_D"]["shared_protocol"]["n_hvg"]
    chance = 50 / (d["d_head"]["measured_fraction"] * n_hvg)
    ax3.axvline(chance, color=ns.C_FLOOR, lw=0.8, ls=(0, (1, 1.4)))
    ax3.text(chance + 0.003, yy[0] + 0.55, f"chance, $k/n$ = {chance:.2f}",
             fontsize=5.2, va="center")
    z = rows[[r[0] for r in rows].index("Predict no change")][4][0]
    ax3.text(z, yy[[r[0] for r in rows].index("Predict no change")] - 0.42,
             "tie artefact (E16)", fontsize=5.0, ha="right", va="center")
    ax3.set_xlabel("Top-50 overlap (as recorded)")
    ax3.tick_params(labelleft=False)
    ax3.set_xlim(0.06, 0.17)
    ax.set_ylim(-0.6, len(rows) - 0.1)

    res = ns.save_nature(fig, "edfig5_external", outdir=outdir, width_mm=w_mm)
    res["font"] = style
    res["n_held_out"] = n_te
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2, default=str))
