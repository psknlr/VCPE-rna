"""Fig. 4 | Robustness: what survives changing the split, the budget and the table.

One message: the head-versus-ridge comparison is split-dependent and
budget-dependent, so any single-split or single-budget claim about it is an
artefact of the slice it was measured on. This figure exists because the project's
own convergence probe nearly produced such a claim.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata as F                                             # noqa: E402
import nature_style as ns                                       # noqa: E402


def main(outdir="figures"):
    style = ns.apply_nature_style()
    d = F.collect()
    fig, w_mm = ns.nature_figure("double", height_mm=80)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.0, 1.18])

    # ---------------- a: per-split head vs ridge -----------------------
    axa = fig.add_subplot(gs[0, 0]); ns.tidy(axa); ns.panel_label(axa, "a")
    reps = d["seeds_graph_80ep"]
    seeds = sorted(reps)
    head = [reps[s]["pearson_dev"] for s in seeds]
    ridge = [reps[s]["baselines"]["ridge_esm2"]["pearson_dev"] for s in seeds]
    x = np.arange(len(seeds)); wdt = 0.34
    axa.bar(x - wdt / 2, head, wdt, color=ns.C_MODEL, linewidth=0, label="P3 head")
    axa.bar(x + wdt / 2, ridge, wdt, color=ns.C_BASE, linewidth=0, label="ESM2 ridge")
    axa.set_xticks(x); axa.set_xticklabels([f"Split {s}" for s in seeds], fontsize=5.6)
    axa.set_ylabel("Held-out $r$ (residual)")
    axa.set_ylim(0, max(head + ridge) * 1.22)
    axa.legend(fontsize=5.2, loc="upper center", ncol=2, handlelength=1.0,
               borderpad=0.2, columnspacing=0.8)
    for xi, h, r in zip(x, head, ridge):
        win = h > r
        axa.text(xi, max(h, r) * 1.035, "head" if win else "ridge", ha="center",
                 fontsize=5.0, color=ns.C_TEXT)
    axa.text(0.5, max(head + ridge) * 0.12,
             "the head leads on 1 of 3 splits;\nsplit 0 is ridge's weakest",
             fontsize=5.0, ha="center", color=ns.C_TEXT)

    # ---------------- b: budget dependence (single split, labelled) ----
    axb = fig.add_subplot(gs[0, 1]); ns.tidy(axb); ns.panel_label(axb, "b")
    bud = d["budget_seed0"]
    eps = sorted(bud)
    hv = [bud[e]["pearson_dev"] for e in eps]
    rv = bud[eps[0]]["baselines"]["ridge_esm2"]["pearson_dev"]
    axb.plot(eps, hv, "-o", ms=3.0, lw=1.0, color=ns.C_MODEL, label="P3 head")
    axb.axhline(rv, color=ns.C_BASE, lw=1.0, ls=(0, (3, 2)), label="ESM2 ridge")
    axb.set_xlabel("Training budget (epochs)")
    axb.set_ylabel("Held-out $r$, split 0 only")
    axb.set_xticks(eps)
    axb.legend(fontsize=5.2, loc="upper left", handlelength=1.3, borderpad=0.2)
    # The selected epoch is deliberately NOT annotated: in these runs the
    # selection slice was also trained on (ERRATA E17), so it tracks training fit
    # and sits at the end of any budget. The held-out values are the evidence.
    for e, v, off in zip(eps, hv, [(12, -3), (0, 5), (-12, 4)]):
        axb.annotate(f"{v:+.3f}", (e, v), textcoords="offset points",
                     xytext=off, ha="center", fontsize=5.0, color=ns.C_TEXT)
    axb.text(np.mean(eps), min(hv + [rv]) * 0.962,
             "held-out $r$ still rising at 150 epochs",
             fontsize=5.0, ha="center", color=ns.C_TEXT)
    axb.set_ylim(min(hv + [rv]) * 0.925, max(hv) * 1.045)

    # ---------------- c: embedding dependence --------------------------
    axc = fig.add_subplot(gs[0, 2]); ns.tidy(axc); ns.panel_label(axc, "c")
    runs = [("ESM2-35M", d["plain_35M"]), ("ESM2-150M", d["plain_150M"]),
            ("150M + STRING", d["graph_80ep"])]
    yy = np.arange(len(runs))[::-1]
    for y, (lab, summ) in zip(yy, runs):
        p = F.paired(summ, "ridge_esm2")
        col = ns.C_MODEL if (p["lo"] or 0) > 0 else ns.C_BASE
        axc.plot([p["lo"], p["hi"]], [y, y], color=col, lw=1.0,
                 solid_capstyle="butt")
        for e in (p["lo"], p["hi"]):
            axc.plot([e, e], [y - .10, y + .10], color=col, lw=0.8)
        axc.plot([p["diff"]], [y], "o", ms=3.2, color=col, zorder=3)
    # the zero line stops above the note so the two never cross
    axc.axvline(0, ymin=0.13, color=ns.C_FLOOR, lw=0.8, ls=(0, (3, 2)))
    axc.set_yticks(yy); axc.set_yticklabels([r[0] for r in runs], fontsize=5.6)
    axc.set_xlabel("Head − ridge, paired by split (95% $t$ interval)")
    axc.set_ylim(-0.75, len(runs) - 0.3)
    n_ex = sum((F.paired(s_, "ridge_esm2")["lo"] > 0) or
               (F.paired(s_, "ridge_esm2")["hi"] < 0) for _, s_ in runs)
    axc.text(0.0, -0.58,
             (f"no representation gives an interval\nthat excludes zero"
              if n_ex == 0 else
              f"{n_ex} of {len(runs)} intervals exclude zero"),
             fontsize=5.0, ha="center", va="center", color=ns.C_TEXT)

    res = ns.save_nature(fig, "fig4_robustness", outdir=outdir, width_mm=w_mm)
    res["font"] = style
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2, default=str))
