"""Fig. 2 | The central result: the gain is representational, not architectural.

One message: folding a gene's interaction neighbourhood into its embedding lifts
every method that reads the embedding, and under those richer features a
closed-form ridge fit matches the conditioned head. Panels are sized so the
load-bearing comparison (a) gets the most area.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata as F                                             # noqa: E402
import nature_style as ns                                       # noqa: E402

ORDER = ["P3 head", "ridge_esm2", "knn_esm2", "train_mean", "zero"]


def main(outdir="figures"):
    style = ns.apply_nature_style()
    d = F.collect()
    fig, w_mm = ns.nature_figure("double", height_mm=78)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.0, 1.0])

    plain, graph = d["plain_150M"], d["graph_80ep"]

    # ---------------- a: grouped bars, plain vs graph features ----------
    axa = fig.add_subplot(gs[0, 0]); ns.tidy(axa); ns.panel_label(axa, "a")

    def val(summ, key):
        if key == "P3 head":
            return summ["model"]["mean"], summ["model"].get("sd")
        b = summ["baselines"].get(key, {})
        return b.get("mean"), b.get("sd")

    x = np.arange(len(ORDER)); wdt = 0.36
    pv = [val(plain, k) for k in ORDER]
    gv = [val(graph, k) for k in ORDER]
    axa.bar(x - wdt / 2, [v[0] for v in pv], wdt, yerr=[v[1] or 0 for v in pv],
            color=ns.C_BASE, linewidth=0, label="ESM2 only (640-d)",
            error_kw=dict(lw=0.8, capsize=1.3, ecolor=ns.C_FLOOR))
    axa.bar(x + wdt / 2, [v[0] for v in gv], wdt, yerr=[v[1] or 0 for v in gv],
            color=ns.C_MODEL, linewidth=0, label="+ STRING neighbourhood (1,280-d)",
            error_kw=dict(lw=0.8, capsize=1.3, ecolor=ns.C_FLOOR))
    axa.axhline(0, color=ns.C_FLOOR, lw=0.8)
    axa.set_xticks(x)
    axa.set_xticklabels([F.pretty(k).replace(" ", "\n", 1) for k in ORDER],
                        fontsize=5.4)
    axa.set_ylabel("Held-out $r$ (residual), mean of 3 splits")
    axa.set_ylim(-0.015, 0.425)
    axa.legend(loc="upper right", fontsize=5.2, handlelength=1.1,
               borderpad=0.25, labelspacing=0.3)
    # the two best values, labelled, since they are the paper's headline numbers
    for xi, (v, sd) in ((x[0] + wdt / 2, gv[0]), (x[1] + wdt / 2, gv[1])):
        # clear the error bar, not just the bar top
        axa.text(xi, v + (sd or 0) + 0.019, f"{v:+.3f}", ha="center", fontsize=5.4,
                 color=ns.C_MODEL, fontweight="bold")

    # ---------------- b: paired difference, head minus ridge -----------
    axb = fig.add_subplot(gs[0, 1]); ns.tidy(axb); ns.panel_label(axb, "b")
    rows = [("ESM2 only", F.paired(plain, "ridge_esm2")),
            ("+ STRING", F.paired(graph, "ridge_esm2"))]
    yy = np.arange(len(rows))[::-1]
    for y, (lab, p) in zip(yy, rows):
        col = ns.C_MODEL if p["lo"] > 0 else ns.C_BASE
        axb.plot([p["lo"], p["hi"]], [y, y], color=col, lw=1.1,
                 solid_capstyle="butt")
        axb.plot([p["lo"], p["lo"]], [y - .09, y + .09], color=col, lw=0.8)
        axb.plot([p["hi"], p["hi"]], [y - .09, y + .09], color=col, lw=0.8)
        axb.plot([p["diff"]], [y], "o", ms=3.0, color=col, zorder=3)
        for s in (p["per_seed"] or []):
            axb.plot([s], [y - 0.20], "|", ms=3.2, color=col, alpha=0.85)
    axb.axvline(0, color=ns.C_FLOOR, lw=0.8, ls=(0, (3, 2)))
    axb.set_yticks(yy); axb.set_yticklabels([r[0] for r in rows], fontsize=5.6)
    axb.set_ylim(-0.55, len(rows) - 0.35)
    axb.set_xlabel("Head − ridge, paired by split\n(95% CI; ticks are splits)")
    axb.text(rows[0][1]["hi"] + 0.004, yy[0], "head\nahead", fontsize=5.0,
             va="center", color=ns.C_MODEL)
    axb.text(rows[1][1]["hi"] + 0.004, yy[1], "tie", fontsize=5.0,
             va="center", color=ns.C_BASE)

    # ---------------- c: what each method gained ------------------------
    axc = fig.add_subplot(gs[0, 2]); ns.tidy(axc); ns.panel_label(axc, "c")
    keys = ["ridge_esm2", "knn_esm2", "P3 head", "train_mean"]
    gains = [val(graph, k)[0] - val(plain, k)[0] for k in keys]
    cols = [ns.C_ACCENT if k == "ridge_esm2" else
            (ns.C_MODEL if k == "P3 head" else ns.C_BASE) for k in keys]
    yy = np.arange(len(keys))[::-1]
    axc.barh(yy, gains, height=0.52, color=cols, linewidth=0)
    axc.axvline(0, color=ns.C_FLOOR, lw=0.8)
    axc.set_yticks(yy)
    axc.set_yticklabels([F.pretty(k) for k in keys], fontsize=5.4)
    axc.set_xlabel("Gain from the graph features\n(Δ held-out $r$)")
    axc.set_xlim(-0.012, max(gains) * 1.34)
    for y, g in zip(yy, gains):
        axc.text(g + max(gains) * 0.035, y, f"{g:+.3f}", va="center", fontsize=5.4)
    axc.set_ylim(-0.62, len(keys) - 0.4)
    axc.text(max(gains) * 0.50, -0.52,
             "train mean does not read the\nembedding, and does not move",
             fontsize=5.0, color=ns.C_BASE, ha="center", va="center")

    res = ns.save_nature(fig, "fig2_main", outdir=outdir, width_mm=w_mm)
    res["font"] = style
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2, default=str))
