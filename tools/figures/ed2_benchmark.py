"""Extended Data Fig. 2 | Every split, every control, every representation.

The full record behind Fig. 2 and Fig. 4c: held-out r for the head and the three
non-trivial controls on each of the three splits under all five configurations
run with the corrected protocol, and the head's paired difference against the two
controls that read the same embedding table.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata as F                                             # noqa: E402
import nature_style as ns                                       # noqa: E402

METHODS = [  # key in the report, label, colour, marker
    ("model", "Conditioned head", ns.C_MODEL, "o"),
    ("ridge_esm2", "Ridge regression", ns.C_ALT, "s"),
    ("knn_esm2", "k-NN", ns.OKABE_ITO["purple"], "^"),
    ("train_mean", "Training mean", ns.C_BASE, "D"),
]


def value(rep, key):
    return rep["pearson_dev"] if key == "model" else rep["baselines"][key]["pearson_dev"]


def forest(ax, d, control, title):
    cfgs = d["configs"]
    yy = np.arange(len(cfgs))[::-1]
    for y, (key, label, _, _) in zip(yy, cfgs):
        p = F.paired(d[key], control)
        col = ns.C_MODEL if p["lo"] > 0 else (ns.C_ACCENT if p["hi"] < 0 else ns.C_BASE)
        ax.plot([p["lo"], p["hi"]], [y, y], color=col, lw=1.0, solid_capstyle="butt")
        ax.plot([p["diff"]], [y], "o", ms=3.0, color=col, zorder=3)
        for s in p["per_seed"]:
            ax.plot([s], [y - 0.22], "|", ms=3.2, color=col)
    ax.axvline(0, color=ns.C_FLOOR, lw=0.8, ls=(0, (3, 2)))
    ax.set_yticks(yy)
    ax.set_yticklabels([c[1] for c in cfgs], fontsize=5.6)
    ax.set_ylim(-0.7, len(cfgs) - 0.4)
    ax.set_xlabel(f"Head − {title}, paired by split (Δ held-out $r$)")
    # at most five ticks: the k-NN panel spans 0.2 and its labels otherwise touch
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))


def main(outdir="figures/extended_data"):
    style = ns.apply_nature_style()
    d = F.collect_ed()
    fig, w_mm = ns.nature_figure("double", height_mm=118)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])

    # ---------------- a: per split, per method ------------------------
    ax = fig.add_subplot(gs[0, :]); ns.tidy(ax); ns.panel_label(ax, "a", dx=-0.04)
    cfgs = d["configs"]
    offs = np.linspace(-0.30, 0.30, len(METHODS))
    for gi, (key, label, _, _) in enumerate(cfgs):
        reps = d["reports"][key]
        for (mk, mlab, col, mar), off in zip(METHODS, offs):
            v = [value(reps[s], mk) for s in sorted(reps)]
            xs = gi + off + np.array([-0.035, 0.0, 0.035])
            ax.plot(xs, v, mar, ms=2.6, mfc="white", mec=col, mew=0.8,
                    ls="none", zorder=3)
            ax.plot([gi + off - 0.06, gi + off + 0.06], [np.mean(v)] * 2,
                    color=col, lw=1.0, solid_capstyle="butt", zorder=2)
    for (mk, mlab, col, mar) in METHODS:
        ax.plot([], [], mar, ms=2.6, mfc="white", mec=col, mew=0.8, ls="none",
                label=mlab)
    ax.axhline(0, color=ns.C_FLOOR, lw=0.6)
    ax.set_xticks(range(len(cfgs)))
    ax.set_xticklabels([c[1].replace(", ", ",\n").replace("+ ", "+ ")
                        for c in cfgs], fontsize=5.6)
    ax.set_xlim(-0.5, len(cfgs) - 0.5)
    ax.set_ylabel("Held-out $r$ (residual)")
    ax.legend(loc="upper left", ncol=4, fontsize=5.6, handletextpad=0.3,
              columnspacing=1.2, borderaxespad=0.1)
    ax.set_ylim(-0.03, 0.42)

    # ---------------- b, c: paired differences ------------------------
    axb = fig.add_subplot(gs[1, 0]); ns.tidy(axb); ns.panel_label(axb, "b", dx=-0.32)
    forest(axb, d, "ridge_esm2", "ridge")
    axc = fig.add_subplot(gs[1, 1]); ns.tidy(axc); ns.panel_label(axc, "c", dx=-0.32)
    forest(axc, d, "knn_esm2", "k-NN")

    res = ns.save_nature(fig, "edfig2_benchmark", outdir=outdir, width_mm=w_mm)
    res["font"] = style
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2, default=str))
