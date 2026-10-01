"""Extended Data Fig. 4 | How the other metrics behave on the real benchmark.

Supports Fig. 5 and the simulation in Extended Data Fig. 1 with the committed real
runs: raw fold-change r barely separates representations that the residual r
separates; the pooled estimator scores every control higher than the
within-perturbation one; and top-50 overlap ranks a model with its conditioning
removed, and a constant prediction, above the intact model.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata as F                                             # noqa: E402
import nature_style as ns                                       # noqa: E402

MARKS = {"plain_35M": "v", "plain_150M": "o", "string_channel": "s",
         "graph_30ep": "^", "graph_80ep": "D"}


def main(outdir="figures/extended_data"):
    style = ns.apply_nature_style()
    d = F.collect_ed()
    fig, w_mm = ns.nature_figure("double", height_mm=66)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.32])

    # ---------------- a: raw fold change compresses the comparison ------
    ax = fig.add_subplot(gs[0, 0]); ns.tidy(ax); ns.panel_label(ax, "a")
    xs, ys = [], []
    for key, label, _, _ in d["configs"]:
        for s, rep in d["reports"][key].items():
            xs.append(rep["pearson_dev"]); ys.append(rep["pearson_delta"])
            ax.plot([xs[-1]], [ys[-1]], MARKS[key], ms=3.0, mfc="white",
                    mec=ns.C_MODEL, mew=0.8, ls="none",
                    label=label if s == 0 else None)
    lo, hi = 0.0, 0.62
    ax.plot([lo, hi], [lo, hi], color=ns.C_FLOOR, lw=0.6, ls=(0, (3, 2)))
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Held-out $r$, residual")
    ax.set_ylabel("Held-out $r$, raw fold change")
    ax.text(0.02, 0.40,
            f"spread across the 15 runs:\nresidual {max(xs) - min(xs):.3f}\n"
            f"raw {max(ys) - min(ys):.3f}", fontsize=5.4, ha="left", va="top")
    ax.legend(fontsize=5.0, loc="lower right", handletextpad=0.2,
              borderaxespad=0.1, labelspacing=0.2)

    # ---------------- b: pooled versus per perturbation ----------------
    ax = fig.add_subplot(gs[0, 1]); ns.tidy(ax); ns.panel_label(ax, "b")
    ctl = [("ridge_esm2", "ESM2 ridge", ns.C_ALT, "s"),
           ("knn_esm2", "ESM2 k-NN", ns.OKABE_ITO["purple"], "^"),
           ("train_mean", "Train mean", ns.C_BASE, "D")]
    for name, label, col, mar in ctl:
        px, py = [], []
        for key, _, _, _ in d["configs"]:
            for s, rep in d["reports"][key].items():
                b = rep["baselines"][name]
                px.append(b["pearson_dev"]); py.append(b["pearson_dev_pooled"])
        ax.plot(px, py, mar, ms=3.0, mfc="white", mec=col, mew=0.8, ls="none",
                label=label)
    lo, hi = -0.02, 0.52
    ax.plot([lo, hi], [lo, hi], color=ns.C_FLOOR, lw=0.6, ls=(0, (3, 2)))
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Within-perturbation $r$, averaged")
    ax.set_ylabel("$r$ pooled over all entries")
    ax.legend(fontsize=5.0, loc="lower right", handletextpad=0.2,
              borderaxespad=0.1, labelspacing=0.2)

    # ---------------- c: top-50 overlap on split 0 ---------------------
    ax = fig.add_subplot(gs[0, 2]); ns.tidy(ax); ns.panel_label(ax, "c", dx=-0.42)
    rep = d["e6_masked"]                       # plain 150M, split 0
    abl = d["ablation_plain"]["report"]
    assert abs(abl["intact"]["top50_dev"] - rep["top50_dev"]) < 1e-9, \
        "the ablation and the report must be the same checkpoint and split"
    rows = [
        ("Predict no change", rep["baselines"]["zero"]["top50_dev"], ns.C_ACCENT),
        ("Train mean", rep["baselines"]["train_mean"]["top50_dev"], ns.C_BASE),
        ("ESM2 k-NN", rep["baselines"]["knn_esm2"]["top50_dev"], ns.C_BASE),
        ("ESM2 ridge", rep["baselines"]["ridge_esm2"]["top50_dev"], ns.C_BASE),
        ("P3 head", rep["top50_dev"], ns.C_MODEL),
        ("Head, target vector → 0", abl["esm_zero"]["top50_dev"], ns.C_MODEL),
        ("Head, all conditioning off", abl["all_off"]["top50_dev"], ns.C_MODEL),
    ]
    k = 50
    n_hvg = d["plain_150M"]["protocol"]["n_hvg"]
    chance = k / (rep["measured_fraction"] * n_hvg)   # mean measured panel size
    yy = np.arange(len(rows))[::-1]
    for y, (lab, v, col) in zip(yy, rows):
        hollow = lab.startswith("Head,")
        ax.barh(y, v, height=0.56, color="white" if hollow else col,
                edgecolor=col, linewidth=0.8 if hollow else 0)
        ax.text(v + 0.006, y, f"{v:.3f}", va="center", fontsize=5.4)
    ax.axvline(chance, color=ns.C_FLOOR, lw=0.8, ls=(0, (1, 1.4)))
    ax.text(chance + 0.004, yy[-1] - 0.62, f"chance, $k/n$ = {chance:.2f}",
            fontsize=5.4, va="center")
    y0 = yy[0]
    ax.annotate("", xy=(chance, y0 + 0.36), xytext=(rows[0][1], y0 + 0.36),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=ns.C_FLOOR,
                                mutation_scale=5))
    ax.text((chance + rows[0][1]) / 2, y0 + 0.50, "tie-aware form (E16)",
            fontsize=5.0, ha="center", va="bottom")
    ax.set_yticks(yy)
    ax.set_yticklabels([r[0] for r in rows], fontsize=5.6)
    ax.set_xlabel(f"Top-{k} overlap, split 0 (as recorded)")
    ax.set_xlim(0, 0.46)
    ax.set_ylim(-1.0, len(rows) - 0.1)

    res = ns.save_nature(fig, "edfig4_metrics", outdir=outdir, width_mm=w_mm)
    res["font"] = style
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2, default=str))
