"""Fig. 1 | Study design: what is measured, on what data, under what controls.

Per the Nature figure guide, Fig. 1 draws the SCIENTIFIC workflow -- screens, QC
gates, the split, the predicted quantity, the controls -- not the software stack.
Nothing here is a software box diagram; every node is a decision that changes what
the reported number means.
"""
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata as F                                             # noqa: E402
import nature_style as ns                                       # noqa: E402

# Dataset facts are read from the ingest provenance written at ingest time.
DATASETS = [
    dict(name="RPE1 (Replogle 2022)", perts=2204, ctrl=3000, genes=8749,
         note="CRISPRi, RPE1"),
    dict(name="Neurons (Tian 2021)", perts=184, ctrl=437, genes=33538,
         note="CRISPRi, iPSC neuron"),
]


def box(ax, x, y, w, h, text, fc="white", ec=ns.C_FLOOR, lw=0.8, fs=5.6,
        weight="normal", tc=None):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            zorder=3, fontweight=weight, color=tc or ns.C_FLOOR, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, lw=0.8, color=None):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", linewidth=lw,
                                color=color or ns.C_FLOOR,
                                shrinkA=0.5, shrinkB=0.5,
                                mutation_scale=5.5), zorder=1)


def main(outdir="figures"):
    style = ns.apply_nature_style()
    d = F.collect()
    fig, w_mm = ns.nature_figure("double", height_mm=92)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.38, 1.0], width_ratios=[1, 1, 1])

    # ---------------- a: the scientific workflow ------------------------
    axa = fig.add_subplot(gs[0, :])
    axa.set_xlim(0, 1); axa.set_ylim(0, 1); axa.axis("off")
    ns.panel_label(axa, "a", dx=-0.012, dy=0.985)

    y0, bh = 0.60, 0.22
    box(axa, 0.005, y0, 0.148, bh,
        "Two CRISPRi\nPerturb-seq screens\n2,388 perturbations", fc="#F2F2F2")
    box(axa, 0.183, y0, 0.148, bh,
        "Per-cell\nnormalisation,\nthen pseudobulk", fc="white")
    box(axa, 0.361, y0, 0.158, bh,
        "Hold out whole\ntarget genes\n340 of 2,269", fc="white", ec=ns.C_ACCENT,
        lw=1.0)
    box(axa, 0.549, y0, 0.158, bh,
        "Predict residual:\nfold change minus\nshared response", fc="white")
    box(axa, 0.737, y0, 0.158, bh,
        "Score once, on\nmeasured genes\nonly", fc="white", ec=ns.C_ACCENT, lw=1.0)
    box(axa, 0.925, y0, 0.070, bh, "Report", fc="#F2F2F2")
    for x0, x1 in ((0.153, 0.183), (0.331, 0.361), (0.519, 0.549),
                   (0.707, 0.737), (0.895, 0.925)):
        arrow(axa, x0, y0 + bh / 2, x1, y0 + bh / 2)

    # the three discipline gates, drawn beneath the stage they constrain
    gy, gh = 0.235, 0.185
    box(axa, 0.345, gy, 0.190, gh,
        "Gene-level split, so a\nheld-out perturbation's\ngene is unseen",
        fc="#FFF4EC", ec=ns.C_ACCENT, lw=0.8, fs=5.2)
    box(axa, 0.545, gy, 0.190, gh,
        "Epoch chosen on an inner\nsplit; held-out set\ntouched exactly once",
        fc="#FFF4EC", ec=ns.C_ACCENT, lw=0.8, fs=5.2)
    box(axa, 0.745, gy, 0.190, gh,
        "Unmeasured genes masked;\nthey carry a per-dataset\nconstant, not signal",
        fc="#FFF4EC", ec=ns.C_ACCENT, lw=0.8, fs=5.2)
    for gx, sx in ((0.440, 0.440), (0.640, 0.628), (0.840, 0.815)):
        arrow(axa, sx, y0 - 0.005, gx, gy + gh + 0.005, lw=0.6, color=ns.C_ACCENT)

    axa.text(0.005, 0.10,
             "Controls scored on the identical split, gene panel, predicted "
             "quantity and estimator:\nno change · training mean · k-NN retrieval · "
             "ridge regression · published models (GEARS, CPA; neuronal screen only)",
             fontsize=5.2, va="bottom", ha="left", linespacing=1.5)

    # ---------------- b: dataset composition ---------------------------
    axb = fig.add_subplot(gs[1, 0])
    ns.tidy(axb); ns.panel_label(axb, "b")
    names = [x["name"] for x in DATASETS]
    perts = [x["perts"] for x in DATASETS]
    yy = np.arange(len(names))
    axb.barh(yy, perts, height=0.5, color=[ns.C_MODEL, ns.C_BASE], linewidth=0)
    for i, v in enumerate(perts):
        axb.text(v * 1.04, i, f"{v:,}", va="center", fontsize=5.6)
    axb.set_yticks(yy)
    axb.set_yticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=5.4)
    axb.set_xlabel("Perturbations retained (n)")
    axb.set_xlim(0, max(perts) * 1.30)
    axb.invert_yaxis()

    # ---------------- c: where the signal comes from -------------------
    axc = fig.add_subplot(gs[1, 1])
    ns.tidy(axc); ns.panel_label(axc, "c")
    both = d["plain_150M"]["model"]["mean"]
    tian = d["head_tian"]["pearson_dev"]
    axc.bar([0, 1], [both, tian], width=0.5,
            color=[ns.C_MODEL, ns.C_BASE], linewidth=0)
    axc.axhline(0, color=ns.C_FLOOR, lw=0.8)
    axc.set_xticks([0, 1])
    axc.set_xticklabels(["Both\nscreens", "Neuronal\nscreen only"], fontsize=5.4)
    axc.set_ylabel("Held-out $r$ (residual)")
    axc.set_ylim(-0.02, 0.36)
    for x, v in ((0, both), (1, tian)):
        axc.text(x, v + 0.012, f"{v:+.3f}", ha="center", fontsize=5.6)
    axc.text(0.5, 0.30, "27 held-out genes:\nevery method at the floor",
             ha="center", fontsize=5.0, color=ns.C_TEXT)

    # ---------------- d: the estimator trap ----------------------------
    axd = fig.add_subplot(gs[1, 2])
    ns.tidy(axd); ns.panel_label(axd, "d")
    rep = d["e6_masked"]
    delta, dev = rep["pearson_delta"], rep["pearson_dev"]
    axd.bar([0, 1], [delta, dev], width=0.5,
            color=[ns.C_ALT, ns.C_MODEL], linewidth=0)
    axd.set_xticks([0, 1])
    axd.set_xticklabels(["Raw fold\nchange", "Residual"], fontsize=5.4)
    axd.set_ylabel("Held-out $r$")
    axd.set_ylim(0, max(delta, dev) * 1.32)
    for x, v in ((0, delta), (1, dev)):
        axd.text(x, v + 0.012, f"{v:+.3f}", ha="center", fontsize=5.6)
    # arrow drawn low, inside the bars, so it cannot cross the data labels
    axd.annotate("", xy=(0.80, dev * 0.52), xytext=(0.20, delta * 0.52),
                 arrowprops=dict(arrowstyle="-|>", lw=0.7, color=ns.C_ACCENT,
                                 mutation_scale=5))
    axd.text(0.5, max(delta, dev) * 1.18,
             f"shared response inflates by {delta - dev:+.3f}", ha="center",
             fontsize=5.0, color=ns.C_TEXT)

    res = ns.save_nature(fig, "fig1_design", outdir=outdir, width_mm=w_mm)
    res["font"] = style
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2, default=str))
