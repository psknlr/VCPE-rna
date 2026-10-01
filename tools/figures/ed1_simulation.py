"""Extended Data Fig. 1 | Four ways a score rises without skill, simulated.

Supports the protocol choices drawn in Fig. 1a. Every predictor here carries no
perturbation-specific information, or a fixed amount of it, so any rise along a
sweep is the size of a trap. Values come from results/simulation/
protocol_traps.json, written by tools/simulate_protocol_traps.py with the
repository's own metric and baseline functions.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata as F                                             # noqa: E402
import nature_style as ns                                       # noqa: E402

DASH = (0, (3.0, 1.6))
DOT = (0, (1.0, 1.4))


def series(ax, x, s, color, ls="-", marker="o", label=None, label_dy=0.0,
           label_side="right"):
    m, sd = np.asarray(s["mean"]), np.asarray(s["sd"])
    ax.errorbar(x, m, yerr=sd, color=color, ls=ls, lw=1.0, marker=marker, ms=2.6,
                mfc=color if ls == "-" else "white", mec=color, mew=0.8,
                elinewidth=0.6, capsize=1.0)
    if label:
        xe = x[-1] if label_side == "right" else x[0]
        ax.text(xe, m[-1 if label_side == "right" else 0] + label_dy, label,
                fontsize=5.4, ha="right" if label_side == "right" else "left",
                va="center", color=ns.C_TEXT)


def main(outdir="figures/extended_data"):
    style = ns.apply_nature_style()
    d = F.collect_ed()
    t = d["sim"]["traps"]
    nrep = d["sim"]["config"]["replicates"]
    fig, w_mm = ns.nature_figure("double", height_mm=112)
    gs = fig.add_gridspec(2, 2)

    # ---------------- a: shared response -----------------------------
    ax = fig.add_subplot(gs[0, 0]); ns.tidy(ax); ns.panel_label(ax, "a")
    a = t["shared"]
    x = np.asarray(a["x"])
    S = a["series"]
    series(ax, x, S["no_info_raw"], ns.C_BASE, DASH, "s")
    series(ax, x, S["informative_raw"], ns.C_MODEL, DASH, "o")
    series(ax, x, S["informative_residual"], ns.C_MODEL, "-", "o")
    series(ax, x, S["no_info_residual"], ns.C_BASE, "-", "s")
    ax.set_xlabel("Share of each gene's variance common to all perturbations")
    ax.set_ylabel("Within-perturbation $r$")
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlim(-0.03, 0.95)
    ax.text(0.30, 0.70, "raw fold change", fontsize=5.4, ha="right")
    ax.text(0.50, 0.37, f"residual, informative (fixed $r$ = {a['informative_r']:g})",
            fontsize=5.4, ha="center")
    ax.text(0.50, 0.06, "residual, no information", fontsize=5.4, ha="center")
    # key for colour, shared by the two raw lines
    ax.plot([0.04, 0.10], [0.95, 0.95], color=ns.C_MODEL, lw=1.0)
    ax.text(0.12, 0.95, "informative predictor", fontsize=5.4, va="center")
    ax.plot([0.04, 0.10], [0.87, 0.87], color=ns.C_BASE, lw=1.0)
    ax.text(0.12, 0.87, "training mean (no information)", fontsize=5.4,
            va="center")

    # ---------------- b: unmeasured columns --------------------------
    ax = fig.add_subplot(gs[0, 1]); ns.tidy(ax); ns.panel_label(ax, "b")
    b = t["unmeasured"]
    x = np.asarray(b["x"])
    series(ax, x, b["series"]["unmasked"], ns.C_ACCENT, DASH, "s",
           label="scored unmasked", label_dy=0.06)
    series(ax, x, b["series"]["masked"], ns.C_MODEL, "-", "o",
           label="masked", label_dy=0.05)
    ax.set_xlabel("Fraction of the panel one screen never measured")
    ax.set_ylabel("Training mean, within-perturbation $r$")
    ax.set_ylim(-0.04, 0.66)

    # ---------------- c: pooled estimator ----------------------------
    ax = fig.add_subplot(gs[1, 0]); ns.tidy(ax); ns.panel_label(ax, "c")
    c = t["pooled"]
    x = np.asarray(c["x"])
    series(ax, x, c["series"]["pooled"], ns.C_ACCENT, DASH, "s",
           label="pooled over all entries", label_dy=-0.09)
    series(ax, x, c["series"]["per_item"], ns.C_MODEL, "-", "o",
           label="within perturbation, averaged", label_dy=0.06)
    ax.set_xlabel("Between-perturbation s.d. of the overall level\n"
                  "(relative to the within-perturbation s.d.)")
    ax.set_ylabel("Pearson $r$ of a level-only predictor")
    ax.set_ylim(-0.04, 1.04)

    # ---------------- d: ties in top-k -------------------------------
    ax = fig.add_subplot(gs[1, 1]); ns.tidy(ax); ns.panel_label(ax, "d")
    e = t["ties"]
    x = np.asarray(e["x"])
    series(ax, x, e["series"]["index_ties_sorted"], ns.C_ACCENT, DASH, "s",
           label="ties by column, panel sorted by variance", label_dy=0.055)
    series(ax, x, e["series"]["index_ties_shuffled"], ns.C_BASE, DOT, "^")
    series(ax, x, e["series"]["tie_aware"], ns.C_MODEL, "-", "o")
    ax.axhline(e["chance"], color=ns.C_FLOOR, lw=0.6, ls=DOT, zorder=0)
    ax.text(x[-1], e["chance"] + 0.035,
            "tie-aware, or ties by column on a shuffled\n"
            f"panel: both at chance, $k/n$ = {e['chance']:g}", fontsize=5.4,
            ha="right",
            va="bottom")
    ax.set_xlabel("Coefficient of variation of the per-gene s.d.")
    ax.set_ylabel(f"Top-{e['k']} overlap of a constant prediction")
    ax.set_ylim(0, 0.70)

    res = ns.save_nature(fig, "edfig1_simulation", outdir=outdir, width_mm=w_mm)
    res["font"] = style
    res["n_replicates"] = nrep
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2, default=str))
