"""Fig. 3 | What the score rests on, and where network information must enter.

One message: the conditioning is real -- destroying target-gene identity collapses
the prediction -- but it flows through a single channel, which is why a sparse
partner indicator could not help and a dense embedding could.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata as F                                             # noqa: E402
import nature_style as ns                                       # noqa: E402

SHOW = ["intact", "esm_shuffle", "esm_zero", "ds_shuffle",
        "is_nb_shuffle", "is_tgt_shuffle", "rna_off", "all_off"]


def main(outdir="figures"):
    style = ns.apply_nature_style()
    d = F.collect()
    fig, w_mm = ns.nature_figure("double", height_mm=82)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.0, 1.08])

    rows = {k: (r, pd_, dl) for k, r, pd_, dl in F.ablation_rows(d["ablation_plain"])}
    interp = d["ablation_plain"]["report"]["interpretation"]

    # ---------------- a: per-axis ablation -----------------------------
    axa = fig.add_subplot(gs[0, 0]); ns.tidy(axa); ns.panel_label(axa, "a")
    labs = [k for k in SHOW if k in rows]
    vals = [rows[k][1] for k in labs]
    yy = np.arange(len(labs))[::-1]
    cols = []
    for k in labs:
        if k == "intact":
            cols.append(ns.C_FLOOR)
        elif k in ("esm_shuffle", "esm_zero", "all_off"):
            cols.append(ns.C_ACCENT)
        elif k == "ds_shuffle":
            cols.append(ns.C_ALT)
        else:
            cols.append(ns.C_BASE)
    axa.barh(yy, vals, height=0.56, color=cols, linewidth=0)
    axa.axvline(rows["intact"][1], color=ns.C_FLOOR, lw=0.8, ls=(0, (3, 2)))
    axa.set_yticks(yy)
    axa.set_yticklabels([F.pretty(k) for k in labs], fontsize=5.4)
    axa.set_xlabel("Held-out $r$ after disabling one channel")
    axa.set_xlim(0, rows["intact"][1] * 1.22)
    for y, v in zip(yy, vals):
        axa.text(v + rows["intact"][1] * 0.02, y, f"{v:+.3f}", va="center",
                 fontsize=5.2)
    axa.text(rows["intact"][1] * 0.40, yy[-1] - 0.95,
             "grey: $r$ changes by less than 0.00001",
             fontsize=5.0, color=ns.C_TEXT, ha="center", va="center")
    axa.set_ylim(-1.5, len(labs) - 0.4)

    # ---------------- b: which floor you quote matters ------------------
    axb = fig.add_subplot(gs[0, 1]); ns.tidy(axb); ns.panel_label(axb, "b")
    # Read the floors from the measured per-axis rows rather than the
    # interpretation block: that block's schema changed when the shuffle-based
    # floor was made primary, so older result files carry the earlier key names
    # while the underlying measurements are identical in both.
    intact = rows["intact"][1]
    fz = rows["all_off"][1]          # zeroed floor: off-distribution, flatters
    fs = rows["esm_shuffle"][1]      # permuted floor: preserves input statistics
    axb.bar([0, 1], [intact - fz, intact - fs], width=0.5,
            color=[ns.C_BASE, ns.C_ACCENT], linewidth=0)
    axb.set_xticks([0, 1])
    axb.set_xticklabels(["vs zeroed\nfloor", "vs permuted\nfloor"], fontsize=5.4)
    axb.set_ylabel("Attributed to conditioning (Δ $r$)")
    axb.set_ylim(0, (intact - fz) * 1.26)
    for xi, v in ((0, intact - fz), (1, intact - fs)):
        axb.text(xi, v + (intact - fz) * 0.03,
                 f"{v:+.3f}\n({v / intact * 100:.0f}% of score)", ha="center",
                 fontsize=5.2)
    axb.text(0.5, (intact - fz) * 1.14,
             "zeroing is off-distribution\nand flatters the floor",
             ha="center", fontsize=5.0, color=ns.C_TEXT)

    # ---------------- c: sparse channel vs dense embedding -------------
    axc = fig.add_subplot(gs[0, 2]); ns.tidy(axc); ns.panel_label(axc, "c")
    base = d["plain_150M"]["model"]["mean"]
    chan = d["string_channel"]["model"]["mean"]
    dense = d["graph_30ep"]["model"]["mean"]   # same 30-epoch budget as both others
    gains = [chan - base, dense - base]
    axc.bar([0, 1], gains, width=0.5, color=[ns.C_BASE, ns.C_MODEL], linewidth=0)
    axc.axhline(0, color=ns.C_FLOOR, lw=0.8)
    axc.set_xticks([0, 1])
    # the coverage contrast belongs in the tick label: placed inside the panel it
    # lands on top of the bar it describes
    axc.set_xticklabels(["Sparse partner indicator\n(0.27% of scored entries)",
                         "Dense neighbourhood\nembedding (all genes)"],
                        fontsize=5.2)
    axc.set_ylabel("Gain over ESM2-150M (Δ held-out $r$)")
    axc.set_ylim(-0.004, max(gains) * 1.30)
    for xi, v in zip((0, 1), gains):
        axc.text(xi, v + max(gains) * 0.035, f"{v:+.4f}", ha="center", fontsize=5.4)
    axc.text(0.5, max(gains) * 1.14,
             "head, all at 30 epochs", ha="center", fontsize=5.0,
             color=ns.C_TEXT)

    res = ns.save_nature(fig, "fig3_mechanism", outdir=outdir, width_mm=w_mm)
    res["font"] = style
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2, default=str))
