"""Extended Data Fig. 3 | Training dynamics, epoch selection and conditioning.

Supports Fig. 4b: read from the per-epoch training logs of every run. The logs
record inner-split scores only -- the held-out split is scored once, after
selection. Panel d is where ERRATA E17 became visible: in these runs the inner
split was also in the training batches, so its score tracks training fit, keeps
rising with the budget, and pulls selection to the last epochs.
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata as F                                             # noqa: E402
import nature_style as ns                                       # noqa: E402

TRAJ = [  # (config key or "probe", label, colour)
    ("plain_150M", "ESM2-150M, 30 epochs", ns.C_BASE),
    ("graph_30ep", "+ STRING neighbourhood, 30 epochs", ns.OKABE_ITO["sky"]),
    ("graph_80ep", "+ STRING neighbourhood, 80 epochs", ns.C_MODEL),
    ("probe", "+ STRING neighbourhood, 150 epochs (split 0)", ns.C_FLOOR),
]


def runs(d, key):
    if key == "probe":
        return {0: d["log_probe150"]}
    return d["logs"][key]


def main(outdir="figures/extended_data"):
    style = ns.apply_nature_style()
    d = F.collect_ed()
    fig, w_mm = ns.nature_figure("double", height_mm=112)
    gs = fig.add_gridspec(2, 2)
    axa = fig.add_subplot(gs[0, 0]); ns.tidy(axa); ns.panel_label(axa, "a")
    axb = fig.add_subplot(gs[0, 1]); ns.tidy(axb); ns.panel_label(axb, "b")
    axc = fig.add_subplot(gs[1, 0]); ns.tidy(axc); ns.panel_label(axc, "c")
    axd = fig.add_subplot(gs[1, 1]); ns.tidy(axd); ns.panel_label(axd, "d")

    for key, label, col in TRAJ:
        for s, L in runs(d, key).items():
            ep = [r["epoch"] for r in L["epochs"]]
            loss = [r["train_loss"] for r in L["epochs"]]
            iv = [r["eval"]["pearson_dev"] for r in L["epochs"]]
            ab = [r["ablation_r_real_vs_zeropert"] for r in L["epochs"]]
            lw = 1.0 if key == "probe" else 0.7
            kw = dict(color=col, lw=lw, label=label if s == 0 else None)
            axa.plot(ep, loss, **kw)
            axb.plot(ep, iv, **kw)
            axc.plot(ep, ab, **kw)
            sel = L["final"]["selected_epoch"]
            axb.plot([sel], [iv[sel - 1]], "o", ms=2.8, mfc="white", mec=col,
                     mew=0.8, zorder=4)

    axa.set_yscale("log")
    # plain decimals: a mathtext exponent would render below the 5 pt floor
    axa.yaxis.set_major_locator(FixedLocator([0.015, 0.02, 0.03, 0.04]))
    axa.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axa.yaxis.set_minor_formatter(NullFormatter())
    axa.set_xlabel("Epoch")
    axa.set_ylabel("Training loss (masked MSE)")
    axa.legend(fontsize=5.4, loc="upper right", handlelength=1.4,
               borderaxespad=0.2)
    axb.set_xlabel("Epoch")
    axb.set_ylabel("Inner-split $r$ (residual; also trained on)")
    axb.text(0.98, 0.04, "open circles: epoch selected", transform=axb.transAxes,
             ha="right", va="bottom", fontsize=5.4)
    axc.set_xlabel("Epoch")
    axc.set_ylabel("$r$(prediction, prediction with the\n"
                   "target embedding set to zero)")
    axc.set_ylim(0, 1.0)
    axc.text(0.98, 0.95, "1 = conditioning has no effect", transform=axc.transAxes,
             ha="right", va="top", fontsize=5.4)

    # ---------------- d: inner validation is optimistic -----------------
    allv = []
    marks = {"plain_35M": "v", "plain_150M": "o", "string_channel": "s",
             "graph_30ep": "^", "graph_80ep": "D"}
    for key, label, _, _ in d["configs"]:
        for s, rep in d["reports"][key].items():
            x, y = rep["pearson_dev"], rep["inner_val_pearson_dev_at_selection"]
            allv += [x, y]
            axd.plot([x], [y], marks[key], ms=3.0, mfc="white",
                     mec=ns.C_MODEL, mew=0.8, ls="none",
                     label=label if s == 0 else None)
    p = d["budget_seed0"][150]
    axd.plot([p["pearson_dev"]], [p["inner_val_pearson_dev_at_selection"]], "*",
             ms=4.5, color=ns.C_FLOOR, ls="none", label="150 epochs (split 0)")
    allv += [p["pearson_dev"], p["inner_val_pearson_dev_at_selection"]]
    lo, hi = 0.0, max(allv) * 1.08
    axd.plot([lo, hi], [lo, hi], color=ns.C_FLOOR, lw=0.6, ls=(0, (3, 2)))
    axd.text(hi * 0.93, hi * 0.86, "identity", fontsize=5.4, ha="left",
             va="top", rotation=0)
    axd.set_xlim(lo, hi); axd.set_ylim(lo, hi)
    axd.set_aspect("equal", adjustable="box")
    axd.set_xlabel("Held-out $r$, scored once")
    axd.set_ylabel("Inner-split $r$ at the selected epoch")
    axd.legend(fontsize=5.2, loc="lower right", handletextpad=0.2,
               borderaxespad=0.2, labelspacing=0.25)

    res = ns.save_nature(fig, "edfig3_training", outdir=outdir, width_mm=w_mm)
    res["font"] = style
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2, default=str))
