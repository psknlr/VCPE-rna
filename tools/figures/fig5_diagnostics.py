"""Fig. 5 | Three ways a perturbation-response number inflates, each measured.

One message: in this task the evaluation protocol moves the headline number by
more than the model does, so a reported score is uninterpretable without the
protocol beside it. Every panel is a measured A/B on real data, not an argument.
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
    fig, w_mm = ns.nature_figure("double", height_mm=76)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.25])

    # ---------------- a: target quantity (shared response) -------------
    axa = fig.add_subplot(gs[0, 0]); ns.tidy(axa); ns.panel_label(axa, "a")
    rep = d["e6_masked"]
    pairs = [("Raw fold\nchange", rep["pearson_delta"], ns.C_ALT),
             ("Residual after\nremoving the\nshared response", rep["pearson_dev"],
              ns.C_MODEL)]
    axa.bar([0, 1], [p[1] for p in pairs], width=0.5,
            color=[p[2] for p in pairs], linewidth=0)
    axa.set_xticks([0, 1]); axa.set_xticklabels([p[0] for p in pairs], fontsize=5.2)
    axa.set_ylabel("Held-out $r$")
    axa.set_ylim(0, pairs[0][1] * 1.30)
    for xi, p in enumerate(pairs):
        axa.text(xi, p[1] + pairs[0][1] * 0.03, f"{p[1]:+.3f}", ha="center",
                 fontsize=5.4)
    axa.set_title("Predicted quantity", fontsize=6.0, pad=3)
    axa.text(0.5, pairs[0][1] * 1.17,
             f"{pairs[0][1] - pairs[1][1]:+.3f} from the response\nshared "
             f"across perturbations", ha="center", fontsize=5.0, color=ns.C_TEXT)

    # ---------------- b: masking unmeasured genes ----------------------
    axb = fig.add_subplot(gs[0, 1]); ns.tidy(axb); ns.panel_label(axb, "b")
    m, u = d["e6_masked"], d["e6_unmasked"]
    keys = [("Conditioned\nhead", m["pearson_dev"], u["pearson_dev"]),
            ("Training\nmean", m["baselines"]["train_mean"]["pearson_dev"],
             u["baselines"]["train_mean"]["pearson_dev"])]
    x = np.arange(len(keys)); wdt = 0.34
    axb.bar(x - wdt / 2, [k[1] for k in keys], wdt, color=ns.C_MODEL, linewidth=0,
            label="Masked")
    axb.bar(x + wdt / 2, [k[2] for k in keys], wdt, color=ns.C_ALT, linewidth=0,
            label="Unmasked")
    axb.set_xticks(x); axb.set_xticklabels([k[0] for k in keys], fontsize=5.4)
    axb.set_ylabel("Held-out $r$")
    axb.set_yscale("symlog", linthresh=0.02)
    axb.set_ylim(0, 0.6)
    # plain decimals: an exponent label would be set at 0.7 x 6 pt = 4.2 pt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    axb.yaxis.set_major_locator(FixedLocator([0, 0.01, 0.02, 0.1, 0.3, 0.6]))
    axb.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axb.yaxis.set_minor_formatter(NullFormatter())
    axb.legend(fontsize=5.2, loc="upper center", ncol=2, handlelength=1.0,
               borderpad=0.2, columnspacing=0.7)
    axb.set_title("Unmeasured genes (0.14% of held-out entries)", fontsize=6.0, pad=3)
    dlt = keys[1][2] - keys[1][1]
    axb.annotate(f"a predictor with no\nperturbation information\ngains {dlt:+.4f}",
                 xy=(1 + wdt / 2, keys[1][2]), xytext=(1.0, 0.10),
                 fontsize=5.0, color=ns.C_TEXT, ha="center",
                 arrowprops=dict(arrowstyle="-|>", lw=0.7, color=ns.C_ACCENT,
                                 mutation_scale=5))

    # ---------------- c: inner-val vs held-out, per design -------------
    axc = fig.add_subplot(gs[0, 2]); ns.tidy(axc); ns.panel_label(axc, "c")
    var = sorted(d["sweep"]["variants"], key=lambda r: r["ridge_inner_val"])
    labs = [v["name"].replace(", norm-matched", "").replace(" norm", "")
            for v in var]
    iv = [v["ridge_inner_val"] for v in var]
    yy = np.arange(len(var))
    axc.barh(yy, iv, height=0.58, color=ns.C_BASE, linewidth=0)
    held = d["graph_80ep"]["baselines"]["ridge_esm2"]["mean"]
    axc.axvline(held, color=ns.C_ACCENT, lw=1.0, ls=(0, (3, 2)))
    axc.set_yticks(yy); axc.set_yticklabels(labs, fontsize=5.0)
    axc.set_xlabel("Ridge $r$ on inner validation")
    axc.set_xlim(0, max(iv) * 1.16)
    axc.set_title("Representation designs, inner validation only",
                  fontsize=6.0, pad=3)
    axc.text(held + max(iv) * 0.012, 0.15,
             f"held-out ridge\n{held:+.3f}", fontsize=5.0, color=ns.C_TEXT,
             va="bottom")
    axc.set_ylim(-0.7, len(var) - 0.3)

    res = ns.save_nature(fig, "fig5_diagnostics", outdir=outdir, width_mm=w_mm)
    res["font"] = style
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2, default=str))
