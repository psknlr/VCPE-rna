"""Extended Data Figs. 1-4, IEEE Transactions style.

Each figure reads only files that are committed under results/ (the per-seed
final_report.json files that the main figures use are gitignored run outputs and
are not needed here). No measurement is typed into this file.

  ED Fig. 1  per-split held-out r for every reported configuration, and the epoch
             inner validation selected against the budget
  ED Fig. 2  the full conditioning-channel ablation, both runs
  ED Fig. 3  the graph-feature search on inner validation, per split
  ED Fig. 4  training-budget probe (split 0) and the smaller-screen comparison
             with the published models
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata as F                                             # noqa: E402
import ieee_style as st                                         # noqa: E402

# marker + fill both differ, so every series survives a greyscale print
SERIES = [("model", "Conditioned head", "o", st.C_MODEL),
          ("ridge_esm2", "ESM2 ridge", "s", st.C_RIDGE),
          ("knn_esm2", "ESM2 k-NN", "^", st.C_KNN),
          ("train_mean", "Train mean", "x", st.C_FLOOR)]

CONFIGS = [("results/real_35M/A_seeds/seed_summary.json", "ESM2-35M\n30 ep"),
           ("results/real_150M/A_seeds/seed_summary.json", "ESM2-150M\n30 ep"),
           ("results/real_150M_string/A_seeds/seed_summary.json",
            "150M + STRING\nflag, 30 ep"),
           ("results/real_150M_graph/A_seeds/seed_summary.json",
            "150M + STRING\nemb., 30 ep"),
           ("results/real_150M_graph_long/A_seeds/seed_summary.json",
            "150M + STRING\nemb., 80 ep")]


def _per_seed(summary, key):
    blk = summary["model"] if key == "model" else summary["baselines"].get(key)
    return None if blk is None else blk["per_seed"]


def ed1(outdir):
    fig, w = st.ieee_figure("double", height_in=2.75)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1.0])
    ax = fig.add_subplot(gs[0])
    sums = [F._summary(p) for p, _ in CONFIGS]
    off = np.linspace(-0.27, 0.27, len(SERIES))
    for (key, lab, mk, c), dx in zip(SERIES, off):
        for i, s in enumerate(sums):
            v = _per_seed(s, key)
            xs = np.full(len(v), i + dx)
            ax.scatter(xs, v, marker=mk, s=14, facecolor=c if mk != "x" else c,
                       edgecolor="black" if mk != "x" else None, linewidth=0.5,
                       zorder=3, label=lab if i == 0 else None)
            ax.plot([i + dx - 0.05, i + dx + 0.05], [np.mean(v)] * 2,
                    color="black", lw=1.0, zorder=4)
    ax.set_xticks(range(len(CONFIGS)))
    ax.set_xticklabels([l for _, l in CONFIGS], fontsize=7)
    ax.set_ylabel("Held-out Pearson $r$ (residual)")
    ax.set_ylim(-0.02, 0.45)
    ax.set_xlim(-0.6, len(CONFIGS) - 0.4)
    ax.legend(loc="upper left", ncol=4, fontsize=7, handletextpad=0.3,
              columnspacing=0.9)
    st.subcaption(ax, "a", "Per-split held-out $r$ (3 splits; bar = mean)",
                  y=-0.25)

    ax2 = fig.add_subplot(gs[1])
    for i, s in enumerate(sums):
        budget = s["protocol"]["epochs"]
        sel = s["selected_epochs"]
        ax2.scatter(np.full(len(sel), i) + np.array([-0.12, 0, 0.12]),
                    np.array(sel) / budget, marker="o", s=14,
                    facecolor=st.C_MODEL, edgecolor="black", linewidth=0.5,
                    zorder=3)
    ax2.axhline(1.0, color=st.C_ACCENT, lw=0.8, ls="--")
    ax2.text(-0.45, 1.005, "budget", color=st.C_ACCENT, fontsize=7,
             ha="left", va="bottom")
    ax2.set_xticks(range(len(CONFIGS)))
    ax2.set_xticklabels(["35M\n30 ep", "150M\n30 ep", "+flag\n30 ep",
                         "+emb.\n30 ep", "+emb.\n80 ep"], fontsize=7)
    ax2.set_xlim(-0.5, len(CONFIGS) - 0.5)
    ax2.set_ylabel("Selected epoch / budget")
    ax2.set_ylim(0.80, 1.04)
    st.subcaption(ax2, "b", "Inner-validation epoch", y=-0.25)
    return st.save_ieee(fig, "ieee_ed_fig1_per_split", outdir, w)


ORDER = ["esm_zero", "esm_shuffle", "all_off", "ds_shuffle", "is_nb_shuffle",
         "is_nb_off", "is_tgt_shuffle", "is_tgt_off", "rna_off"]


def ed2(outdir):
    runs = [("ESM2-150M", F._load("results/real_150M/C_ablation.json")["report"],
             st.C_MODEL, None),
            ("+ STRING flag", F._load("results/real_150M_string/C_ablation.json")
             ["report"], st.C_RIDGE, "////")]
    fig, w = st.ieee_figure("double", height_in=2.9)
    gs = fig.add_gridspec(1, 2)
    y = np.arange(len(ORDER))
    for j, (metric, xl, letter, cap) in enumerate((
            ("delta_vs_intact", "$\\Delta r$ vs. intact model", "a",
             "Change in held-out $r$"),
            ("r_vs_real", "Pearson $r$ with intact predictions", "b",
             "Similarity to intact predictions"))):
        ax = fig.add_subplot(gs[j])
        for k, (lab, rep, c, h) in enumerate(runs):
            v = [rep[a][metric] for a in ORDER]
            ax.barh(y + (k - 0.5) * 0.38, v, height=0.38, color=c, hatch=h,
                    edgecolor="black", linewidth=0.5, label=lab, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([F.pretty(a) for a in ORDER] if j == 0 else [],
                           fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(xl)
        ax.axvline(0 if j == 0 else 1, color="black", lw=0.6)
        if j == 0:
            for i, a in enumerate(ORDER):
                m = max(abs(rep[a][metric]) for _, rep, _, _ in runs)
                if m < 1e-3:
                    ax.text(-0.004, i, "$|\\Delta r|<10^{-3}$", fontsize=7,
                            ha="right", va="center")
            ax.set_xlim(-0.32, 0.02)
            ax.legend(loc="lower left", fontsize=7)
        else:
            ax.set_xlim(0, 1.05)
        st.subcaption(ax, letter, cap, y=-0.22)
    return st.save_ieee(fig, "ieee_ed_fig2_ablation", outdir, w)


def ed3(outdir):
    sw = F._load("results/feature_sweep_innerval.json")
    vs = sorted(sw["variants"], key=lambda v: v["ridge_inner_val"])
    fig, w = st.ieee_figure("double", height_in=2.6)
    gs = fig.add_gridspec(1, 2)
    y = np.arange(len(vs))
    for j, (key, ps, lab, letter) in enumerate((
            ("ridge_inner_val", "ridge_per_seed", "Ridge", "a"),
            ("knn_inner_val", "knn_per_seed", "k-NN", "b"))):
        ax = fig.add_subplot(gs[j])
        for i, v in enumerate(vs):
            ax.scatter(v[ps], [i] * len(v[ps]), marker="o", s=10,
                       facecolor="white", edgecolor="black", linewidth=0.5,
                       zorder=3)
            ax.scatter([v[key]], [i], marker="D", s=18,
                       facecolor=st.C_MODEL if v["name"] == sw["candidate"]
                       else st.C_KNN, edgecolor="black", linewidth=0.5,
                       zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{v['name']} ({v['dim']}-d)" for v in vs]
                           if j == 0 else [], fontsize=7)
        ax.set_xlabel(f"{lab} inner-validation $r$")
        if j == 0:
            ax.axvline(sw["plain_reference"], color=st.C_ACCENT, lw=0.8, ls="--")
            ax.text(sw["plain_reference"], len(vs) - 1.5, " no-graph reference",
                    color=st.C_ACCENT, fontsize=7, va="center")
        st.subcaption(ax, letter, f"{lab} readout (diamond = mean of 3 "
                      "splits, circles = splits)", y=-0.22)
    return st.save_ieee(fig, "ieee_ed_fig3_feature_search", outdir, w)


def ed4(outdir):
    g30 = F._summary("results/real_150M_graph/A_seeds/seed_summary.json")
    g80 = F._summary("results/real_150M_graph_long/A_seeds/seed_summary.json")
    probe = F._load("results/real_150M_graph_probe/seed0/final_report.json")
    budgets = [30, 80, 150]
    head = [g30["model"]["per_seed"][0], g80["model"]["per_seed"][0],
            probe["pearson_dev"]]
    sel = [g30["selected_epochs"][0], g80["selected_epochs"][0],
           probe["selected_epoch"]]
    ridge0 = probe["baselines"]["ridge_esm2"]["pearson_dev"]
    knn0 = probe["baselines"]["knn_esm2"]["pearson_dev"]

    fig, w = st.ieee_figure("double", height_in=2.6)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25])
    ax = fig.add_subplot(gs[0])
    ax.plot(budgets, head, "-o", color=st.C_MODEL, mfc=st.C_MODEL, mec="black",
            mew=0.5, ms=4, label="Conditioned head")
    ax.axhline(ridge0, color="black", lw=0.8, ls="--", label="ESM2 ridge")
    ax.axhline(knn0, color=st.C_GREY, lw=0.8, ls=":", label="ESM2 k-NN")
    for b, h, s in zip(budgets, head, sel):
        ax.annotate(f"sel. {s}", (b, h), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=7)
    ax.set_xticks(budgets)
    ax.set_xlim(15, 165)
    ax.set_xlabel("Training budget (epochs)")
    ax.set_ylabel("Held-out $r$, split 0")
    ax.set_ylim(0.20, 0.38)
    ax.legend(loc="lower right", fontsize=7)
    st.subcaption(ax, "a", "Budget probe (single split)", y=-0.25)

    ax2 = fig.add_subplot(gs[1])
    rows = sorted(F._load("results/real_35M/table_D.json")["rows"],
                  key=lambda r: r["value"])
    names = {"P3 head": "Conditioned head", "zero": "Predict no change",
             "train_mean": "Train mean", "ridge_esm2": "ESM2 ridge",
             "knn_esm2": "ESM2 k-NN"}
    yy = np.arange(len(rows))
    for i, r in enumerate(rows):
        ext = r["label"] in ("GEARS", "CPA")
        ax2.barh(i, r["value"], height=0.55, color=st.C_RIDGE if ext else st.C_KNN,
                 hatch="////" if ext else None, edgecolor="black", linewidth=0.5,
                 zorder=3)
        if r.get("lo") is not None:
            ax2.errorbar(r["value"], i, xerr=[[r["value"] - r["lo"]],
                                              [r["hi"] - r["value"]]],
                         fmt="none", ecolor="black", elinewidth=0.6, zorder=4)
        ax2.text(0.052, i, f"{r['value']:+.4f}" + (f"  (n={r['n']})"
                                                     if r["n"] > 1 else ""),
                 va="center", fontsize=7)
    ax2.set_yticks(yy)
    ax2.set_yticklabels([names.get(r["label"], r["label"]) for r in rows],
                        fontsize=7)
    ax2.axvline(0, color="black", lw=0.6)
    ax2.set_xlim(-0.06, 0.10)
    ax2.set_xlabel("Held-out $r$, smaller screen only")
    st.subcaption(ax2, "b", "Published models on the smaller screen", y=-0.25)
    return st.save_ieee(fig, "ieee_ed_fig4_budget_external", outdir, w)


def main(outdir="figures"):
    st.apply_ieee_style()
    return [f(outdir) for f in (ed1, ed2, ed3, ed4)]


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=1, default=str))
