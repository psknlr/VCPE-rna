#!/usr/bin/env python3
"""IEEE-style architecture figure for the conditioned per-gene deviation head.

Nothing architectural is typed into this file. Layer shapes, dropout, the
activation, the dataset embedding and the parameter count are read from a
`DeviationModel` instantiated with the configuration the reported runs used
(recorded in their provenance); the gate constants come from `model_dev`; the
optimiser, scheduler, gradient clip, loss and selection rule are matched in the
training source by regular expression, and the build will not proceed if any of
them has changed. A figure that drifts from the code is the failure this
repository documents in docs/ERRATA.md (E11), so drift is made a build error.

Run from the repository root:  python3 tools/figures/ieee_architecture.py
"""
from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src" / "maprna_p3"))

import matplotlib.patches as mpatches                           # noqa: E402
import ieee_diagram as S                                          # noqa: E402

MAIN_RUN = "results/real_150M_graph_long/A_seeds"


def _find(pattern, rel, cast=str):
    src = (ROOT / rel).read_text()
    m = re.search(pattern, src)
    if not m:
        raise SystemExit(f"architecture fact not found in {rel}: /{pattern}/ -- "
                         f"the code changed; update the figure, do not guess")
    return cast(m.group(1)) if m.groups() else True


def architecture_facts() -> dict:
    """Every number and name the figure prints, with where it came from."""
    import numpy as np
    import torch
    import model_dev
    import baselines

    rep = json.loads((ROOT / MAIN_RUN / "seed0" / "final_report.json").read_text())
    args = rep["provenance"]["args"]
    summ = json.loads((ROOT / MAIN_RUN / "seed_summary.json").read_text())
    proto = summ["protocol"]
    for k in ("n_hvg", "d_model", "inner_val_frac", "knn_k", "epochs"):
        assert proto[k] == args[k], f"seed 0 and the summary disagree on {k}"
    sweep = json.loads((ROOT / "results/feature_sweep_innerval.json").read_text())
    cand = next(v for v in sweep["variants"] if v["name"] == sweep["candidate"])
    assert cand["mode"] == "concat" and cand["norm"], cand
    d_table = int(cand["dim"])
    d_own = d_table // 2                       # concat: [own || neighbourhood]
    blend_dims = {int(v["dim"]) for v in sweep["variants"] if v["mode"] == "blend"}
    assert blend_dims == {d_own}, "blend variants should keep the own dimension"

    n_ds = len(args["data_dirs"])
    model = model_dev.DeviationModel(torch.zeros(4, d_table),
                                     np.arange(args["n_hvg"]), n_ds=n_ds,
                                     d_model=args["d_model"])
    lin = [m for m in model.mlp if isinstance(m, torch.nn.Linear)]
    acts = {type(m).__name__ for m in model.mlp
            if not isinstance(m, (torch.nn.Linear, torch.nn.Dropout))}
    drops = {m.p for m in model.mlp if isinstance(m, torch.nn.Dropout)}
    assert len(acts) == 1 and len(drops) == 1, (acts, drops)
    assert model.proj_r is None, "the main configuration has no RNA branch"
    d = args["d_model"]
    n_scalar = lin[0].in_features - 4 * d - model.ds_emb.embedding_dim
    assert n_scalar == 3, "expected control mean, is-neighbour, is-target"

    esm_name = _find(r'"--model", default="([^"]+)"', "tools/build_esm2_gene_table.py")
    mm = re.search(r"esm2_t(\d+)_(\d+[MB])_", esm_name)
    knn_sig = inspect.signature(baselines.baseline_knn_esm2).parameters

    plain = model_dev.DeviationModel(torch.zeros(4, d_own), np.arange(args["n_hvg"]),
                                     n_ds=n_ds, d_model=args["d_model"])

    # ERRATA E17: the figure states that the inner split is held out of training,
    # so check that this is what the training code does by default.
    import train_p3
    iv = np.zeros(10, dtype=bool); iv[[2, 7]] = True
    held = train_p3.training_indices(dict(train_items=list(range(10)),
                                          is_inner_val=iv))
    inner_val_held_out = not set(held) & {2, 7}
    assert inner_val_held_out, "training code trains on the selection slice"
    committed_legacy = rep.get("inner_val_in_training") is None

    return dict(
        source_run=MAIN_RUN,
        inner_val_held_out=inner_val_held_out,
        reported_runs_predate_e17=committed_legacy,
        n_params_plain=int(sum(p.numel() for p in plain.parameters())),
        n_mlp_linear=len(lin),
        n_hvg=args["n_hvg"], n_ds=n_ds, d_model=d, d_table=d_table, d_own=d_own,
        hidden=[(m.in_features, m.out_features) for m in lin],
        activation=acts.pop(), dropout=drops.pop(),
        ds_emb=(model.ds_emb.num_embeddings, model.ds_emb.embedding_dim),
        proj=(model.proj_p.in_features, model.proj_p.out_features),
        concat_dim=lin[0].in_features,
        n_params=int(sum(p.numel() for p in model.parameters())),
        gate_tau=model_dev.GATE_TAU, gate_slope=model_dev.GATE_SLOPE,
        batch=args["batch_size"], lr=args["lr"], wd=args["weight_decay"],
        epochs=args["epochs"], inner_val_frac=args["inner_val_frac"],
        knn_k=args["knn_k"], knn_weighted=knn_sig["weighted"].default,
        use_mask=args["use_mask"], split_by=args["split_by"],
        hvg_from=args["hvg_from"],
        optimiser=_find(r"torch\.optim\.(\w+)\(", "src/maprna_p3/train_p3.py"),
        scheduler=_find(r"lr_scheduler\.(\w+)\(", "src/maprna_p3/train_p3.py"),
        clip=_find(r"clip_grad_norm_\(model\.parameters\(\), ([0-9.]+)\)",
                   "src/maprna_p3/train_p3.py", float),
        masked_mse=_find(r"loss = \(\(\(pred - dev_tr_t\[idx\]\) \*\* 2\) \* m\)"
                         r"\.sum\(\) / m\.sum\(\)", "src/maprna_p3/train_p3.py"),
        selection=_find(r'"selection": "(inner_val_pearson_dev)"',
                        "src/maprna_p3/train_p3.py"),
        ridge_alpha=_find(r'kw\.get\("ridge_alpha", ([0-9.]+)\)',
                          "src/maprna_p3/baselines.py", float),
        string_min_score=_find(r'"--min_score", type=int, default=(\d+)',
                               "tools/build_string_neighbors.py", int),
        string_top_k=int(cand["top_k"]),
        esm_model=esm_name, esm_layers=int(mm.group(1)), esm_size=mm.group(2),
    )


def _fmt_int(n):
    return f"{n:,}"


def _sci(x):
    e = int(round(__import__("math").log10(x)))
    assert abs(x - 10 ** e) < 1e-12, x
    return rf"10^{{{e}}}"


PROFILES = {
    # IEEE Transactions: two-column 7.16 in, Times-metric serif at 9 pt, "(a)
    # Title" beneath each sub-figure.
    "ieee": dict(family="serif", fs=9.0, scale=1.0, stem="fig_architecture"),
    # The same layout re-typeset for the Nature portfolio: double column 183 mm,
    # Arial-metric sans at 7 pt (the top of the 5-7 pt band, so that sub- and
    # superscripts, set at 0.7 of it, stay near 5 pt), bold lowercase panel
    # letters. Without it, dropping the IEEE figure into an npj manuscript would
    # put 9 pt Times beside 5-7 pt Arial, against both guides.
    "nature": dict(family="sans-serif", fs=7.0, scale=183.0 / (7.16 * 25.4),
                   stem="fig_architecture_nature"),
}


def draw(F, outdir, profile="ieee"):
    P = PROFILES[profile]
    if profile == "ieee":
        style = S.apply_ieee_style(family="serif", size=P["fs"])
    else:
        import nature_style as ns
        style = ns.apply_nature_style()
    S.set_scale(fs=P["fs"], lw=P["scale"])
    H = 5.02
    fig, ax, W = S.ieee_canvas("double", H, scale=P["scale"])
    fs = P["fs"]
    OFF = (0, (0.9, 1.5))

    # ------------------------------------------------------------ key
    # Nature puts the panel letter top-left, so the key starts to its right
    S.legend_row(ax, 0.10 if profile == "ieee" else 0.30, H - 0.15,
                 ["data", "train", "fixed", "off"])

    # ============================================================ (a) head
    # Columns, left to right, as (centre, width); gaps are what is left over.
    xin, win = 0.67, 1.22             # inputs
    xpr, wpr = 1.93, 0.94             # projections
    xod = 2.555                       # Hadamard product node
    xcc, wcc = 3.11, 0.82             # concatenated feature cells
    xml, wml = 4.73, 1.06             # per-gene MLP
    xgt, wgt = 5.995, 0.97            # self-response gate
    xo, wo = 6.86, 0.48               # output
    top, gap = H - 0.42, 0.075
    h2, h1 = 0.36, 0.215

    rows = {}
    y = top - h2 / 2
    rows["t"] = y
    block(ax, xin, y, win, h2,
          f"Target gene $t$\n$\\mathbf{{x}}_t\\in\\mathbb{{R}}^{{{F['d_table']}}}$")
    y -= h2 / 2 + gap + h2 / 2
    rows["i"] = y
    block(ax, xin, y, win, h2,
          f"Panel gene $i$, $N$ = {F['n_hvg']}\n"
          f"$\\mathbf{{x}}_i\\in\\mathbb{{R}}^{{{F['d_table']}}}$")
    y -= h2 / 2 + gap + h1 / 2
    rows["r"] = y
    block(ax, xin, y, win, h1, "RNA-sequence input", kind="off")
    y -= h1 / 2 + gap + h2 / 2
    rows["c"] = y
    block(ax, xin, y, win, h2,
          "$c_i$: control mean ($z$)\n$\\tau_i$: $i$ is the target")
    y -= h2 / 2 + gap + h1 / 2
    rows["a"] = y
    block(ax, xin, y, win, h1, "$a_i$: STRING partner", kind="off")
    y -= h1 / 2 + gap + h1 / 2
    rows["d"] = y
    block(ax, xin, y, win, h1, f"Screen $d$ (of {F['n_ds']})")
    y_bottom_a = y - h1 / 2

    pin, pout = F["proj"]
    block(ax, xpr, rows["t"], wpr, h2, f"Linear\n{pin}$\\to${pout}", kind="train")
    block(ax, xpr, rows["i"], wpr, h2, f"Linear, shared\n{pin}$\\to${pout}",
          kind="train")
    nd, ed = F["ds_emb"]
    block(ax, xpr, rows["d"], wpr, h1, f"Embedding {nd}$\\to${ed}", kind="train")

    seg = [("P", rf"$\mathbf{{P}}$ ({pout})", rows["t"]),
           ("GP", rf"$\mathbf{{G}}_i\odot\mathbf{{P}}$ ({pout})",
            (rows["t"] + rows["i"]) / 2),
           ("G", rf"$\mathbf{{G}}_i$ ({pout})", rows["i"]),
           ("R", rf"$\mathbf{{R}}$ ({pout}, $\mathbf{{0}}$)", rows["r"]),
           ("c", r"$c_i,\ \tau_i$ (2)", rows["c"]),
           ("a", r"$a_i$ (1)", rows["a"]),
           ("s", rf"$\mathbf{{s}}_d$ ({ed})", rows["d"])]
    cell_h = 0.215
    for key, lab, yy in seg:
        block(ax, xcc, yy, wcc, cell_h, lab,
              kind="off" if key in ("R", "a") else "data")
    xb = xcc + wcc / 2 + 0.06
    ytop_c, ybot_c = rows["t"] + cell_h / 2, rows["d"] - cell_h / 2
    ax.plot([xb, xb + 0.05, xb + 0.05, xb], [ytop_c, ytop_c, ybot_c, ybot_c],
            color=S.EDGE, lw=S.BOX_LW * P["scale"])

    def harrow(x0, x1, yy, ls="solid"):
        S.arrow(ax, (x0, yy), (x1, yy), ls=ls)
    for k in ("t", "i", "d"):
        harrow(xin + win / 2, xpr - wpr / 2, rows[k])
        harrow(xpr + wpr / 2, xcc - wcc / 2, rows[k])
    for k in ("r", "c", "a"):
        harrow(xin + win / 2, xcc - wcc / 2, rows[k],
               ls=OFF if k in ("r", "a") else "solid")
    yod = (rows["t"] + rows["i"]) / 2
    S.op_node(ax, xod, yod, r"$\odot$")
    ax.plot([xod, xod], [rows["t"], yod + 0.085], color=S.EDGE,
            lw=S.ARROW_LW * P["scale"])
    ax.plot([xod, xod], [rows["i"], yod - 0.085], color=S.EDGE,
            lw=S.ARROW_LW * P["scale"])
    for yy in (rows["t"], rows["i"]):
        ax.add_patch(mpatches.Circle((xod, yy), 0.022, color=S.EDGE, zorder=3))
    S.arrow(ax, (xod + 0.085, yod), (xcc - wcc / 2, yod))
    xlab = (xpr + wpr / 2 + xod) / 2
    ax.text(xlab, rows["t"] + 0.035, r"$\mathbf{P}$", ha="center",
            va="bottom", fontsize=fs)
    ax.text(xlab, rows["i"] - 0.035, r"$\mathbf{G}_i$", ha="center",
            va="top", fontsize=fs)

    hid = F["hidden"]
    act = F["activation"]
    layers = [(f"Linear {hid[0][0]}$\\to${hid[0][1]}", "train"),
              (f"{act}, dropout {F['dropout']:g}", "fixed"),
              (f"Linear {hid[1][0]}$\\to${hid[1][1]}", "train"),
              (f"{act}, dropout {F['dropout']:g}", "fixed"),
              (f"Linear {hid[2][0]}$\\to${hid[2][1]}", "train")]
    lh, lg = 0.215, 0.105
    span = len(layers) * lh + (len(layers) - 1) * lg
    ymid = (ytop_c + ybot_c) / 2
    y0 = ymid + span / 2 - lh / 2
    ys = []
    for j, (lab, kind) in enumerate(layers):
        yy = y0 - j * (lh + lg)
        ys.append(yy)
        block(ax, xml, yy, wml, lh, lab, kind=kind)
        if j:
            S.arrow(ax, (xml, ys[j - 1] - lh / 2), (xml, yy + lh / 2))
    fx0, fx1 = xml - wml / 2 - 0.06, xml + wml / 2 + 0.06
    S.frame(ax, fx0, ys[-1] - lh / 2 - 0.06, fx1, ys[0] + lh / 2 + 0.06)
    ax.text(xml, ys[0] + lh / 2 + 0.10, "Shared over all $N$ genes",
            ha="center", va="bottom", fontsize=fs)
    xv = fx0 - 0.07
    S.polyline_arrow(ax, [(xb + 0.05, ymid), (xv, ymid), (xv, ys[0]),
                          (xml - wml / 2, ys[0])])
    xm = (xb + 0.05 + xv) / 2
    ax.text(xm, ymid + 0.04, r"$\mathbf{h}_{t,i}$", ha="center", va="bottom",
            fontsize=fs)
    ax.text(xm, ymid - 0.04, f"{F['concat_dim']}", ha="center", va="top",
            fontsize=fs)

    yg = ys[-1]
    hg = 0.52
    block(ax, xgt, yg, wgt, hg,
          "Gate, row $i=t$:\n"
          f"$\\times\\,\\sigma({F['gate_slope']:g}(e_t-{F['gate_tau']:g}))$\n"
          "(not trained)", kind="fixed")
    S.arrow(ax, (fx1, yg), (xgt - wgt / 2, yg))
    ax.text((fx1 + xgt - wgt / 2) / 2, yg + 0.035, r"$\hat{\delta}_{t,i}$",
            ha="center", va="bottom", fontsize=fs)
    ye = yg + hg / 2 + 0.30 + h2 / 2
    block(ax, xgt, ye, wgt, h2, "$e_t$: control\nexpression of $t$")
    S.arrow(ax, (xgt, ye - h2 / 2), (xgt, yg + hg / 2))
    block(ax, xo, yg, wo, h2,
          "$\\hat{\\boldsymbol{\\delta}}_t$\n$\\in\\mathbb{R}^{N}$")
    S.arrow(ax, (xgt + wgt / 2, yg), (xo - wo / 2, yg))

    if profile == "ieee":
        S.sublabel(ax, W / 2, y_bottom_a - 0.07,
                   "(a) Conditioned per-gene deviation head "
                   f"({_fmt_int(F['n_params'])} trained parameters)")

    # ============================================================ (b) table
    yb_top = y_bottom_a - 0.50
    r1 = yb_top - h2 / 2
    r2 = r1 - h2 - 0.30
    c1, w1 = 0.49, 0.86
    c2, w2 = 1.71, 1.22
    c3, w3 = 2.83, 0.70
    block(ax, c1, r1, w1, h2, "Protein of $g$\n(UniProt)")
    block(ax, c2, r1, w2, h2,
          f"ESM2-{F['esm_size']}, frozen;\nmean over residues", kind="fixed")
    block(ax, c3, r1, w3, h2, f"$\\mathbf{{e}}_g\\in\\mathbb{{R}}^{{{F['d_own']}}}$")
    block(ax, c1, r2, w1, h2,
          f"STRING v12.0,\nscore $\\geq$ {F['string_min_score']}")
    block(ax, c2, r2, w2, h2,
          f"Norm-matched mean\nof top-{F['string_top_k']} partners",
          kind="fixed")
    block(ax, c3, r2, w3, h2, f"$\\mathbf{{n}}_g\\in\\mathbb{{R}}^{{{F['d_own']}}}$")
    for rr in (r1, r2):
        S.arrow(ax, (c1 + w1 / 2, rr), (c2 - w2 / 2, rr))
        S.arrow(ax, (c2 + w2 / 2, rr), (c3 - w3 / 2, rr))
    S.arrow(ax, (c2, r1 - h2 / 2), (c2, r2 + h2 / 2))
    ax.text(c2 + 0.05, (r1 + r2) / 2, r"$\mathbf{e}_u,\ u\in\mathcal{N}(g)$",
            ha="left", va="center", fontsize=fs)
    xcat = c3 + w3 / 2 + 0.22
    ycat = (r1 + r2) / 2
    S.op_node(ax, xcat, ycat, r"$\Vert$")
    S.polyline_arrow(ax, [(c3 + w3 / 2, r1), (xcat, r1), (xcat, ycat + 0.085)])
    S.polyline_arrow(ax, [(c3 + w3 / 2, r2), (xcat, r2), (xcat, ycat - 0.085)])
    r3 = r2 - h2 - 0.20
    bx0, bx1 = c1 - w1 / 2, xcat + 0.10
    block(ax, (bx0 + bx1) / 2, r3, bx1 - bx0, h2,
          f"Gene table row $\\mathbf{{x}}_g\\in\\mathbb{{R}}^{{{F['d_table']}}}$, "
          "one per gene;\nthe head, ridge and $k$-NN read the same table")
    xr = xcat + 0.20
    S.polyline_arrow(ax, [(xcat + 0.085, ycat), (xr, ycat), (xr, r3), (bx1, r3)])
    xb_mid = (bx0 + xr) / 2
    y_bottom_b = r3 - h2 / 2

    # ============================================================ (c) protocol
    x0c, x1c = xr + 0.30, W - 0.06
    gx = 0.24
    wc = (x1c - x0c - gx) / 2
    cA, cB = x0c + wc / 2, x1c - wc / 2
    hb, gy = 0.50, 0.17
    q1 = yb_top - hb / 2
    q2 = q1 - hb - gy
    hc = 0.36
    q3 = q2 - hb / 2 - gy - hc / 2
    block(ax, cA, q1, wc, hb,
          "Residual target\n"
          r"$\boldsymbol{\delta}_t=\mathbf{f}_t-\bar{\mathbf{f}}$"
          "\n" r"$\bar{\mathbf{f}}$: training-split mean")
    block(ax, cB, q1, wc, hb,
          "MSE on measured genes;\n"
          f"{F['optimiser']}, lr ${_sci(F['lr'])}$, wd ${_sci(F['wd'])}$;\n"
          f"cosine, clip {F['clip']:g}, batch {F['batch']}", kind="train")
    block(ax, cB, q2, wc, hb,
          f"Epoch chosen on {F['inner_val_frac'] * 100:.0f}% of\n"
          "training target genes,\nheld out of training", kind="fixed")
    block(ax, cA, q2, wc, hb,
          "Held-out target genes,\nscored once: mean\nwithin-perturbation $r$")
    block(ax, (x0c + x1c) / 2, q3, x1c - x0c, hc,
          "Controls on the same split: ridge ($\\alpha$ = "
          f"{F['ridge_alpha']:g}), $k$-NN ($k$ = {F['knn_k']},\n"
          "cosine-weighted), train mean, predict no change", kind="fixed")
    S.arrow(ax, (cA + wc / 2, q1), (cB - wc / 2, q1))
    S.arrow(ax, (cB, q1 - hb / 2), (cB, q2 + hb / 2))
    S.arrow(ax, (cB - wc / 2, q2), (cA + wc / 2, q2))
    S.arrow(ax, (cA, q3 + hc / 2), (cA, q2 - hb / 2))
    y_bottom_c = q3 - hc / 2

    ylab = min(y_bottom_b, y_bottom_c) - 0.07
    if profile == "ieee":
        S.sublabel(ax, xb_mid, ylab, "(b) Gene representation")
        S.sublabel(ax, (x0c + x1c) / 2, ylab,
                   "(c) Objective, model selection and controls")
        res = S.save_ieee(fig, P["stem"], outdir)
    else:
        # Nature: 8 pt bold lowercase letters, top-left of each panel
        for letter, (lx, ly) in (("a", (0.0, H - 0.15)),
                                 ("b", (0.0, yb_top + 0.17)),
                                 ("c", (x0c - 0.12, yb_top + 0.17))):
            ax.text(lx, ly, letter, fontsize=8, fontweight="bold", ha="left",
                    va="center", color=S.TEXT)
        res = ns.save_nature(fig, P["stem"], outdir=str(outdir), width_mm=183.0)
        over = S.check_overflow(fig)
        if over:
            res["warnings"].append(f"text overflows its box ({len(over)}): "
                                   f"{over[:6]}")
    res["font"] = style
    res["profile"] = profile
    return res


def block(ax, *a, **k):
    return S.block(ax, *a, **k)


def captions(F):
    """IEEE caption (LaTeX) and Nature legend, every number from `F`."""
    n, d, h = F["n_hvg"], F["d_model"], F["concat_dim"]
    common = dict(
        head=(f"the rows of the target ($\\mathbf{{x}}_t$) and of each of the "
              f"$N={n}$ panel genes ($\\mathbf{{x}}_i$) are projected to {d} "
              f"dimensions by trained linear maps, the panel-gene map shared "
              f"across genes"),
    )
    ieee = (
        "\\begin{figure*}[!t]\n\\centering\n"
        "\\includegraphics[width=7.16in]{fig_architecture}\n"
        "\\caption{Architecture and training protocol of the conditioned "
        "per-gene deviation head. (a) For a perturbation of target gene $t$, "
        f"{common['head']}. Their element-wise product ($\\odot$) carries the "
        "target--gene interaction. Concatenated with the control expression "
        "$c_i$, the target indicator $\\tau_i$ and a learned screen embedding "
        f"$\\mathbf{{s}}_d$, they form the {h}-dimensional feature "
        f"$\\mathbf{{h}}_{{t,i}}$, which a {F['n_mlp_linear']}-layer perceptron "
        "with weights shared across genes maps to the residual response "
        "$\\hat{\\delta}_{t,i}$. A fixed sigmoid gate scales the target's own "
        "row by its control expression $e_t$. Dotted elements, the RNA-sequence "
        "branch and the STRING-partner indicator $a_i$, are inactive in the main "
        "configuration and enter as zeros. "
        f"(b) Each gene's ESM2-{F['esm_size']} embedding $\\mathbf{{e}}_g$, the "
        "mean over residues, is concatenated ($\\Vert$) with the norm-matched "
        f"mean embedding of its top {F['string_top_k']} STRING v12.0 partners "
        f"(combined score $\\geq {F['string_min_score']}$; zero for genes "
        "without partners). The head and the ridge and $k$-NN controls read the "
        "same table. "
        "(c) The head predicts the residual left after subtracting the mean "
        "response of the training split, is trained by mean squared error over "
        "measured genes, and its epoch is chosen on an inner validation split of "
        "training target genes; held-out target genes are scored once, beside "
        "controls evaluated on the identical split. "
        + ("The inner split is held out of training as drawn; the runs reported "
           "in Table~I predate that correction and also trained on it "
           "(ERRATA E17). " if F["reported_runs_predate_e17"] else "")
        + "The head has "
        f"{F['n_params']:,} trained parameters with the {F['d_table']}-dimensional "
        f"table ({F['n_params_plain']:,} with the {F['d_own']}-dimensional "
        "ESM2-only table).}\n\\label{fig:architecture}\n\\end{figure*}\n")
    nature = (
        "Fig. X | Architecture and training protocol of the conditioned per-gene "
        "deviation head.\n"
        f"a, For a perturbation of target gene t, the gene-table rows of the "
        f"target (x_t) and of each of the N = {n} panel genes (x_i) are projected "
        f"to {d} dimensions by trained linear maps, the panel-gene map shared "
        "across genes; their element-wise product carries the target-gene "
        "interaction. Concatenated with the control expression c_i, the target "
        f"indicator tau_i and a learned screen embedding, they form a {h}-"
        f"dimensional feature that a {F['n_mlp_linear']}-layer perceptron with "
        "weights shared across genes maps to the residual response. A fixed "
        "sigmoid gate scales the target's own row by its control expression e_t. "
        "Dotted elements are inactive in the main configuration and enter as "
        f"zeros. b, Each gene's ESM2-{F['esm_size']} embedding (mean over "
        "residues) is concatenated with the norm-matched mean embedding of its "
        f"top {F['string_top_k']} STRING v12.0 partners (combined score >= "
        f"{F['string_min_score']}; zero for genes without partners); the head "
        "and the ridge and k-NN controls read the same table. c, The head "
        "predicts the residual after subtracting the training-split mean "
        "response, is trained by mean squared error over measured genes, and "
        "its epoch is chosen on an inner validation split of training target "
        "genes; held-out target genes are scored once, beside controls on the "
        "identical split. "
        + ("The inner split is held out of training as drawn; the runs reported "
           "in Table 1 predate that correction and also trained on it (ERRATA "
           "E17). " if F["reported_runs_predate_e17"] else "")
        + f"{F['n_params']:,} trained parameters with the "
        f"{F['d_table']}-dimensional table ({F['n_params_plain']:,} with the "
        f"{F['d_own']}-dimensional ESM2-only table).\n")
    return ieee, nature


def main(outdir=None):
    F = architecture_facts()
    out = Path(outdir) if outdir else ROOT / "figures" / "ieee"
    res = [draw(F, out, "ieee"), draw(F, out, "nature")]
    (out / "architecture_facts.json").write_text(json.dumps(F, indent=2) + "\n")
    ieee, nature = captions(F)
    (out / "caption_ieee.tex").write_text(ieee)
    (out / "legend_nature.md").write_text(nature)
    return res, F


if __name__ == "__main__":
    rs, _ = main()
    print(json.dumps(rs, indent=2, default=str))
    raise SystemExit(1 if any(r["warnings"] for r in rs) else 0)
