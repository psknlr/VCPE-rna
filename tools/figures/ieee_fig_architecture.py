"""IEEE Fig. | Architecture of the conditioned per-gene deviation head.

Drawn from src/maprna_p3/model_dev.py (DeviationModel) and the training loop in
src/maprna_p3/train_p3.py, not from a description of them. Every dimension in the
diagram is computed below from the constructor arguments the reported runs used
(seed_summary.json -> protocol), so the drawing cannot drift from the code:

    d_model = 256, hidden = 512, DS_EMB = 8, n_hvg = 500, n_ds = 2,
    D = 640 (ESM2-150M) or 1,280 (ESM2-150M + STRING neighbourhood).

The RNA channel is drawn dashed because no RNA-encoder checkpoint was passed in
any reported run: its 256-d slot is filled with zeros (model_dev.py, forward) and
the C_ablation "rna_off" row is bit-identical to the intact model.
"""
import sys
from pathlib import Path

import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata as F                                             # noqa: E402
import ieee_style as st                                         # noqa: E402

# ---- constructor constants, mirrored from model_dev.py / train_p3.py --------
D_MODEL, HIDDEN, DS_EMB, N_DS = 256, 512, 8, 2
GATE_TAU, GATE_SLOPE = 0.7, 6.0
FEAT = 4 * D_MODEL + 3 + DS_EMB                                # 1035


def n_params(esm_dim, d=D_MODEL, h=HIDDEN, n_ds=N_DS, rna=False):
    lin = lambda i, o: i * o + o                                # noqa: E731
    n = 2 * lin(esm_dim, d)                                     # proj_g, proj_p
    n += lin(d, d) if rna else 0                                # proj_r
    n += n_ds * DS_EMB                                          # ds_emb
    n += lin(FEAT, h) + lin(h, h) + lin(h, 1)                   # mlp
    return n


def box(ax, x, y, w, h, text, fc=st.FILL_OP, ls="-", fs=7, lw=0.6, ec="black"):
    ax.add_patch(mpatches.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                                    linewidth=lw, linestyle=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            zorder=3, linespacing=1.15)


def arrow(ax, p0, p1, ls="-", lw=0.6):
    ax.annotate("", xy=p1, xytext=p0, zorder=1,
                arrowprops=dict(arrowstyle="-|>", lw=lw, linestyle=ls,
                                color="black", shrinkA=0, shrinkB=0,
                                mutation_scale=6))


def line(ax, xs, ys, ls="-", lw=0.6):
    ax.plot(xs, ys, color="black", lw=lw, ls=ls, zorder=1,
            solid_capstyle="butt")


def node(ax, x, y, sym, r=1.9):
    ax.add_patch(mpatches.Circle((x, y), r, facecolor="white",
                                 edgecolor="black", linewidth=0.6, zorder=2))
    ax.text(x, y - 0.1, sym, ha="center", va="center", fontsize=8, zorder=3)


def main(outdir="figures"):
    style = st.apply_ieee_style()
    d = F._summary("results/real_150M/A_seeds/seed_summary.json")
    n_hvg = d["protocol"]["n_hvg"]
    p640, p1280 = n_params(640), n_params(1280)

    fig, w = st.ieee_figure("double", height_in=4.15, constrained=False)

    # ===================== (a) data flow ====================================
    ax = fig.add_axes([0.012, 0.300, 0.976, 0.690])
    ax.set_xlim(0, 140); ax.set_ylim(0, 57); ax.axis("off")

    hdr = dict(fontsize=7.5, ha="center", va="center", style="italic")
    for x, t in ((12, "Inputs (one perturbation)"), (36, "Projection"),
                 (72, "Shared per-gene MLP"), (97, "Output"),
                 (124, "Training objective")):
        ax.text(x, 56.0, t, **hdr)

    # -- inputs ---------------------------------------------------------------
    bw, bh, x0 = 24, 5.4, 0.5
    rows = {"p": 50.5, "G": 43.0, "R": 35.5, "c": 28.0, "nb": 20.5,
            "tgt": 13.0, "ds": 5.5}
    box(ax, x0, rows["p"] - bh / 2, bw, bh,
        "Target gene ESM2\n$\\mathbf{e}_p\\in\\mathbb{R}^{D}$", fc=st.FILL_IN)
    box(ax, x0, rows["G"] - bh / 2, bw, bh,
        f"Panel-gene ESM2 (frozen)\n$\\mathbf{{E}}\\in\\mathbb{{R}}^{{N\\times D}}$, "
        f"$N={n_hvg}$", fc=st.FILL_IN)
    box(ax, x0, rows["R"] - bh / 2, bw, bh,
        "Transcript RNA encoder\n$\\mathbf{h}_p\\in\\mathbb{R}^{256}$ (off)",
        fc="white", ls="--")
    box(ax, x0, rows["c"] - bh / 2, bw, bh,
        "Control expression\n$\\mathbf{c}\\in\\mathbb{R}^{N}$ (z-scored)",
        fc=st.FILL_IN)
    box(ax, x0, rows["nb"] - bh / 2, bw, bh,
        "STRING-neighbour flag\n$\\mathbf{a}^{\\mathrm{nb}}\\in\\{0,1\\}^{N}$",
        fc=st.FILL_IN)
    box(ax, x0, rows["tgt"] - bh / 2, bw, bh,
        "Self (is-target) flag\n$\\mathbf{a}^{\\mathrm{tgt}}\\in\\{0,1\\}^{N}$",
        fc=st.FILL_IN)
    box(ax, x0, rows["ds"] - bh / 2, bw, bh,
        "Screen index\n$k\\in\\{1,%d\\}$" % N_DS, fc=st.FILL_IN)

    # -- projections ----------------------------------------------------------
    px, pw = 29, 14
    box(ax, px, rows["p"] - bh / 2, pw, bh,
        "$W_p$: Linear\n$D\\rightarrow%d$" % D_MODEL, fc=st.FILL_LEARN)
    box(ax, px, rows["G"] - bh / 2, pw, bh,
        "$W_g$: Linear\n$D\\rightarrow%d$" % D_MODEL, fc=st.FILL_LEARN)
    box(ax, px, rows["R"] - bh / 2, pw, bh,
        "$W_r$: Linear\n$256\\rightarrow%d$" % D_MODEL, fc="white", ls="--")
    box(ax, px, rows["ds"] - bh / 2, pw, bh,
        "Embedding\n$%d\\rightarrow%d$" % (N_DS, DS_EMB), fc=st.FILL_LEARN)
    for k in ("p", "G", "ds"):
        arrow(ax, (x0 + bw, rows[k]), (px, rows[k]))
    arrow(ax, (x0 + bw, rows["R"]), (px, rows["R"]), ls="--")

    # Hadamard interaction
    hx, hy = 47.5, (rows["p"] + rows["G"]) / 2
    node(ax, hx, hy, "$\\odot$")
    line(ax, [px + pw, hx - 1.4], [rows["p"], hy + 1.3])
    line(ax, [px + pw, hx - 1.4], [rows["G"], hy - 1.3])

    # -- concatenation bar ----------------------------------------------------
    cx, cw = 52.0, 4.2
    ax.add_patch(mpatches.Rectangle((cx, 2.5), cw, 51.0, facecolor=st.FILL_OP,
                                    edgecolor="black", linewidth=0.6, zorder=2))
    ax.text(cx + cw / 2, 28.0,
            "Broadcast over $N$ genes and concatenate:  "
            "$\\mathbf{x}_i\\in\\mathbb{R}^{%d}$" % FEAT,
            rotation=90, ha="center", va="center", fontsize=7, zorder=3)
    arrow(ax, (px + pw, rows["p"]), (cx, rows["p"]))
    arrow(ax, (px + pw, rows["G"]), (cx, rows["G"]))
    arrow(ax, (hx + 1.9, hy), (cx, hy))
    arrow(ax, (px + pw, rows["R"]), (cx, rows["R"]), ls="--")
    arrow(ax, (px + pw, rows["ds"]), (cx, rows["ds"]))
    for k in ("c", "nb", "tgt"):
        arrow(ax, (x0 + bw, rows[k]), (cx, rows[k]))
    lab = dict(fontsize=6.5, ha="center", va="bottom")
    ax.text(45.5, rows["p"] + 0.4, "$\\mathbf{p}$", **lab)
    ax.text(45.5, rows["G"] - 3.0, "$\\mathbf{g}_i$", **lab)
    ax.text(45.5, rows["R"] + 0.4, "$\\mathbf{r}$ = 0", **lab)
    ax.text(45.5, rows["ds"] + 0.4, "$\\mathbf{u}_k$", **lab)

    # -- shared MLP -----------------------------------------------------------
    mx, mw, mh = 61.5, 21, 4.6
    ys = [47.0, 40.0, 33.0, 26.0, 19.0]
    txt = [f"Linear {FEAT}$\\rightarrow${HIDDEN}", "GELU, Dropout 0.1",
           f"Linear {HIDDEN}$\\rightarrow${HIDDEN}", "GELU, Dropout 0.1",
           f"Linear {HIDDEN}$\\rightarrow$1"]
    fills = [st.FILL_LEARN, st.FILL_OP, st.FILL_LEARN, st.FILL_OP, st.FILL_LEARN]
    ax.add_patch(mpatches.Rectangle((mx - 2, 11.6), mw + 4, 40.4, fill=False,
                                    edgecolor="black", linewidth=0.6,
                                    linestyle=(0, (4, 2)), zorder=1))
    ax.text(mx + mw / 2, 14.2, "weights tied across all $N$ genes",
            fontsize=6.5, ha="center", va="center", style="italic")
    for y, t, f in zip(ys, txt, fills):
        box(ax, mx, y - mh / 2, mw, mh, t, fc=f)
    for y0_, y1_ in zip(ys[:-1], ys[1:]):
        arrow(ax, (mx + mw / 2, y0_ - mh / 2), (mx + mw / 2, y1_ + mh / 2))
    arrow(ax, (cx + cw, ys[0]), (mx, ys[0]))

    # -- output and fixed self-response gate ---------------------------------
    gx, gy = 88.5, ys[-1]
    node(ax, gx, gy, "$\\otimes$")
    arrow(ax, (mx + mw, gy), (gx - 1.9, gy))
    box(ax, 86.0, 28.0, 22.0, 9.5,
        "Self-response gate (fixed)\n"
        "$s=\\sigma(%g\\,(x_p-%g))$\n"
        "on the target's own row only" % (GATE_SLOPE, GATE_TAU),
        fc=st.FILL_FIXED, fs=6.5)
    arrow(ax, (gx, 28.0), (gx, gy + 1.9))
    ox, ow = 92.5, 11.5
    box(ax, ox, gy - bh / 2, ow, bh,
        "$\\hat{\\mathbf{y}}\\in\\mathbb{R}^{N}$", fc=st.FILL_IN)
    arrow(ax, (gx + 1.9, gy), (ox, gy))

    # -- objective ------------------------------------------------------------
    lx, lw_ = 109.0, 30.5
    box(ax, lx, 14.0, lw_, 10.0,
        "Masked MSE\n"
        "$\\mathcal{L}=\\Sigma_i\\, m_i(\\hat{y}_i-y_i)^2\\,/\\,\\Sigma_i\\, m_i$",
        fc=st.FILL_FIXED)
    arrow(ax, (ox + ow, gy), (lx, gy))
    box(ax, lx, 30.5, 14.5, 9.5,
        "Residual target\n$y_i=\\Delta_i-\\bar{\\Delta}_i$\n($\\bar{\\Delta}$: train mean)",
        fc=st.FILL_IN, fs=6.5)
    box(ax, lx + 16.0, 30.5, 14.5, 9.5,
        "Measured mask\n$m_i=1$ iff gene $i$\nmeasured in screen",
        fc=st.FILL_IN, fs=6.5)
    arrow(ax, (lx + 7.25, 30.5), (lx + 7.25, 24.0))
    arrow(ax, (lx + 23.25, 30.5), (lx + 23.25, 24.0))
    ax.text(lx + lw_ / 2, 44.5,
            "AdamW, lr $10^{-3}$, wd $10^{-2}$\ncosine schedule, clip 1.0, "
            "batch 16\nepoch chosen on inner split",
            fontsize=6.5, ha="center", va="center", linespacing=1.25)

    # -- key ------------------------------------------------------------------
    ky = 6.5
    items = [(st.FILL_IN, "-", "input / tensor"),
             (st.FILL_LEARN, "-", "learned layer"),
             (st.FILL_FIXED, "-", "fixed (not learned)"),
             ("white", "--", "inactive in reported runs")]
    for (fc, ls, t), kx in zip(items, (61.5, 76.0, 92.0, 113.0)):
        ax.add_patch(mpatches.Rectangle((kx, ky - 1.2), 3.2, 2.4, facecolor=fc,
                                        edgecolor="black", linewidth=0.6,
                                        linestyle=ls))
        ax.text(kx + 4.2, ky, t, fontsize=6.5, va="center")
    ax.text(61.5, 1.8,
            f"Trainable parameters: {p640:,} ($D$ = 640, ESM2-150M); "
            f"{p1280:,} ($D$ = 1,280, + STRING neighbourhood)",
            fontsize=6.5, va="center")
    ax.text(70, -1.6, "(a)", fontsize=8, ha="center", va="center")

    # ===================== (b) per-gene feature vector ======================
    axb = fig.add_axes([0.012, 0.035, 0.976, 0.215])
    axb.set_xlim(0, 140); axb.set_ylim(0, 15); axb.axis("off")
    axb.text(70, 13.4,
             "$\\hat{y}_i=f_\\theta(\\,[\\,W_g\\mathbf{e}_i,\\; W_p\\mathbf{e}_p,\\;"
             "W_g\\mathbf{e}_i\\odot W_p\\mathbf{e}_p,\\; W_r\\mathbf{h}_p,\\; c_i,\\;"
             "a^{\\mathrm{nb}}_i,\\; a^{\\mathrm{tgt}}_i,\\; \\mathbf{u}_k\\,])"
             "\\cdot(1-a^{\\mathrm{tgt}}_i(1-s))$",
             fontsize=8, ha="center", va="center")
    segs = [("$\\mathbf{g}_i$", D_MODEL, st.FILL_LEARN, "-"),
            ("$\\mathbf{p}$", D_MODEL, st.FILL_LEARN, "-"),
            ("$\\mathbf{g}_i\\odot\\mathbf{p}$", D_MODEL, st.FILL_LEARN, "-"),
            ("$\\mathbf{r}$ (zeros)", D_MODEL, "white", "--"),
            ("$c_i$", 1, st.FILL_IN, "-"),
            ("$a^{\\mathrm{nb}}_i$", 1, st.FILL_IN, "-"),
            ("$a^{\\mathrm{tgt}}_i$", 1, st.FILL_IN, "-"),
            ("$\\mathbf{u}_k$", DS_EMB, st.FILL_LEARN, "-")]
    assert sum(s[1] for s in segs) == FEAT
    # widths: the four 256-d blocks to scale; the 11 scalar dims magnified
    widths = [24 if n == D_MODEL else (6.5 if n == 1 else 8.5) for _, n, _, _ in segs]
    x = 70 - sum(widths) / 2
    for (t, n, fc, ls), wd in zip(segs, widths):
        axb.add_patch(mpatches.Rectangle((x, 4.6), wd, 4.6, facecolor=fc,
                                         edgecolor="black", linewidth=0.6,
                                         linestyle=ls))
        axb.text(x + wd / 2, 6.9, t, fontsize=7, ha="center", va="center")
        axb.text(x + wd / 2, 3.4, f"{n}", fontsize=6.5, ha="center", va="center")
        x += wd
    x_end = x
    x_small = x_end - sum(widths[4:])
    line(axb, [x_small, x_end], [10.0, 10.0])
    axb.text((x_small + x_end) / 2, 10.4, "not to scale", fontsize=6.5,
             ha="center", va="bottom", style="italic")
    axb.text(x_end + 1.5, 6.9, f"= {FEAT}", fontsize=7, va="center")
    axb.text(70, 0.6, "(b)", fontsize=8, ha="center", va="center")

    res = st.save_ieee(fig, "ieee_fig_architecture", outdir=outdir, width_in=w)
    res["font"] = style
    res["params"] = {"D=640": p640, "D=1280": p1280}
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2, default=str))
