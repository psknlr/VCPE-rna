"""IEEE Transactions graphics specification, implemented for matplotlib.

What IEEE states (IEEE Author Center, "Create Graphics for Your Article",
pages "Resolution and Size" and "File Formatting", checked 2026-10-01):

* one column 3.5 in (88.9 mm, 21 pc); two columns 7.16 in (182 mm, 43 pc);
  graphics no larger than 7.16 x 8.8 in (182 x 220 mm); avoid less than one column;
* typefaces Helvetica, Times New Roman, Arial, Cambria or Symbol; type
  "approximately 9-10 point when viewed at full size", consistent across all
  graphics and tables;
* PS, EPS, PDF, PNG or TIFF; in EPS/PS/PDF the fonts are embedded or outlined;
* raster colour and greyscale above 300 dpi, black-and-white line art above
  600 dpi;
* do not rely on colour alone: pair colour with shape (solid versus dashed
  lines, different fills) and contrast elements in brightness, not hue only.

What this module adds as house rules, stated as such because IEEE does not
specify them: strokes of at least 0.5 pt, so hairlines survive print; fills that
stay distinguishable once printed in greyscale (CIELAB lightness at least 8
apart for any two block kinds that share a border style); black text on every
fill with a WCAG contrast of at least 4.5:1; and no text overflowing the box it
labels. The caption ("Fig. 1. ...") belongs to the LaTeX source, not to the
graphic; sub-figure labels "(a)", "(b)" sit centred beneath each sub-figure.

Font note: Times New Roman is not installed in this container. Liberation Serif
is metrically identical to it (same advance widths), so the layout is unchanged
when production substitutes Times New Roman; mathematics is set in STIX, a
Times-compatible face. Both substitutions are recorded in the provenance rather
than passed off as the named fonts. Text is embedded as Type 42, never outlined.
"""
from __future__ import annotations

import struct
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

WIDTHS_IN = {"single": 3.5, "double": 7.16}
MAX_W_IN, MAX_H_IN = 7.16, 8.8
TEXT_PT = (9.0, 10.0)
MIN_STROKE_PT = 0.5                      # house rule (IEEE states none)
MIN_DL = 8.0                             # house rule: greyscale separation
MIN_CONTRAST = 4.5                       # house rule: WCAG AA for body text
DPI = 600

SERIF_STACK = ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman",
               "DejaVu Serif"]
SANS_STACK = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]

# Block kinds for architecture diagrams. Each kind differs from every other in
# BOTH fill lightness and border style where it can, so the figure reads the same
# in greyscale print (IEEE: "use both color and shape to convey the same
# meaning"). The blue is a tint of Okabe-Ito blue, the grey is neutral.
KINDS = {
    "data":  dict(fc="#FFFFFF", ls="solid", label="Input or tensor"),
    "train": dict(fc="#9CC8E2", ls="solid", label="Trained layer"),
    "fixed": dict(fc="#E4E4E4", ls=(0, (3.2, 1.6)),
                  label="Fixed, no trained parameters"),
    "off":   dict(fc="#FFFFFF", ls=(0, (0.9, 1.5)),
                  label="Inactive in the main configuration"),
}
EDGE = "#000000"
TEXT = "#000000"
BOX_LW = 0.6
ARROW_LW = 0.7

# Type size and stroke scale for the primitives below. The IEEE build uses the
# defaults; a re-typeset variant for another venue (for example the Nature
# portfolio's 5-7 pt band) sets them once with `set_scale` rather than threading
# a size through every call.
SCALE = {"fs": 9.0, "lw": 1.0}


def set_scale(fs=9.0, lw=1.0):
    SCALE.update(fs=float(fs), lw=float(lw))


def _fs(size):
    return SCALE["fs"] if size is None else size


def resolved_font(stack) -> str:
    from matplotlib import font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    for n in stack:
        if n in have:
            return n
    return stack[-1]


def apply_ieee_style(family="serif", size=9.0) -> dict:
    stack = SERIF_STACK if family == "serif" else SANS_STACK
    f = resolved_font(stack)
    mpl.rcParams.update({
        "font.family": family,
        f"font.{family}": stack,
        "mathtext.fontset": "stix" if family == "serif" else "stixsans",
        "font.size": size,
        "axes.labelsize": size,
        "axes.titlesize": size,
        "xtick.labelsize": size,
        "ytick.labelsize": size,
        "legend.fontsize": size,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "patch.linewidth": BOX_LW,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "standard",     # never "tight": it changes the width
        "savefig.dpi": DPI,
        "pdf.fonttype": 42,             # embedded, live text
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "legend.frameon": False,
    })
    return {"font": f, "family": family,
            "metric_clone_of": ("Times New Roman" if f == "Liberation Serif" else
                                "Arial" if f == "Liberation Sans" else None),
            "math": mpl.rcParams["mathtext.fontset"]}


def ieee_canvas(width="double", height_in=4.0, scale=1.0):
    """A figure with one axes in layout inches: one data unit is `scale` inches
    on paper (1.0 for the IEEE build), so a layout can be re-typeset at another
    physical size without moving anything."""
    w = WIDTHS_IN[width] if isinstance(width, str) else float(width)
    fig = plt.figure(figsize=(w * scale, height_in * scale))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, height_in)
    ax.set_aspect("equal")
    ax.axis("off")
    fig._ieee_boxes = []                # (text artist, box patch) for audits
    fig._ieee_kinds = set()
    return fig, ax, w


# ------------------------------------------------------------- primitives

def block(ax, cx, cy, w, h, text, kind="data", size=None, ha="center",
          pad=0.035, rounding=0.03):
    """A labelled box centred at (cx, cy), in layout inches. Returns the patch."""
    k = KINDS[kind]
    p = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        linewidth=BOX_LW * SCALE["lw"], edgecolor=EDGE, facecolor=k["fc"],
        linestyle=k["ls"], zorder=2)
    ax.add_patch(p)
    tx = cx if ha == "center" else cx - w / 2 + pad
    t = ax.text(tx, cy, text, ha=ha, va="center", fontsize=_fs(size), color=TEXT,
                zorder=3, linespacing=1.18)
    ax.figure._ieee_boxes.append((t, p, k["fc"]))
    ax.figure._ieee_kinds.add(kind)
    return p


def op_node(ax, cx, cy, symbol, r=0.085, size=None):
    """A circled operator (element-wise product, concatenation, addition)."""
    c = mpatches.Circle((cx, cy), r, facecolor="white", edgecolor=EDGE,
                        linewidth=BOX_LW * SCALE["lw"], zorder=4)
    ax.add_patch(c)
    t = ax.text(cx, cy, symbol, ha="center", va="center_baseline",
                fontsize=_fs(size), color=TEXT, zorder=5)
    ax.figure._ieee_boxes.append((t, c, "#FFFFFF"))
    return c


def arrow(ax, xy0, xy1, lw=ARROW_LW, ls="solid", head=5.0, connection=None):
    ax.annotate("", xy=xy1, xytext=xy0, zorder=1,
                arrowprops=dict(arrowstyle="-|>", linewidth=lw * SCALE["lw"],
                                color=EDGE, linestyle=ls, shrinkA=0, shrinkB=0,
                                mutation_scale=head * SCALE["lw"],
                                connectionstyle=connection or "arc3,rad=0"))


def polyline_arrow(ax, pts, lw=ARROW_LW, ls="solid", head=5.0):
    """Right-angled connector through the given points, arrow head at the end."""
    xs, ys = zip(*pts[:-1])
    ax.plot(list(xs) + [pts[-2][0]], list(ys) + [pts[-2][1]], color=EDGE,
            lw=lw * SCALE["lw"], ls=ls, solid_capstyle="butt", zorder=1)
    arrow(ax, pts[-2], pts[-1], lw=lw, ls=ls, head=head)


def frame(ax, x0, y0, x1, y1, ls=(0, (4, 2)), lw=0.6):
    """Dashed grouping frame, no fill."""
    ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                    edgecolor=EDGE, linewidth=lw * SCALE["lw"],
                                    linestyle=ls, zorder=0.5))


def sublabel(ax, cx, y, text, size=None):
    """IEEE sub-figure label, '(a) Title', centred beneath its sub-figure."""
    ax.text(cx, y, text, ha="center", va="top", fontsize=_fs(size), color=TEXT)


def legend_row(ax, x, y, kinds, size=None, sw=0.26, sh=0.13, gap=0.07,
               between=0.22):
    """Swatch-and-label key for the block kinds, laid out left to right.

    Text widths are measured, not guessed, so the row cannot run into itself.
    """
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    for kind in kinds:
        k = KINDS[kind]
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y - sh / 2), sw, sh, boxstyle="round,pad=0,rounding_size=0.02",
            linewidth=BOX_LW * SCALE["lw"], edgecolor=EDGE, facecolor=k["fc"],
            linestyle=k["ls"], zorder=2))
        t = ax.text(x + sw + gap, y, k["label"], ha="left", va="center",
                    fontsize=_fs(size), color=TEXT)
        bb = t.get_window_extent(renderer=r)
        x = inv.transform((bb.x1, bb.y0))[0] + between
    return x


# ------------------------------------------------------------- audits

def _srgb_to_lin(u):
    return u / 12.92 if u <= 0.04045 else ((u + 0.055) / 1.055) ** 2.4


def rel_luminance(c) -> float:
    r, g, b = (_srgb_to_lin(u) for u in mpl.colors.to_rgb(c))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def lightness(c) -> float:
    """CIELAB L* (D65), what a greyscale printer preserves."""
    y = rel_luminance(c)
    f = y ** (1 / 3) if y > (6 / 29) ** 3 else y / (3 * (6 / 29) ** 2) + 4 / 29
    return 116 * f - 16


def contrast(c1, c2) -> float:
    a, b = sorted((rel_luminance(c1), rel_luminance(c2)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def check_greyscale(kinds) -> list:
    bad = []
    ks = sorted(kinds)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            same_border = KINDS[a]["ls"] == KINDS[b]["ls"]
            dl = abs(lightness(KINDS[a]["fc"]) - lightness(KINDS[b]["fc"]))
            if same_border and dl < MIN_DL:
                bad.append(f"{a}/{b}: same border and dL*={dl:.1f} < {MIN_DL}")
    return bad


def check_text(fig, lo=TEXT_PT[0], hi=TEXT_PT[1]) -> list:
    bad = []
    for t in fig.findobj(mpl.text.Text):
        s = t.get_text()
        if not s or not s.strip():
            continue
        if not (lo - 1e-6 <= t.get_fontsize() <= hi + 1e-6):
            bad.append((s[:24], t.get_fontsize()))
    return bad


def check_strokes(fig, floor=MIN_STROKE_PT) -> list:
    bad = []
    for a in fig.findobj(lambda o: isinstance(o, (mpl.lines.Line2D,
                                                  mpl.patches.Patch))):
        if not a.get_visible():
            continue
        lw = a.get_linewidth()
        if isinstance(a, mpl.patches.Patch):
            ec = a.get_edgecolor()
            if lw == 0 or (ec is not None and len(ec) == 4 and ec[3] == 0):
                continue
            if a is a.figure.patch or (a.axes is not None and a is a.axes.patch):
                continue
        if 0 < lw < floor - 1e-9:
            bad.append((type(a).__name__, lw))
    return bad


def check_contrast(fig) -> list:
    bad = []
    for t, _, fc in getattr(fig, "_ieee_boxes", []):
        c = contrast(t.get_color(), fc)
        if c < MIN_CONTRAST:
            bad.append((t.get_text()[:24], round(c, 2)))
    return bad


def check_overflow(fig, slack_px=0.5) -> list:
    """Every boxed label must sit inside its box (and inside the canvas)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = []
    W, H = fig.canvas.get_width_height()
    for t, p, _ in getattr(fig, "_ieee_boxes", []):
        tb = t.get_window_extent(renderer=r)
        pb = p.get_window_extent(renderer=r)
        if (tb.x0 < pb.x0 - slack_px or tb.x1 > pb.x1 + slack_px or
                tb.y0 < pb.y0 - slack_px or tb.y1 > pb.y1 + slack_px):
            bad.append(t.get_text().replace("\n", " ")[:32])
    for t in fig.findobj(mpl.text.Text):
        if not t.get_text().strip():
            continue
        tb = t.get_window_extent(renderer=r)
        if tb.x0 < -slack_px or tb.y0 < -slack_px or tb.x1 > W + slack_px \
                or tb.y1 > H + slack_px:
            bad.append("off-canvas: " + t.get_text().replace("\n", " ")[:24])
    return bad


def save_ieee(fig, stem, outdir, formats=("pdf", "eps", "png")) -> dict:
    """Write at exact physical size and audit. Empty `warnings` = pass."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    w, h = fig.get_size_inches()
    warnings = []
    cls = next((k for k, v in WIDTHS_IN.items() if abs(w - v) <= 0.01), None)
    if cls is None:
        warnings.append(f"width {w:.3f} in is not an IEEE column width "
                        f"(3.5 or 7.16 in)")
    if h > MAX_H_IN + 1e-6:
        warnings.append(f"height {h:.2f} in exceeds {MAX_H_IN} in")
    if mpl.rcParams["savefig.bbox"] not in (None, "standard"):
        warnings.append("savefig.bbox is not 'standard'; physical width will shift")
    if mpl.rcParams["pdf.fonttype"] != 42 or mpl.rcParams["ps.fonttype"] != 42:
        warnings.append("fonts would not be embedded as live Type 42 text")
    for name, res in (("text outside 9-10 pt", check_text(fig)),
                      ("strokes below 0.5 pt", check_strokes(fig)),
                      ("low text contrast on fill", check_contrast(fig)),
                      ("text overflows its box", check_overflow(fig)),
                      ("greyscale-ambiguous block kinds",
                       check_greyscale(getattr(fig, "_ieee_kinds", set())))):
        if res:
            warnings.append(f"{name} ({len(res)}): {res[:6]}")
    for a in fig.findobj(lambda o: hasattr(o, "get_alpha")):
        al = a.get_alpha()
        if al is not None and al < 1:
            warnings.append("transparency present; EPS cannot carry it")
            break

    paths = []
    for ext in formats:
        p = outdir / f"{stem}.{ext}"
        fig.savefig(p, **({"dpi": DPI} if ext == "png" else {}))
        paths.append(str(p))
    png = outdir / f"{stem}.png"
    dpi_eff = None
    if png.exists():
        with open(png, "rb") as fh:
            fh.read(16)
            w_px, _ = struct.unpack(">II", fh.read(8))
        dpi_eff = w_px / w
        if dpi_eff < 600 - 1:
            warnings.append(f"raster {dpi_eff:.0f} dpi is below 600 dpi for line art")
    return dict(stem=stem, paths=paths, width_in=round(w, 3), height_in=round(h, 3),
                width_mm=round(w * 25.4, 1), height_mm=round(h * 25.4, 1),
                column_class=cls, raster_dpi=round(dpi_eff) if dpi_eff else None,
                warnings=warnings)
