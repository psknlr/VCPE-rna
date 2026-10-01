"""IEEE Transactions display-item specification, implemented for matplotlib.

Mechanical spec enforced here (IEEE Author Center, "Preparing graphics"):
single column 3.5 in (88.9 mm), double column 7.16 in (181.9 mm), height <= the
9.4 in text block; Times (serif) throughout, 8 pt at final size with nothing
below 6 pt; strokes >= 0.5 pt; vector output with embedded, live text; raster
fall-back at 600 dpi (line art). Sub-figure labels are "(a)", "(b)" set in the
body font below or at the top-left of each panel -- never bold lowercase on its
own as in Nature. Every figure must stay readable when printed in greyscale, so
fills are distinguished by luminance and hatching as well as hue.

Font note: Times New Roman is not installed in this container. Liberation Serif
is metrically identical to Times New Roman (same advance widths), so the layout
is unchanged when the production system substitutes the real face; the
substitution is recorded in the manifest rather than hidden.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

MM = 1.0 / 25.4
WIDTHS_IN = {"single": 3.5, "double": 7.16}
MAX_HEIGHT_IN = 9.4

FONT_STACK = ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman",
              "DejaVu Serif"]

# Greyscale-safe roles: each fill differs in luminance by >= ~20%, so the
# figure survives a monochrome print. Hue is a bonus, not the carrier.
C_MODEL = "#1F4E79"     # dark blue   (L ~ 30%)
C_RIDGE = "#8FAADC"     # light blue  (L ~ 66%)
C_KNN = "#BFBFBF"       # light grey  (L ~ 75%)
C_FLOOR = "#000000"
C_GREY = "#7F7F7F"
C_ACCENT = "#C00000"    # dark red, used for a single emphasised element
FILL_IN = "#F2F2F2"     # input blocks
FILL_OP = "#FFFFFF"     # operators
FILL_LEARN = "#DDEBF7"  # learned layers
FILL_FIXED = "#FBE5D6"  # fixed (non-learned) priors


def resolved_font() -> str:
    from matplotlib import font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    for n in FONT_STACK:
        if n in have:
            return n
    return "serif"


def apply_ieee_style() -> dict:
    f = resolved_font()
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": FONT_STACK,
        "mathtext.fontset": "stix",    # STIX is a Times-design math face
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "in",       # IEEE plots conventionally tick inward
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.grid": True,
        "grid.linewidth": 0.5,
        "grid.linestyle": ":",
        "grid.color": "#A6A6A6",
        "axes.axisbelow": True,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "standard",    # never "tight": it changes physical width
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "legend.frameon": True,
        "legend.edgecolor": "#000000",
        "legend.fancybox": False,
        "legend.framealpha": 1.0,
        "errorbar.capsize": 2.0,
        "hatch.linewidth": 0.5,
    })
    return {"font": f, "font_is_metric_clone_of_times": f == "Liberation Serif"}


def ieee_figure(width="double", height_in=3.0, constrained=True):
    w = WIDTHS_IN[width] if isinstance(width, str) else float(width)
    fig = plt.figure(figsize=(w, height_in),
                     layout="constrained" if constrained else None)
    if constrained:
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04,
                                    hspace=0.05)
    return fig, w


def subcaption(ax, letter, text="", y=-0.30):
    """IEEE sub-figure label, '(a) text', centred beneath the panel."""
    ax.text(0.5, y, f"({letter}) {text}".rstrip(), transform=ax.transAxes,
            ha="center", va="top", fontsize=8)


def check_text_sizes(fig, lo=6.0, hi=10.0):
    bad = []
    for t in fig.findobj(mpl.text.Text):
        s = t.get_text()
        if not s or not s.strip():
            continue
        sz = t.get_fontsize()
        if sz < lo or sz > hi:
            bad.append((s[:28], round(sz, 2)))
    return bad


def check_line_widths(fig, lo=0.5):
    bad = []
    for ln in fig.findobj(mpl.lines.Line2D):
        if ln.get_visible() and 0 < ln.get_linewidth() < lo and ln.get_xydata().size:
            bad.append(round(ln.get_linewidth(), 2))
    return sorted(set(bad))


def save_ieee(fig, stem, outdir="figures", width_in=None,
              formats=("pdf", "png")):
    """Write vector + 600 dpi raster at the exact physical size and audit it."""
    from pathlib import Path
    import struct
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    warnings = []
    w_in, h_in = fig.get_size_inches()
    if width_in is not None and abs(w_in - width_in) > 0.02:
        warnings.append(f"width {w_in:.2f} in != requested {width_in:.2f} in")
    cls = next((k for k, v in WIDTHS_IN.items() if abs(w_in - v) <= 0.02), None)
    if cls is None:
        warnings.append(f"width {w_in:.2f} in is not an IEEE column (3.5 / 7.16 in)")
    if h_in > MAX_HEIGHT_IN + 0.01:
        warnings.append(f"height {h_in:.2f} in exceeds {MAX_HEIGHT_IN} in")
    if mpl.rcParams["pdf.fonttype"] != 42:
        warnings.append("pdf.fonttype != 42; fonts would not embed as TrueType")
    bad_t = check_text_sizes(fig)
    if bad_t:
        warnings.append(f"text outside 6-10 pt: {bad_t[:4]}")
    bad_l = check_line_widths(fig)
    if bad_l:
        warnings.append(f"strokes thinner than 0.5 pt: {bad_l[:4]}")

    paths = []
    for ext in formats:
        p = outdir / f"{stem}.{ext}"
        fig.savefig(p, **({} if ext == "pdf" else {"dpi": 600}))
        paths.append(str(p))
    png = outdir / f"{stem}.png"
    if png.exists():
        with open(png, "rb") as fh:
            fh.read(16)
            w_px, _ = struct.unpack(">II", fh.read(8))
        if w_px / w_in < 600 - 1:
            warnings.append(f"raster {w_px / w_in:.0f} dpi is below IEEE's 600 dpi")
    plt.close(fig)
    return dict(stem=stem, paths=[str(Path(p).relative_to(outdir.parent))
                                  if outdir.parent in Path(p).parents else p
                                  for p in paths],
                width_in=round(w_in, 3), height_in=round(h_in, 3),
                width_mm=round(w_in / MM, 1), height_mm=round(h_in / MM, 1),
                column_class=cls, warnings=warnings)
