"""Nature Portfolio display-item specification, implemented for matplotlib.

Mechanical spec enforced here: single column 89 mm, column-and-a-half 120-136 mm,
double column 183 mm; height <= 170 mm; Arial/Helvetica throughout; body, tick and
legend text 5-7 pt at final size; panel labels 8 pt bold lowercase upright
top-left; strokes 0.25-1 pt (axis 0.8, data 1.0-1.2, error bars 0.8); white
background with no gridlines; RGB; vector output with live (selectable) text.

Two traps this module exists to avoid, because both silently defeat careful work:

* `bbox_inches="tight"` changes the physical width, so a figure built at 183 mm
  stops being 183 mm on save. `savefig.bbox` is pinned to "standard" and layout is
  handled by constrained layout instead.
* Text must stay editable for the production editor, so `pdf.fonttype` is 42 and
  `svg.fonttype` is "none". Glyphs are never outlined to curves.

Font note: Arial and Helvetica are not installed in this container. Liberation
Sans is metrically identical to Arial -- same advance widths, so a layout built
with it is unchanged when the production system substitutes real Arial -- and is
used with that substitution recorded in the figure provenance rather than passed
off as Arial.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

MM = 1.0 / 25.4

WIDTHS_MM = {"single": 89.0, "onehalf": 136.0, "double": 183.0}
MAX_HEIGHT_MM = 170.0

# Okabe-Ito, the colour-blind-safe palette the Nature figure guide publishes.
OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#999999",
}
# Role-based assignment, so colour marks the variable under test and
# everything else stays grey. Never red+green, never rainbow.
C_MODEL = OKABE_ITO["blue"]
C_BASE = OKABE_ITO["grey"]
C_ACCENT = OKABE_ITO["vermillion"]
C_ALT = OKABE_ITO["orange"]
C_FLOOR = OKABE_ITO["black"]

FONT_STACK = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]


def resolved_font() -> str:
    from matplotlib import font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    for n in FONT_STACK:
        if n in have:
            return n
    return "sans-serif"


def apply_nature_style() -> dict:
    f = resolved_font()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "font.size": 6.0,              # body text inside the 5-7 pt band
        "axes.labelsize": 7.0,
        "axes.titlesize": 7.0,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "legend.fontsize": 6.0,
        "figure.titlesize": 7.0,
        "axes.linewidth": 0.8,         # within 0.25-1 pt
        "lines.linewidth": 1.1,
        "patch.linewidth": 0.6,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.grid": False,            # the guide forbids gridlines
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "standard",    # never "tight": it changes physical width
        "savefig.dpi": 600,
        "pdf.fonttype": 42,            # keep text live and editable
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "legend.frameon": False,
        "errorbar.capsize": 1.4,
    })
    return {"font": f, "font_is_metric_clone_of_arial": f == "Liberation Sans"}


def nature_figure(width="double", height_mm=90.0, **gridspec):
    w_mm = WIDTHS_MM[width] if isinstance(width, str) else float(width)
    fig = plt.figure(figsize=(w_mm * MM, height_mm * MM), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.012, h_pad=0.012, wspace=0.03, hspace=0.03)
    return fig, w_mm


def panel_label(ax, letter, dx=-0.085, dy=1.045):
    """8 pt bold lowercase, upright, top-left, in reading order."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left")


def tidy(ax, ygrid=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid:  # kept available but off by default; the guide forbids gridlines
        ax.grid(False)
    return ax


def check_text_sizes(fig, lo=5.0, hi=8.0):
    """Flag rendered text outside the 5-7 pt band (8 pt allowed for panel labels)."""
    bad = []
    for t in fig.findobj(mpl.text.Text):
        s = t.get_text()
        if not s or not s.strip():
            continue
        sz = t.get_fontsize()
        if sz < lo or sz > hi:
            bad.append((s[:28], round(sz, 2)))
    return bad


def save_nature(fig, stem, outdir="figures", width_mm=None, formats=("pdf", "png")):
    """Write vector + raster at the exact physical size and audit the result."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    warnings = []
    w_in, h_in = fig.get_size_inches()
    w_mm, h_mm = w_in / MM, h_in / MM

    if width_mm is not None and abs(w_mm - width_mm) > 0.6:
        warnings.append(f"width {w_mm:.1f} mm != requested {width_mm:.1f} mm")
    classes = {k: v for k, v in WIDTHS_MM.items()}
    matched = next((k for k, v in classes.items() if abs(w_mm - v) <= 0.6), None)
    if matched is None and not (120.0 - 0.6 <= w_mm <= 136.0 + 0.6):
        warnings.append(
            f"width {w_mm:.1f} mm is not a Nature column class "
            f"(89 / 120-136 / 183 mm)")
    if h_mm > MAX_HEIGHT_MM + 0.6:
        warnings.append(f"height {h_mm:.1f} mm exceeds {MAX_HEIGHT_MM:.0f} mm")
    if mpl.rcParams["savefig.bbox"] not in (None, "standard"):
        warnings.append("savefig.bbox is not 'standard'; physical width will shift")
    if mpl.rcParams["pdf.fonttype"] != 42:
        warnings.append("pdf.fonttype != 42; text would not stay editable")
    oversize = check_text_sizes(fig)
    if oversize:
        warnings.append(f"text outside the 5-8 pt band: {oversize[:4]}")

    paths = []
    for ext in formats:
        p = outdir / f"{stem}.{ext}"
        dpi = 600 if ext != "pdf" else None
        fig.savefig(p, **({"dpi": dpi} if dpi else {}))
        paths.append(str(p))
    png = outdir / f"{stem}.png"
    if png.exists():
        import struct
        with open(png, "rb") as fh:
            fh.read(8)
            struct.unpack(">I", fh.read(4))
            fh.read(4)
            w_px, h_px = struct.unpack(">II", fh.read(8))
        dpi_eff = w_px / w_in
        if dpi_eff < 300:
            warnings.append(f"raster {dpi_eff:.0f} dpi is below the 300 dpi floor")
    return dict(stem=stem, paths=paths, width_mm=round(w_mm, 2),
                height_mm=round(h_mm, 2), column_class=matched,
                warnings=warnings)


def git_rev():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              cwd=str(Path(__file__).resolve().parents[2])
                              ).stdout.strip() or None
    except Exception:
        return None
