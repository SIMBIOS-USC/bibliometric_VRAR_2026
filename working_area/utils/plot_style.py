"""
utils/plot_style.py
===================
Shared plotting configuration and style utilities.

All analysis modules import from this file to ensure visual consistency
across all figures in the manuscript.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib as mpl


# ---------------------------------------------------------------------------
# 1. COLOUR PALETTE
# ---------------------------------------------------------------------------
CATEGORY_COLORS: dict[str, str] = {
    "VR":                       "#e53935",   # Deep Red
    "AR":                       "#1e88e5",   # Deep Blue
    "MR/XR":                    "#43a047",   # Deep Green
    "Hybrid/Multi-technology":  "#8e24aa",   # Deep Purple
}
"""
Canonical colours for the four technology categories.
Must match the colours used in ALL manuscript figures.
"""

CATEGORY_ORDER: list[str] = ["VR", "AR", "MR/XR", "Hybrid/Multi-technology"]
"""Canonical display order for categories."""

ORIENTATION_COLORS: dict[str, str] = {
    "Technical":    "#0288d1",
    "Pedagogical":  "#e65100",
}


# ---------------------------------------------------------------------------
# 2. MATPLOTLIB STYLE
# ---------------------------------------------------------------------------
def apply_global_style(font_size: int = 12) -> None:
    """
    Apply the project-wide Matplotlib style.
    Call this once at the top of each analysis module.
    """
    mpl.rcParams.update({
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":            font_size,
        "axes.titlesize":       font_size + 2,
        "axes.labelsize":       font_size,
        "xtick.labelsize":      font_size - 1,
        "ytick.labelsize":      font_size - 1,
        "legend.fontsize":      font_size - 1,
        "figure.facecolor":     "white",
        "axes.facecolor":       "white",
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.grid":            True,
        "grid.linestyle":       ":",
        "grid.alpha":           0.5,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
    })


def save_figure(fig: plt.Figure, path: str | None, close: bool = True) -> None:
    """Save a figure to *path* (if not None) and optionally close it."""
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
        # Keep the PNG for previews and also export a vector PDF for journal
        # layout and later editing in Illustrator or Inkscape.
        if path.lower().endswith(".png"):
            fig.savefig(path[:-4] + ".pdf", format="pdf", bbox_inches="tight")
        print(f"  → Saved: {path}")
    if close:
        plt.close(fig)
