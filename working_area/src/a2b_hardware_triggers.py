"""Plot orientation counts by technology category with hardware milestones.

The vertical lines mark selected product releases for context; the figure does
not estimate their effect on publication counts.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.classifier import CATEGORY_ORDER
from utils.plot_style import save_figure

# Plot range
START_YEAR = 1991
END_YEAR   = 2025

_PALETTE = {
    "Pedagogical": "#1976D2",
    "Technical":   "#E64A19",
}

# Legend labels
_LABEL = {
    "Pedagogical": "Pedagogical Research",
    "Technical":   "Technical Development",
}

# Selected hardware milestones, by category
TRIGGERS: dict[str, list[tuple[int, str]]] = {
    "VR": [
        (2012, "Oculus Kickstarter"),
        (2015, "Samsung Gear VR"),
        (2016, "Oculus/Vive/PSVR"),
        (2019, "Meta Quest / Valve Index"),
        (2024, "Apple Vision Pro"),
    ],
    "AR": [
        (2008, "HTC Dream (Wikitude)"),
        (2012, "Oculus Kickstarter"),
        (2013, "Google Glass"),
        (2016, "Pokémon GO"),
        (2024, "Apple Vision Pro"),
    ],
    "MR/XR": [
        (2012, "Oculus Kickstarter"),
        (2016, "HoloLens Dev"),
        (2024, "Apple Vision Pro"),
    ],
    "Hybrid/Multi-technology": [],
}


def run(df: pd.DataFrame, results_dir: Path) -> None:
    """
    Plot the annual orientation counts and milestone annotations.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned + classified DataFrame with columns:
        'Tech_Category', 'Orientation', 'Year'.
    results_dir : Path
        Output directory.
    """
    print("\n[A2b] Orientation counts and hardware milestones")
    print("-" * 55)

    results_dir.mkdir(parents=True, exist_ok=True)

    col_cat  = "Tech_Category"
    col_ori  = "Orientation"
    col_year = "Year"

    df_w = df[df[col_cat].notna() & df[col_year].notna()].copy()
    df_w[col_year] = pd.to_numeric(df_w[col_year], errors="coerce")
    df_w = df_w.dropna(subset=[col_year])
    df_w[col_year] = df_w[col_year].astype(int)
    df_w = df_w[(df_w[col_year] >= START_YEAR) & (df_w[col_year] <= END_YEAR)]

    # Map orientation to display key
    df_w["_ori_key"] = df_w[col_ori].map({
        "Pedagogical": "Pedagogical",
        "Technical":   "Technical",
    }).fillna("Pedagogical")   # fallback: pedagogical

    years_range = np.arange(START_YEAR, END_YEAR + 1)

    # ── Create figure ────────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid",
                  rc={"axes.facecolor": "white", "figure.facecolor": "white"})
    fig, axes = plt.subplots(4, 1, figsize=(11.69, 16.54),
                             sharex=True, facecolor="white")
    fig.subplots_adjust(hspace=0.4)

    for i, cat in enumerate(CATEGORY_ORDER):
        ax = axes[i]
        df_cat = df_w[df_w[col_cat] == cat]

        # Build year × orientation pivot
        summary = (
            df_cat.groupby([col_year, "_ori_key"])
            .size()
            .unstack(fill_value=0)
        )
        for key in _PALETTE:
            if key not in summary.columns:
                summary[key] = 0
        summary = summary.reindex(years_range, fill_value=0)

        # Plot lines + shade
        for key, color in _PALETTE.items():
            data = summary[key]
            ax.plot(summary.index, data,
                    label=_LABEL[key],
                    marker="o", markersize=4,
                    linewidth=2.5, color=color, zorder=4)
            ax.fill_between(summary.index, 0, data,
                            alpha=0.10, color=color, zorder=2)

        # Title
        ax.set_title(
            f"Technical and Pedagogical Research in {cat} Education "
            f"({START_YEAR}–{END_YEAR})",
            fontsize=13, fontweight="bold", pad=8,
        )
        ax.set_ylabel("Publications Count", fontsize=11,
                      fontweight="bold", labelpad=8)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(True, linestyle="--", alpha=0.55)

        # Hardware milestone lines
        y_max = max(summary.max().max(), 1)
        panel_triggers = TRIGGERS.get(cat, [])
        for year, label in panel_triggers:
            if year < START_YEAR:
                continue
            ax.axvline(x=year, color="red", linestyle="--",
                       alpha=0.55, linewidth=3, zorder=5)
            ax.text(year, y_max * 0.52, label,
                    rotation=90, color="black",
                    fontweight="bold", fontsize=12,
                    va="center", ha="center",
                    bbox=dict(facecolor="white", alpha=0.65,
                              edgecolor="none", pad=2),
                    zorder=6)

        ax.legend(title="Research Type", fontsize=10,
                  title_fontsize=11, loc="upper left")

    # Shared x-axis
    axes[-1].set_xlabel("Year", fontsize=13, fontweight="bold", labelpad=12)
    axes[-1].tick_params(axis="x", labelsize=12)
    plt.xticks(np.arange(START_YEAR, END_YEAR + 1, 2),
               rotation=45, ha="right", fontsize=12)

    plt.tight_layout()
    save_figure(fig, str(results_dir / "fig_A2b_hardware_triggers.png"))

    print("[A2b] Done.")
