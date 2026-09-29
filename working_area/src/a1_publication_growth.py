"""
src/a1_publication_growth.py
============================
Analysis Module 1: Publication Growth & Citation Evolution
-----------------------------------------------------------
Produces Figure: Annual production and citation evolution per technology category
Corresponds to: Manuscript RQ1 (growth trends)

Output files (saved to results/):
  - fig_A1_publication_growth.png  → 4-panel figure (one panel per category)
  - table_A1_yearly_stats.csv      → yearly statistics per category

Methodology:
  For each of the four technology categories (VR, AR, MR/XR, Hybrid), the module
  computes per year:
    N  = annual article count
    TC = total citations accumulated by articles published that year
    Impact = TC / N (citations per article)
  A first-article annotation is added to each panel showing the earliest year
  with at least one record in that category.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Allow running from working_area/ directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.classifier import add_classifications, CATEGORY_ORDER
from utils.plot_style import apply_global_style, save_figure, CATEGORY_COLORS


# ---------------------------------------------------------------------------
def run(df: pd.DataFrame, results_dir: Path) -> None:
    """
    Execute Publication Growth & Citation Evolution analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned and classified Scopus DataFrame (output of main pipeline).
        Must have columns: 'Tech_Category', 'Year', 'Cited by', 'Title'.
    results_dir : Path
        Directory where output figures and tables will be saved.
    """
    print("\n[A1] Publication Growth & Citation Evolution")
    print("-" * 50)

    apply_global_style(font_size=12)
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Validate required columns ---
    col_year = "Year"
    col_cit = "Cited by" if "Cited by" in df.columns else "Citations"
    col_cat = "Tech_Category"

    df_work = df.copy()
    df_work[col_cit] = pd.to_numeric(df_work[col_cit], errors="coerce").fillna(0)
    df_work = df_work[df_work[col_cat].notna()].copy()

    years = np.arange(1991, 2026)

    # --- Figure: 4-panel evolution ---
    fig, axes = plt.subplots(4, 1, figsize=(11.69, 16.54), sharex=True, facecolor="white")
    fig.subplots_adjust(hspace=0.3, right=0.82)

    all_stats = []

    for i, cat in enumerate(CATEGORY_ORDER):
        ax = axes[i]
        color = CATEGORY_COLORS[cat]

        df_cat = df_work[df_work[col_cat] == cat]

        stats = (
            df_cat.groupby(col_year)
            .agg(N=(col_year, "count"), TC=(col_cit, "sum"))
            .reindex(years, fill_value=0)
        )
        stats["Impact"] = (stats["TC"] / stats["N"].replace(0, np.nan)).fillna(0)
        stats["Category"] = cat
        stats["Year"] = stats.index
        all_stats.append(stats.reset_index(drop=True))

        # Twin axes
        par1 = ax.twinx()
        par2 = ax.twinx()
        par2.spines["right"].set_position(("axes", 1.15))
        par2.spines["right"].set_visible(True)
        par2.spines["right"].set_color("#c62828")

        # Plots
        b = ax.bar(years, stats["N"], color=color, alpha=0.55, label="Annual Production (N)")
        (l1,) = par1.plot(years, stats["TC"], color="#1565c0", lw=2.5, marker="o", ms=3, label="Total Citations")
        (l2,) = par2.plot(years, stats["Impact"], color="#c62828", lw=2, ls="--", marker="s", ms=3, label="Impact (TC/N)")

        # Labels
        ax.set_ylabel("N (articles)", fontweight="bold", color="#455a64", fontsize=10)
        par1.set_ylabel("Total Citations", fontweight="bold", color="#1565c0", fontsize=10)
        par2.set_ylabel("Impact (TC/N)", fontweight="bold", color="#c62828", fontsize=10)

        letter = chr(65 + i)
        ax.set_title(f"{letter}. {cat}", fontsize=12, fontweight="bold", loc="left", pad=6)
        ax.grid(True, axis="y", ls=":", alpha=0.5)
        ax.tick_params(axis="both", labelsize=9)
        par1.tick_params(axis="y", labelsize=9)
        par2.tick_params(axis="y", labelsize=9)

        # First article annotation
        first_year = df_cat[col_year].min() if not df_cat.empty else None
        if first_year and not np.isnan(first_year):
            ax.axvline(x=first_year, color="black", ls=":", lw=2)
            y_pos = ax.get_ylim()[1] * 0.65
            offset = 1.5 if i == 0 else -4
            ax.annotate(
                f"First article:\n{int(first_year)}",
                xy=(first_year, y_pos * 0.6),
                xytext=(first_year + offset, y_pos),
                arrowprops=dict(facecolor="black", edgecolor="none", width=2, headwidth=7, shrink=0.05),
                fontsize=8,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", lw=1),
            )

    # Global legend
    fig.legend(
        [b, l1, l2],
        ["Annual Production (N)", "Total Citations (TC)", "Impact (TC/N)"],
        loc="upper center",
        ncol=3,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
    )
    plt.xlabel("Publication Year", fontsize=11, fontweight="bold")
    plt.xticks(years[::5], rotation=45, ha="right", fontsize=9)

    save_figure(fig, str(results_dir / "fig_A1_publication_growth.png"))

    # --- CSV export ---
    df_all = pd.concat(all_stats, ignore_index=True)
    df_all = df_all[["Category", "Year", "N", "TC", "Impact"]]
    df_all.to_csv(results_dir / "table_A1_yearly_stats.csv", index=False)
    print(f"  → Saved: {results_dir / 'table_A1_yearly_stats.csv'}")

    print("[A1] Done.")
