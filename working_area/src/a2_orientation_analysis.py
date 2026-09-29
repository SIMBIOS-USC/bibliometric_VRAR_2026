"""
src/a2_orientation_analysis.py
================================
Analysis Module 2: Technical vs Pedagogical Orientation
---------------------------------------------------------
Produces Figure: Temporal trends in research orientation per category
Corresponds to: Manuscript Figure 2 & RQ4

This module implements the MISSING classifier from the original repository.
The manuscript states that records were classified by dominant orientation
(Technical vs Pedagogical) based on title + abstract + keywords.
That logic is now in utils/classifier.py:classify_orientation().

Output files (saved to results/):
  - fig_A2_orientation_trend.png       → stacked 100% bar chart per year
  - fig_A2_orientation_by_category.png → 4-panel breakdown per tech category
  - table_A2_orientation_counts.csv    → raw orientation counts per year x category

Methodology:
  classify_orientation() scores a record's Title + Abstract + Author Keywords
  against two keyword lists (TECHNICAL_TERMS, PEDAGOGICAL_TERMS) and assigns
  the dominant label. This is a transparent, reproducible implementation of
  the methodology described in the manuscript.
"""

from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.classifier import CATEGORY_ORDER
from utils.plot_style import apply_global_style, save_figure, CATEGORY_COLORS, ORIENTATION_COLORS


# ---------------------------------------------------------------------------
def run(df: pd.DataFrame, results_dir: Path) -> None:
    """
    Execute Technical vs Pedagogical Orientation analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned and classified DataFrame. Must have columns:
        'Tech_Category', 'Orientation', 'Year'.
    results_dir : Path
        Output directory.
    """
    print("\n[A2] Technical vs Pedagogical Orientation Analysis")
    print("-" * 50)

    apply_global_style(font_size=11)
    results_dir.mkdir(parents=True, exist_ok=True)

    required = {"Tech_Category", "Orientation", "Year"}
    if not required.issubset(df.columns):
        print(f"  [ERROR] Missing columns: {required - set(df.columns)}")
        return

    df_work = df[df["Tech_Category"].notna() & df["Orientation"].notna()].copy()

    # -----------------------------------------------------------------------
    # Figure A: Overall orientation trend (all categories combined)
    # -----------------------------------------------------------------------
    pivot_all = (
        df_work.groupby(["Year", "Orientation"]).size().unstack(fill_value=0)
    )
    # Ensure both columns exist
    for ori in ["Technical", "Pedagogical"]:
        if ori not in pivot_all.columns:
            pivot_all[ori] = 0
    pivot_pct = pivot_all.div(pivot_all.sum(axis=1), axis=0) * 100

    years_available = pivot_pct.index[(pivot_pct.index >= 1991) & (pivot_pct.index <= 2025)]
    pivot_pct = pivot_pct.loc[years_available]

    fig, ax = plt.subplots(figsize=(14, 5), facecolor="white")
    pivot_pct[["Technical", "Pedagogical"]].plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[ORIENTATION_COLORS["Technical"], ORIENTATION_COLORS["Pedagogical"]],
        width=0.85,
        edgecolor="white",
        linewidth=0.3,
    )
    ax.set_title("Research Orientation Over Time (All XR Categories)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Publication Year", fontsize=11)
    ax.set_ylabel("Percentage of Documents (%)", fontsize=11)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylim(0, 105)
    ax.legend(title="Orientation", fontsize=10)
    ax.set_xticks(range(len(pivot_pct)))
    ax.set_xticklabels([str(y) for y in pivot_pct.index], rotation=45, ha="right", fontsize=8)

    save_figure(fig, str(results_dir / "fig_A2_orientation_trend.png"))

    # -----------------------------------------------------------------------
    # Figure B: Orientation breakdown per technology category
    # -----------------------------------------------------------------------
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="white")
    axes = axes.flatten()

    for i, cat in enumerate(CATEGORY_ORDER):
        ax = axes[i]
        df_cat = df_work[df_work["Tech_Category"] == cat]

        if df_cat.empty:
            ax.set_title(f"{cat}\n(no data)", fontsize=10)
            ax.axis("off")
            continue

        pivot_cat = (
            df_cat.groupby(["Year", "Orientation"]).size().unstack(fill_value=0)
        )
        for ori in ["Technical", "Pedagogical"]:
            if ori not in pivot_cat.columns:
                pivot_cat[ori] = 0
        pivot_cat_pct = pivot_cat.div(pivot_cat.sum(axis=1), axis=0) * 100
        pivot_cat_pct = pivot_cat_pct.loc[
            (pivot_cat_pct.index >= 1991) & (pivot_cat_pct.index <= 2025)
        ]

        pivot_cat_pct[["Technical", "Pedagogical"]].plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=[ORIENTATION_COLORS["Technical"], ORIENTATION_COLORS["Pedagogical"]],
            width=0.85,
            edgecolor="white",
            linewidth=0.3,
            legend=i == 0,
        )
        ax.set_title(cat, fontsize=11, fontweight="bold", color=CATEGORY_COLORS[cat])
        ax.set_xlabel("")
        ax.set_ylabel("% Documents" if i % 2 == 0 else "")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_ylim(0, 105)
        ax.set_xticks(range(0, len(pivot_cat_pct), max(1, len(pivot_cat_pct) // 8)))
        ax.set_xticklabels(
            [str(pivot_cat_pct.index[j]) for j in range(0, len(pivot_cat_pct), max(1, len(pivot_cat_pct) // 8))],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax.tick_params(axis="both", labelsize=8)

    fig2.suptitle("Technical vs Pedagogical Orientation by Technology Category", fontsize=13, fontweight="bold")
    fig2.tight_layout()
    save_figure(fig2, str(results_dir / "fig_A2_orientation_by_category.png"))

    # -----------------------------------------------------------------------
    # CSV export
    # -----------------------------------------------------------------------
    counts = (
        df_work.groupby(["Year", "Tech_Category", "Orientation"]).size().reset_index(name="Count")
    )
    counts.to_csv(results_dir / "table_A2_orientation_counts.csv", index=False)
    print(f"  → Saved: {results_dir / 'table_A2_orientation_counts.csv'}")

    print("[A2] Done.")
