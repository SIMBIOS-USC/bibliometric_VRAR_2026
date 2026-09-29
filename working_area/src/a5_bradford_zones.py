"""
src/a5_bradford_zones.py
========================
Analysis Module 5: Bradford's Law & Source Analysis
-----------------------------------------------------
Produces Figure: Bradford zone visualization + top-10 source evolution per category
Corresponds to: Manuscript source concentration analysis

Output files (saved to results/):
  - fig_A5_bradford_{category}.png   → 4 files, one per category
  - fig_A5_bradford_full.png         → full corpus (all categories)
  - table_A5_bradford_zones.csv      → zone counts per category

Methodology:
  Bradford's Law divides a corpus of publications into three equal-sized zones
  by cumulative article count. The number of journals required to cover each
  third follows a geometric progression (the Bradford multiplier).
  Zone 1 (Core): fewest journals, most articles per journal.
  Zone 2 (Middle): more journals, moderate concentration.
  Zone 3 (Periphery): many journals, few articles each.
"""

from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.classifier import CATEGORY_ORDER
from utils.plot_style import apply_global_style, save_figure, CATEGORY_COLORS


BRADFORD_COLORS: list[str] = ["#1f4e79", "#2e75b6", "#89bceb"]


def _bradford_analysis(df_subset: pd.DataFrame, label: str) -> dict:
    """
    Compute Bradford zones for a subset of the corpus.

    Returns a dict with zone counts, and saves a figure at the given path.
    """
    source_counts = df_subset["Source title"].value_counts()
    total = source_counts.sum()
    cumsum = source_counts.cumsum()

    # Assign the journal that crosses each cumulative one-third boundary to
    # the zone it completes. This keeps every journal intact and each zone as
    # close as possible to one-third of the corpus (rather than dropping the
    # boundary-crossing journal into the next zone).
    end_1 = int((cumsum < total / 3).sum())
    end_1 = min(end_1, len(source_counts) - 1)
    end_2 = int((cumsum < 2 * total / 3).sum())
    end_2 = max(end_1, min(end_2, len(source_counts) - 1))
    zone1 = source_counts.iloc[:end_1 + 1]
    zone2 = source_counts.iloc[end_1 + 1:end_2 + 1]
    zone3 = source_counts.iloc[end_2 + 1:]

    return {
        "category": label,
        "total_articles": int(total),
        "zone1_journals": len(zone1),
        "zone1_articles": int(zone1.sum()),
        "zone2_journals": len(zone2),
        "zone2_articles": int(zone2.sum()),
        "zone3_journals": len(zone3),
        "zone3_articles": int(zone3.sum()),
        "top10_sources": source_counts.head(10),
    }


def _plot_bradford(result: dict, df_subset: pd.DataFrame, out_path: str) -> None:
    """Plot Bradford pyramid + top-10 source evolution for one group."""
    if len(df_subset) < 10:
        print(f"    [SKIP] Too few records ({len(df_subset)}) for {result['category']}")
        return

    source_counts = result["top10_sources"]
    top10_names = source_counts.index.tolist()
    df_top = df_subset[df_subset["Source title"].isin(top10_names)]
    pivot = pd.crosstab(df_top["Year"], df_top["Source title"]).reindex(columns=top10_names, fill_value=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), facecolor="white")

    # === Bradford pyramid ===
    zones = [
        {"zone": "Zone 1 (Core)", "journals": result["zone1_journals"], "articles": result["zone1_articles"]},
        {"zone": "Zone 2 (Middle)", "journals": result["zone2_journals"], "articles": result["zone2_articles"]},
        {"zone": "Zone 3 (Periphery)", "journals": result["zone3_journals"], "articles": result["zone3_articles"]},
    ]
    widths = [3, 5, 7]
    bottoms = [2, 1, 0]

    for i, zone in enumerate(zones):
        rect = mpatches.Rectangle(
            (-widths[i] / 2, bottoms[i]), widths[i], 1,
            linewidth=1, edgecolor="white", facecolor=BRADFORD_COLORS[i],
        )
        ax1.add_patch(rect)
        ax1.text(
            0, bottoms[i] + 0.5,
            f"{zone['zone']}\n{zone['journals']} journals  |  {zone['articles']:,} articles",
            ha="center", va="center", color="white", fontweight="bold", fontsize=11,
        )
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(0, 3.3)
    ax1.axis("off")
    ax1.set_title(f"Bradford's Law Zones\n{result['category']}  (N={result['total_articles']:,})",
                  fontsize=13, fontweight="bold", pad=15)

    # === Top-10 source evolution ===
    colors_top = sns.color_palette("tab10", n_colors=min(10, len(top10_names)))
    pivot.plot(kind="bar", stacked=True, ax=ax2, color=colors_top, width=0.82, edgecolor="white", linewidth=0.3)
    ax2.set_title(f"Annual Evolution – Top 10 Sources\n{result['category']}", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Year", fontsize=11)
    ax2.set_ylabel("Number of Publications", fontsize=11)
    ax2.grid(axis="y", linestyle="--", alpha=0.45)
    handles, labels = ax2.get_legend_handles_labels()
    new_labels = [f"{lbl} ({source_counts.get(lbl, 0)})" for lbl in labels]
    ax2.legend(handles, new_labels, title="Source (Total)", bbox_to_anchor=(1.02, 1),
               loc="upper left", fontsize=9)

    fig.tight_layout()
    save_figure(fig, out_path)


# ---------------------------------------------------------------------------
def run(df: pd.DataFrame, results_dir: Path) -> None:
    """
    Execute Bradford Zone analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned and classified DataFrame with columns:
        'Tech_Category', 'Source title', 'Year'.
    results_dir : Path
        Output directory.
    """
    print("\n[A5] Bradford's Law & Source Analysis")
    print("-" * 50)

    apply_global_style(font_size=11)
    results_dir.mkdir(parents=True, exist_ok=True)

    df_work = df[df["Tech_Category"].notna() & df["Source title"].notna()].copy()
    df_work["Source title"] = df_work["Source title"].astype(str).str.strip()
    df_work["Year"] = pd.to_numeric(df_work["Year"], errors="coerce")
    df_work = df_work.dropna(subset=["Year"]).copy()
    df_work["Year"] = df_work["Year"].astype(int)

    all_zone_records = []

    # Full corpus
    result_full = _bradford_analysis(df_work, "Full Corpus")
    _plot_bradford(result_full, df_work, str(results_dir / "fig_A5_bradford_full.png"))
    zone_row = {k: v for k, v in result_full.items() if k != "top10_sources"}
    all_zone_records.append(zone_row)

    # Per category
    for cat in CATEGORY_ORDER:
        df_cat = df_work[df_work["Tech_Category"] == cat]
        if df_cat.empty:
            continue
        safe_name = cat.replace("/", "_").replace(" ", "_")
        result = _bradford_analysis(df_cat, cat)
        _plot_bradford(result, df_cat, str(results_dir / f"fig_A5_bradford_{safe_name}.png"))
        zone_row = {k: v for k, v in result.items() if k != "top10_sources"}
        all_zone_records.append(zone_row)

    # Save zone summary table
    df_zones = pd.DataFrame(all_zone_records)
    df_zones.to_csv(results_dir / "table_A5_bradford_zones.csv", index=False)
    print(f"  → Saved: {results_dir / 'table_A5_bradford_zones.csv'}")

    print("[A5] Done.")
