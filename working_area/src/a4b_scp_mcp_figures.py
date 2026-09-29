"""SCP/MCP figures using the final classified corpus.

Countries are read from the address suffixes in Scopus's Affiliations field.
The first listed affiliation supplies the country used in country-level plots.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.classifier import CATEGORY_ORDER
from utils.plot_style import apply_global_style, save_figure
from utils.countries import affiliation_countries, leading_affiliation_country, COUNTRY_ORDER

# ── Category colours (SCP vivid, MCP pastel) ─────────────────────────────────
_SCP_COLORS = {
    "VR":                      "#D32F2F",
    "AR":                      "#1565C0",
    "MR/XR":                   "#2E7D32",
    "Hybrid/Multi-technology": "#6A1B9A",
}
_MCP_COLORS = {
    "VR":                      "#FFCDD2",
    "AR":                      "#BBDEFB",
    "MR/XR":                   "#C8E6C9",
    "Hybrid/Multi-technology": "#E1BEE7",
}

def _get_countries(aff_str: str) -> set[str]:
    """Use normalized country suffixes shared with the other figures."""
    return affiliation_countries(aff_str)


def _is_mcp(row: pd.Series) -> bool:
    return len(_get_countries(row.get('Affiliations', ''))) > 1


def _leader_country(aff_str: str) -> str | None:
    """Country in the first affiliation, parsed from its address suffix."""
    return leading_affiliation_country(aff_str)


# ─────────────────────────────────────────────────────────────────────────────
def run(df_classified: pd.DataFrame, results_dir: Path) -> None:
    """
    Create SCP/MCP figures from the classified corpus.

    Parameters
    ----------
    df_classified : pd.DataFrame
        Cleaned and classified corpus from the pipeline.
    results_dir : Path
        Output directory.
    """
    print("\n[A4b] SCP vs MCP figures")
    print("-" * 50)

    apply_global_style(font_size=12)
    results_dir.mkdir(parents=True, exist_ok=True)

    if "Tech_Category" not in df_classified.columns:
        print("  [ERROR] Tech_Category column not found.")
        return
    df = df_classified.copy()
    df['_cat'] = df['Tech_Category']
    df = df.dropna(subset=['_cat']).copy()
    print(f"  After classification: {len(df):,} records")

    # SCP / MCP flags
    df['_countries'] = df['Affiliations'].apply(_get_countries)
    df['_collab'] = df['_countries'].apply(
        lambda countries: 'MCP' if len(countries) > 1 else ('SCP' if len(countries) == 1 else 'Unresolved')
    )
    df['_leader'] = df['Affiliations'].apply(_leader_country)

    global_scp = int((df['_collab'] == 'SCP').sum())
    global_mcp = int((df['_collab'] == 'MCP').sum())
    global_unresolved = int((df['_collab'] == 'Unresolved').sum())
    print(f"  Global SCP: {global_scp:,}   MCP: {global_mcp:,}   "
          f"Unresolved: {global_unresolved:,}   Corpus: {len(df):,}")

    # ── Build pivot: country × category × collab ─────────────────────────────
    df_w = df[df['_leader'].notna()].copy()
    pivot = (
        df_w.groupby(['_leader', '_cat', '_collab'])
        .size()
        .unstack(['_cat', '_collab'], fill_value=0)
    )
    pivot.columns = [f"{cat}|{col}" for cat, col in pivot.columns]
    for cat in CATEGORY_ORDER:
        for col in ('SCP', 'MCP'):
            key = f"{cat}|{col}"
            if key not in pivot.columns:
                pivot[key] = 0

    pivot['Total_SCP'] = pivot[[f"{c}|SCP" for c in CATEGORY_ORDER]].sum(axis=1)
    pivot['Total_MCP'] = pivot[[f"{c}|MCP" for c in CATEGORY_ORDER]].sum(axis=1)
    pivot['Total']     = pivot['Total_SCP'] + pivot['Total_MCP']

    # Reindex to fixed country order (only those present)
    ordered = [c for c in COUNTRY_ORDER if c in pivot.index]
    pivot = pivot.loc[ordered]
    shown_total = int(pivot["Total"].sum())
    corpus_total = len(df)
    others_total = corpus_total - shown_total

    pivot.to_csv(results_dir / "table_A4b_country_collab.csv")
    print(f"  → Saved: {results_dir / 'table_A4b_country_collab.csv'}")

    # ─────────────────────────────────────────────────────────────────────────
    # Stacked SCP/MCP counts by leading country.
    # ─────────────────────────────────────────────────────────────────────────
    n = len(pivot)
    x = np.arange(n)
    w = 0.55

    fig, ax = plt.subplots(figsize=(22, 9), facecolor="white")
    ax.set_facecolor("white")

    bottom = np.zeros(n)
    # SCP layers (vivid)
    for cat in CATEGORY_ORDER:
        vals = pivot[f"{cat}|SCP"].values.astype(float)
        ax.bar(x, vals, w, bottom=bottom, color=_SCP_COLORS[cat],
               edgecolor="white", linewidth=0.4, zorder=3)
        bottom += vals
    scp_tops = bottom.copy()

    # MCP layers (pastel) on top
    for cat in CATEGORY_ORDER:
        vals = pivot[f"{cat}|MCP"].values.astype(float)
        ax.bar(x, vals, w, bottom=bottom, color=_MCP_COLORS[cat],
               edgecolor="white", linewidth=0.4, zorder=3)
        bottom += vals
    total_tops = bottom.copy()
    # Residual group keeps the figure numerically closed to the complete
    # corpus while preserving the original SCP/MCP arrows for named countries.
    x_other = n
    ax.bar(x_other, others_total, w, color="#B8C1CC", edgecolor="white",
           linewidth=0.5, zorder=2)
    y_max = max(total_tops.max(), others_total)

    # Annotations
    for i in range(n):
        scp_h  = float(pivot["Total_SCP"].iloc[i])
        mcp_h  = float(pivot["Total_MCP"].iloc[i])
        tot_h  = scp_h + mcp_h

        # TOTAL label
        ax.text(x[i], tot_h + y_max * 0.013,
                f"TOTAL\n{int(tot_h)}",
                ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color="black")

        # SCP label (left arrow)
        if scp_h > 0:
            ax.annotate(f"SCP: {int(scp_h)}",
                        xy=(x[i] - w/2 - 0.02, scp_h/2),
                        xytext=(x[i] - w/2 - 0.28, scp_h/2),
                        fontsize=8.5, ha="right", va="center", color="black",
                        arrowprops=dict(arrowstyle="-", color="black", lw=1.1))

        # MCP brace (right double arrow)
        if mcp_h > 0:
            xb = x[i] + w/2 + 0.07
            ax.annotate("",
                        xy=(xb, tot_h), xytext=(xb, scp_h),
                        arrowprops=dict(arrowstyle="<->", color="red", lw=2.0))
            ax.text(xb + 0.06, (scp_h + tot_h) / 2,
                    f"MCP:\n{int(mcp_h)}",
                    ha="left", va="center",
                    fontsize=8.5, fontweight="bold", color="red")

    ax.text(x_other, others_total + y_max * 0.013,
            f"OTHERS /\nUNRESOLVED\n{others_total:,}",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold",
            color="#4B5563")

    # Legend
    handles = [mpatches.Patch(facecolor="white",
                              label=r"$\bf{International\ (MCP)}$")]
    handles += [mpatches.Patch(facecolor=_MCP_COLORS[c], label=c)
                for c in CATEGORY_ORDER]
    handles += [mpatches.Patch(facecolor="white",
                               label=r"$\bf{Domestic\ (SCP)}$")]
    handles += [mpatches.Patch(facecolor=_SCP_COLORS[c], label=c)
                for c in CATEGORY_ORDER]
    handles += [mpatches.Patch(facecolor="#B8C1CC", label="Others / unresolved")]

    ax.legend(handles=handles, loc="upper left", ncol=2,
              fontsize=9.5, frameon=True,
              title="Category & Type", title_fontsize=10.5)

    ax.set_title("Scientific Production Structure: Domestic (SCP) vs. International (MCP)\n"
                 f"Named countries: {shown_total:,}; Others/unresolved: {others_total:,}; Corpus: {corpus_total:,}",
                 fontsize=16, fontweight="bold", pad=20)
    ax.set_ylabel("Documents", fontsize=13, fontweight="bold")
    ax.set_xticks(np.arange(n + 1))
    ax.set_xticklabels(list(pivot.index) + ["Others"], rotation=20, ha="right",
                       fontsize=12, fontweight="bold")
    ax.set_ylim(0, y_max * 1.20)
    ax.yaxis.grid(True, linestyle=":", alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_figure(fig, str(results_dir / "fig_A4b_scp_mcp_braces.png"))

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE B — Global pie (SCP vs MCP)
    # ─────────────────────────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 9), facecolor="white")
    ax2.set_facecolor("white")

    wedges, texts, autotexts = ax2.pie(
        [global_scp, global_mcp, global_unresolved],
        labels=[f"Domestic (SCP)\n{global_scp:,} docs",
                f"International (MCP)\n{global_mcp:,} docs",
                f"Unresolved\n{global_unresolved:,} docs"],
        colors=["#4CAF50", "#F44336", "#B8C1CC"],
        autopct="%1.1f%%",
        startangle=90,
        explode=(0, 0.06, 0),
        wedgeprops={"linewidth": 2, "edgecolor": "grey"},
        textprops={"fontsize": 16, "fontweight": "bold"},
        pctdistance=0.65,
    )
    for at in autotexts:
        at.set_fontsize(18)
        at.set_fontweight("bold")

    ax2.set_title("Global Ratio: Domestic (SCP) vs. International (MCP) (%)",
                  fontsize=18, fontweight="bold", pad=20)
    plt.tight_layout()
    save_figure(fig2, str(results_dir / "fig_A4b_scp_mcp_global_pie.png"))

    print("[A4b] Done.")
