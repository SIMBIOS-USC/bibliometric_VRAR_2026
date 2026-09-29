"""
src/a4_country_collaboration.py
================================
Analysis Module 4: Country Collaboration (SCP vs MCP)
------------------------------------------------------
Produces Figure: Stacked bar chart of Single-Country Publications (SCP) vs
                 Multi-Country Publications (MCP) per top-12 countries,
                 broken down by technology category.
Corresponds to: Manuscript collaboration analysis section

Output files (saved to results/):
  - fig_A4_scp_mcp.png               → main SCP/MCP stacked bar figure
  - fig_A4_scp_mcp_normalized.png    → normalized (%) version
  - fig_A4_country_pies.png          → SCP vs MCP country distribution pies
  - table_A4_country_collab.csv      → raw country collaboration counts

Methodology:
  A publication is classified as MCP (Multi-Country Publication) if the
  Affiliations field contains affiliation strings from more than one distinct
  country (identified using a keyword-to-country mapping dictionary).
  Otherwise it is SCP (Single-Country Publication).
  The leading country is determined as the country of the FIRST listed affiliation.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.classifier import CATEGORY_ORDER
from utils.plot_style import apply_global_style, save_figure, CATEGORY_COLORS
from utils.countries import affiliation_countries, leading_affiliation_country


# ---------------------------------------------------------------------------
# Country parsing is centralized in utils/countries.py.
# ---------------------------------------------------------------------------
def _detect_countries(aff_string: str) -> set[str]:
    """Return normalized country suffixes from affiliation addresses."""
    return affiliation_countries(aff_string)


def _leading_country(aff_string: str) -> str | None:
    """Return the country of the first listed affiliation."""
    return leading_affiliation_country(aff_string)


# ---------------------------------------------------------------------------
def run(df: pd.DataFrame, results_dir: Path) -> None:
    """
    Execute Country Collaboration analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned and classified DataFrame with columns:
        'Tech_Category', 'Affiliations'.
    results_dir : Path
        Output directory.
    """
    print("\n[A4] Country Collaboration (SCP vs MCP)")
    print("-" * 50)

    apply_global_style(font_size=11)
    results_dir.mkdir(parents=True, exist_ok=True)

    col_aff = "Affiliations"
    col_cat = "Tech_Category"

    df_work = df[df[col_cat].notna()].copy()

    # --- Determine leading country and SCP/MCP ---
    df_work["_countries"] = df_work[col_aff].apply(_detect_countries)
    df_work["_leader"] = df_work[col_aff].apply(_leading_country)
    df_work["_collab"] = df_work["_countries"].apply(
        lambda s: "MCP" if len(s) > 1 else ("SCP" if len(s) == 1 else "Unresolved")
    )
    df_base = df_work.copy()

    # Full country contribution count: count a document once for every
    # distinct country represented in its affiliation list. This differs from
    # the SCP/MCP table below, which attributes each record to its first-listed
    # affiliation country.
    contribution_rows = [
        {"Country": country, "Category": row[col_cat]}
        for _, row in df_base.iterrows()
        for country in row["_countries"]
    ]
    contributions = pd.DataFrame(contribution_rows)
    if not contributions.empty:
        contribution_counts = (contributions.groupby("Country").size()
                               .rename("Articles with any affiliation in country")
                               .reset_index())
        lead_counts = (df_base[df_base["_leader"].notna()]
                       .groupby("_leader").size().rename("Articles assigned by first affiliation")
                       .rename_axis("Country").reset_index())
        contribution_counts = contribution_counts.merge(lead_counts, on="Country", how="left")
        contribution_counts["Articles assigned by first affiliation"] = (
            contribution_counts["Articles assigned by first affiliation"].fillna(0).astype(int)
        )
        contribution_counts["Percent of corpus (any affiliation)"] = (
            100 * contribution_counts["Articles with any affiliation in country"] / max(len(df), 1)
        ).round(2)
        contribution_counts = contribution_counts.sort_values(
            "Articles with any affiliation in country", ascending=False
        )
        contribution_counts.to_csv(results_dir / "table_A4_country_contributions.csv", index=False)
        print(f"  → Saved: {results_dir / 'table_A4_country_contributions.csv'}")

    df_work = df_work[df_work["_leader"].notna()]

    # --- Build pivot table ---
    records = []
    for _, row in df_work.iterrows():
        records.append({
            "Country": row["_leader"],
            "Category": row[col_cat],
            "Collab": row["_collab"],
        })
    df_long = pd.DataFrame(records)

    pivot = (
        df_long.groupby(["Country", "Category", "Collab"])
        .size()
        .unstack(["Category", "Collab"], fill_value=0)
    )
    # Flatten columns
    pivot.columns = [f"{cat}_{collab}" for cat, collab in pivot.columns]

    # Totals
    pivot["Total_SCP"] = sum(pivot.get(f"{cat}_SCP", 0) for cat in CATEGORY_ORDER)
    pivot["Total_MCP"] = sum(pivot.get(f"{cat}_MCP", 0) for cat in CATEGORY_ORDER)
    pivot["Total"] = pivot["Total_SCP"] + pivot["Total_MCP"]
    pivot = pivot.sort_values("Total", ascending=False)

    # Keep a complete global denominator separate from the displayed top-12
    # country subset. Otherwise the figure can be misread as if the visible
    # bars represented the whole corpus.
    global_scp = int((df_base["_collab"] == "SCP").sum())
    global_mcp = int((df_base["_collab"] == "MCP").sum())
    country_identifiable = global_scp + global_mcp
    corpus_total = len(df)
    shown = pivot.head(12).copy()
    shown_total = int(shown["Total"].sum())
    others_total = int(corpus_total - shown_total)
    pivot = shown

    # --- Save CSV ---
    pivot.to_csv(results_dir / "table_A4_country_collab.csv")
    pd.DataFrame([{
        "Global_SCP": global_scp,
        "Global_MCP": global_mcp,
        "Corpus_Total": corpus_total,
        "Country_Identifiable": country_identifiable,
        "Shown_Top12": shown_total,
        "Shown_Coverage_of_Country_Identifiable": shown_total / country_identifiable if country_identifiable else 0,
        "Unassigned_or_not_shown": corpus_total - shown_total,
        "Unresolved_collaboration_status": int((df_base["_collab"] == "Unresolved").sum()),
    }]).to_csv(results_dir / "table_A4_global_collab_summary.csv", index=False)

    # Wilson score intervals for the MCP proportion, reported only for
    # country-category groups with at least 10 identified documents.
    ci_rows = []
    for (country, category), group in df_long.groupby(["Country", "Category"]):
        n = len(group)
        if n < 10:
            continue
        k = int((group["Collab"] == "MCP").sum())
        p = k / n
        z = 1.96
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        ci_rows.append({"Country": country, "Category": category, "Documents": n,
                        "MCP documents": k, "MCP proportion": p,
                        "Wilson 95% CI lower": max(0.0, center - half),
                        "Wilson 95% CI upper": min(1.0, center + half),
                        "Formula": "MCP = >1 distinct parsed country; Wilson score interval; n>=10"})
    pd.DataFrame(ci_rows).to_csv(results_dir / "table_A4_scp_mcp_rates_ci.csv", index=False)
    print(f"  → Saved: {results_dir / 'table_A4_country_collab.csv'}")

    # --- Main SCP/MCP stacked bar ---
    colors_full = {cat: CATEGORY_COLORS[cat] for cat in CATEGORY_ORDER}
    # SCP = vivid, MCP = pastel (75% lighter)
    scp_colors = {cat: CATEGORY_COLORS[cat] for cat in CATEGORY_ORDER}
    mcp_colors = {cat: CATEGORY_COLORS[cat] + "88" for cat in CATEGORY_ORDER}  # hex alpha

    x = np.arange(len(pivot))
    width = 0.55

    fig, ax = plt.subplots(figsize=(18, 8), facecolor="white")
    ax.set_facecolor("white")
    bottom = np.zeros(len(pivot))

    for collab_type, shade in [("SCP", scp_colors), ("MCP", mcp_colors)]:
        for cat in CATEGORY_ORDER:
            col = f"{cat}_{collab_type}"
            vals = pivot.get(col, pd.Series(0, index=pivot.index)).values
            ax.bar(x, vals, width, bottom=bottom,
                   color=shade[cat],
                   edgecolor="white", linewidth=0.4)
            bottom += vals

    # Residual bar: countries outside the top 12 plus country-unresolved
    # records. This makes the displayed bars sum to the full corpus.
    x_other = len(pivot)
    ax.bar(x_other, others_total, width, color="#B8C1CC", edgecolor="white",
           linewidth=0.5, zorder=2)

    # Divider lines + annotations
    for i, country in enumerate(pivot.index):
        scp_h = pivot.loc[country, "Total_SCP"]
        total_h = pivot.loc[country, "Total"]
        ax.hlines(y=scp_h, xmin=i - width / 2, xmax=i + width / 2,
                  color="black", linestyle="--", linewidth=1.2, zorder=5)
        ax.text(i, total_h + pivot["Total"].max() * 0.02,
                f"TOTAL\n{int(total_h)}", ha="center", va="bottom",
                fontsize=10, fontweight="black", color="black")

    ax.text(x_other, others_total + pivot["Total"].max() * 0.02,
            f"OTHERS /\nUNRESOLVED\n{others_total:,}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color="#4B5563")

    ax.set_title("Domestic (SCP) vs International (MCP) Collaboration\nTop 12 leading countries by first affiliation",
                 fontsize=14, fontweight="bold", pad=20)
    ax.text(0.5, 1.015,
            f"Corpus: {corpus_total:,}; country identifiable: {country_identifiable:,}; "
            f"displayed top-12: {shown_total:,} ({shown_total / country_identifiable:.1%} of country-identifiable)",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color="#5B6573")
    ax.set_ylabel("Documents", fontsize=12, fontweight="bold")
    ax.set_xticks(np.arange(len(pivot) + 1))
    ax.set_xticklabels(list(pivot.index) + ["Others"], rotation=35, ha="right", fontsize=11, fontweight="bold")
    ax.yaxis.grid(True, linestyle=":", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    legend_elements = [
        Patch(facecolor="white", alpha=0, label=r"$\bf{Domestic\ (SCP)}$"),
        *[Patch(facecolor=scp_colors[cat], label=cat) for cat in CATEGORY_ORDER],
        Patch(facecolor="white", alpha=0, label=r"$\bf{International\ (MCP)}$"),
        *[Patch(facecolor=mcp_colors[cat], label=cat) for cat in CATEGORY_ORDER],
        Patch(facecolor="#B8C1CC", label="Others / unresolved"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", ncol=2, fontsize=9,
              frameon=True, title="Category & Type", title_fontsize=10)

    plt.tight_layout()
    save_figure(fig, str(results_dir / "fig_A4_scp_mcp.png"))

    # --- Normalized version ---
    fig2, ax2 = plt.subplots(figsize=(14, 6), facecolor="white")
    ax2.set_facecolor("white")
    bottom2 = np.zeros(len(pivot))

    for collab_type, shade in [("SCP", scp_colors), ("MCP", mcp_colors)]:
        for cat in CATEGORY_ORDER:
            col = f"{cat}_{collab_type}"
            vals = pivot.get(col, pd.Series(0, index=pivot.index)).values / pivot["Total"].values
            ax2.bar(x, vals, width, bottom=bottom2, color=shade[cat], edgecolor="white", linewidth=0.4)
            bottom2 += vals

    ax2.set_title("Normalized SCP vs MCP by Technology Category and Country", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Fraction of Documents", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(pivot.index, rotation=35, ha="right", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.grid(True, linestyle=":", alpha=0.45)
    plt.tight_layout()
    save_figure(fig2, str(results_dir / "fig_A4_scp_mcp_normalized.png"))

    # --- Pie charts ---
    scp_vals = list(pivot["Total_SCP"])
    scp_labs = list(pivot.index)
    scp_other = max(0, global_scp - sum(scp_vals))
    if scp_other > 0:
        scp_vals.append(scp_other)
        scp_labs.append("Others")

    mcp_vals = list(pivot["Total_MCP"])
    mcp_labs = list(pivot.index)
    mcp_other = max(0, global_mcp - sum(mcp_vals))
    if mcp_other > 0:
        mcp_vals.append(mcp_other)
        mcp_labs.append("Others")

    fig3, (ax_s, ax_m) = plt.subplots(1, 2, figsize=(16, 7), facecolor="white")
    ax_s.pie(scp_vals, labels=scp_labs, autopct="%1.1f%%", shadow=False, startangle=140)
    ax_s.set_title(f"SCP (n={global_scp:,})", fontsize=14, fontweight="bold")
    ax_m.pie(mcp_vals, labels=mcp_labs, autopct="%1.1f%%", shadow=False, startangle=140)
    ax_m.set_title(f"MCP (n={global_mcp:,})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig3, str(results_dir / "fig_A4_country_pies.png"))

    print("[A4] Done.")
