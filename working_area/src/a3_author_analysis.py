"""
src/a3_author_analysis.py
=========================
Analysis Module 3: Top Authors per Technology Category
-------------------------------------------------------
Produces Figure: Horizontal bar charts of top-10 authors by publications,
                 citations, and h-index per category.
Corresponds to: Manuscript author productivity tables

Output files (saved to results/):
  - fig_A3_top_authors_pubs.png     → top-10 by publications (4 panels)
  - fig_A3_top_authors_cit.png      → top-10 by total citations (4 panels)
  - fig_A3_top_authors_hindex.png   → top-10 by h-index (4 panels)
  - table_A3_author_stats.csv       → full author stats table

Methodology:
  For each author, metrics are computed within each technology category using
  the category labels already assigned by the canonical corpus pipeline.
  Authors may therefore appear in more than one category. The displayed
  affiliation is estimated by majority vote over the first affiliation listed
  on records where the author appears; Scopus does not provide an author-to-
  affiliation link in the fields used here.
"""

from __future__ import annotations
import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.classifier import CATEGORY_ORDER
from utils.plot_style import apply_global_style, save_figure, CATEGORY_COLORS


# ---------------------------------------------------------------------------
# INSTITUTION NAME NORMALIZATION
# ---------------------------------------------------------------------------
# Key mappings for well-known institutions (expand as needed)
INSTITUTION_MAPPINGS: dict[str, str] = {
    "la jolla": "UC San Diego (USA)",
    "san diego": "UC San Diego (USA)",
    "ucsd": "UC San Diego (USA)",
    "ann arbor": "Univ. of Michigan (USA)",
    "ithaca": "Cornell Univ. (USA)",
    "harvard": "Harvard Univ. (USA)",
    "stanford": "Stanford Univ. (USA)",
    "washington": "Univ. of Washington (USA)",
    "mit": "MIT (USA)",
    "massachusetts institute": "MIT (USA)",
    "johns hopkins": "Johns Hopkins (USA)",
    "mayo clinic": "Mayo Clinic (USA)",
    # Keep this specific: "college london" also matches Imperial College London.
    "university college london": "UCL (UK)",
    "imperial college": "Imperial College (UK)",
    "oxford": "Univ. of Oxford (UK)",
    "cambridge": "Univ. of Cambridge (UK)",
    "toronto": "Univ. of Toronto (CAN)",
    "tsinghua": "Tsinghua Univ. (CHN)",
    "nanyang": "Nanyang Tech. Univ. (SGP)",
    "nus": "National Univ. Singapore (SGP)",
    "hong kong polytechnic": "HK PolyU (HKG)",
    "tokyo": "Univ. of Tokyo (JPN)",
    "munich": "Tech. Univ. Munich (DEU)",
    "barcelona": "Univ. de Barcelona (ESP)",
    "seville": "Univ. de Sevilla (ESP)",
    "sevilla": "Univ. de Sevilla (ESP)",
    "michigan": "Univ. of Michigan (USA)",
    "zhejiang": "Zhejiang Univ. (CHN)",
    "cornell": "Cornell Univ. (USA)",
}


def _clean_institution(aff_string: str) -> str | None:
    """Return a cleaned institution name from a raw Scopus affiliation string."""
    if not isinstance(aff_string, str) or len(aff_string) < 3:
        return None
    aff_lower = aff_string.lower()
    for key, val in INSTITUTION_MAPPINGS.items():
        if key in aff_lower:
            return val
    # Fallback: extract from comma-separated parts
    parts = [p.strip() for p in aff_string.split(",")]
    country = parts[-1] if parts else ""
    target_tokens = ["university", "universidad", "institute", "hospital", "clinic", "foundation"]
    skip_tokens = ["department", "dept", "faculty", "school", "division", "unit", "center", "lab"]
    best = None
    for part in parts:
        if len(part) < 3:
            continue
        if any(b in part.lower() for b in skip_tokens) and not any(g in part.lower() for g in target_tokens):
            continue
        if any(g in part.lower() for g in target_tokens):
            best = part
            break
    if not best:
        best = parts[1] if len(parts) > 1 and len(parts[1]) > 3 else parts[0]
    label = f"{best} ({country})" if country else best
    return label[:40] + ".." if len(label) > 40 else label


def _h_index(citations: list[float]) -> int:
    """Compute h-index from a list of citation counts."""
    cites = sorted([int(c) for c in citations], reverse=True)
    h = 0
    for i, c in enumerate(cites):
        if c >= i + 1:
            h = i + 1
        else:
            break
    return h


# ---------------------------------------------------------------------------
def run(df: pd.DataFrame, results_dir: Path) -> None:
    """
    Execute Top Author Analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned and classified DataFrame with columns:
        'Tech_Category', 'Author full names' (or 'Authors'), 'Cited by', 'Affiliations'.
    results_dir : Path
        Output directory.
    """
    print("\n[A3] Top Author Analysis")
    print("-" * 50)

    apply_global_style(font_size=10)
    results_dir.mkdir(parents=True, exist_ok=True)

    col_authors = "Author full names" if "Author full names" in df.columns else "Authors"
    col_cit = "Cited by" if "Cited by" in df.columns else "Citations"
    col_aff = "Affiliations"

    df_work = df[df["Tech_Category"].notna()].copy()
    df_work[col_cit] = pd.to_numeric(df_work[col_cit], errors="coerce").fillna(0)

    # --- Build affiliation majority-vote map ---
    author_aff_history: dict[str, list[str]] = {}
    for _, row in df_work.iterrows():
        if pd.isna(row.get(col_aff)):
            continue
        aff_raw = str(row[col_aff]).split(";")[0].strip()
        inst = _clean_institution(aff_raw)
        if not inst:
            continue
        for auth_raw in str(row[col_authors]).split(";"):
            auth = auth_raw.split("(")[0].strip()
            if auth:
                author_aff_history.setdefault(auth, []).append(inst)

    author_aff_map: dict[str, str] = {}
    for auth, affs in author_aff_history.items():
        if affs:
            author_aff_map[auth] = Counter(affs).most_common(1)[0][0]

    # --- Build per-category author stats ---
    stats_by_cat: dict[str, pd.DataFrame] = {}

    for cat in CATEGORY_ORDER:
        df_cat = df_work[df_work["Tech_Category"] == cat]
        auth_data: dict[str, list[float]] = {}

        for _, row in df_cat.iterrows():
            for auth_raw in str(row[col_authors]).split(";"):
                auth = auth_raw.split("(")[0].strip()
                if auth:
                    auth_data.setdefault(auth, []).append(row[col_cit])

        records = []
        for auth, cites in auth_data.items():
            records.append({
                "Author": auth,
                "Affiliation": author_aff_map.get(auth, "Unknown"),
                "Pubs": len(cites),
                "Citations": int(sum(cites)),
                "h_index": _h_index(cites),
                "Category": cat,
            })

        stats_by_cat[cat] = pd.DataFrame(records) if records else pd.DataFrame(
            columns=["Author", "Affiliation", "Pubs", "Citations", "h_index", "Category"]
        )

    # --- Save full table ---
    df_all_authors = pd.concat(stats_by_cat.values(), ignore_index=True)
    df_all_authors.to_csv(results_dir / "table_A3_author_stats.csv", index=False)
    print(f"  → Saved: {results_dir / 'table_A3_author_stats.csv'}")

    # --- Plotting function ---
    def _plot_metric(metric: str, label: str, filename: str) -> None:
        global_max = max(
            (df_r[metric].max() for df_r in stats_by_cat.values() if not df_r.empty and metric in df_r.columns),
            default=1,
        )
        x_limit = global_max * 1.25

        n_cats = len(CATEGORY_ORDER)
        fig, axes = plt.subplots(n_cats, 1, figsize=(11.69, n_cats * 4.1), facecolor="white")
        for i, cat in enumerate(CATEGORY_ORDER):
            ax = axes[i]
            ax.set_facecolor("white")
            color = CATEGORY_COLORS[cat]
            df_r = stats_by_cat[cat]
            if df_r.empty:
                ax.set_title(f"{cat} (no data)", fontsize=10, color=color)
                ax.axis("off")
                continue

            top10 = df_r.sort_values(metric, ascending=False).head(10)
            labels = [f"{r['Author']}  |  {r['Affiliation']}" for _, r in top10.iterrows()]
            bars = ax.barh(range(len(top10)), top10[metric].values, color=color, alpha=0.88,
                           edgecolor="black", height=0.65, linewidth=1.2)
            ax.set_yticks(range(len(top10)))
            ax.set_yticklabels(labels, fontsize=8.5, fontweight="bold")
            ax.set_title(cat, fontsize=11, fontweight="bold", loc="center", color=color, pad=6)
            ax.invert_yaxis()
            ax.set_xlim(0, x_limit)
            ax.grid(axis="x", linestyle="--", alpha=0.45)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            for bar in bars:
                w = bar.get_width()
                ax.text(
                    w + x_limit * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{int(w)}", va="center", fontweight="bold", fontsize=9,
                    path_effects=[PathEffects.withStroke(linewidth=2, foreground="white")],
                )
        axes[-1].set_xlabel(label, fontsize=11, fontweight="bold", labelpad=10)
        plt.subplots_adjust(left=0.42, right=0.97, top=0.96, bottom=0.04, hspace=0.38)
        save_figure(fig, str(results_dir / filename))

    _plot_metric("Pubs", "Number of Publications", "fig_A3_top_authors_pubs.png")
    _plot_metric("Citations", "Total Citations", "fig_A3_top_authors_cit.png")
    _plot_metric("h_index", "H-index", "fig_A3_top_authors_hindex.png")

    print("[A3] Done.")
