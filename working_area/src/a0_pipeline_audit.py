"""Audit table and flow figure for the reproducible corpus definition."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils.plot_style import apply_global_style, save_figure


def run(report: list[dict], results_dir: Path) -> None:
    """Write a stage-by-stage audit table and a compact flow figure."""
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    previous = None
    for item in report:
        n_after = int(item.get("records", 0))
        dropped = int(item.get("dropped", 0) or 0)
        n_before = n_after + dropped if previous is None else previous
        rows.append({
            "Stage": item.get("stage", ""),
            "Records before": n_before,
            "Removed": dropped,
            "Records after": n_after,
            "Criterion": item.get("note", ""),
        })
        previous = n_after

    audit = pd.DataFrame(rows)
    audit.to_csv(results_dir / "table_A0_pipeline_audit.csv", index=False)
    audit.to_markdown(results_dir / "table_A0_pipeline_audit.md", index=False)

    apply_global_style(font_size=11)
    spacing = 1.35
    y_positions = [(len(audit) - 1 - i) * spacing for i in range(len(audit))]
    fig_height = max(8, len(audit) * 1.15)
    fig, ax = plt.subplots(figsize=(12, fig_height), facecolor="white")
    ax.axis("off")
    colors = ["#27364B", "#3D5A80", "#4F7CAC", "#6C9BCF", "#8DB5D9",
              "#B6CCE2", "#D6A756", "#3A8D7A"]
    for i, (_, row) in enumerate(audit.iterrows()):
        y = y_positions[i]
        label = {
            "0_raw_load": "Raw Scopus records",
            "1_col_standardize": "Normalize column names",
            "2_exact_dedup": "Remove exact duplicates",
            "3_doi_dedup": "Deduplicate by DOI",
            "4_title_dedup": "Deduplicate by title, year, and first author",
            "5_missing_metadata": "Remove incomplete essential metadata",
            "6_thematic_screen": "XR thematic screen",
            "7_year_filter": "Publication-year window",
            "8_education_eligibility_algorithm": "Education eligibility",
            "9_classification_eligibility": "Technology classification eligibility",
        }.get(row["Stage"], row["Stage"].replace("_", " ").title())
        text = f"{label}\n{int(row['Records after']):,} records"
        ax.text(0.46, y, text, ha="center", va="center", color="white",
                fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.65", facecolor=colors[i % len(colors)],
                          edgecolor="none"))
        if i < len(audit) - 1:
            removed = int(audit.iloc[i + 1]["Removed"])
            next_y = y_positions[i + 1]
            ax.annotate("", xy=(0.46, next_y + 0.40), xytext=(0.46, y - 0.40),
                        arrowprops=dict(arrowstyle="-|>", color="#9AA6B2", lw=1.5))
            ax.text(0.62, (y + next_y) / 2, f"−{removed:,} removed",
                    ha="left", va="center", fontsize=9, color="#5B6573")

    top = y_positions[0] if y_positions else 0
    ax.text(0.5, top + 0.82,
            "Corpus construction audit trail",
            ha="center", va="bottom", fontsize=18, fontweight="bold", color="#172033")
    ax.text(0.5, top + 0.48,
            "Each arrow connects consecutive stages; labels show records removed",
            ha="center", va="bottom", fontsize=10, color="#5B6573")
    ax.set_xlim(0.04, 0.96)
    ax.set_ylim(y_positions[-1] - 0.75 if y_positions else -0.75, top + 1.3)
    save_figure(fig, str(results_dir / "fig_A0_pipeline_audit.png"))
