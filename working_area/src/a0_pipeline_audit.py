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
    fig, ax = plt.subplots(figsize=(15, 7), facecolor="white")
    ax.axis("off")
    colors = ["#27364B", "#3D5A80", "#4F7CAC", "#6C9BCF", "#8DB5D9",
              "#B6CCE2", "#D6A756", "#3A8D7A"]
    y_positions = list(range(len(audit) - 1, -1, -1))
    for i, (_, row) in enumerate(audit.iterrows()):
        y = y_positions[i]
        label = row["Stage"].replace("_", " ").title()
        text = f"{label}\n{int(row['Records after']):,} records"
        ax.text(0.05, y, text, ha="left", va="center", color="white",
                fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.65", facecolor=colors[i % len(colors)],
                          edgecolor="none"))
        if i < len(audit) - 1:
            removed = int(audit.iloc[i + 1]["Removed"])
            ax.annotate(f"−{removed:,}", xy=(0.5, y - 0.25), xytext=(0.5, y - 0.72),
                        ha="center", va="center", fontsize=10, color="#5B6573",
                        arrowprops=dict(arrowstyle="-|>", color="#9AA6B2", lw=1.5))

    ax.text(0.5, len(audit) - 0.25,
            "Corpus construction audit trail",
            ha="center", va="bottom", fontsize=18, fontweight="bold", color="#172033")
    ax.text(0.5, len(audit) - 0.58,
            "Each arrow shows the number of records removed at the next stage",
            ha="center", va="bottom", fontsize=10, color="#5B6573")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.8, len(audit) + 0.5)
    save_figure(fig, str(results_dir / "fig_A0_pipeline_audit.png"))

