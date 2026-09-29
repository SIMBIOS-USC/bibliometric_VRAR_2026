"""Count keyword co-occurrence in the classified corpus.

The matrix diagonal contains keyword document frequencies; off-diagonal cells
count documents containing both terms. Manual normalization is followed by
fuzzy matching among the 200 most common terms (similarity threshold 90).
"""

from __future__ import annotations
import itertools
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plot_style import save_figure

# Keyword normalization rules
MANUAL_MAP = {
    'virtual reality (vr)':   'virtual reality',
    'vr':                     'virtual reality',
    'augmented reality (ar)': 'augmented reality',
    'ar':                     'augmented reality',
    'elearning':              'e-learning',
    'e learning':             'e-learning',
    'electronic learning':    'e-learning',
    'student':                'students',
    'mixed reality (mr)':     'mixed reality',
    'mr':                     'mixed reality',
    'extended reality (xr)':  'extended reality',
    'xr':                     'extended reality',
    'computer aided instruction': 'computer-aided instruction',
    'learning system':        'learning systems',
    'simulation':             'simulations',
}


def _extract_papers_kws(df_sub: pd.DataFrame) -> list[list[str]]:
    papers = []
    for kw_str in df_sub['Author Keywords'].dropna():
        kws = [k.strip().lower() for k in str(kw_str).split(';') if len(k.strip()) > 2]
        if kws:
            papers.append(kws)
    return papers


def _build_fuzzy_map(raw_papers: list[list[str]],
                     top_n_candidates: int = 200) -> dict[str, str]:
    from thefuzz import process, fuzz
    all_flat = list(itertools.chain(*raw_papers))
    mapped = [MANUAL_MAP.get(k, k) for k in all_flat]
    counts = Counter(mapped)
    top_candidates = [w for w, _ in counts.most_common(top_n_candidates)]

    fuzzy_map: dict[str, str] = {}
    processed: set[str] = set()
    for term in top_candidates:
        if term in processed:
            continue
        matches = process.extract(term, top_candidates, limit=10,
                                  scorer=fuzz.ratio)
        primary = term
        group: list[str] = []
        for match_term, score in matches:
            if score >= 90:
                group.append(match_term)
                if len(match_term) > len(primary) or '-' in match_term:
                    primary = match_term
        for g in group:
            fuzzy_map[g] = primary
            processed.add(g)
    return fuzzy_map


def _clean_papers_kws(raw_papers: list[list[str]],
                      fuzzy_map: dict[str, str]) -> list[list[str]]:
    cleaned = []
    for paper in raw_papers:
        new_paper: list[str] = []
        for k in paper:
            k1 = MANUAL_MAP.get(k, k)
            k2 = fuzzy_map.get(k1, k1)
            if k2 not in new_paper:
                new_paper.append(k2)
        cleaned.append(new_paper)
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
def run(df_classified: pd.DataFrame, results_dir: Path) -> None:
    """
    Execute the Keyword Co-occurrence Matrix figure on the canonical corpus.

    Parameters
    ----------
    df_classified : pd.DataFrame  — canonical corpus with Tech_Category
    results_dir   : Path          — output directory
    """
    print("\n[A5b] Keyword Co-occurrence Matrix")
    print("-" * 50)
    results_dir.mkdir(parents=True, exist_ok=True)

    if "Tech_Category" not in df_classified.columns:
        print("  [ERROR] Tech_Category column not found.")
        return
    df_cl = df_classified[df_classified['Tech_Category'].notna()].copy()
    df_cl['Tech'] = df_cl['Tech_Category']
    print(f"  Classified records: {len(df_cl):,}")

    # Build fuzzy map from ALL classified papers
    print("  Building fuzzy normalisation map...")
    all_raw   = _extract_papers_kws(df_cl)
    fuzzy_map = _build_fuzzy_map(all_raw)
    all_clean = _clean_papers_kws(all_raw, fuzzy_map)

    # Keyword frequencies + top-10
    all_kws       = list(itertools.chain(*all_clean))
    keyword_counts = Counter(all_kws)
    top_n         = 10
    top_keywords  = [k for k, _ in keyword_counts.most_common(top_n)]
    print(f"  Top {top_n} keywords: {top_keywords}")

    # ── Build co-occurrence matrix ────────────────────────────────────────────
    matrix = np.zeros((len(top_keywords), len(top_keywords)))

    for paper in all_clean:
        present = [k for k in paper if k in top_keywords]
        for i in range(len(present)):
            for j in range(len(present)):
                if i != j:
                    idx1 = top_keywords.index(present[i])
                    idx2 = top_keywords.index(present[j])
                    matrix[idx1, idx2] += 1

    # Diagonal = self-frequency
    for i, k in enumerate(top_keywords):
        matrix[i, i] = keyword_counts[k]

    cooccurrence_df = pd.DataFrame(matrix,
                                   index=top_keywords,
                                   columns=top_keywords)
    cooccurrence_df.to_csv(results_dir / "table_A5b_cooccurrence_matrix.csv")
    print(f"  → Saved: {results_dir / 'table_A5b_cooccurrence_matrix.csv'}")

    # Plot the keyword co-occurrence matrix.
    plt.figure(figsize=(14, 12), facecolor='white')
    ax = sns.heatmap(
        cooccurrence_df,
        annot=True,
        annot_kws={"size": 14, "weight": "bold"},
        cmap="YlGnBu",
        fmt=".0f",
        linewidths=1,
        linecolor='white',
        cbar_kws={"label": "Co-occurrence Frequency"},
    )
    plt.xticks(fontsize=14, rotation=45, ha='right')
    plt.yticks(fontsize=14, rotation=0)
    plt.title("Keyword Co-occurrence Matrix (Top Terms)",
              fontsize=22, pad=20, fontweight='bold')
    plt.tight_layout()

    out = str(results_dir / "fig_A5b_keyword_cooccurrence.png")
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  → Saved: {out}")
    print("[A5b] Done.")
