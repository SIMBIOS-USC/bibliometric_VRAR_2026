"""Summarize common author keywords overall and by technology category.

A fuzzy normalization map is built from the full corpus and reused in each
category panel. The output includes the top ten terms for each group.
"""

from __future__ import annotations
import itertools
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

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
    'simulation':             'simulation',
}

# Figure colors
GLOBAL_COLOR = '#1a1a2e'
VR_COLOR     = '#4a90d9'
AR_COLOR     = '#e8773a'
MR_COLOR     = '#3ab795'
XR_COLOR     = '#2E7D32'
HYBRID_COLOR = '#9b59b6'

CATEGORY_ORDER = ['VR', 'AR', 'MR', 'XR', 'Hybrid/Multi-technology']


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


def _count_from_papers(cleaned_papers: list[list[str]],
                       top_n: int = 10) -> pd.DataFrame:
    all_kws = list(itertools.chain(*cleaned_papers))
    counts = Counter(all_kws)
    return pd.DataFrame(counts.most_common(top_n), columns=['Keyword', 'Count'])


# ─────────────────────────────────────────────────────────────────────────────
def run(df_classified: pd.DataFrame, results_dir: Path) -> None:
    """
    Execute the Bibliometric Semantic Landscape figure.

    Uses the canonical cleaned and classified corpus from the master pipeline.

    Parameters
    ----------
    df_classified : pd.DataFrame  — canonical corpus with Tech_Category
    results_dir   : Path          — output directory
    """
    print("\n[A5c] Bibliometric Semantic Landscape")
    print("-" * 50)
    results_dir.mkdir(parents=True, exist_ok=True)

    if "Tech_Category" not in df_classified.columns:
        print("  [ERROR] Tech_Category column not found.")
        return
    df_cl = df_classified[df_classified['Tech_Category'].notna()].copy()
    df_cl['Tech'] = df_cl['Tech_Category']
    print(f"  Classified records: {len(df_cl):,}")

    # ── Build fuzzy map from ALL classified papers ────────────────────────────
    print("  Constructing fuzzy map...")
    all_raw   = _extract_papers_kws(df_cl)
    fuzzy_map = _build_fuzzy_map(all_raw)
    all_clean = _clean_papers_kws(all_raw, fuzzy_map)

    # ── Global top-10 ─────────────────────────────────────────────────────────
    top_global = _count_from_papers(all_clean, top_n=10)
    print(f"  Top 10 global keywords: {list(top_global['Keyword'])}")

    # ── Per-category ─────────────────────────────────────────────────────────
    def count_classified(df_sub: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        raw     = _extract_papers_kws(df_sub)
        cleaned = _clean_papers_kws(raw, fuzzy_map)
        return _count_from_papers(cleaned, top_n=top_n)

    top_vr = count_classified(df_cl[df_cl['Tech'] == 'VR'])
    top_ar = count_classified(df_cl[df_cl['Tech'] == 'AR'])
    top_mr = count_classified(df_cl[df_cl['Tech'] == 'MR'])
    top_xr = count_classified(df_cl[df_cl['Tech'] == 'XR'])
    top_hy = count_classified(df_cl[df_cl['Tech'] == 'Hybrid/Multi-technology'])

    # Save tables
    top_global.to_csv(results_dir / "table_A5c_global_keywords.csv", index=False)
    cat_rows = []
    for cat, df_t in [('VR', top_vr), ('AR', top_ar),
                      ('MR', top_mr), ('XR', top_xr),
                      ('Hybrid/Multi-technology', top_hy)]:
        for _, row in df_t.iterrows():
            cat_rows.append({'category': cat, 'keyword': row['Keyword'],
                             'count': row['Count']})
    pd.DataFrame(cat_rows).to_csv(
        results_dir / "table_A5c_category_keywords.csv", index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Overall and category keyword panels.
    # ─────────────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    fig = plt.figure(figsize=(20, 11.69), facecolor='white')  # wider for 5 panels
    outer     = gridspec.GridSpec(2, 1, figure=fig,
                                  height_ratios=[1.3, 1.0], hspace=0.45)
    inner_top = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0])
    inner_bot = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[1],
                                                 wspace=0.85)

    # ── Panel Global ──────────────────────────────────────────────────────────
    ax_g = fig.add_subplot(inner_top[0])
    bars_g = ax_g.barh(top_global['Keyword'][::-1], top_global['Count'][::-1],
                       color=GLOBAL_COLOR, edgecolor='white',
                       linewidth=0.5, height=0.7)
    ax_g.set_facecolor('white')
    ax_g.set_title("GLOBAL AUTHOR KEYWORDS  (1991–2025)",
                   fontsize=20, fontweight='bold', color=GLOBAL_COLOR,
                   pad=16, loc='left')
    ax_g.set_xlabel("Number of papers", fontsize=13, color='#555')
    ax_g.tick_params(axis='y', labelsize=13)
    ax_g.tick_params(axis='x', labelsize=11)
    for bar, val in zip(bars_g, top_global['Count'][::-1]):
        ax_g.text(val + 20, bar.get_y() + bar.get_height() / 2,
                  f'{val:,}', va='center', fontsize=10,
                  color=GLOBAL_COLOR, fontweight='bold')
    ax_g.set_xlim(0, top_global['Count'].max() * 1.18)
    ax_g.grid(axis='x', linestyle='--', alpha=0.4)

    # ── Subplots por tecnología ───────────────────────────────────────────────
    subplot_data = [
        (top_vr, VR_COLOR,     'Virtual Reality'),
        (top_ar, AR_COLOR,     'Augmented Reality'),
        (top_mr, MR_COLOR,     'Mixed Reality'),
        (top_xr, XR_COLOR,     'Extended Reality'),
        (top_hy, HYBRID_COLOR, 'Hybrid/Multi-tech'),
    ]

    for i, (top_t, color, title) in enumerate(subplot_data):
        ax = fig.add_subplot(inner_bot[i])
        if top_t.empty:
            ax.set_visible(False)
            continue
        bars_t = ax.barh(top_t['Keyword'][::-1], top_t['Count'][::-1],
                         color=color, edgecolor='white',
                         linewidth=0.4, height=0.7)
        ax.set_facecolor('white')
        ax.set_title(title, fontsize=12, fontweight='bold',
                     color=color, pad=12, loc='left')
        ax.tick_params(axis='y', labelsize=9)
        ax.tick_params(axis='x', labelsize=8)
        for bar, val in zip(bars_t, top_t['Count'][::-1]):
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{val:,}', va='center', fontsize=8, color='#333')
        max_val = top_t['Count'].max()
        ax.set_xlim(0, max_val * 1.25 if max_val > 0 else 1)
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        ax.set_xlabel("Papers", fontsize=10, color='#555')

    fig.suptitle("Bibliometric Semantic Landscape",
                 fontsize=26, fontweight='bold', color=GLOBAL_COLOR, y=1.01)

    out = str(results_dir / "fig_A5c_semantic_landscape.png")
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  → Saved: {out}")
    print("[A5c] Done.")
