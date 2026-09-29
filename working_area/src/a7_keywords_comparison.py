"""
src/a7_keywords_comparison.py
==============================
Analysis Module 7: Author vs Index Keyword Frequencies
------------------------------------------------------
Generates a comparative table of the most frequent Author Keywords and Index
Keywords in the canonical analytical corpus.

Author keywords use the module's manual and fuzzy normalization rules. Index
keywords use the manual normalization defined below.

Output:
  - table_A7_keywords_comparison.csv
  - table_A7_keywords_comparison.md
"""

import sys
import itertools
from pathlib import Path
from collections import Counter
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

# Normalisation for Author Keywords (matching a5b/a5c)
MANUAL_MAP_AUTHOR = {
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

# Index keyword normalization
MANUAL_MAP_INDEX = {
    'human': 'humans',
}

def extract_author_kws(df: pd.DataFrame, fuzzy_map: dict = None) -> list:
    kws_list = []
    for kw_str in df['Author Keywords'].dropna():
        kws = [k.strip().lower() for k in str(kw_str).split(';') if len(k.strip()) > 2]
        for k in kws:
            k1 = MANUAL_MAP_AUTHOR.get(k, k)
            if fuzzy_map:
                k1 = fuzzy_map.get(k1, k1)
            kws_list.append(k1)
    return kws_list

def extract_index_kws(df: pd.DataFrame) -> list:
    kws_list = []
    for kw_str in df['Index Keywords'].dropna():
        kws = [k.strip().lower() for k in str(kw_str).split(';') if len(k.strip()) > 2]
        for k in kws:
            k1 = MANUAL_MAP_INDEX.get(k, k)
            kws_list.append(k1)
    return kws_list

def run(df_classified: pd.DataFrame, results_dir: Path) -> None:
    print("\n[A7] Author vs Index Keywords Comparison")
    print("-" * 50)
    
    if "Tech_Category" not in df_classified.columns:
        print("  [ERROR] Tech_Category column not found.")
        return
    df_cl = df_classified[df_classified['Tech_Category'].notna()].copy()
    df_cl['Tech'] = df_cl['Tech_Category']
    
    # Author Keywords (need fuzzy map from a5b logic to match exact numbers)
    all_raw_papers = []
    for kw_str in df_cl['Author Keywords'].dropna():
        all_raw_papers.append([k.strip().lower() for k in str(kw_str).split(';') if len(k.strip()) > 2])
    
    # We apply MANUAL_MAP first
    all_flat = list(itertools.chain(*all_raw_papers))
    mapped = [MANUAL_MAP_AUTHOR.get(k, k) for k in all_flat]
    
    from thefuzz import process, fuzz
    counts = Counter(mapped)
    top_candidates = [w for w, _ in counts.most_common(200)]
    
    fuzzy_map = {}
    processed = set()
    for term in top_candidates:
        if term in processed: continue
        matches = process.extract(term, top_candidates, limit=10, scorer=fuzz.ratio)
        primary = term
        group = []
        for match_term, score in matches:
            if score >= 90:
                group.append(match_term)
                if len(match_term) > len(primary) or '-' in match_term:
                    primary = match_term
        for g in group:
            fuzzy_map[g] = primary
            processed.add(g)
            
    # Apply fuzzy map to flattened list
    final_author_kws = []
    for paper in all_raw_papers:
        # We use a set per paper so we don't count the same keyword twice in a single paper (co-occurrence rule)
        paper_kws = set()
        for k in paper:
            k1 = MANUAL_MAP_AUTHOR.get(k, k)
            k2 = fuzzy_map.get(k1, k1)
            paper_kws.add(k2)
        final_author_kws.extend(list(paper_kws))
        
    author_counts = Counter(final_author_kws)
    
    # Index Keywords (also counted on classified papers)
    final_index_kws = extract_index_kws(df_cl)
    index_counts = Counter(final_index_kws)
    
    top_10_author = author_counts.most_common(10)
    top_10_index = index_counts.most_common(10)
    
    # Build dataframe
    df_compare = pd.DataFrame({
        'Author Keywords': [k for k, v in top_10_author],
        'Freq (Auth)': [v for k, v in top_10_author],
        'Index Keywords': [k for k, v in top_10_index],
        'Freq (Index)': [v for k, v in top_10_index]
    })
    
    csv_path = results_dir / 'table_A7_keywords_comparison.csv'
    md_path = results_dir / 'table_A7_keywords_comparison.md'
    
    df_compare.to_csv(csv_path, index=False)
    with open(md_path, 'w') as f:
        f.write(df_compare.to_markdown(index=False))
        
    print(f"  → Saved: {csv_path}")
    print(df_compare.to_markdown(index=False))
    print("[A7] Done.")
