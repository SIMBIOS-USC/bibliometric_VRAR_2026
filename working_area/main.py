"""
main.py
=======
XR Bibliometrics Analysis Pipeline — Master Orchestrator
=========================================================

Project structure
-----------------
xr_bibliometrics/
├── scopus_raw.csv              ← Raw Scopus export (INPUT — do not modify)
├── results/                    ← All figures and tables (OUTPUT)
│   ├── pipeline_report.csv     ← Stage-by-stage record counts (audit trail)
│   ├── classified_corpus.csv   ← Cleaned & classified full dataset
│   ├── fig_A1_*.png / table_A1_*.csv  ← Publication growth analysis
│   ├── fig_A2_*.png / table_A2_*.csv  ← Orientation analysis
│   ├── fig_A3_*.png / table_A3_*.csv  ← Author analysis
│   ├── fig_A4_*.png / table_A4_*.csv  ← Country collaboration
│   ├── fig_A5_*.png / table_A5_*.csv  ← Bradford zones
│   └── table_A6_*.csv / .md           ← Main bibliometric table
└── working_area/
    ├── main.py                 ← THIS FILE
    ├── utils/
    │   ├── classifier.py       ← CANONICAL technology + orientation classifiers
    │   ├── preprocess.py       ← Scopus cleaning pipeline
    │   └── plot_style.py       ← Shared Matplotlib style
    └── src/
        ├── a1_publication_growth.py
        ├── a2_orientation_analysis.py
        ├── a3_author_analysis.py
        ├── a4_country_collaboration.py
        ├── a5_bradford_zones.py
        └── a6_bibliometric_table.py

Reviewer issues addressed
-------------------------
Issue 1 — Classification inconsistency (VR+AR → MR/XR bug):
    ALL modules now call utils/classifier.classify_technology().
    The canonical function implements the MANUSCRIPT definition:
      VR+AR without MR/XR terms → 'Hybrid/Multi-technology'  (NOT 'MR/XR')

Issue 2 — No single canonical classifier:
    utils/classifier.py is the single source of truth.
    Keyword dictionaries, field combinations, and priority logic are
    defined ONCE and imported by all modules.

Issue 3 — Missing technical vs pedagogical classifier:
    utils/classifier.classify_orientation() now implements this.
    Module a2_orientation_analysis.py uses it to reproduce Figure 2 / RQ4.

Issue 4 — Incomplete deduplication pipeline:
    utils/preprocess.run_pipeline() implements all stages:
      Stage 0: Raw load
      Stage 1: Column normalization
      Stage 2: Exact duplicate removal
      Stage 3: DOI-based deduplication
      Stage 4: Exact normalized-title deduplication (non-empty titles)
      Stage 5: Essential metadata filter
      Stage 6: Thematic screening
      Stage 7: Year-window filter
    The full stage-by-stage count is saved to results/pipeline_report.csv.

Issue 5 — Reproducibility claims without a master workflow:
    THIS FILE is the master workflow. Running `python main.py` reproduces
    all analyses in a single call. All outputs go to results/.

Issue 6 — 'immersive' alone triggering VR:
    'immersive' is intentionally excluded from VR_TERMS in classifier.py.
    Explicit VR terms trigger VR; hardware is not required when technology is named.

Issue 7 — MR/XR terminological mixing:
    The keyword dictionary in classifier.py is explicit and documented.
    'metaverse', 'digital twin', 'passthrough' are intentionally excluded.
    The grouping is terminological (not ontological), as stated in the manuscript.

Usage
-----
    python main.py [--input PATH] [--results-dir PATH] [--skip MODULES]

    Defaults:
      --input       ../scopus_raw.csv
      --results-dir ../../results
      --skip        (none)

    Examples:
      python main.py
      python main.py --input /path/to/custom.csv
      python main.py --skip a3,a5          # skip author and bradford analyses

Dependencies
------------
    pip install pandas numpy matplotlib seaborn tabulate
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Ensure working_area is in the Python path so `from utils...` works
# regardless of where the script is invoked from.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from utils.preprocess import run_pipeline, print_pipeline_report
from utils.classifier import add_classifications, print_classification_report
from utils.plot_style import apply_global_style

import src.a1_publication_growth    as a1
import src.a0_pipeline_audit        as a0
import src.a2b_hardware_triggers    as a2b
import src.a3_author_analysis       as a3
import src.a4_country_collaboration as a4
import src.a4b_scp_mcp_figures      as a4b
import src.a5_bradford_zones        as a5
import src.a5b_keyword_cooccurrence as a5b
import src.a5c_semantic_landscape   as a5c
import src.a6_bibliometric_table    as a6
import src.a7_keywords_comparison   as a7
import src.a8_network_graphs        as a8
import src.a9_additional_outputs    as a9
import src.a10_classification_audit as a10
import src.a11_education_eligibility as a11


# ---------------------------------------------------------------------------
# DEFAULT PATHS (relative to working_area/)
# ---------------------------------------------------------------------------
DEFAULT_INPUT = _HERE.parent / "scopus_raw.csv"
DEFAULT_RESULTS = _HERE.parent / "results"


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _banner(msg: str) -> None:
    """Print a prominent section banner to stdout."""
    width = 70
    print("\n" + "═" * width)
    print(f"  {msg}")
    print("═" * width)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="XR Bibliometrics Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"Path to raw Scopus CSV. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS,
        help=f"Output directory for figures and tables. Default: {DEFAULT_RESULTS}",
    )
    parser.add_argument(
        "--skip", type=str, default="",
        help="Comma-separated module IDs to skip (e.g. 'a3,a5'). "
             "Valid IDs: a1, a2b, a3, a4, a4b, a5, a5b, a5c, a6, a7, a8, a9, a10",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()

    input_path: Path = args.input
    results_dir: Path = args.results_dir
    skip_modules = {s.strip().lower() for s in args.skip.split(",") if s.strip()}

    results_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # -----------------------------------------------------------------------
    # STEP 1: SCOPUS CLEANING PIPELINE
    # -----------------------------------------------------------------------
    _banner("STEP 1 — Scopus Cleaning Pipeline")

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    print(f"  Input  : {input_path}")
    print(f"  Results: {results_dir}\n")

    df_clean, pipeline_report = run_pipeline(input_path)
    print_pipeline_report(pipeline_report)

    # -----------------------------------------------------------------------
    # STEP 2: CLASSIFICATION
    # -----------------------------------------------------------------------
    _banner("STEP 2 — Technology & Orientation Classification")

    df_all_classifications = add_classifications(df_clean)
    df_screened = a11.run(df_all_classifications, results_dir)
    df_with_classifications = df_screened[
        df_screened["Education_Screen_Disposition"].eq("include_algorithm")
    ].copy()
    print_classification_report(df_with_classifications)
    a10.run(df_with_classifications, results_dir)
    # Keep education eligibility and subsequent technology eligibility as
    # separate, deterministic corpus-construction stages.
    n_before_education_screen = len(df_clean)
    n_after_education_screen = len(df_with_classifications)
    pipeline_report.append({
        "stage": "8_education_eligibility_algorithm",
        "records": n_after_education_screen,
        "dropped": n_before_education_screen - n_after_education_screen,
        "note": "Binary deterministic title + abstract + author-keyword lexical rule; see A11 trace and algorithm specification",
    })
    # The analytical corpus then contains records assigned to a canonical
    # technology category. Every figure uses this final denominator.
    n_before_classification_screen = len(df_with_classifications)
    df_classified = df_with_classifications[df_with_classifications["Tech_Category"].notna()].copy()
    pipeline_report.append({
        "stage": "9_classification_eligibility",
        "records": len(df_classified),
        "dropped": n_before_classification_screen - len(df_classified),
        "note": "Records assigned to one canonical technology category",
    })
    # Save the complete audit trail, including the explicit classification
    # eligibility step that defines the final analytical denominator.
    df_report = pd.DataFrame(pipeline_report)
    df_report.to_csv(results_dir / "pipeline_report.csv", index=False)
    print(f"  → Saved: {results_dir / 'pipeline_report.csv'}")
    a0.run(pipeline_report, results_dir)

    # Save classified corpus
    df_classified.to_csv(results_dir / "classified_corpus.csv", index=False)
    print(f"  → Saved: {results_dir / 'classified_corpus.csv'}")

    # -----------------------------------------------------------------------
    # STEP 3: ANALYSIS MODULES
    # -----------------------------------------------------------------------
    _banner("STEP 3 — Analysis Modules")

    modules = [
        ("a1",  "Publication Growth & Citation Evolution",                a1.run),
        ("a2b", "Tech & Ped + Hardware Triggers (4-panel)",               a2b.run),
        ("a3",  "Top Author Analysis",                                    a3.run),
        ("a4",  "Country Collaboration (SCP vs MCP)",                     a4.run),
        ("a4b", "SCP/MCP Canonical Figures (braces + pie)",              a4b.run),
        ("a5",  "Bradford's Law & Source Analysis",                       a5.run),
        ("a5b", "Keyword Co-occurrence Matrix (Top 10)",                  a5b.run),
        ("a5c", "Bibliometric Semantic Landscape",                        a5c.run),
        ("a6",  "Main Bibliometric Information Table",                    a6.run),
        ("a7",  "Author vs Index Keywords Comparison",                    a7.run),
        ("a8",  "Network Graphs (Keywords & Authors)",                    a8.run),
        ("a9",  "Additional Manuscript Tables & Figures",                  a9.run),
    ]

    for module_id, module_name, module_fn in modules:
        if module_id in skip_modules:
            print(f"\n  [SKIP] {module_id.upper()} — {module_name}")
            continue
        try:
            # raw_csv is retained only for backwards-compatible function
            # signatures; all modules consume the canonical corpus.
            if module_id in ("a4b", "a5b", "a5c", "a7"):
                module_fn(df_classified, results_dir, raw_csv=input_path)
            else:
                module_fn(df_classified, results_dir)
        except Exception as exc:
            print(f"\n  [ERROR] Module {module_id.upper()} failed: {exc}")
            import traceback
            traceback.print_exc()

    # -----------------------------------------------------------------------
    # DONE
    # -----------------------------------------------------------------------
    elapsed = time.time() - t_start
    _banner(f"ALL ANALYSES COMPLETE  ({elapsed:.1f}s)")
    print(f"  Results saved to: {results_dir}\n")


if __name__ == "__main__":
    main()
