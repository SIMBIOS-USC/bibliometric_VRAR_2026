"""Run the XR bibliometric corpus-construction and analysis pipeline.

Usage from the repository root::

    python working_area/main.py [--input PATH] [--results-dir PATH]
                                 [--skip MODULES]

The default input is ``scopus_raw.csv`` and the default output directory is
``results/``. Corpus construction and its audit steps always run; ``--skip``
applies only to the optional analysis modules.
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
SKIPPABLE_MODULE_IDS = {
    "a1", "a2b", "a3", "a4", "a4b", "a5", "a5b", "a5c", "a6", "a7", "a8", "a9",
}


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
        help="Comma-separated analysis module IDs to skip (e.g. 'a3,a5'). "
             "Valid IDs: " + ", ".join(sorted(SKIPPABLE_MODULE_IDS)),
    )
    args = parser.parse_args()
    requested = {s.strip().lower() for s in args.skip.split(",") if s.strip()}
    unknown = requested - SKIPPABLE_MODULE_IDS
    if unknown:
        parser.error(f"unknown --skip module ID(s): {', '.join(sorted(unknown))}")
    return args


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
        ("a4b", "SCP/MCP Figures",                                      a4b.run),
        ("a5",  "Bradford's Law & Source Analysis",                       a5.run),
        ("a5b", "Keyword Co-occurrence Matrix (Top 10)",                  a5b.run),
        ("a5c", "Bibliometric Semantic Landscape",                        a5c.run),
        ("a6",  "Main Bibliometric Information Table",                    a6.run),
        ("a7",  "Author vs Index Keywords Comparison",                    a7.run),
        ("a8",  "Network Graphs (Keywords & Authors)",                    a8.run),
        ("a9",  "Additional Tables & Figures",                             a9.run),
    ]

    module_failures: list[tuple[str, str]] = []
    for module_id, module_name, module_fn in modules:
        if module_id in skip_modules:
            print(f"\n  [SKIP] {module_id.upper()} — {module_name}")
            continue
        try:
            module_fn(df_classified, results_dir)
        except Exception as exc:
            print(f"\n  [ERROR] Module {module_id.upper()} failed: {exc}")
            module_failures.append((module_id, str(exc)))
            import traceback
            traceback.print_exc()

    # -----------------------------------------------------------------------
    # DONE
    # -----------------------------------------------------------------------
    elapsed = time.time() - t_start
    if module_failures:
        _banner(f"ANALYSIS RUN FAILED  ({elapsed:.1f}s)")
        for module_id, message in module_failures:
            print(f"  {module_id.upper()}: {message}")
        print(f"  Partial results may be present in: {results_dir}\n")
        sys.exit(1)

    _banner(f"ALL ANALYSES COMPLETE  ({elapsed:.1f}s)")
    print(f"  Results saved to: {results_dir}\n")


if __name__ == "__main__":
    main()
