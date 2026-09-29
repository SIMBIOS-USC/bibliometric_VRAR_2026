"""
utils/preprocess.py
===================
Scopus CSV Cleaning & Deduplication Pipeline.

This module implements the reproducible cleaning pipeline from the raw Scopus
export. Final corpus size is determined by the current input and is written to
the stage-by-stage audit outputs; it is not assumed to be a fixed number.

The pipeline stages are:
  Stage 0: Load raw CSV
  Stage 1: Standardize column names (strip BOM, trailing spaces)
  Stage 2: Drop exact duplicates (pandas drop_duplicates on all columns)
  Stage 3: DOI-based deduplication (same normalized DOI → keep richest record)
  Stage 4: Conservative exact-title matching using year and first author
  Stage 5: Remove records with missing essential metadata
  Stage 6: Thematic screening (apply XR keyword filter to retain only XR records)
  Stage 7: Year-window filter (1991–2025)

Each stage logs its record count to reproduce the PRISMA-style flow chart.

Usage
-----
    from utils.preprocess import run_pipeline
    df_clean, report = run_pipeline("path/to/scopus_raw.csv")
"""

from __future__ import annotations
import re
import unicodedata
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. CONSTANTS
# ---------------------------------------------------------------------------

ESSENTIAL_FIELDS: list[str] = ["Title", "Year", "Source title"]
"""
Minimum fields required for a record to be included in the corpus.
Records missing ANY of these are excluded in Stage 5.
"""

THEMATIC_SCREEN_TERMS: list[str] = [
    "virtual reality",
    "augmented reality",
    "mixed reality",
    "extended reality",
    " vr ",
    " ar ",
    " xr ",
    " mr ",
    "hololens",
    "oculus",
    "head-mounted",
    "immersive environment",
    "immersive simulation",
    "360 video",
    "360° video",
    "stereoscopic",
]
"""
Terms used in Stage 6 (thematic screening). A record must contain at least
one of these terms in Title + Abstract + Author Keywords to be retained.
This is a broad filter (any XR signal), not the canonical 4-category classifier.
"""

YEAR_MIN = 1991
YEAR_MAX = 2025
"""
Year range applied after thematic screening. Records outside this prespecified
window are excluded.
"""

COMPLETENESS_FIELDS = [
    "Title", "Year", "Source title", "DOI", "Abstract", "Author Keywords",
    "Index Keywords", "Affiliations", "Cited by", "EID", "Author(s) ID",
]


# ---------------------------------------------------------------------------
# 2. HELPER: TITLE NORMALIZER (for normalized exact matching)
# ---------------------------------------------------------------------------
def _normalize_title(title: str) -> str:
    """
    Normalize a title string for exact normalized-title comparison.

    Operations:
      1. Convert to lowercase
      2. Remove accents (NFD decomposition + strip combining chars)
      3. Remove all non-alphanumeric characters
      4. Collapse multiple spaces into one
    """
    if not isinstance(title, str):
        return ""
    # Lowercase
    s = title.lower()
    # Remove accents
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    # Remove non-alphanumeric
    s = re.sub(r"[^a-z0-9\s]", "", s)
    # Collapse spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# 3. PIPELINE STAGES
# ---------------------------------------------------------------------------

def _stage0_load(csv_path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Stage 0: Load raw Scopus CSV export."""
    df = pd.read_csv(csv_path, low_memory=False, encoding="utf-8-sig")
    n = len(df)
    return df, {"stage": "0_raw_load", "records": n, "note": "Raw Scopus export loaded"}


def _stage1_standardize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Stage 1: Standardize column names.
    Scopus CSVs often have a BOM character in the first column name and
    trailing spaces. This stage strips both.
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lstrip("\ufeff")
    n = len(df)
    return df, {"stage": "1_col_standardize", "records": n, "note": "Column names cleaned"}


def _stage2_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Stage 2: Drop exact duplicate rows (all columns identical).
    This handles cases where the same record was exported from Scopus multiple times.
    """
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_after = len(df)
    return df, {
        "stage": "2_exact_dedup",
        "records": n_after,
        "dropped": n_before - n_after,
        "note": "Exact row duplicates removed",
    }


def _stage3_doi_deduplication(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Stage 3: DOI-based deduplication.
    If two records share the same non-null DOI, keep only the first occurrence.
    DOI is the most reliable unique identifier in Scopus.
    """
    n_before = len(df)
    doi_col = "DOI"
    if doi_col in df.columns:
        # Normalize DOI (lowercase, strip whitespace)
        df = df.copy()
        doi_mask = df[doi_col].notna() & (df[doi_col].astype(str).str.strip() != "")
        df.loc[doi_mask, "_doi_norm"] = df.loc[doi_mask, doi_col].astype(str).str.lower().str.strip()
        df.loc[doi_mask, "_doi_norm"] = df.loc[doi_mask, "_doi_norm"].str.replace(
            r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", regex=True
        )
        # Among identical DOIs, retain the row with the richest bibliographic
        # metadata; ties keep the original order for reproducibility.
        quality_fields = [f for f in COMPLETENESS_FIELDS if f in df.columns]
        df["_quality"] = df[quality_fields].apply(
            lambda col: col.notna() & col.astype(str).str.strip().ne("")
        ).sum(axis=1)
        df_has_doi = (df.loc[doi_mask]
                      .sort_values("_quality", ascending=False, kind="stable")
                      .drop_duplicates(subset=["_doi_norm"], keep="first"))
        df_no_doi = df[~doi_mask]
        df = pd.concat([df_has_doi, df_no_doi], ignore_index=True)
        df = df.drop(columns=["_doi_norm", "_quality"], errors="ignore")
    n_after = len(df)
    return df.reset_index(drop=True), {
        "stage": "3_doi_dedup",
        "records": n_after,
        "dropped": n_before - n_after,
        "note": "DOI-based duplicates removed",
    }


def _stage4_title_deduplication(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Stage 4: Conservative normalized-title deduplication.
    Candidate duplicates must share a non-empty normalized title, publication
    year, and first author identifier (or first-author text if IDs are absent).
    Records with distinct non-empty DOIs are retained as distinct works.
    This catches punctuation, capitalization, and diacritic variants such as:
      - "Virtual Reality in Education" vs "Virtual reality in education."

    This is exact matching after normalization, not fuzzy similarity matching.
    Rows without a usable title, year, or first-author key are retained here:
    without all three fields there is not enough evidence to call two records
    title duplicates. Stage 5 later reports missing essential metadata.
    """
    n_before = len(df)
    if "Title" in df.columns:
        df = df.copy()
        df["_title_norm"] = df["Title"].apply(_normalize_title)
        if "Year" in df.columns:
            df["_year_key"] = pd.to_numeric(df["Year"], errors="coerce").fillna(-1).astype(int)
        else:
            df["_year_key"] = -1
        author_col = "Authors" if "Authors" in df.columns else "Author full names"
        author_names = (df[author_col].fillna("").astype(str).str.split(";").str[0]
                        .str.lower().str.strip() if author_col in df.columns else "")
        if "Author(s) ID" in df.columns:
            author_ids = df["Author(s) ID"].fillna("").astype(str).str.split(";").str[0].str.strip()
            df["_first_author_key"] = author_ids.where(author_ids.ne(""), author_names)
        else:
            df["_first_author_key"] = author_names
        if "DOI" in df.columns:
            df["_doi_key"] = df["DOI"].fillna("").astype(str).str.lower().str.strip()
            df["_doi_key"] = df["_doi_key"].str.replace(
                r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", regex=True
            )
        else:
            df["_doi_key"] = ""
        quality_fields = [f for f in COMPLETENESS_FIELDS if f in df.columns]
        df["_quality"] = df[quality_fields].apply(
            lambda col: col.notna() & col.astype(str).str.strip().ne("")
        ).sum(axis=1)
        keys = ["_title_norm", "_year_key", "_first_author_key"]
        keep = []
        # Do not treat two missing author/year values as evidence that records
        # are duplicates. Only compare rows with a complete matching key.
        candidate_mask = (
            df["_title_norm"].ne("")
            & df["_year_key"].ge(0)
            & df["_first_author_key"].ne("")
        )
        candidates = df[candidate_mask]
        for _, group in candidates.groupby(keys, sort=False, dropna=False):
            nonempty_dois = set(group.loc[group["_doi_key"].ne(""), "_doi_key"])
            if len(group) == 1:
                keep.extend(group.index.tolist())
            elif len(nonempty_dois) > 1:
                # Different DOI records are different works even when title,
                # year, and first author coincide. Keep each DOI once, plus at
                # most one record without a DOI.
                for _, doi_group in group[group["_doi_key"].ne("")].groupby("_doi_key", sort=False):
                    keep.append(doi_group.sort_values("_quality", ascending=False, kind="stable").index[0])
                no_doi = group[group["_doi_key"].eq("")]
                if not no_doi.empty:
                    keep.append(no_doi.sort_values("_quality", ascending=False, kind="stable").index[0])
            else:
                keep.append(group.sort_values("_quality", ascending=False, kind="stable").index[0])
        not_comparable = df[~candidate_mask]
        df = pd.concat([df.loc[keep], not_comparable], axis=0).sort_index()
        df = df.drop(columns=["_title_norm", "_year_key", "_first_author_key", "_doi_key", "_quality"])
    n_after = len(df)
    return df.reset_index(drop=True), {
        "stage": "4_title_dedup",
        "records": n_after,
        "dropped": n_before - n_after,
        "note": "Exact normalized title + year + first-author match; incomplete keys retained; distinct non-empty DOIs preserved",
    }


def _stage5_missing_metadata(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Stage 5: Remove records with missing essential metadata.
    Records missing Title, Year, or Source title cannot be analysed and are excluded.
    """
    n_before = len(df)
    missing_fields = [field for field in ESSENTIAL_FIELDS if field not in df.columns]
    if missing_fields:
        raise ValueError(f"Required Scopus columns are missing: {missing_fields}")
    for field in ESSENTIAL_FIELDS:
        if field in df.columns:
            df = df[df[field].notna() & (df[field].astype(str).str.strip() != "")]
    # Ensure Year is numeric
    if "Year" in df.columns:
        df = df.copy()
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df = df.dropna(subset=["Year"])
        df["Year"] = df["Year"].astype(int)
    n_after = len(df)
    return df.reset_index(drop=True), {
        "stage": "5_missing_metadata",
        "records": n_after,
        "dropped": n_before - n_after,
        "note": f"Records missing {ESSENTIAL_FIELDS} removed",
    }


def _stage6_thematic_screening(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Stage 6: Thematic screening.
    Retain only records that contain at least one XR-related term in
    Title + Abstract + Author Keywords.
    The year filter is a separate auditable stage.
    """
    n_before = len(df)

    def _has_xr_signal(row: pd.Series) -> bool:
        parts = []
        # Index Keywords are retained for descriptive analyses, but are not
        # used as an inclusion signal because Scopus assigns broad biomedical
        # descriptors that can inflate an education corpus.
        for field in ["Title", "Abstract", "Author Keywords"]:
            val = row.get(field, "")
            if pd.notna(val):
                parts.append(str(val))
        text = f" {' '.join(parts).lower()} "
        return any(term in text for term in THEMATIC_SCREEN_TERMS)

    df = df[df.apply(_has_xr_signal, axis=1)].copy()
    n_after_thematic = len(df)

    n_after = len(df)
    return df.reset_index(drop=True), {
        "stage": "6_thematic_screen",
        "records": n_after,
        "dropped_thematic": n_before - n_after_thematic,
        "dropped": n_before - n_after,
        "note": "XR signal in Title, Abstract, or Author Keywords",
    }


def _stage7_year_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Stage 7: Keep records inside the prespecified publication-year window."""
    n_before = len(df)
    if "Year" in df.columns:
        df = df[(df["Year"] >= YEAR_MIN) & (df["Year"] <= YEAR_MAX)].copy()
    n_after = len(df)
    return df.reset_index(drop=True), {
        "stage": "7_year_filter",
        "records": n_after,
        "dropped": n_before - n_after,
        "note": f"Publication year window {YEAR_MIN}–{YEAR_MAX} (inclusive)",
    }


# ---------------------------------------------------------------------------
# 4. MAIN PIPELINE ENTRY POINT
# ---------------------------------------------------------------------------

def run_pipeline(csv_path: str | Path) -> tuple[pd.DataFrame, list[dict]]:
    """
    Run the full Scopus cleaning pipeline.

    Parameters
    ----------
    csv_path : str or Path
        Path to the raw Scopus CSV export.

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned DataFrame ready for analysis.
    report : list of dict
        Stage-by-stage record counts for audit/reproducibility logging.
        Each dict has keys: 'stage', 'records', 'dropped' (if applicable), 'note'.
    """
    report: list[dict] = []

    df, info = _stage0_load(csv_path)
    report.append(info)

    df, info = _stage1_standardize_columns(df)
    report.append(info)

    df, info = _stage2_exact_duplicates(df)
    report.append(info)

    df, info = _stage3_doi_deduplication(df)
    report.append(info)

    df, info = _stage4_title_deduplication(df)
    report.append(info)

    df, info = _stage5_missing_metadata(df)
    report.append(info)

    df, info = _stage6_thematic_screening(df)
    report.append(info)

    df, info = _stage7_year_filter(df)
    report.append(info)

    return df, report


def print_pipeline_report(report: list[dict]) -> None:
    """Print the pipeline report (stage-by-stage record counts) to stdout."""
    print("\n" + "=" * 65)
    print("  SCOPUS CLEANING PIPELINE REPORT")
    print("=" * 65)
    for info in report:
        stage = info.get("stage", "?")
        records = info.get("records", "?")
        dropped = info.get("dropped", None)
        note = info.get("note", "")
        if dropped is not None:
            print(f"  [{stage:>25}]  {records:>7,} records  (−{dropped:,} dropped)")
        else:
            print(f"  [{stage:>25}]  {records:>7,} records")
        if note:
            print(f"  {'':>27}  → {note}")
    print("=" * 65)
    print()
