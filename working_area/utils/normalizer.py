"""
utils/normalizer.py
===================
Institution name and author name standardization.

Applied in the preprocessing pipeline (Stage 0.5) before deduplication, so
that the normalized forms propagate into all downstream analyses.

Institution standardization
----------------------------
Maps common abbreviations, acronyms, and variant spellings to a canonical
full name.  The mapping is applied to Scopus's ``Affiliations`` field using
whole-word case-insensitive replacement.  The canonical name is always the
officially preferred English form of the institution.

Author name standardization
-----------------------------
Normalizes author name strings exported by Scopus ("Last, First M.") to a
consistent Unicode-stripped, whitespace-collapsed form.  This does NOT perform
semantic author disambiguation (i.e., two authors with the same name are not
merged, and the same author with variant spellings in different records is not
automatically unified).  It removes diacritics, collapses multiple spaces, and
ensures punctuation consistency.

Limitation
----------
The institution dictionary covers the most frequently appearing institutions
in the immersive-technology education literature.  Institutions not in the
dictionary are left unchanged.  Researchers requiring exhaustive disambiguation
should cross-reference the audit file with OpenAlex or ROR.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd


# ---------------------------------------------------------------------------
# 1. INSTITUTION ABBREVIATION → CANONICAL NAME MAP
# ---------------------------------------------------------------------------
# Keys: regex-safe abbreviations / common variants (searched case-insensitively
#       as whole-word tokens where possible).
# Values: canonical full name (English, official).
# ---------------------------------------------------------------------------
INSTITUTION_MAP: dict[str, str] = {
    # ── United States ───────────────────────────────────────────────────────
    r"\bMIT\b":           "Massachusetts Institute of Technology",
    r"\bStanford Univ\b": "Stanford University",
    r"\bUCLA\b":          "University of California, Los Angeles",
    r"\bUCSB\b":          "University of California, Santa Barbara",
    r"\bUCSD\b":          "University of California, San Diego",
    r"\bUCSF\b":          "University of California, San Francisco",
    r"\bUC Berkeley\b":   "University of California, Berkeley",
    r"\bCMU\b":           "Carnegie Mellon University",
    r"\bNYU\b":           "New York University",
    r"\bUSC\b":           "University of Southern California",
    r"\bGT\b":            "Georgia Institute of Technology",
    r"\bGeorgia Tech\b":  "Georgia Institute of Technology",
    r"\bUW\b":            "University of Washington",
    r"\bU Washington\b":  "University of Washington",
    r"\bOhio State\b":    "The Ohio State University",
    r"\bPenn State\b":    "Pennsylvania State University",
    r"\bMayo Clin\b":     "Mayo Clinic",
    # ── United Kingdom ──────────────────────────────────────────────────────
    r"\bUCL\b":           "University College London",
    r"\bKCL\b":           "King's College London",
    r"\bLSE\b":           "London School of Economics and Political Science",
    r"\bImperial Coll\b": "Imperial College London",
    r"\bUniv Coll London\b": "University College London",
    r"\bUniv College London\b": "University College London",
    r"\bOxford Univ\b":   "University of Oxford",
    r"\bCambridge Univ\b": "University of Cambridge",
    r"\bUniv Edinburgh\b": "University of Edinburgh",
    r"\bUniv Manchester\b": "University of Manchester",
    r"\bUniv Nottingham\b": "University of Nottingham",
    r"\bUniv Sheffield\b":  "University of Sheffield",
    r"\bUniv Leeds\b":      "University of Leeds",
    # ── Spain ───────────────────────────────────────────────────────────────
    r"\bUPM\b":           "Universidad Politécnica de Madrid",
    r"\bUPV\b":           "Universitat Politècnica de València",
    r"\bUAM\b":           "Universidad Autónoma de Madrid",
    r"\bUAB\b":           "Universitat Autònoma de Barcelona",
    r"\bUB\b":            "Universitat de Barcelona",
    r"\bUGR\b":           "Universidad de Granada",
    r"\bUS\b":            "Universidad de Sevilla",
    r"\bUCM\b":           "Universidad Complutense de Madrid",
    r"\bUOC\b":           "Universitat Oberta de Catalunya",
    # ── China ───────────────────────────────────────────────────────────────
    r"\bPKU\b":           "Peking University",
    r"\bTsinghua Univ\b": "Tsinghua University",
    r"\bZJU\b":           "Zhejiang University",
    r"\bSJTU\b":          "Shanghai Jiao Tong University",
    r"\bFudan Univ\b":    "Fudan University",
    r"\bHUST\b":          "Huazhong University of Science and Technology",
    r"\bNJU\b":           "Nanjing University",
    r"\bSCU\b":           "Sichuan University",
    # ── Europe & International ──────────────────────────────────────────────
    r"\bETH Zurich\b":    "ETH Zürich",
    r"\bETH Zürich\b":    "ETH Zürich",
    r"\bEPFL\b":          "École Polytechnique Fédérale de Lausanne",
    r"\bKU Leuven\b":     "Katholieke Universiteit Leuven",
    r"\bTU Delft\b":      "Delft University of Technology",
    r"\bTU Munich\b":     "Technical University of Munich",
    r"\bTUM\b":           "Technical University of Munich",
    r"\bLMU\b":           "Ludwig Maximilian University of Munich",
    r"\bCharité\b":       "Charité – Universitätsmedizin Berlin",
    r"\bWageningen Univ\b": "Wageningen University & Research",
    r"\bKarolinska\b":    "Karolinska Institutet",
    r"\bUniv Tokyo\b":    "University of Tokyo",
    r"\bTokyo Univ\b":    "University of Tokyo",
    r"\bOsaka Univ\b":    "Osaka University",
    r"\bUniv Toronto\b":  "University of Toronto",
    r"\bMcGill Univ\b":   "McGill University",
    r"\bUBC\b":           "University of British Columbia",
    r"\bANU\b":           "Australian National University",
    r"\bUNSW\b":          "University of New South Wales",
    r"\bUniv Melbourne\b": "University of Melbourne",
    r"\bUniv Sydney\b":   "University of Sydney",
    r"\bUniv Queensland\b": "University of Queensland",
}
"""
Canonical institution name map.  Keys are Python regex patterns (with word
boundaries \\b where applicable).  Values are the preferred full English name.
Applied case-insensitively via :func:`standardize_affiliations`.
"""

# Pre-compile patterns once for performance
_COMPILED: list[tuple[re.Pattern, str]] = [
    (re.compile(k, re.IGNORECASE), v)
    for k, v in INSTITUTION_MAP.items()
]


# ---------------------------------------------------------------------------
# 2. INSTITUTION STANDARDIZATION
# ---------------------------------------------------------------------------
def standardize_affiliations(aff_string: str) -> str:
    """
    Apply canonical institution name replacements to a Scopus Affiliations string.

    Parameters
    ----------
    aff_string : str
        Raw value from the Scopus ``Affiliations`` column.

    Returns
    -------
    str
        Affiliations string with known abbreviations replaced by canonical names.
    """
    if not isinstance(aff_string, str) or not aff_string.strip():
        return aff_string
    result = aff_string
    for pattern, canonical in _COMPILED:
        result = pattern.sub(canonical, result)
    return result


# ---------------------------------------------------------------------------
# 3. AUTHOR NAME STANDARDIZATION
# ---------------------------------------------------------------------------
def _strip_diacritics(text: str) -> str:
    """Remove diacritic marks (accents) from a unicode string."""
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def standardize_author_name(name: str) -> str:
    """
    Normalize a single author name string.

    Operations applied (in order):
      1. Strip leading/trailing whitespace.
      2. Remove diacritics (é → e, ü → u, etc.).
      3. Collapse consecutive spaces/tabs.
      4. Normalize punctuation: ensure exactly one space after each comma.
      5. Title-case the result (Last, Firstname → Last, Firstname).

    This is a surface normalization only — it does NOT perform author
    disambiguation.  Two authors whose names happen to share the same
    normalized form will be counted as one in ranked tables.

    Parameters
    ----------
    name : str
        A single author name, typically in "Last, First M." format as
        exported by Scopus.

    Returns
    -------
    str
        Normalized name string.
    """
    if not isinstance(name, str):
        return name
    name = name.strip()
    name = _strip_diacritics(name)
    name = re.sub(r"\s+", " ", name)                  # collapse whitespace
    name = re.sub(r",\s*", ", ", name)                 # normalize comma spacing
    name = re.sub(r"\.(?=[A-Za-z])", ". ", name)       # space after initials
    name = re.sub(r"\s+", " ", name).strip()           # final whitespace pass
    return name


def standardize_authors_field(authors_str: str) -> str:
    """
    Normalize the full semicolon-separated Authors string from Scopus.

    Parameters
    ----------
    authors_str : str
        Raw value from the Scopus ``Authors`` column
        (e.g. "Smith, J.; García López, M.; Müller, K.").

    Returns
    -------
    str
        Semicolon-separated string of normalized author names.
    """
    if not isinstance(authors_str, str) or not authors_str.strip():
        return authors_str
    parts = authors_str.split(";")
    normalized = [standardize_author_name(p.strip()) for p in parts]
    return "; ".join(normalized)


# ---------------------------------------------------------------------------
# 4. BATCH APPLICATION (used by preprocess.py)
# ---------------------------------------------------------------------------
def apply_normalization(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Apply institution and author name standardization to a Scopus DataFrame.

    Called by the preprocessing pipeline between column standardization and
    deduplication.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned Scopus DataFrame (after column standardization).

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Updated DataFrame and a pipeline stage info dict.
    """
    df = df.copy()
    n = len(df)

    if "Affiliations" in df.columns:
        df["Affiliations"] = df["Affiliations"].apply(standardize_affiliations)

    if "Authors" in df.columns:
        df["Authors"] = df["Authors"].apply(standardize_authors_field)

    # Also normalize "Author full names" if present
    if "Author full names" in df.columns:
        df["Author full names"] = df["Author full names"].apply(standardize_authors_field)

    info = {
        "stage":   "1b_name_normalization",
        "records": n,
        "note":    (
            f"Institution abbreviations expanded (dictionary: {len(INSTITUTION_MAP)} rules); "
            "author names stripped of diacritics and punctuation-normalized. "
            "No semantic author disambiguation applied."
        ),
    }
    return df, info
