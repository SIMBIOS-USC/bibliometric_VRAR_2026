"""
utils/classifier.py
===================
CANONICAL Technology Classifier for XR Bibliometric Analysis.

This module defines the SINGLE authoritative classification function used
across ALL analyses in this project. Having one canonical classifier solves
the methodological inconsistency flagged in the peer-review (Reviewer Issue #2):
different scripts were using different keyword dictionaries and different field
combinations, meaning figures could be classifying articles with subtly different rules.

Classification Scheme (aligned with the manuscript):
------------------------------------------------------
  VR   – Virtual Reality only (no AR or MR/XR signals)
  AR   – Augmented Reality only (no VR or MR/XR signals)
  MR/XR – Mixed/Extended Reality (any MR/XR term present)
  Hybrid/Multi-technology – co-mention of VR + AR without MR/XR terms
          (this is the definition from the manuscript, NOT "VR+AR → MR/XR"
           which was the bug in paper_clasificados.py and institutions_and_journals.py)

Fields searched (applied consistently across ALL modules):
  Title + Abstract + Author Keywords + Index Keywords

Note on "immersive":
  "immersive" alone is NOT treated as a VR signal because it can appear in
  "immersive learning" or "immersive education" contexts unrelated to VR hardware.
  Explicit VR terms such as "virtual reality" are sufficient; hardware is not
  required when the technology itself is named.

Note on MR/XR grouping:
  Mixed Reality and Extended Reality are grouped under one terminological category
  because they overlap heavily in the corpus. This is a data-driven decision,
  NOT a claim that MR and XR are conceptually equivalent. The manuscript should
  state this explicitly.
"""

from __future__ import annotations
import pandas as pd


# ---------------------------------------------------------------------------
# 1. KEYWORD DICTIONARIES
# ---------------------------------------------------------------------------
# These are the CANONICAL term sets. Do NOT add terms here without updating
# the methodology section of the manuscript accordingly.

VR_TERMS: list[str] = [
    "virtual reality",
    "virtual environment",
    "virtual world",
    " vr ",
    "head-mounted display",
    " hmd ",
    "oculus",
    "htc vive",
    " vive ",
    "oculus rift",
    "oculus quest",
    "cave automatic",
    " cave ",
    "stereoscopic",
    "360 video",
    "360° video",
    "360-degree video",
    " ive ",          # immersive virtual environment
    " cve ",          # collaborative virtual environment
]
"""
VR-specific terms. Note: 'immersive' alone is intentionally excluded because it
appears in 'immersive learning' contexts unrelated to VR technology.
"""

AR_TERMS: list[str] = [
    "augmented reality",
    " ar ",
    "smart glasses",
    "google glass",
    "epson moverio",
    "heads-up display",
    " hud ",
    "marker-based",
    "marker-less",
    "mobile ar",
    "ar application",
    "ar-based",
]
"""AR-specific terms."""

MR_XR_TERMS: list[str] = [
    "mixed reality",
    "extended reality",
    " xr ",
    " mr ",
    "spatial computing",
    "hololens",
    "magic leap",
    "windows mixed reality",
]
"""
MR/XR terms. This is a terminological grouping. Mixed Reality and Extended Reality
are grouped because they overlap heavily in the corpus. Metaverse, digital twin,
and passthrough are intentionally excluded because they are broader concepts that
do not necessarily imply MR/XR hardware.
"""

EXCLUSION_TERMS: list[str] = [
    "agent-mediated",
    "multi-agent",
    "auction mechanism",
    "negotiation protocol",
    "market mechanism",
    "mixed integer programming",
    "mixed integer linear",
    "mixed integer nonlinear",
]
"""
Terms that, when present WITHOUT any XR hardware signal, indicate the record
is not relevant to the XR corpus (e.g., operations research papers that use
'mixed' in a mathematical sense, or multi-agent system papers where 'AR' refers
to an agent role, not Augmented Reality).
"""

CATEGORY_ORDER: list[str] = ["VR", "AR", "MR/XR", "Hybrid/Multi-technology"]
"""
Canonical display order for the four technology categories.
Used consistently across all analysis modules for table rows and figure panels.
"""

XR_HARDWARE_ANCHOR: list[str] = [
    "hololens",
    "magic leap",
    "oculus",
    "htc vive",
    "head-mounted display",
    " hmd ",
    "quest ",
    "microsoft hololens",
]
"""
Hardware-specific terms recorded as corroborating evidence in the per-record
audit. Exclusion handling uses any explicit category signal, not hardware alone.
"""


# ---------------------------------------------------------------------------
# 2. CLASSIFICATION FIELDS
# ---------------------------------------------------------------------------
CLASSIFICATION_FIELDS: list[str] = [
    "Title",
    "Abstract",
    "Author Keywords",
    "Index Keywords",
]
"""
Fields combined to build the classification text. This is the canonical set.
Using Title + Abstract + Author Keywords + Index Keywords provides the most
complete view of a record's topic without introducing noise from Keywords Plus
(which are algorithmically generated and can mismatch).
"""


# ---------------------------------------------------------------------------
# 3. ORIENTATION (TECHNICAL vs PEDAGOGICAL) CLASSIFIER
# ---------------------------------------------------------------------------
TECHNICAL_TERMS: list[str] = [
    "system design",
    "system architecture",
    "hardware",
    "framework design",
    "implementation",
    "interface design",
    "prototype",
    "software development",
    "rendering",
    "tracking",
    "latency",
    "haptic",
    "3d modeling",
    "simulation engine",
    "network protocol",
    "bandwidth",
    "fps",
    "frame rate",
    "calibration",
    "sensor",
    "depth camera",
    "point cloud",
    "motion capture",
    "eye tracking",
    "hand tracking",
]

PEDAGOGICAL_TERMS: list[str] = [
    "learning",
    "education",
    "teaching",
    "instruction",
    "curriculum",
    "pedagogy",
    "student",
    "classroom",
    "assessment",
    "feedback",
    "engagement",
    "motivation",
    "collaboration",
    "cognitive load",
    "knowledge acquisition",
    "training",
    "e-learning",
    "blended learning",
    "higher education",
    "k-12",
    "constructivism",
    "problem-based",
    "project-based",
    "inquiry-based",
    "self-regulated",
    "metacognition",
    "comprehension",
    "retention",
    "academic performance",
]
"""
Pedagogical orientation terms. Used for the Technical vs Pedagogical
classification that supports Figure 2 and RQ4 in the manuscript.
"""


# ---------------------------------------------------------------------------
# 4. CANONICAL CLASSIFICATION FUNCTION
# ---------------------------------------------------------------------------
def classify_technology(row: pd.Series) -> str | None:
    """
    Classify a Scopus record into one of four XR technology categories.

    Parameters
    ----------
    row : pd.Series
        A single row from the cleaned Scopus DataFrame. Must contain columns
        defined in CLASSIFICATION_FIELDS.

    Returns
    -------
    str or None
        One of: 'VR', 'AR', 'MR/XR', 'Hybrid/Multi-technology'
        Returns None if the record should be excluded (non-XR noise) or
        cannot be classified into any category.

    Classification Logic (manuscript-aligned):
    ------------------------------------------
    Priority 1 (EXCLUSION): If EXCLUSION_TERMS found AND no explicit technology
                             term from any category → return None
    Priority 2 (MR/XR):     If any MR_XR_TERMS found → 'MR/XR'
    Priority 3 (Hybrid):    If VR_TERMS AND AR_TERMS both found (but NOT MR/XR)
                             → 'Hybrid/Multi-technology'
                (Manuscript definition: co-mention of VR+AR without explicit MR/XR)
    Priority 4 (AR):        If only AR_TERMS found → 'AR'
    Priority 5 (VR):        If only VR_TERMS found → 'VR'
    Default:                 None (unclassifiable)

    Note: The original bug in paper_clasificados.py and institutions_and_journals.py
    was that VR+AR was classified as 'Mixed/Extended Reality' instead of
    'Hybrid/Multi-technology'. This function implements the CORRECT manuscript logic.
    """
    # Build combined text from canonical fields
    parts = []
    for field in CLASSIFICATION_FIELDS:
        val = row.get(field, "")
        if pd.notna(val):
            parts.append(str(val))
    text = " ".join(parts).lower()

    # Pad text with spaces to enable reliable whole-word matching for short tokens like ' ar ', ' vr '
    text = f" {text} "

    # Detect explicit category evidence before applying broad exclusion phrases.
    has_mr_xr = any(term in text for term in MR_XR_TERMS)
    has_ar = any(term in text for term in AR_TERMS)
    has_vr = any(term in text for term in VR_TERMS)

    # Terms such as "multi-agent" should not remove a record that separately
    # names an XR technology. The previous hardware-only rescue rule wrongly
    # discarded genuine VR/AR records without a device name.
    has_exclusion = any(ex in text for ex in EXCLUSION_TERMS)
    if has_exclusion and not (has_mr_xr or has_ar or has_vr):
        return None  # Non-XR record, exclude

    # --- Priority 2: MR/XR (highest specificity) ---
    if has_mr_xr:
        return "MR/XR"

    # --- Priority 3: Hybrid (VR + AR co-mention, no MR/XR) ---
    # This implements the MANUSCRIPT definition of Hybrid/Multi-technology
    if has_vr and has_ar:
        return "Hybrid/Multi-technology"

    # --- Priority 4: Pure AR ---
    if has_ar:
        return "AR"

    # --- Priority 5: Pure VR ---
    if has_vr:
        return "VR"

    # Cannot be classified
    return None


def classify_orientation(row: pd.Series) -> str:
    """
    Classify a record as 'Technical' or 'Pedagogical' based on dominant orientation.

    This classification supports Figure 2 and RQ4 in the manuscript.
    Records are classified by counts of distinct dictionary terms present in
    Title + Abstract + Author Keywords. Ties are assigned Pedagogical. This is
    an exploratory automated label, not a validated human coding scheme.

    Parameters
    ----------
    row : pd.Series
        A single row from the cleaned Scopus DataFrame.

    Returns
    -------
    str
        'Pedagogical' or 'Technical'
    """
    # Only use Title + Abstract + Author Keywords for orientation (not Index Keywords,
    # which are subject descriptors and may introduce systematic bias)
    orientation_fields = ["Title", "Abstract", "Author Keywords"]
    parts = []
    for field in orientation_fields:
        val = row.get(field, "")
        if pd.notna(val):
            parts.append(str(val))
    text = f" {' '.join(parts).lower()} "

    tech_score = sum(1 for term in TECHNICAL_TERMS if term in text)
    ped_score = sum(1 for term in PEDAGOGICAL_TERMS if term in text)

    return "Pedagogical" if ped_score >= tech_score else "Technical"


# ---------------------------------------------------------------------------
# 5. BATCH CLASSIFICATION (CONVENIENCE WRAPPER)
# ---------------------------------------------------------------------------
def add_classifications(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply both technology and orientation classifiers to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned Scopus DataFrame (output of preprocess.py pipeline).

    Returns
    -------
    pd.DataFrame
        Same DataFrame with two new columns:
          - 'Tech_Category': 'VR' | 'AR' | 'MR/XR' | 'Hybrid/Multi-technology' | NaN
          - 'Orientation': 'Technical' | 'Pedagogical'
    """
    df = df.copy()
    df["Tech_Category"] = df.apply(classify_technology, axis=1)
    # Only apply orientation to classified records
    classified_mask = df["Tech_Category"].notna()
    df.loc[classified_mask, "Orientation"] = df[classified_mask].apply(
        classify_orientation, axis=1
    )
    return df


# ---------------------------------------------------------------------------
# 6. CLASSIFICATION REPORT
# ---------------------------------------------------------------------------
def print_classification_report(df: pd.DataFrame) -> None:
    """
    Print a summary of classification results to stdout.
    Useful for auditing and reproducing manuscript counts.
    """
    if "Tech_Category" not in df.columns:
        print("[WARNING] 'Tech_Category' column not found. Run add_classifications() first.")
        return

    total = len(df)
    classified = df["Tech_Category"].notna().sum()
    unclassified = total - classified

    print("\n" + "=" * 60)
    print("  TECHNOLOGY CLASSIFICATION REPORT")
    print("=" * 60)
    print(f"  Total records     : {total:>7,}")
    print(f"  Classified        : {classified:>7,} ({100*classified/total:.1f}%)")
    print(f"  Excluded/Unknown  : {unclassified:>7,} ({100*unclassified/total:.1f}%)")
    print("-" * 60)
    counts = df["Tech_Category"].value_counts()
    for cat in ["VR", "AR", "MR/XR", "Hybrid/Multi-technology"]:
        n = counts.get(cat, 0)
        print(f"  {cat:<30}: {n:>6,} ({100*n/total:.1f}%)")
    print("=" * 60)

    if "Orientation" in df.columns:
        print("\n  ORIENTATION CLASSIFICATION REPORT (classified records only)")
        print("-" * 60)
        orient_counts = df.loc[df["Tech_Category"].notna(), "Orientation"].value_counts()
        for ori in ["Technical", "Pedagogical"]:
            n = orient_counts.get(ori, 0)
            print(f"  {ori:<30}: {n:>6,} ({100*n/classified:.1f}%)")
        print("=" * 60)
    print()
