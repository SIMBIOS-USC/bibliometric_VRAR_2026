"""Keyword rules for technology categories and research orientation.

Technology labels use the title, abstract, author keywords, and Index Keywords.
MR/XR takes priority; otherwise a record mentioning both VR and AR is labelled
Hybrid/Multi-technology. The single word "immersive" is not a VR signal.
Mixed Reality and Extended Reality share a label here; that is a coding choice,
not a claim that the terms mean the same thing.
"""

from __future__ import annotations
import pandas as pd


# ---------------------------------------------------------------------------
# 1. KEYWORD DICTIONARIES
# ---------------------------------------------------------------------------
# Keep these lists in step with the methods description.

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

MR_TERMS: list[str] = [
    "mixed reality",
    " mr ",
    "hololens",
    "magic leap",
    "windows mixed reality",
    "spatial computing",
]
"""
Mixed Reality-specific terms (Milgram-Kishino continuum, 1994).
Includes devices that are canonically associated with MR environments
(HoloLens, Magic Leap). Spatial computing is included here because it
describes the same blended physical-digital interaction model.
"""

XR_TERMS: list[str] = [
    "extended reality",
    " xr ",
]
"""
Extended Reality-specific terms. XR is a broader umbrella concept that
encompasses VR, AR, and MR under a single label. Records using this
term without a more specific technology signal are classified as XR.
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

CATEGORY_ORDER: list[str] = ["VR", "AR", "MR", "XR", "Hybrid/Multi-technology"]
"""
Canonical display order for the five technology categories.
MR (Mixed Reality) and XR (Extended Reality) are now separated to allow
finer-grained analysis of each sub-community.
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
    Classify a Scopus record into one of five XR technology categories.

    Parameters
    ----------
    row : pd.Series
        A single row from the cleaned Scopus DataFrame. Must contain columns
        defined in CLASSIFICATION_FIELDS.

    Returns
    -------
    str or None
        One of: 'VR', 'AR', 'MR', 'XR', 'Hybrid/Multi-technology'
        Returns None if the record should be excluded (non-XR noise) or
        cannot be classified into any category.

    Classification priority (highest → lowest specificity):
      1. MR (Mixed Reality — device-grounded: HoloLens, Magic Leap, etc.)
      2. XR (Extended Reality — umbrella label without a more specific signal)
      3. Hybrid/Multi-technology (explicit VR + AR co-mention, no MR/XR)
      4. Pure AR
      5. Pure VR
    Exclusion phrases suppress a record only when no technology signal matches.
    """
    # Combine the fields used for classification.
    parts = []
    for field in CLASSIFICATION_FIELDS:
        val = row.get(field, "")
        if pd.notna(val):
            parts.append(str(val))
    text = " ".join(parts).lower()

    # Pad with spaces for reliable whole-word matching of short tokens (' ar ', ' vr ', ' mr ', ' xr ')
    text = f" {text} "

    # Detect explicit category signals.
    has_mr  = any(term in text for term in MR_TERMS)
    has_xr  = any(term in text for term in XR_TERMS)
    has_ar  = any(term in text for term in AR_TERMS)
    has_vr  = any(term in text for term in VR_TERMS)

    # Suppress non-XR records (e.g. operations-research uses of 'mixed')
    # but only when no XR technology signal is present at all.
    has_exclusion = any(ex in text for ex in EXCLUSION_TERMS)
    if has_exclusion and not (has_mr or has_xr or has_ar or has_vr):
        return None

    # --- Priority 1: MR (device-level specificity, highest confidence) ---
    if has_mr:
        return "MR"

    # --- Priority 2: XR (umbrella label, no more specific signal found) ---
    if has_xr:
        return "XR"

    # --- Priority 3: Hybrid (VR + AR co-mention, no MR/XR signal) ---
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
          - 'Tech_Category': 'VR' | 'AR' | 'MR' | 'XR' | 'Hybrid/Multi-technology' | NaN
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
    for cat in CATEGORY_ORDER:
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
