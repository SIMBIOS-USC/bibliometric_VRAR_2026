"""Record-level decision trace and human validation worksheet.

This module documents what the deterministic keyword rules matched. It cannot
estimate intercoder agreement: two independent raters must code the worksheet
before kappa or validation claims can be made.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from utils.classifier import (
    CLASSIFICATION_FIELDS, EXCLUSION_TERMS, MR_XR_TERMS, PEDAGOGICAL_TERMS,
    TECHNICAL_TERMS, VR_TERMS, AR_TERMS, XR_HARDWARE_ANCHOR,
)


def _text(row: pd.Series, fields: list[str]) -> str:
    return " ".join(str(row.get(field, "")) for field in fields
                    if pd.notna(row.get(field, ""))).lower()


def _matched_terms(text: str, terms: list[str]) -> str:
    return "; ".join(term.strip() for term in terms if term in text)


def run(df: pd.DataFrame, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in df.iterrows():
        tech_text = _text(row, CLASSIFICATION_FIELDS)
        ori_text = _text(row, ["Title", "Abstract", "Author Keywords"])
        rows.append({
            "EID": row.get("EID", ""),
            "Title": row.get("Title", ""),
            "Year": row.get("Year", ""),
            "Tech_Category": row.get("Tech_Category", ""),
            "Orientation": row.get("Orientation", ""),
            "Matched_VR_terms": _matched_terms(tech_text, VR_TERMS),
            "Matched_AR_terms": _matched_terms(tech_text, AR_TERMS),
            "Matched_MR_XR_terms": _matched_terms(tech_text, MR_XR_TERMS),
            "Matched_Exclusion_terms": _matched_terms(tech_text, EXCLUSION_TERMS),
            "Matched_Hardware_rescue_terms": _matched_terms(tech_text, XR_HARDWARE_ANCHOR),
            "Matched_Technical_terms": _matched_terms(ori_text, TECHNICAL_TERMS),
            "Matched_Pedagogical_terms": _matched_terms(ori_text, PEDAGOGICAL_TERMS),
        })
    trace = pd.DataFrame(rows)
    trace.to_csv(results_dir / "table_A10_classification_trace.csv", index=False)

    # Stratified, fixed-seed worksheet. It is a review aid, not a validation
    # result; prevalence estimates require appropriate sampling weights.
    sample_parts = []
    classified = df[df["Tech_Category"].notna()]
    for category, group in classified.groupby("Tech_Category", dropna=False):
        sample_parts.append(group.sample(n=min(50, len(group)), random_state=20260929))
    sample = pd.concat(sample_parts).drop_duplicates(subset=["EID"] if "EID" in df.columns else ["Title"])
    worksheet_cols = [c for c in ["EID", "Title", "Year", "Abstract", "Author Keywords",
                                    "Index Keywords", "Tech_Category", "Orientation"] if c in sample.columns]
    worksheet = sample[worksheet_cols].copy().rename(columns={
        "Tech_Category": "Automated_Tech_Category",
        "Orientation": "Automated_Orientation",
    })
    for col in ["Rater1_Tech_Category", "Rater2_Tech_Category", "Adjudicated_Tech_Category",
                "Rater1_Orientation", "Rater2_Orientation", "Adjudicated_Orientation",
                "Coding_Rationale"]:
        worksheet[col] = ""
    worksheet.to_csv(results_dir / "table_A10_human_validation_worksheet.csv", index=False)

    counts = df.groupby(["Tech_Category", "Orientation"], dropna=False).size().rename("Documents").reset_index()
    counts.to_csv(results_dir / "table_A10_classifier_counts.csv", index=False)

    education_terms = ["education", "educational", "learning", "teaching", "student",
                       "curriculum", "pedagogy", "school", "university", "training"]
    health_terms = ["health", "medical", "clinical", "nursing", "surgery", "surgical",
                    "patient", "hospital", "healthcare", "medicine"]
    scope_text = df.apply(lambda row: _text(row, ["Title", "Abstract", "Author Keywords"]), axis=1)
    scope = pd.DataFrame({
        "EID": df.get("EID", pd.Series("", index=df.index)),
        "Education_signal": scope_text.map(lambda text: any(term in text for term in education_terms)),
        "Health_or_clinical_signal": scope_text.map(lambda text: any(term in text for term in health_terms)),
    })
    scope_summary = pd.DataFrame([
        {"Indicator": label, "Documents": int(scope[column].sum()),
         "Percent_of_corpus": round(100 * float(scope[column].mean()), 2),
         "Rule": "Any listed term in Title, Abstract, or Author Keywords; non-exclusive descriptive flag"}
        for label, column in [("Education signal", "Education_signal"),
                              ("Health/clinical signal", "Health_or_clinical_signal")]
    ])
    scope_summary.to_csv(results_dir / "table_A10_scope_signal_counts.csv", index=False)
    print("  → Saved classifier trace, automated counts, and double-coding worksheet")
