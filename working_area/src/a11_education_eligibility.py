"""Run the deterministic education-eligibility screen and write its audit."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from utils.education_screen import (
    SCREEN_FIELDS, EDUCATION_PATTERNS, CLINICAL_USE_PATTERNS,
    MACHINE_LEARNING_PATTERNS, screen_education_eligibility,
)


def run(df: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    results_dir.mkdir(parents=True, exist_ok=True)
    screened = screen_education_eligibility(df)

    trace_columns = [c for c in [
        "EID", "Title", "Year", "Source title", "Tech_Category",
        "Education_Screen_Status", "Education_Screen_Disposition",
        "Has_Education_Cue", "Has_Clinical_Use_Cue", "Matched_Education_Cues",
        "Matched_Clinical_Use_Cues", *SCREEN_FIELDS,
    ] if c in screened.columns]
    screened[trace_columns].to_csv(results_dir / "table_A11_education_screen_trace.csv", index=False)

    counts = screened.groupby(
        ["Education_Screen_Disposition", "Has_Education_Cue", "Has_Clinical_Use_Cue"],
        dropna=False,
    ).size().rename("Documents").reset_index()
    counts["Percent of screened corpus"] = (
        100 * counts["Documents"] / max(len(screened), 1)
    ).round(2)
    counts.to_csv(results_dir / "table_A11_education_screen_counts.csv", index=False)

    included = screened["Education_Screen_Disposition"].eq("include_algorithm")
    classified = included & screened.get("Tech_Category", pd.Series(index=screened.index, dtype=object)).notna()
    n_include = int(included.sum())
    n_exclude = int((~included).sum())
    n_clinical_only = int((~included & screened["Has_Clinical_Use_Cue"]).sum())
    n_no_cue = n_exclude - n_clinical_only
    n_classified = int(classified.sum())

    spec = pd.DataFrame([
        {"Component": "Input fields", "Operational rule": "; ".join(SCREEN_FIELDS)},
        {"Component": "Education cue dictionary", "Operational rule": "; ".join(EDUCATION_PATTERNS)},
        {"Component": "Clinical-use dictionary (audit only)", "Operational rule": "; ".join(CLINICAL_USE_PATTERNS)},
        {"Component": "Computational-language safeguard", "Operational rule": "; ".join(MACHINE_LEARNING_PATTERNS)},
        {"Component": "Include", "Operational rule": "At least one qualifying education/training cue in any input field; a clinical-use cue does not override inclusion."},
        {"Component": "Exclude", "Operational rule": "No qualifying education/training cue in any input field, including records with clinical-use cues only."},
        {"Component": "Decision process", "Operational rule": "Case-insensitive regex matching; deterministic binary output; no human adjudication or Index Keywords."},
    ])
    spec.to_csv(results_dir / "table_A11_algorithm_specification.csv", index=False)

    methods = f"""# Education-eligibility algorithm — manuscript methods text

After deduplication, essential-metadata cleaning, XR thematic screening, and the 1991–2025 year restriction, **{len(screened):,} records** were screened using a deterministic lexical algorithm. The algorithm searched only the **title, abstract, and author keywords**; Scopus Index Keywords were not used. Text was matched case-insensitively against a prespecified dictionary of explicit education and training terms (dictionary and per-record matches are supplied in the repository audit files). A record was included if at least one qualifying education/training cue occurred in any searched field. This rule retained health-professions and clinical-training studies when an educational purpose was explicit, even if the record also mentioned diagnosis, treatment, patients, or clinical procedures. A record was excluded if it contained no qualifying education/training cue; clinical-use terminology alone did not establish educational eligibility. Generic computational phrases such as “machine learning,” “deep learning,” and “training data” were not accepted as education evidence without another substantive education cue. “Learning” alone was not an inclusion term.

The algorithm included **{n_include:,}** records and excluded **{n_exclude:,}** ({n_clinical_only:,} with clinical-use cues but no education cue; {n_no_cue:,} with neither qualifying education nor clinical-use cues). Of the included records, **{n_classified:,}** received one of the prespecified technology categories and formed the analytical corpus used to regenerate the results. The complete decision trace, cue dictionaries, and counts are provided in `table_A11_education_screen_trace.csv`, `table_A11_algorithm_specification.csv`, and `table_A11_education_screen_counts.csv`.

This is an operational lexical definition of educational relevance, not a manual semantic assessment. No human adjudication or inter-rater coefficient is part of the algorithm. Its reproducibility comes from the fixed fields, dictionary, matching rules, and record-level audit trail. As a limitation, relevant educational studies that do not use any dictionary cue may be excluded, while incidental mentions can trigger inclusion; conclusions should therefore be framed as applying to records meeting this explicit lexical eligibility rule.
"""
    (results_dir / "education_screen_paper_methods.md").write_text(methods, encoding="utf-8")
    print(f"[A11] Deterministic education eligibility: {n_include:,} included; "
          f"{n_exclude:,} excluded ({n_clinical_only:,} clinical-only, {n_no_cue:,} no-cue); "
          f"{n_classified:,} included and technology-classifiable.")
    return screened
