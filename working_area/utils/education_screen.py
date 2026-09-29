"""Deterministic education-eligibility screen using title/abstract/author keywords."""

from __future__ import annotations

import re
import pandas as pd


SCREEN_FIELDS = ["Title", "Abstract", "Author Keywords"]

# Labels are included in the audit trace; expressions use case-insensitive regex.
EDUCATION_PATTERNS = {
    "education/educational": r"\beducation(?:al)?\b",
    "teaching": r"\bteaching\b",
    "teacher(s)": r"\bteachers?\b",
    "instruction/instructional": r"\binstruction(?:al)?\b",
    "curriculum/curricula": r"\bcurricul(?:um|a)\b",
    "pedagogy/pedagogical": r"\bpedagog(?:y|ical)\b",
    "student(s)": r"\bstudents?\b",
    "learner(s)": r"\blearners?\b",
    "classroom(s)": r"\bclassrooms?\b",
    "school(s)": r"\bschools?\b",
    "course(s)": r"\bcourses?\b",
    "training": r"\btraining\b",
    "trainee(s)": r"\btrainees?\b",
    "resident(s)": r"\bresidents?\b",
    "medical education": r"\bmedical education\b",
    "nursing education": r"\bnursing education\b",
    "health professions education": r"\bhealth professions education\b",
    "clinical education": r"\bclinical education\b",
    "learning outcome(s)": r"\blearning outcomes?\b",
    "learning performance": r"\blearning performance\b",
    "learning gain(s)": r"\blearning gains?\b",
    "learning experience": r"\blearning experience\b",
    "learning environment": r"\blearning environment\b",
    "learning objective(s)": r"\blearning objectives?\b",
    "knowledge acquisition": r"\bknowledge acquisition\b",
    "skill(s) acquisition": r"\bskills? acquisition\b",
    "professional development": r"\bprofessional development\b",
    "resident training": r"\bresident training\b",
}

CLINICAL_USE_PATTERNS = {
    "diagnosis/diagnostic": r"\bdiagnos(?:is|tic)\b",
    "treatment": r"\btreatment\b",
    "therapy": r"\btherapy\b",
    "rehabilitation": r"\brehabilitation\b",
    "patient care": r"\bpatient care\b",
    "patient outcome(s)": r"\bpatient outcomes?\b",
    "surgical planning": r"\bsurgical planning\b",
    "treatment planning": r"\btreatment planning\b",
    "intraoperative": r"\bintraoperative\b",
    "preoperative planning": r"\bpreoperative planning\b",
    "surgery planning": r"\bsurgery planning\b",
}

MACHINE_LEARNING_PATTERNS = [
    r"\bmachine learning\b", r"\bdeep learning\b", r"\btraining data\b",
    r"\btraining dataset\b", r"\btraining set\b", r"\btrain(?:ing)? the model\b",
]

# These phrases are insufficient by themselves when the only apparent cue is
# generic "learning" or "training"; concrete education cues still qualify.
GENERIC_LEARNING_LABELS = {"training", "learning performance", "learning gain(s)"}


def _text(row: pd.Series) -> str:
    return " ".join(str(row.get(field, "")) for field in SCREEN_FIELDS
                    if pd.notna(row.get(field, ""))).lower()


def _matches(text: str, patterns: dict[str, str]) -> list[str]:
    return [label for label, pattern in patterns.items()
            if re.search(pattern, text, flags=re.IGNORECASE)]


def screen_education_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    """Apply a binary lexical rule; no human decision or clinical veto is used."""
    out = df.copy()
    statuses, dispositions, ed_matches, care_matches = [], [], [], []
    has_education, has_clinical = [], []
    for _, row in out.iterrows():
        text = _text(row)
        education = _matches(text, EDUCATION_PATTERNS)
        clinical = _matches(text, CLINICAL_USE_PATTERNS)
        machine_context = any(re.search(pattern, text, flags=re.IGNORECASE)
                              for pattern in MACHINE_LEARNING_PATTERNS)

        # Prevent generic computational-learning/training wording from acting
        # as educational evidence unless another explicit education cue exists.
        substantive = [cue for cue in education if cue not in GENERIC_LEARNING_LABELS]
        if machine_context and not substantive:
            education = []

        eligible = bool(education)
        dispositions.append("include_algorithm" if eligible else "exclude_algorithm")
        statuses.append("Included — education/training cue detected" if eligible else
                        "Excluded — no qualifying education/training cue")
        ed_matches.append("; ".join(education))
        care_matches.append("; ".join(clinical))
        has_education.append(eligible)
        has_clinical.append(bool(clinical))

    out["Education_Screen_Status"] = statuses
    out["Education_Screen_Disposition"] = dispositions
    out["Has_Education_Cue"] = has_education
    out["Has_Clinical_Use_Cue"] = has_clinical
    out["Matched_Education_Cues"] = ed_matches
    out["Matched_Clinical_Use_Cues"] = care_matches
    out["Education_Screen_Fields"] = ", ".join(SCREEN_FIELDS)
    return out
