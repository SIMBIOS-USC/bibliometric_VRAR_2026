"""Conservative shared institution-name normalization for tables and networks."""

from __future__ import annotations

import re


ALIASES = {
    "University of Toronto": ("university of toronto", "univ of toronto"),
    "University of Washington": ("university of washington", "univ. of washington"),
    "Harvard University": ("harvard university", "harvard medical school"),
    "Stanford University": ("stanford university",),
    "Massachusetts Institute of Technology (MIT)": (
        "massachusetts institute of technology", "massachusetts institute"
    ),
    "Imperial College London": ("imperial college london",),
    # Requested consolidation of the Danish university / Copenhagen hospital variants.
    "Københavns Universitet": (
        "københavns universitet", "university of copenhagen",
        "copenhagen university hospital"
    ),
    "Rigshospitalet": ("rigshospitalet",),
}

SCHOOL_TO_PARENT = {
    "harvard medical school": "Harvard University",
}

UNIVERSITY_MARKERS = ("university", "universit", "universidad", "università",
                      "universidade", "universität", "universiteit", "universitet")
INSTITUTION_MARKERS = UNIVERSITY_MARKERS + (
    "hospital", "institute", "institut", "clinic", "medical center",
    "medical centre", "research center", "research centre", "academy",
)
NON_PARENT_UNIT = re.compile(
    r"^(department|school|college|faculty|division|centre|center|unit|laboratory|lab)\b",
    re.IGNORECASE,
)
GENERIC_ORG_LABELS = {
    "university", "the university", "national", "college", "school",
    "hospital", "institute", "institut", "clinic", "academy", "center",
    "centre", "medical center", "medical centre",
}
AFFILIATED_UNIT_SUFFIX = re.compile(
    r"\s+(?:school|college|faculty|department|division)\b.*$",
    re.IGNORECASE,
)


def _canonical_parent(label: str) -> str:
    """Remove an attached sub-unit from a university label."""
    label = re.sub(r"\s+", " ", label).strip()
    if "university college" not in label.lower():
        label = AFFILIATED_UNIT_SUFFIX.sub("", label).strip()
    lower = label.lower()
    if lower in {"the university of hong kong", "university of hong kong"}:
        return "The University of Hong Kong"
    return label


def institution_items(value: object) -> list[str]:
    """Return parent-level institutions, not departments, schools, or colleges.

    For each Scopus affiliation segment, prefer an explicitly named university.
    If none is supplied, retain a named hospital, institute, clinic, or academy;
    generic academic sub-units are omitted rather than misreported as institutions.
    """
    found: set[str] = set()
    for segment in str(value).split(";"):
        segment = segment.strip()
        lower = segment.lower()
        if not segment or lower == "nan":
            continue
        matched = next(
            (canonical for canonical, variants in ALIASES.items()
             if any(variant in lower for variant in variants)
             or (canonical == "Massachusetts Institute of Technology (MIT)"
                 and re.search(r"\bmit\b", lower))),
            None,
        )
        if matched:
            found.add(matched)
            continue
        parts = [part.strip() for part in segment.split(",") if part.strip()]
        # Scopus may export the city name without the university name. This
        # explicit fallback is retained for the user's requested Copenhagen merge.
        if parts and parts[0].lower() in {"københavn", "copenhagen"}:
            found.add("Københavns Universitet")
            continue
        universities = [part for part in parts
                        if any(marker in part.lower() for marker in UNIVERSITY_MARKERS)
                        and part.strip().lower() not in GENERIC_ORG_LABELS]
        if universities:
            parent = _canonical_parent(universities[0])
            if parent.lower() not in GENERIC_ORG_LABELS:
                found.add(parent)
            continue
        organizations = [part for part in parts
                         if any(marker in part.lower() for marker in INSTITUTION_MARKERS)
                         and not NON_PARENT_UNIT.search(part)
                         and part.strip().lower() not in GENERIC_ORG_LABELS]
        # Retain named, standalone collegiate institutions, not generic
        # sub-units such as "College of Nursing".
        if not organizations:
            organizations = [part for part in parts
                             if "college" in part.lower()
                             and not re.match(r"^college\s+of\b", part, re.IGNORECASE)]
        if organizations:
            found.add(organizations[0])
            continue
        # A named school that is a recognizable parent institution is retained
        # only when no parent university appears in the same affiliation string.
        parent = next((name for alias, name in SCHOOL_TO_PARENT.items() if alias in lower), None)
        if parent:
            found.add(parent)
    return sorted(found)
