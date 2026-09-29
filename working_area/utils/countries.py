"""Country parsing shared by all collaboration and country plots.

Scopus affiliation strings are semicolon-separated affiliations, usually with
the country as the final comma-separated field. Parsing the same explicit
field in every figure avoids keyword/city substring guesses (e.g. "London",
"Toronto", or "Harvard") being mistaken for country evidence.
"""

from __future__ import annotations


COUNTRY_ALIASES = {
    "united states": "USA", "united states of america": "USA", "usa": "USA",
    "u.s.a.": "USA", "us": "USA",
    "united kingdom": "UK", "uk": "UK", "great britain": "UK",
    "russian federation": "Russia", "viet nam": "Vietnam",
    "korea, republic of": "South Korea", "republic of korea": "South Korea",
    "hong kong": "Hong Kong", "macao": "Macao", "taiwan": "Taiwan",
}

COUNTRY_ORDER = [
    "USA", "China", "UK", "Spain", "Taiwan", "South Korea", "Germany",
    "Australia", "Canada", "Italy", "Japan", "France",
]


def _canonical_country(value: str) -> str | None:
    key = " ".join(value.strip().lower().split())
    if not key or key in {"nan", "none", "null", "n/a"}:
        return None
    if key in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[key]
    # Scopus exports the country in English; title casing is a display-only
    # normalization that retains country names outside the paper's top 12.
    return " ".join(part.capitalize() for part in key.split())


def affiliation_countries(affiliations: object) -> set[str]:
    """Return distinct countries parsed from affiliation address suffixes."""
    countries: set[str] = set()
    for segment in str(affiliations).split(";"):
        parts = [part.strip() for part in segment.split(",") if part.strip()]
        if parts:
            country = _canonical_country(parts[-1])
            if country:
                countries.add(country)
    return countries


def leading_affiliation_country(affiliations: object) -> str | None:
    """Return the country suffix of the first Scopus affiliation segment."""
    first = str(affiliations).split(";")[0]
    parts = [part.strip() for part in first.split(",") if part.strip()]
    return _canonical_country(parts[-1]) if parts else None
