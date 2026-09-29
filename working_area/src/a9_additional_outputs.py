"""Additional manuscript-ready tables and figures.

All outputs are computed from the canonical classified corpus.  Journal
metrics that are not present in the Scopus export are deliberately reported
as NA rather than reconstructed or invented.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.classifier import CATEGORY_ORDER
from utils.plot_style import apply_global_style, save_figure
from utils.plot_style import CATEGORY_COLORS
from src.a4_country_collaboration import _leading_country
from utils.countries import affiliation_countries
from utils.institutions import institution_items


def _citations(df: pd.DataFrame) -> pd.Series:
    if "Cited by" in df.columns:
        return pd.to_numeric(df["Cited by"], errors="coerce").fillna(0)
    return pd.Series(0, index=df.index, dtype=float)


def _tokens(value: object) -> list[str]:
    return [re.sub(r"\s+", " ", str(x)).strip().lower()
            for x in str(value).split(";") if str(x).strip() and str(x).lower() != "nan"]


def _institution_table(df: pd.DataFrame) -> pd.DataFrame:
    counts: Counter[str] = Counter()
    cites: Counter[str] = Counter()
    citation_series = _citations(df)
    for idx, raw in df.get("Affiliations", pd.Series("", index=df.index)).items():
        institutions = institution_items(raw)
        for institution in institutions:
            counts[institution] += 1
            cites[institution] += float(citation_series.loc[idx])
    rows = [{"Institution": k, "Documents": v, "Total Citations": int(cites[k]),
             "CPP": round(cites[k] / v, 2) if v else 0} for k, v in counts.items()]
    result = pd.DataFrame(rows).sort_values(["Documents", "Total Citations"], ascending=False)
    must_include = {
        "University of Toronto", "University of Washington", "Harvard Medical School",
        "Stanford University", "Massachusetts Institute of Technology (MIT)",
        "Imperial College London", "Københavns Universitet", "Rigshospitalet",
    }
    keep_names = set(result.head(20)["Institution"]) | must_include
    return result[result["Institution"].isin(keep_names)]


def _h_index(values: pd.Series) -> int:
    ordered = sorted(pd.to_numeric(values, errors="coerce").fillna(0).astype(int), reverse=True)
    return int(sum(c >= i for i, c in enumerate(ordered, 1)))


def _source_table(df: pd.DataFrame) -> pd.DataFrame:
    citation_series = _citations(df)
    rows = []
    for source, sub in df.groupby("Source title", dropna=True):
        source = str(source).strip()
        if not source or source.lower() == "nan":
            continue
        cites = citation_series.loc[sub.index]
        rows.append({
            "Source": source,
            "Articles": len(sub),
            "Total Citations": int(cites.sum()),
            "CPP": round(float(cites.mean()), 2) if len(sub) else 0,
            "H-index": _h_index(cites),
            "Q": np.nan,
            "SJR": np.nan,
        })
    return pd.DataFrame(rows).sort_values(["Articles", "Total Citations"], ascending=False).head(20)


def _keyword_table(df: pd.DataFrame) -> pd.DataFrame:
    author = Counter()
    index = Counter()
    for value in df.get("Author Keywords", pd.Series(dtype=object)).dropna():
        author.update(_tokens(value))
    for value in df.get("Index Keywords", pd.Series(dtype=object)).dropna():
        index.update(_tokens(value))
    n = max(len(author), len(index), 10)
    a = author.most_common(10)
    b = index.most_common(10)
    rows = []
    for i in range(10):
        rows.append({"Rank": i + 1,
                     "Author Keywords": a[i][0] if i < len(a) else "",
                     "Author Freq.": a[i][1] if i < len(a) else 0,
                     "Index Keywords": b[i][0] if i < len(b) else "",
                     "Index Freq.": b[i][1] if i < len(b) else 0})
    return pd.DataFrame(rows)


def _bar_table(table: pd.DataFrame, out: Path, label: str, value: str, title: str) -> None:
    shown = table.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    palette = sns.color_palette("viridis", len(shown))
    ax.barh(shown[label], shown[value], color=palette, edgecolor="white", linewidth=0.7)
    for y, v in enumerate(shown[value]):
        ax.text(float(v), y, f"  {v:,.0f}", va="center", fontsize=9)
    ax.set_title(title, weight="bold", pad=14)
    ax.set_xlabel(value)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=.25)
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=.34, right=.96, top=.88, bottom=.12)
    save_figure(fig, str(out))


def _keyword_figure(table: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=False)
    for ax, name, freq, color, title in [
        (axes[0], "Author Keywords", "Author Freq.", "#1565c0", "Author keywords"),
        (axes[1], "Index Keywords", "Index Freq.", "#00897b", "Index keywords"),
    ]:
        part = table.iloc[::-1]
        ax.barh(part[name], part[freq], color=color, alpha=.88)
        ax.set_title(title, weight="bold")
        ax.set_xlabel("Frequency")
        ax.grid(axis="x", alpha=.25)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=9)
    fig.suptitle("Most frequent keywords in the canonical corpus", weight="bold", y=.98)
    fig.tight_layout(rect=(0, 0, 1, .95))
    save_figure(fig, str(out))


def _rca(df: pd.DataFrame, out_csv: Path, out_fig: Path) -> None:
    work = df.copy()
    work["Country"] = work["Affiliations"].map(_leading_country)
    work = work[work["Country"].notna() & work["Tech_Category"].notna()]
    all_counts = pd.crosstab(work["Country"], work["Tech_Category"]).reindex(columns=CATEGORY_ORDER, fill_value=0)
    # Standard RCA: a country's category share divided by that category's
    # share in the full country-identifiable corpus. Do not recalculate the
    # world benchmark after trimming the display to the top countries.
    global_category_share = all_counts.sum(axis=0) / all_counts.values.sum()
    all_rca = all_counts.div(all_counts.sum(axis=1), axis=0).div(global_category_share, axis=1)
    display_counts = all_counts.loc[all_counts.sum(axis=1) >= 10]
    display_counts = display_counts.loc[display_counts.sum(axis=1).sort_values(ascending=False).index].head(15)
    rca = all_rca.loc[display_counts.index]
    rca.to_csv(out_csv)

    # Document-level nonparametric bootstrap (fixed seed) for uncertainty in
    # RCA. Resample the full country-identifiable corpus so the country and
    # global category shares are estimated from the same bootstrap draw.
    country_codes, country_labels = pd.factorize(work["Country"], sort=True)
    category_codes = pd.Categorical(work["Tech_Category"], categories=CATEGORY_ORDER).codes
    n_country = len(country_labels)
    n_categories = len(CATEGORY_ORDER)
    n_docs = len(work)
    observed_countries = set(display_counts.index)
    row_for_country = {country: i for i, country in enumerate(country_labels)}
    bootstrap = np.full((1000, n_country, n_categories), np.nan, dtype=float)
    rng = np.random.default_rng(20260929)
    for b in range(bootstrap.shape[0]):
        sampled = rng.integers(0, n_docs, size=n_docs)
        matrix = np.bincount(
            country_codes[sampled] * n_categories + category_codes[sampled],
            minlength=n_country * n_categories,
        ).reshape(n_country, n_categories)
        country_n = matrix.sum(axis=1)
        global_counts = matrix.sum(axis=0)
        global_share = global_counts / n_docs
        country_share = np.divide(matrix, country_n[:, None],
                                  out=np.full_like(matrix, np.nan, dtype=float),
                                  where=country_n[:, None] > 0)
        bootstrap[b] = country_share / global_share[None, :]
    ci_rows = []
    for country in display_counts.index:
        idx = row_for_country[country]
        country_n = int(all_counts.loc[country].sum())
        for j, category in enumerate(CATEGORY_ORDER):
            draws = bootstrap[:, idx, j]
            draws = draws[np.isfinite(draws)]
            ci_rows.append({
                "Country": country,
                "Category": category,
                "Country documents": country_n,
                "Country-category documents": int(all_counts.loc[country, category]),
                "RCA": float(rca.loc[country, category]),
                "Bootstrap 95% CI lower": float(np.quantile(draws, .025)) if len(draws) else np.nan,
                "Bootstrap 95% CI upper": float(np.quantile(draws, .975)) if len(draws) else np.nan,
                "Bootstrap replicates": 1000,
                "Minimum country N": 10,
            })
    pd.DataFrame(ci_rows).to_csv(out_csv.with_name("table_A9_rca_confidence_intervals.csv"), index=False)
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    sns.heatmap(rca, annot=True, fmt=".2f", cmap="YlGn", center=1, linewidths=.7,
                linecolor="white", cbar_kws={"label": "RCA value"}, ax=ax)
    ax.set_title("Revealed comparative advantage by country and technology", weight="bold", pad=14)
    ax.set_xlabel("Technology category")
    ax.set_ylabel("Country")
    fig.tight_layout()
    save_figure(fig, str(out_fig))


def _donuts(df: pd.DataFrame, out: Path) -> None:
    work = df[df["Affiliations"].notna()].copy()
    work["Country"] = work["Affiliations"].map(_leading_country)
    work["Countries"] = work["Affiliations"].map(affiliation_countries)
    work["Collab"] = work["Countries"].map(
        lambda countries: "MCP" if len(countries) > 1 else ("SCP" if len(countries) == 1 else "Unresolved")
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    colors = ["#1976d2", "#ff8f00", "#43a047", "#8e24aa", "#00838f", "#6d4c41", "#9e9d24", "#546e7a", "#d81b60", "#5e35b1", "#90a4ae"]
    for ax, kind in zip(axes, ["SCP", "MCP"]):
        subset = work.loc[work["Collab"] == kind, "Country"]
        unknown = int(subset.isna().sum())
        s = subset.value_counts().dropna()
        s = s.sort_values(ascending=False)
        top = s.head(10)
        others = int(s.iloc[10:].sum()) + unknown
        if others:
            top.loc["Others / unresolved"] = others
        wedges, _, autotexts = ax.pie(top.values, labels=None, startangle=90, counterclock=False,
                                      colors=colors[:len(top)], autopct=lambda p: f"{p:.1f}%" if p >= 4 else "",
                                      pctdistance=.72, wedgeprops={"width": .42, "edgecolor": "white"})
        ax.text(0, 0, f"{kind}\n(n={int(subset.shape[0])})", ha="center", va="center", weight="bold", fontsize=14)
        ax.set_title(f"{kind} by country", weight="bold")
        ax.legend(wedges, top.index, loc="center left", bbox_to_anchor=(1.0, .5), frameon=False, fontsize=9)
    fig.suptitle("Country distribution of single- and multi-country publications", weight="bold", y=.98)
    fig.tight_layout(rect=(0, 0, 1, .94))
    save_figure(fig, str(out))


def run(df: pd.DataFrame, results_dir: Path) -> None:
    print("\n[A9] Additional manuscript tables and figures")
    apply_global_style(font_size=11)
    results_dir.mkdir(parents=True, exist_ok=True)

    institutions = _institution_table(df)
    institutions.to_csv(results_dir / "table_A9_institutions.csv", index=False)
    _bar_table(institutions, results_dir / "fig_A9_institutions.png", "Institution", "Documents",
               "Leading institutions by number of documents")

    sources = _source_table(df)
    sources.to_csv(results_dir / "table_A9_sources.csv", index=False)
    _bar_table(sources, results_dir / "fig_A9_sources.png", "Source", "Articles",
               "Leading sources by number of articles")

    # Reuse A7's per-document de-duplicated and normalized keyword counts so
    # the manuscript table and its figure cannot report different frequencies.
    a7_path = results_dir / "table_A7_keywords_comparison.csv"
    if a7_path.exists():
        a7 = pd.read_csv(a7_path)
        keywords = pd.DataFrame({
            "Rank": range(1, len(a7) + 1),
            "Author Keywords": a7["Author Keywords"],
            "Author Freq.": a7["Freq (Auth)"],
            "Index Keywords": a7["Index Keywords"],
            "Index Freq.": a7["Freq (Index)"],
        })
    else:
        keywords = _keyword_table(df)
    keywords.to_csv(results_dir / "table_A9_keywords.csv", index=False)
    _keyword_figure(keywords, results_dir / "fig_A9_keywords.png")

    _rca(df, results_dir / "table_A9_rca.csv", results_dir / "fig_A9_rca.png")
    _donuts(df, results_dir / "fig_A9_scp_mcp_donuts.png")
    print("  → A9 tables and figures saved")
