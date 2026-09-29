# XR bibliometrics

Scripts and analysis outputs for *Extended Reality (XR) in Education: A Systematic Bibliometric Analysis*. The pipeline starts from `scopus_raw.csv`; it does not run a search in Scopus.

## Run the pipeline

Use Python 3.10 or later. From the repository root:

```bash
python -m pip install \
  pandas numpy matplotlib seaborn tabulate networkx python-louvain \
  adjustText scikit-learn thefuzz
python working_area/main.py
```

By default, results go to `results/`. To use another export or output folder:

```bash
python working_area/main.py --input /path/to/scopus_export.csv --results-dir /path/to/results
```

You can skip analysis modules, for example `--skip a3,a5`. The accepted IDs are `a1`, `a2b`, `a3`, `a4`, `a4b`, `a5`, `a5b`, `a5c`, `a6`, `a7`, `a8`, and `a9`. Corpus construction and its audit files always run. Skipped modules leave any existing files untouched, so use a new output folder for a partial run if needed.

## How the corpus is built

`working_area/main.py` runs these steps:

1. Load the CSV and clean its column names.
2. Remove exact duplicate rows, repeated DOIs, then exact title/year/first-author matches. Rows missing a usable match key are kept; records with different DOIs are kept separately.
3. Remove records without a title, year, or source title.
4. Keep records with an XR term in the title, abstract, or author keywords, then restrict publication years to 1991–2025.
5. Apply the education screen and retain records assigned to a technology category.

The stage counts are in `results/pipeline_report.csv` and `results/table_A0_pipeline_audit.csv`; `fig_A0_pipeline_audit.png` shows the flow. Education-screen decisions and matched terms are in `table_A11_education_screen_trace.csv`.

## Screening and labels

The education screen looks for terms from the dictionary in `working_area/utils/education_screen.py` in titles, abstracts, and author keywords. A match includes the record. Clinical terms are logged but do not override an education match. Phrases such as “machine learning” and “training data” do not qualify by themselves. This screen is automatic; no human decisions or inter-rater scores are included.

Technology labels are assigned by the term lists in `working_area/utils/classifier.py`. The classifier searches title, abstract, author keywords, and Index Keywords.

| Label | Rule |
|---|---|
| VR | VR terms, with no AR or MR/XR term |
| AR | AR terms, with no VR or MR/XR term |
| MR/XR | Any MR/XR term; this rule takes priority |
| Hybrid/Multi-technology | Both VR and AR terms, with no MR/XR term |

The orientation figure (A2b) scores technical and pedagogical terms in the title, abstract, and author keywords. It assigns ties to Pedagogical. This is a dictionary score, not a human-coded measure.

## Outputs

The `results/` folder contains the cleaned corpus, audit files, tables, and figures. Modules cover yearly production (A1), author metrics (A3), country collaboration (A4/A4b), Bradford zones (A5), keyword summaries (A5b/A5c/A7), corpus metrics (A6), collaboration and co-citation networks (A8), and additional country, source, institution, and RCA analyses (A9). Figures are saved as PNG; most also have PDF copies.

## Before release

The Scopus export in this repository does not record the search query or retrieval date. Add them from the original search log; without them, the search cannot be reproduced and 2025 coverage cannot be checked. Citation counts are the values in this export and are not adjusted for publication age.

Author names and affiliations are taken from Scopus fields and can be ambiguous. The education and technology labels are dictionary-based, so incidental mentions and unlisted terminology can affect inclusion. The orientation score is especially uneven in this run: 9,472 of 9,535 classified records are labeled Pedagogical. Treat A2b as descriptive unless that measure is independently validated.

There is no `LICENSE` file yet. Choose a code license before release and check whether the Scopus export can be redistributed.

## Citation

<!-- Add the published citation and DOI here. -->
