"""
src/__init__.py
===============
Analysis modules package for the XR Bibliometrics analysis suite.

Module index:
  a1_publication_growth     → Publication & citation evolution (4 categories × year)
  a2_orientation_analysis   → Technical vs Pedagogical orientation (Figure 2 / RQ4)
  a3_author_analysis        → Top author productivity per category
  a4_country_collaboration  → SCP vs MCP country analysis
  a5_bradford_zones         → Bradford's law & source concentration
  a6_bibliometric_table     → Main bibliometric information table

Each module exposes a single `run(df, results_dir)` entry point called by main.py.
"""
