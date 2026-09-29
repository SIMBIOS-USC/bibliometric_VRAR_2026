"""
utils/__init__.py
=================
Utility package for the XR Bibliometrics analysis suite.

Public API:
  - classifier.classify_technology       Canonical 4-category technology classifier
  - classifier.classify_orientation      Technical vs Pedagogical classifier
  - classifier.add_classifications       Batch wrapper (adds both columns to DataFrame)
  - classifier.print_classification_report
  - preprocess.run_pipeline             Full Scopus cleaning pipeline
  - preprocess.print_pipeline_report
  - plot_style.apply_global_style        Shared Matplotlib configuration
  - plot_style.save_figure
  - plot_style.CATEGORY_COLORS
  - plot_style.CATEGORY_ORDER
"""
