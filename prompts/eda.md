NODE: EXPLORATORY ANALYSIS

Ingestion is done (see state.json for columns/dtypes). Your job this node:

- Univariate: distributions of numeric columns (mean/median/std/skew, min/max),
  value counts for categoricals.
- Bivariate: correlation matrix for numeric columns (Pearson + Spearman if
  relationships look non-linear); relationship between candidate features and
  the target (if task_type != "eda").
- Outliers: flag via IQR or z-score; decide case-by-case whether to cap,
  remove, or keep them, and say why.
- Save 1-3 of the most informative matplotlib figures to
  /home/user/plot_<name>.png (don't save more than that — pick the ones that
  actually matter for the report).

When done, update /home/user/state.json with:
- "eda_findings": a list of 2-5 short bullet-string insights (correlations
  that matter, notable distributions, outlier decisions)

Then print your findings list, and stop.
