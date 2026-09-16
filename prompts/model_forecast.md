NODE: FORECASTING

Features are prepared. task_type is "forecast" — predicting beyond the
observed time range. Your job this node:

- Never use a random train/test split for time series — split chronologically
  to avoid leakage. Establish a naive baseline (e.g. last-value or seasonal
  naive) FIRST and record it as a model entry with "is_baseline": true.
- Try a small set of approaches appropriate to what's actually installed
  (check before assuming a library exists): e.g. a regression with time trend
  + exogenous variables, and a classical time-series model (ARIMA/SARIMAX).
- Backtest with a walk-forward validation window if you have enough history,
  rather than trusting a single split.
- When projecting beyond the last observed date, this is extrapolation:
  report a point estimate AND an uncertainty range (prediction interval, or
  spread across your candidate models). Name the main risks (regime change,
  missing drivers, non-stationarity). Never present a single confident number
  as if it were interpolation.

When done, update /home/user/state.json with:
- "models_tried": list of {"name", "metrics": {...}, "is_baseline": bool}
  using a consistent metric key across baseline and candidates (e.g. "mae")
  computed on the backtest/holdout window.
- "forecast": a list of {"period", "point_estimate", "lower", "upper"} for
  the requested future point(s).

Then print the forecast table, and stop.
