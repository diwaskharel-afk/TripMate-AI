NODE: FORECASTING

Features are prepared. task_type is "forecast" — predicting beyond the
observed time range. You have a limited tool-call budget for this node — fit
AND evaluate/backtest each model (baseline or candidate) in ONE run_python
call each, not separate calls per step, so you have room to finish and write
state.json. Your job this node:

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

When done, run a run_python call that updates /home/user/state.json with:
- "models_tried": list of {"name", "metrics": {...}, "is_baseline": bool}
  using a consistent metric key across baseline and candidates (e.g. "mae")
  computed on the backtest/holdout window.
- "forecast": a list of {"period", "point_estimate", "lower", "upper"} for
  the requested future point(s).

Printing the forecast table in your final answer text does not save it — you
must actually execute the write. An empty or unwritten models_tried/forecast
means the report will incorrectly say no forecast was produced, so do this
before you stop.

Then print the forecast table, and stop.
