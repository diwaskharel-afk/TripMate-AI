NODE: SUPERVISED PREDICTION

Features are prepared. task_type is "predict" — non-temporal, held-out rows
from the same population. You have a limited tool-call budget for this node —
fit AND evaluate each model (baseline or candidate) in ONE run_python call
each, not separate calls per step, so you have room to finish and write
state.json. Your job this node:

- Split train/test (e.g. 80/20), stratified if classification and classes are
  imbalanced.
- Establish a naive baseline (majority class / mean predictor) FIRST, and
  record it in state.json as a model entry with "is_baseline": true, so later
  numbers mean something.
- Try at least two model families appropriate to the task (e.g.
  logistic/linear regression as an interpretable baseline model, plus a tree
  ensemble like Random Forest or Gradient Boosting).
- Evaluate with metrics matched to the task: classification -> accuracy,
  precision/recall/F1; regression -> RMSE, MAE, R².
- Pick the best model on held-out performance, not training performance. Flag
  overfitting if train/test performance diverge sharply (note it in
  feature_prep_notes or a new "modeling_notes" list if you add one).

When done, run a run_python call that updates /home/user/state.json with:
- "models_tried": a list of {"name", "metrics": {...}, "is_baseline": bool},
  baseline first, then each model you tried, in the order tried. The metric
  the eval harness checks against the baseline should use the SAME metric
  name key across every entry (e.g. always "rmse" or always "accuracy").

Printing the metrics table in your final answer text does not save it — you
must actually execute the write. An empty or unwritten models_tried list means
the report will incorrectly say no models were trained even though you did
the work, so do this before you stop.

Then print the metrics table, and stop.
