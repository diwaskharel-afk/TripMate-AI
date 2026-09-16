NODE: SUPERVISED PREDICTION

Features are prepared. task_type is "predict" — non-temporal, held-out rows
from the same population. Your job this node:

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

When done, update /home/user/state.json with:
- "models_tried": a list of {"name", "metrics": {...}, "is_baseline": bool},
  baseline first, then each model you tried, in the order tried. The metric
  the eval harness checks against the baseline should use the SAME metric
  name key across every entry (e.g. always "rmse" or always "accuracy").

Then print the metrics table, and stop.
