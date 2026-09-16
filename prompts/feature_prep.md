NODE: FEATURE PREP

EDA is done. This node only runs when task_type is "predict" or "forecast"
(see state.json). Your job this node:

- Handle missing values (impute or drop, justified by how much is missing).
- Encode categoricals, scale numeric features if the candidate models need it
  (linear/distance-based models do; tree ensembles don't).
- If task_type == "forecast": engineer lag features, rolling stats, and
  calendar features from the time column, and sort by time. Do NOT create a
  random shuffle split later — that's handled in the next node.

When done, update /home/user/state.json with:
- "feature_prep_notes": a list of short strings describing what you did

Then print a summary of the resulting feature set (columns + dtypes), and stop.
