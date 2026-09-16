NODE: CLASSIFY TASK

Your only job this node: decide which of three tasks this analysis is, before
any heavy work happens.

1. "eda" — EXPLORATION ONLY. No prediction target is implied by the user's
   question. Report findings and stop after EDA.
2. "predict" — SUPERVISED PREDICTION. A target column is named or implied,
   predicting held-out rows from the SAME population/time period.
3. "forecast" — FORECASTING. The data has a time dimension and the user wants
   a value at a future point beyond the observed range (e.g. "predict gold
   price in 2029" from historical price + other variables).

If the request doesn't clearly say which of these it is, infer the most
reasonable one from the wording and the data's shape (peek at the raw file
and/or load it into `df` to check for a date column and an obvious target),
and record that assumption.

When you're done, write into /home/user/state.json:
- "task_type": one of "eda" | "predict" | "forecast"
- "task_type_reasoning": one sentence explaining why
- "target_column": the column name if applicable, else null
- "shape": [n_rows, n_cols]

The user's question is: {question}

Then print the final state.json content so it's visible in your output, and stop.
