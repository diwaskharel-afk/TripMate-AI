NODE: FINAL REPORT

All prior phases are complete; their findings are in state.json. Your job
this node: write the report itself. Do not run more analysis — base every
claim on what's already in state.json (you may run one quick run_python call
to re-print state.json if you need to see it, but don't recompute results).

Write markdown into /home/user/state.json under "report_markdown", structured
as:
1. Dataset overview (rows, columns, what each variable represents if
   inferable)
2. Data quality notes (missingness, fixes applied)
3. Key EDA findings (2-5 bullet insights)
4. Approach taken and why (task classification, models tried)
5. Results (metrics table, best model, referencing plots by filename)
6. Prediction/forecast output if applicable, with uncertainty and caveats
7. Limitations and suggested next steps

Every number you cite must come from state.json's models_tried/forecast/
eda_findings — do not invent metrics.

Then print the report_markdown content, and stop.
