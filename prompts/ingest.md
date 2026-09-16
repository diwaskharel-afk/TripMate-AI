NODE: INGEST & VALIDATE

The task has already been classified (see state.json for task_type and
target_column). Your job this node:

- Load the dataset into a DataFrame called `df` (if not already loaded).
- Check shape, dtypes, duplicate rows, missing values (count + %), and
  obvious parsing issues (mixed types in a column, encoding artifacts).
- Fix only what's clearly necessary to proceed (parse date columns, coerce
  numeric-looking object columns). Note every change you make.

When done, update /home/user/state.json with:
- "columns": a list of {"name", "dtype", "missing_count", "missing_pct"} for
  every column
- "assumptions": append any fixes/assumptions you made (list of strings)

Then print the updated columns section, and stop.
