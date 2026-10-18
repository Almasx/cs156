# Predicting My Reps from My Gym Logs

Code: https://github.com/Almasx/cs156/tree/main/first-assignment

This project uses three gym app exports from 2023 to 2026. The pipeline cleans the data, builds features that use only past sessions, and compares simple baselines with linear regression, kNN, a regression tree, a random forest and gradient boosting. The models are evaluated with random, grouped by session and time forward splits, and once on a final test set from 2026-07-05 onward, which the code calls the holdout. The write-up and all results are in `pipeline.ipynb`.

## Files

* `pipeline.ipynb` is the write-up, with all outputs saved from one run from top to bottom that includes the final test set. The figure cell right after the final test cell reads only `data/results_dev.csv` and final test errors copied from the saved output above it.
* `clean.py` contains all cleaning rules, and `name_map.py` maps the exercise names.
* `requirements.txt` lists the Python packages needed to run everything.
* `overrides.csv` is an empty list of dropped rows to restore, which `clean.py` reads. After I froze it, I reworded only its comment lines. It still has no data rows, and its new hash is recorded in `data/clean_meta.json`.
* `data/sets_clean.csv` holds the cleaned sets that the notebook uses.
* `data/dropped.csv` lists every dropped row with the name of its rule, and `data/rule_table.csv` gives the row counts before and after each rule.
* `data/clean_meta.json` records the kept and dropped row counts and the hash of `overrides.csv` from the last cleaning run.
* `data/results_dev.csv` holds the development errors of every model and split, which the notebook writes and the final figure reads.
* `demo/` holds one real workout example. `predict_today.py` trains the frozen models on all cleaned data up to 2026-10-08 and predicts the filmed lift for 17 October 2026, writing `predictions_rerun.csv` so that the committed predictions stay unchanged, `prediction_sheet.md` and `predictions_raw.csv` keep the prediction for the filmed set of machine curls at 110 lb with the time it was made, and `machine_curl_2026-10-17.mp4` is the filmed machine curl set, with audio and all metadata removed.
* The pipeline diagram is embedded in `pipeline.ipynb` as a cell attachment.
* `pipeline.pdf` is the paper as a PDF.

## Cleaning details

* In `clean.py` the anomaly rules are `typo_above_3x_earlier_max`, `typo_above_3x_session_median` and `typo_below_quarter_session_top` (typos far above the earlier max, far above that day's other sets, or far below that day's top set), `more_than_30_reps` and `sudden_jump_vs_recent_sessions`. The three typo rules and the jump rule compare a set only with other sets of the same lift in the same session or in earlier sessions, so they never use a later session or the final test set. Anomalies are treated as probable errors and dropped.
* The notebook shows every rule with a plain name. The mapping from the rule names in `clean.py` to these plain names is `RULE_NAMES` in the first code cell after the diagram.
* Every dropped row goes to `data/dropped.csv` with the name of its rule, so each one can be checked by hand. Flagged rows stay in `data/sets_clean.csv` with one column for each flag.
* The cleaned file contains no bodyweight, location or device columns.

The folder `raw/` contains the original app exports `src1.csv`, `src2.csv` and `src3.csv`, so a reader can run `clean.py` on the real data. Running it reproduces `data/sets_clean.csv`, `data/dropped.csv` and `data/rule_table.csv` exactly.

## Run

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python clean.py                      # needs raw/src1.csv raw/src2.csv raw/src3.csv
jupyter nbconvert --to notebook --execute --output rerun.ipynb pipeline.ipynb
```

The nbconvert command writes a new file `rerun.ipynb`. Do not use `--inplace`, because that would replace the saved final test output in `pipeline.ipynb` with `holdout locked`.

Without `raw/`, for example in a copy that has only `data/`, the notebook skips the cleaning step and uses `data/sets_clean.csv`.

The final test cell, which comes just before the Final Test Set Results section, runs only when `FINAL=1` is set. Otherwise it prints `holdout locked`. Its outputs are saved in the notebook, so it should not be run again.
