"""Predict reps for a session on 17 October 2026 with the frozen notebook code.

The script reads the model code from pipeline.ipynb and the cleaned sets from data/sets_clean.csv, trains the
random forest and gradient boosting on every cleaned row with history up to DATA_CUTOFF, and predicts three
sets of each lift as the first lift of the day. It never changes the notebook, data/ or the committed
demo/predictions_raw.csv.

Usage:
    python demo/predict_today.py                 # the filmed lift, Bicep Curl (Machine)
    python demo/predict_today.py "Lift name" ...  # other lifts by their cleaned exercise name

Writes:
    demo/predictions_rerun.csv, with one row per predicted set, and prints the same table.
"""
import ast, os, sys, datetime
import nbformat, numpy as np, pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(OUT)
TODAY = pd.Timestamp('2026-10-17')
DATA_CUTOFF = pd.Timestamp('2026-10-08')   # the committed predictions used cleaned data up to this date
LIFTS = sys.argv[1:] or ['Bicep Curl (Machine)']
RERUN_CSV = os.path.join(OUT, 'predictions_rerun.csv')
made_at = datetime.datetime.now().astimezone()

os.chdir(REPO)  # the frozen code reads data/ by relative path; nothing is written there
cells = nbformat.read('pipeline.ipynb', 4).cells
ns = {}
exec(cells[1].source, ns)                        # imports, SEED, clip_reps, EPLEY_DIVISOR
exec(cells[4].source.split('\n\ndev = ')[0], ns)  # read_clean_sets, load_dev (defs only)
exec(cells[6].source.split('\nfeats = build_features(dev)')[0], ns)  # feature code, FEATS
tree = ast.parse(cells[16].source)                # model settings and models(), defs only
keep = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))
        or (isinstance(n, ast.Assign) and n.targets[0].id in ('KNN_GRID', 'TREE_GRID', 'HGB_SETTINGS', 'ROLLING_STARTS', 'FOLD_MONTHS'))]
exec(compile(ast.Module(keep, []), 'cell16', 'exec'), ns)

sets = ns['read_clean_sets']()                    # ALL cleaned rows, development plus final test period
sets = sets[sets.date <= DATA_CUTOFF].reset_index(drop=True)
feats_all = ns['build_features'](sets)
train = feats_all[feats_all.has_hist == 1].reset_index(drop=True)
FEATS = ns['FEATS']
fitted = {}
for name in ['rf', 'hgb']:
    m = ns['models']()[name]
    m.fit(train[FEATS], train.reps)
    fitted[name] = m

rows = []
for lift in LIFTS:
    hist = sets[sets.exercise == lift]
    last_date = hist.date.max()
    last = hist[hist.date == last_date]
    normal = last[last.set_type != 'drop']
    weight = normal.weight_lb.mode().max()        # most common working weight in the latest session
    fake = pd.DataFrame({'session_id': 'demo_2026-10-17', 'date': TODAY, 'sess_ts': TODAY, 'source': 'demo',
                         'exercise': lift, 'set_idx': [0, 1, 2], 'set_type': 'normal', 'weight_lb': weight,
                         'reps': np.nan, 'row_id': sets.row_id.max() + 1 + np.arange(3)})
    f = ns['build_features'](pd.concat([sets, fake], ignore_index=True))
    f = f[f.session_id == 'demo_2026-10-17'].sort_values('set_idx')
    assert (f.has_hist == 1).all() and len(f) == 3
    for name, m in fitted.items():
        f[name] = ns['clip_reps'](m.predict(f[FEATS]))
    f['repeat_last'] = ns['clip_reps'](f.last_mean_reps)
    for _, r in f.iterrows():
        rows.append({'made_at': made_at.isoformat(timespec='seconds'), 'exercise': lift, 'weight_lb': weight,
                     'weight_kg': round(weight / 2.20462, 1), 'last_session': last_date.date(),
                     'last_session_sets': ' '.join(f'{w:g}x{int(x)}' for w, x in zip(last.weight_lb, last.reps)),
                     'set': int(r.set_idx) + 1, 'n_earlier_sets_session': int(r.n_earlier_sets_session),
                     'days_since_ex': int(r.days_since_ex), 'days_since_any': int(r.days_since_any),
                     'rf': round(r.rf, 2), 'hgb': round(r.hgb, 2), 'repeat_last': round(r.repeat_last, 2)})
out = pd.DataFrame(rows)
out['status'] = 'active'
out.to_csv(RERUN_CSV, index=False)
print(len(train), 'training rows,', train.session_id.nunique(), 'sessions, last', sets.date.max().date())
print(out.to_string())
