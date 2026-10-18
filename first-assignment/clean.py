# clean + merge the three gym app exports -> data/sets_clean.csv, data/dropped.csv, data/rule_table.csv
# usage: python clean.py   (reads raw/src1.csv raw/src2.csv raw/src3.csv)
# every anomaly rule only looks at the same session or EARLIER sessions of the same lift, never later ones
import os, json, hashlib
import numpy as np
import pandas as pd
from collections import Counter, namedtuple
from name_map import canon

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.environ.get('RAW_DIR', os.path.join(HERE, 'raw'))
OUT = os.path.join(HERE, 'data')
KG2LB = 2.20462262185
HOLDOUT_START = pd.Timestamp('2026-07-05')
FIRST_REAL_SRC2 = pd.Timestamp('2025-07-27')
KG_ERAS = [(pd.Timestamp('2025-07-27'), pd.Timestamp('2025-09-10')),
           (pd.Timestamp('2026-01-05'), pd.Timestamp('2026-04-20'))]
SRC1_BW_TOTAL = {'Pull Up', 'Parallel Bar Triceps Dip', 'Hybrid Pullup', 'Narrow pull up', 'decline Pushup',
                 'muscle up', 'Dips', 'Pushup', 'Crunch', 'Back extension'}
DB_PAIR_EXERCISES = {'Incline Bench Press (Dumbbell)', 'Bench Press (Dumbbell)', 'Seated Overhead Press (Dumbbell)',
                     'Shrug (Dumbbell)', 'Hammer Curl (Dumbbell)'}
ADDED_LOAD_BW = {'Pull Up (Weighted)', 'Triceps Dip (Weighted)', 'Wide Pull Up (Weighted)',
                 'Close Grip Pull Up (Weighted)', 'Back Extension'}
# thresholds for the chronological per-lift scan (typo, kg_typed_in_lb_field and sudden jump rules). History = earlier sessions only
TYPO_WINDOW_DAYS = 180         # typo_above_3x_earlier_max compares with the lift's max over the last 180 days
TYPO_FACTOR = 3                # typo_above_3x_earlier_max and typo_above_3x_session_median: more than 3x is a typo
MIN_HISTORY_SESSIONS = 2       # need at least 2 earlier sessions before judging
LOW_TYPO_SHARE = 0.25          # typo_below_quarter_session_top: below 1/4 of the session top is a typo
KG_REF_DAYS, KG_REF_DAYS_LONG = 120, 400   # kg_typed_in_lb_field reference window, longer fallback
KG_RATIO_LOW, KG_RATIO_HIGH = 0.363, 0.567 # kg_typed_in_lb_field: raw number looks like kg if it is this share of the lb reference
KG_FIXED_LOW, KG_FIXED_HIGH = 0.8, 1.25    # ...and x2.20462 lands this close to the reference
JUMP_WINDOW_DAYS = 365         # sudden_jump_vs_recent_sessions reference window
JUMP_LOW, JUMP_HIGH = 0.58, 1.7  # sudden_jump_vs_recent_sessions: session top outside this ratio of the past median is a jump
HISTORY_SESSIONS = 5           # kg_typed_in_lb_field and sudden_jump_vs_recent_sessions use the median of the last 5 sessions
RESTORABLE = {'typo_above_3x_earlier_max', 'typo_above_3x_session_median', 'typo_below_quarter_session_top',
              'more_than_30_reps', 'sudden_jump_vs_recent_sessions'}


def load1():
    """Read the FitNotes export (src1) into the common columns.

    Returns:
        DataFrame, src1 rows; kg weights converted to lb.
    """
    d = pd.read_csv(os.path.join(RAW, 'src1.csv'))
    d['src_row'] = np.arange(len(d)) + 2
    d['source'] = 'src1'
    d['date'] = pd.to_datetime(d['Date'])
    d['session_key'] = d['Date']
    d['raw_exercise'] = d['Exercise']
    d['set_type_raw'] = 'normal'
    d['weight_raw'] = d['Weight']
    d['weight_lb'] = d['Weight'] * KG2LB
    d['reps'] = d['Reps'].astype(float)
    d['is_timer'] = False
    d['time_only'] = d['Weight'].isna() & (d['Time'].notna() | d['Distance'].notna())
    return d


def load2():
    """Read the second app export (src2) into the common columns.

    Returns:
        DataFrame, src2 rows; weights already lb, Set Order mapped to set types.
    """
    d = pd.read_csv(os.path.join(RAW, 'src2.csv'))
    d['src_row'] = np.arange(len(d)) + 2
    d['source'] = 'src2'
    ts = pd.to_datetime(d['Date'].str.replace('\u202f', ' '), format='%Y-%m-%d %I:%M:%S %p')
    d['date'] = ts.dt.normalize()
    d['session_key'] = ts.dt.strftime('%Y-%m-%dT%H:%M:%S')
    d['raw_exercise'] = d['Exercise Name']
    so = d['Set Order'].astype(str)
    d['set_type_raw'] = so.map(lambda s: {'Rest Timer': 'rest_timer', 'D': 'drop', 'W': 'warmup', 'F': 'failure'}.get(s, 'normal'))
    d['weight_raw'] = d['Weight']
    d['weight_lb'] = d['Weight']
    d['reps'] = d['Reps'].astype(float)
    d['is_timer'] = so.eq('Rest Timer')
    d['time_only'] = False
    return d


def load3():
    """Read the third app export (src3) into the common columns.

    Returns:
        DataFrame, src3 rows; kg weights converted to lb.
    """
    d = pd.read_csv(os.path.join(RAW, 'src3.csv'))
    d['src_row'] = np.arange(len(d)) + 2
    d['source'] = 'src3'
    ts = pd.to_datetime(d['start_time'], format='%d %b %Y, %H:%M')
    d['date'] = ts.dt.normalize()
    d['session_key'] = ts.dt.strftime('%Y-%m-%dT%H:%M:%S')
    d['raw_exercise'] = d['exercise_title']
    d['set_type_raw'] = d['set_type'].map({'normal': 'normal', 'dropset': 'drop', 'warmup': 'warmup', 'failure': 'failure'})
    d['weight_raw'] = d['weight_kg']
    d['weight_lb'] = d['weight_kg'] * KG2LB
    d['reps'] = d['reps'].astype(float)
    d['is_timer'] = False
    d['time_only'] = False
    return d


cols = ['source', 'src_row', 'date', 'session_key', 'raw_exercise', 'set_type_raw', 'weight_raw', 'weight_lb', 'reps', 'is_timer', 'time_only']
parts = []
for i, f in enumerate([load1, load2, load3]):
    x = f()[cols].copy()
    x['exercise'] = [canon(i + 1, n)[0] for n in x.raw_exercise]
    x['map_conf'] = [canon(i + 1, n)[1] for n in x.raw_exercise]
    parts.append(x)
d = pd.concat(parts, ignore_index=True)
d['session_id'] = d.source + '_' + d.session_key
d['file_order'] = d.groupby('source').cumcount()
d['key'] = d.source + ':' + d.src_row.astype(str)
d['is_dev'] = d.date < HOLDOUT_START

# overrides = rows I looked at by hand in data/dropped.csv and want back. dev rows only, anomaly rules only.
ov_path = os.path.join(HERE, 'overrides.csv')
ov = pd.read_csv(ov_path, comment='#') if os.path.exists(ov_path) else pd.DataFrame(columns=['source', 'src_row', 'note'])
ov_keys = set(ov.source.astype(str) + ':' + ov.src_row.astype(str))
bad_ov = ov_keys & set(d.loc[~d.is_dev, 'key'])
if bad_ov:
    raise SystemExit(f'overrides point at holdout rows, not allowed: {sorted(bad_ov)}')
ov_hash = hashlib.sha256(open(ov_path, 'rb').read()).hexdigest() if os.path.exists(ov_path) else 'none'

log = []
dropped = []


def note(rule, desc, before, after, changed=0, restored=0):
    """Add one line to the rule table.

    Args:
        rule: rule id.
        desc: short description.
        before: row count before the rule.
        after: row count after the rule.
        changed: rows changed in place.
        restored: rows kept back by overrides.

    Returns:
        None.
    """
    log.append(dict(rule=rule, rows_before=before, rows_after=after, dropped=before - after, changed=changed,
                    restored=restored, description=desc))


def drop(d, mask, rule, desc):
    """Remove rows matched by a rule and log the counts.

    Args:
        d: current rows.
        mask: rows the rule hits.
        rule: rule id.
        desc: short description.

    Returns:
        DataFrame, remaining rows. Dev rows in overrides.csv are kept for restorable rules; dropped rows go to the dropped list.
    """
    mask = pd.Series(mask, index=d.index)
    restored = 0
    if rule in RESTORABLE:
        r = mask & d.key.isin(ov_keys) & d.is_dev
        restored = int(r.sum())
        mask = mask & ~r
    sub = d[mask].copy()
    sub['rule'] = rule
    dropped.append(sub)
    note(rule, desc, len(d), int((~mask).sum()), restored=restored)
    return d[~mask].copy()


note('raw_rows_loaded', 'raw rows from the three exports', len(d), len(d))
d = drop(d, d.is_timer, 'rest_timer_saved_as_set', "src2 'Rest Timer' pseudo rows (weight 0, reps 0), not sets")

# same_workout_in_two_apps, the same workout logged in two apps (within +-1 day, >= half of the smaller session's sets match)
SessionSig = namedtuple('SessionSig', 'source date sets_raw sets_5lb n_sets')
signature = {}
for (source, sid, date), sess in d.groupby(['source', 'session_id', 'date']):
    reps = sess.reps.fillna(-1).astype(int)
    sets_raw = Counter(zip(sess.exercise, reps, sess.weight_raw.fillna(-1).round(1)))
    sets_5lb = Counter(zip(sess.exercise, reps, (sess.weight_lb / 5).round().fillna(-1).astype(int)))
    signature[sid] = SessionSig(source, date, sets_raw, sets_5lb, len(sess))
dup = set()
ids = list(signature)
for i, first in enumerate(ids):
    for second in ids[i + 1:]:
        A, B = signature[first], signature[second]
        if A.source == B.source or abs((A.date - B.date).days) > 1:
            continue
        shared = max(sum((A.sets_raw & B.sets_raw).values()), sum((A.sets_5lb & B.sets_5lb).values()))
        smaller = first if A.n_sets <= B.n_sets else second
        if shared >= 1 and shared / signature[smaller].n_sets >= 0.5:
            dup.add(smaller)
d = drop(d, d.session_id.isin(dup), 'same_workout_in_two_apps', 'smaller of two sessions from different apps (+-1 day) sharing >= half of its sets')
d = drop(d, (d.source == 'src2') & (d.date < FIRST_REAL_SRC2), 'test_entry_before_current_app', 'src2 rows before its first real session 2025-07-27 (app test entries)')
d = drop(d, d.time_only, 'time_or_distance_only', 'src1 rows with time/distance and no weight (stretches, sprints, holds)')
m = ((d.source == 'src1') & d.raw_exercise.isin(SRC1_BW_TOTAL)) | \
    ((d.source == 'src3') & d.exercise.isin(['Pull Up (Weighted)', 'Triceps Dip (Weighted)']) & (d.weight_raw >= 60))
d = drop(d, m, 'bodyweight_in_weight_field', 'pull-up/dip/pushup rows where the weight field is bodyweight (+added); bodyweight stays out')
d = drop(d, d.set_type_raw == 'warmup', 'marked_warmup', 'sets marked warmup (src2 W, src3 warmup)')
d = drop(d, d.weight_raw.isna(), 'missing_weight', 'no weight (unweighted bodyweight sets)')
d = drop(d, d.weight_lb <= 3.0, 'placeholder_weight', 'weight <= 3 lb (0 or placeholder values)')
d = drop(d, d.reps.isna() | (d.reps <= 0), 'missing_reps', 'reps missing or 0')

# unit fixes (no rows removed)
n = len(d)
snap = (d.source == 'src3') & ((d.weight_lb * 2 - np.round(d.weight_lb * 2)).abs() < 0.12) & (d.weight_lb != np.round(d.weight_lb * 2) / 2)
d.loc[snap, 'weight_lb'] = np.round(d.loc[snap, 'weight_lb'] * 2) / 2
note('kg_export_converted_to_lb', 'src1 and src3 are kg -> x2.20462; src2 export already lb', n, n, changed=int(d.source.isin(['src1', 'src3']).sum()))
note('rounded_lb_value_fixed', 'src3 values that were typed in lb come back as x.99/x.01 -> snapped to 0.5 lb', n, n, changed=int(snap.sum()))


def kg_converted(x):
    """Check whether a src2 lb value is the app's own kg to lb conversion.

    Args:
        x: weight in lb.

    Returns:
        bool, True if x equals round(kg * 2.20462, 2) for a kg value on a 0.25 grid.
    """
    if not np.isfinite(x) or x <= 0 or x == round(x):
        return False
    kg = np.round(x / KG2LB * 4) / 4
    return abs(x - np.round(kg * KG2LB, 2)) < 0.0061


# chronological pass per lift: kg-typed-as-lb fix, typos, jumps. history = earlier sessions only
d['sess_sort'] = d.date.dt.strftime('%Y-%m-%d') + '_' + d.session_key.str[-8:]
d['flag_kg_fixed'] = 0
d['flag_kg_not_checkable'] = 0
src2_raw = (d.source == 'src2') & ~d.weight_lb.apply(kg_converted)
in_era = d.date.apply(lambda x: any(a <= x <= b for a, b in KG_ERAS))


def scan_lift_history(d):
    """Mark typos, kg typed as lb and jumps, comparing each session only with earlier ones.

    Args:
        d: rows after the basic drops; kg_typed_in_lb_field fixes weights in place.

    Returns:
        dict, row index to rule name (one of the three typo rules or sudden_jump_vs_recent_sessions) for rows to drop.
    """
    rule_of = {}
    for lift, lift_rows in d.sort_values(['sess_sort', 'file_order']).groupby('exercise', sort=False):
        history = []   # (date, top_weight) of earlier sessions of this lift
        for _, sess in lift_rows.groupby('sess_sort', sort=True):
            idx = list(sess.index)
            today = sess.date.iloc[0]
            near = [top for date, top in history if (today - date).days <= TYPO_WINDOW_DAYS]
            if len(near) >= MIN_HISTORY_SESSIONS:
                past_max = max(near)
                bad = [i for i in idx if d.at[i, 'weight_lb'] > TYPO_FACTOR * past_max]
                if 0 < len(bad) <= len(idx) / 2:   # only isolated sets, a whole heavy session is a jump
                    for i in bad:
                        rule_of[i] = 'typo_above_3x_earlier_max'
            idx = [i for i in idx if i not in rule_of]
            if len(idx) >= 3:
                for i in list(idx):
                    others = d.loc[[j for j in idx if j != i], 'weight_lb']
                    if d.at[i, 'weight_lb'] > TYPO_FACTOR * others.median():
                        rule_of[i] = 'typo_above_3x_session_median'
                idx = [i for i in idx if i not in rule_of]
            if idx:
                session_top = d.loc[idx, 'weight_lb'].max()
                for i in idx:
                    if d.at[i, 'weight_lb'] < session_top * LOW_TYPO_SHARE:
                        rule_of[i] = 'typo_below_quarter_session_top'
                idx = [i for i in idx if i not in rule_of]
            if not idx:
                continue
            if sess.source.iloc[0] == 'src2':
                recent = [h for h in history if (today - h[0]).days <= KG_REF_DAYS][-HISTORY_SESSIONS:]
                if len(recent) < MIN_HISTORY_SESSIONS:
                    recent = [h for h in history if (today - h[0]).days <= KG_REF_DAYS_LONG][-HISTORY_SESSIONS:]
                raw_idx = [i for i in idx if src2_raw[i]]
                if raw_idx and in_era[idx[0]]:
                    if len(recent) >= MIN_HISTORY_SESSIONS:
                        reference = np.median([top for _, top in recent])
                        ratio = d.loc[raw_idx, 'weight_lb'].max() / reference
                        if KG_RATIO_LOW <= ratio <= KG_RATIO_HIGH and KG_FIXED_LOW <= ratio * KG2LB <= KG_FIXED_HIGH:
                            d.loc[raw_idx, 'weight_lb'] = (d.loc[raw_idx, 'weight_lb'] * KG2LB).round(2)
                            d.loc[raw_idx, 'flag_kg_fixed'] = 1
                    else:
                        d.loc[idx, 'flag_kg_not_checkable'] = 1
            session_top = d.loc[idx, 'weight_lb'].max()
            recent = [top for date, top in history if (today - date).days <= JUMP_WINDOW_DAYS][-HISTORY_SESSIONS:]
            if len(recent) >= MIN_HISTORY_SESSIONS:
                ratio = session_top / np.median(recent)
                if ratio < JUMP_LOW or ratio > JUMP_HIGH:
                    for i in idx:
                        rule_of[i] = 'sudden_jump_vs_recent_sessions'
            # the session still goes into history even if it gets dropped, so a real new level is accepted after a few sessions
            history.append((today, session_top))
    return rule_of


rule_of = scan_lift_history(d)

n = len(d)
note('kg_typed_in_lb_field', 'src2 raw numbers in the two kg eras that are 0.36-0.57x the recent median of the lift and x2.2 lands near it -> multiplied by 2.20462', n, n, changed=int(d.flag_kg_fixed.sum()))
desc = {'typo_above_3x_earlier_max': 'set weight > 3x max of the lift in earlier sessions (last 180 d, >= 2 sessions), isolated sets only',
        'typo_above_3x_session_median': 'set weight > 3x median of the other sets of that lift in the same session',
        'typo_below_quarter_session_top': 'set weight < 1/4 of the session top for that lift'}
for rule in desc:
    d = drop(d, d.index.isin([i for i, r in rule_of.items() if r == rule]), rule, desc[rule])
d = drop(d, d.reps > 30, 'more_than_30_reps', 'reps > 30 treated as typo')
d = drop(d, d.index.isin([i for i, r in rule_of.items() if r == 'sudden_jump_vs_recent_sessions']), 'sudden_jump_vs_recent_sessions',
         'session top weight < 0.58x or > 1.7x the median top of the last 5 earlier sessions of the lift (<= 365 d); unresolved scale jump, dropped as a likely entry error')

# flags on kept rows
d = d.sort_values(['sess_sort', 'file_order']).reset_index(drop=True)
d['set_type'] = d.set_type_raw.where(d.set_type_raw.isin(['drop', 'failure']), 'normal')
d['set_idx'] = d.groupby(['session_id', 'exercise']).cumcount()
d['flag_dumbbell_pair_or_single_unclear'] = d.exercise.isin(DB_PAIR_EXERCISES).astype(int)
d['flag_added_load_only'] = d.exercise.isin(ADDED_LOAD_BW).astype(int)
d['flag_uncertain_exercise_name'] = d.map_conf.isin(['low', 'identity']).astype(int)
ss = d.drop_duplicates('session_id')[['session_id', 'date']]
gap = ss.date.diff().dt.days
d['flag_after_long_break'] = d.session_id.isin(ss.session_id[gap > 30]).astype(int)
n = len(d)
for f, t in [('flag_dumbbell_pair_or_single_unclear', 'dumbbell lifts where apps disagree on per-dumbbell vs pair weight (kept, flagged)'),
             ('flag_added_load_only', 'weighted pull-up/dip/back ext: weight = added load only (kept, flagged)'),
             ('flag_kg_not_checkable', 'src2 raw number in a kg era with no history to check the unit (kept, flagged)'),
             ('flag_after_long_break', 'first session after a > 30 day break (kept, flagged)'),
             ('flag_uncertain_exercise_name', 'exercise name mapping is low confidence (kept, flagged)')]:
    note(f, t, n, n, changed=int(d[f].sum()))
d['flag_any'] = d[['flag_dumbbell_pair_or_single_unclear', 'flag_added_load_only', 'flag_kg_not_checkable', 'flag_kg_fixed']].max(axis=1)

os.makedirs(OUT, exist_ok=True)
keep = ['session_id', 'date', 'source', 'src_row', 'exercise', 'set_idx', 'set_type', 'weight_lb', 'reps',
        'flag_kg_fixed', 'flag_kg_not_checkable', 'flag_dumbbell_pair_or_single_unclear', 'flag_added_load_only', 'flag_uncertain_exercise_name', 'flag_after_long_break', 'flag_any']
o = d[keep].copy()
o['date'] = o.date.dt.strftime('%Y-%m-%d')
o['weight_lb'] = o.weight_lb.round(2)
o['reps'] = o.reps.astype(int)
o.to_csv(os.path.join(OUT, 'sets_clean.csv'), index=False)

dr = pd.concat(dropped, ignore_index=True)
dr['weight_lb'] = dr.weight_lb.round(2)
dr.loc[dr.rule == 'bodyweight_in_weight_field', 'weight_lb'] = np.nan   # would leak bodyweight
dr['date'] = dr.date.dt.strftime('%Y-%m-%d')
dr['restorable'] = dr.rule.isin(RESTORABLE).astype(int)
dr['is_dev'] = dr.is_dev.astype(int)
dr[['source', 'src_row', 'session_id', 'date', 'exercise', 'set_type_raw', 'weight_lb', 'reps', 'rule', 'restorable', 'is_dev']].to_csv(
    os.path.join(OUT, 'dropped.csv'), index=False)

rt = pd.DataFrame(log)
rt.to_csv(os.path.join(OUT, 'rule_table.csv'), index=False)
json.dump({'overrides_sha256': ov_hash, 'n_overrides': len(ov_keys), 'kept_rows': len(o), 'dropped_rows': len(dr)},
          open(os.path.join(OUT, 'clean_meta.json'), 'w'), indent=1)
print(rt[['rule', 'rows_before', 'rows_after', 'dropped', 'changed', 'restored']].to_string(index=False))
print('kept', len(o), 'dropped', len(dr), 'overrides', len(ov_keys))
