# Prediction sheet for Sat 17 Oct 2026

Made 2026-10-17T19:02:54-07:00 (America/Los_Angeles), before the lift. Latest logged session in the cleaned data: 2026-10-08.

## Bicep Curl (Machine), 110 lb (49.9 kg)

Made 2026-10-17T19:02:54-07:00. Last session counted as history: 2026-10-03 (110x9 110x5, lb x reps). Days since this lift 14, since any session 9.

| set | RF | HGB | repeat last | RF raw | HGB raw | repeat last raw |
|---|---|---|---|---|---|---|
| 1 | 9 | 9 | 7 | 9.17 | 9.40 | 7.00 |
| 2 | 8 | 8 | 7 | 7.95 | 8.09 | 7.00 |
| 3 | 8 | 8 | 7 | 7.79 | 7.78 | 7.00 |

## Assumptions

* RF and HGB use the frozen notebook code read straight from pipeline.ipynb, with the same features, hyperparameters and seed 156. They are trained on all 6,130 cleaned rows with history (398 sessions), covering development and the final test period up to 2026-10-08.
* I chose machine curls as part of a normal arms day. It is a machine lift where the weight field is the full load.
* The weight is the most common working weight in my latest session of that lift.
* The lift is treated as the first lift of today's session, so sets 1, 2 and 3 have set_idx 0, 1, 2 and 0, 1, 2 earlier sets in the session. If I do it later in the session, the models would predict somewhat fewer reps.
* Gaps are counted to today. No drop set, and the sets are normal (not warm-ups).
* Repeat last is the mean reps of the whole last session on the lift, so it is the same for all three sets.
* Rounded values use normal rounding of the raw prediction. Predictions are clipped to 1 to 30 reps.
