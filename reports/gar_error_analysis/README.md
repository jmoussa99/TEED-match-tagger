# Why T-DEED fails on SoccerNet-GAR — error analysis

Follow-up to `reports/gar_eval_v1` (balanced accuracy 37.6% on 436 GAR validation clips). Data: per-frame T-DEED scores and 7-frame strips (−1.5 s … +1.5 s around the action) for the same 436 clips, from `notebooks/tdeed_gar_analysis.ipynb` (part A). Recomputing the evaluation from the per-frame scores reproduces gar_eval_v1 exactly (37.6%, 100% identical predictions).

## 1. Most errors are not near misses

Of 248 wrong clips, only 24 are "close calls" (true class at least half the winner's score). Tackles: in all 50 clips T-DEED fires *some* event, but the tackle score itself never exceeds 0.05 anywhere in the clip. Free kicks: the true class exceeds 0.05 in 1 of 50 clips. So re-weighting or thresholds cannot recover these classes.

## 2. Much of the gap is a labeling-convention mismatch, not blindness

Looking at the frame strips (`goals.jpg`, `free_kicks.jpg`, `tackles.jpg`):

- **Goals** are shots that go in. GAR puts the GOAL label on the shot; T-DEED calls that moment SHOT (78% of goals), which is a sensible answer.
- **Free kicks** in GAR include restarts such as a goalkeeper's goal kick; the restart itself is a pass, and T-DEED calls it PASS.
- **Tackles** in wide broadcast shots are a few small players colliding; afterwards the ball moves on and T-DEED reports PASS or OUT.

Accepting the parent action (GOAL→SHOT, FREE KICK→PASS/HIGH PASS/CROSS, HIGH PASS→PASS) raises balanced accuracy from **37.6% to 59.6%** (goal 0→0.78, free kick 0→0.88, high pass 0→0.54). Tackle stays at 0: that one is a genuine recognition failure.

## 3. Timing is not the problem

When the true class does fire, it peaks close to the labeled moment (median offset −0.35 s to +0.37 s for header, out, cross, throw-in, shot). Widening the ±1 s window does not help: ±0.5 s 39.4%, ±1 s 37.6%, ±1.5 s 36.2%, ±2.4 s 34.4%.

## 4. Padding creates a false event at the start of every clip

GAR clips are 4.8 s and T-DEED reads 8 s windows, so each clip is padded with blank frames. The probability of "some event" is highest at the first real frame (0.31 on average, versus 0.25 at the labeled action). This is partly real (in 18% of clips the previous GAR event falls there), but even in the 152 clips with no labeled event within 4 s before, the first frames score 0.23 — higher than the action itself (0.15). The black-to-video cut triggers spurious detections. This makes the padding vs context experiment (part B) necessary to measure the model fairly.

## 5. Confidence is informative

| Winning score | Accuracy |
|---|---|
| ≥ 0.4 | 76% |
| 0.1–0.4 | ~46% |
| < 0.1 | ~28% |

Keeping only predictions with confidence ≥ 0.2 keeps 31% of clips at 68% accuracy (vs 43% overall).

## What this means

- For the app: show confidence prominently, and explain that GOAL usually appears as SHOT and set pieces as PASS. Tackles are not detected at all.
- For evaluation: report both strict and relaxed (parent-class) scores, and use continuous footage rather than padded clips.
- To recover tackles and GAR's definitions of goal/free kick, the model would need fine-tuning on GAR labels.
