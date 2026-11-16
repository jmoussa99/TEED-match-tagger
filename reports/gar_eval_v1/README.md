# T-DEED on SoccerNet-GAR — first evaluation

**Question:** how accurate is the off-the-shelf T-DEED model (winner of the SoccerNet 2024 Ball Action Spotting challenge, checkpoint `SoccerNetBall_challenge2`) when used as our match tagger on footage it was not trained on?

## Setup

- **Data:** 436 labeled clips from the SoccerNet-GAR **validation** split (World Cup 2022 broadcast footage from PFF, 1280×720, 4.8 s each). Up to 50 random clips per class; all 23 goals and 13 high passes. T-DEED was trained on SoccerNet (English football league broadcasts), so there is no overlap.
- **Run:** Google Colab, T4 GPU, `notebooks/tdeed_gar_eval.ipynb` → `scripts/evaluate_gar.py`, using the app's own scoring pipeline (`app/tdeed_runner.compute_scores`). 2.6 s per clip.
- **Decision rule:** each clip's labeled action is at 2.43 s. For each class we take T-DEED's highest score within ±1 s of that moment (same tolerance as SoccerNet's mAP@1) and predict the highest-scoring of the 10 GAR classes. T-DEED's two extra classes (DRIVE, BALL PLAYER BLOCK) are not allowed as predictions.

## Results

**Balanced accuracy 37.6%** (accuracy 43.1%). For reference, models trained on GAR's own tracking data: official GIN baseline 78.6% (test), our Milestone 1 features 58.3% (valid).

| Action | T-DEED recall (video) | GIN baseline (tracking, test) | Our M1 features (tracking, valid) |
|---|---|---|---|
| PASS | 0.86 | 0.80 | 0.66 |
| OUT | 0.76 | 0.91 | 0.63 |
| SHOT | 0.74 | 0.77 | 0.64 |
| THROW IN | 0.60 | 0.84 | 0.81 |
| HEADER | 0.50 | 0.66 | 0.81 |
| CROSS | 0.30 | 0.89 | 0.81 |
| TACKLE | **0.00** | 0.54 | 0.38 |
| FREE KICK | **0.00** | 0.90 | 0.76 |
| GOAL | **0.00** | 0.73 | 0.17 |
| HIGH PASS | **0.00** | 0.83 | 0.15 |

With 50 clips per class, a per-class recall has a 95% margin of about ±14 points.

## What we found

1. **Four classes are never predicted.** Tackles, free kicks, goals and high passes get recall 0. For tackles and free kicks, T-DEED's score for the true class is essentially zero (median < 0.001), so it is not a tie-breaking problem: the model does not recognise these events in this footage.
2. **Goals are recognised as shots (78%).** This is likely a labeling-convention difference rather than a model error: the GAR goal label sits at the moment of the shot, which T-DEED correctly calls a shot.
3. **Most mistakes fall into PASS.** 62% of tackles, 70% of free kicks, 54% of high passes and 42% of crosses are predicted as PASS. T-DEED's DRIVE (dribble) score is also high on most GAR passes; it would beat every GAR class on 88% of pass clips if allowed.
4. **Confident classes transfer well.** Pass, out, shot and throw-in reach 0.60–0.86, close to the tracking baseline for pass and shot.
5. **Simple recalibration helps only a little.** Dividing each class score by its average (a per-class prior correction, cross-validated) raises balanced accuracy to about 41%, mostly by recovering goals. Mapping DRIVE→PASS or BALL PLAYER BLOCK→TACKLE does not help.

**Takeaway:** a challenge-winning model does not transfer cleanly to a different broadcast dataset with the same action names. The gap comes mostly from rare and definition-sensitive classes (tackle, free kick, high pass, goal), not from the common ones.

## Caveats

- GAR clips are only 4.8 s; T-DEED looks at 8 s windows, so each clip is padded with blank frames. On full match video (how the app is normally used) the model sees real context, so these numbers may underestimate it.
- The sample is class-balanced, so precision values in `per_class.csv` do not reflect real match frequencies (passes dominate real games).
- Validation split only; the test split has not been used.

## Files

`metrics.json`, `per_class.csv`, `comparison.csv`, `confusion.csv`, `confusion_matrix.png`, `predictions.csv` (per clip: label, prediction, and T-DEED's peak score for all 12 classes).
