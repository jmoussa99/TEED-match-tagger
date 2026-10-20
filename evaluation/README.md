# GAR evaluation

The [CodaBench evaluation server](https://www.codabench.org/competitions/18326/) accepts result submissions as ZIP archives containing an OpenSportsLib JSON prediction file. The scorer reads the OpenSportsLib ground-truth JSON and the submitted OpenSportsLib prediction JSON, matches records by sample ID, and calls OpenSportsLib's `compute_classification_metrics(..., mode="labels")`. It reports balanced accuracy, macro F1, accuracy, and ten per-class recalls as percentages (0–100). **Balanced accuracy** determines the single leaderboard. The CodaBench scoring image pins OpenSportsLib in the [Dockerfile](Dockerfile).

## Metrics and leaderboard

The existing `balanced_accuracy`, `macro_f1`, and `accuracy` scores retain their names, calculations, and percentage scale. Balanced accuracy remains the primary descending ranking metric, with the existing best-submission rule and four decimal places in the leaderboard. The [competition configuration](competition.yaml) appends these diagnostic columns after the three existing metrics:

| Score key | Ground-truth action label |
| --- | --- |
| `recall_pass` | `PASS` |
| `recall_tackle` | `PLAYER SUCCESSFUL TACKLE` |
| `recall_out` | `OUT` |
| `recall_header` | `HEADER` |
| `recall_throw_in` | `THROW IN` |
| `recall_cross` | `CROSS` |
| `recall_free_kick` | `FREE KICK` |
| `recall_shot` | `SHOT` |
| `recall_goal` | `GOAL` |
| `recall_high_pass` | `HIGH PASS` |

For each class, recall is `100 × correctly classified samples / ground-truth samples in that class`. The test reference contains every class, as required by the existing scorer. The arithmetic mean of the ten unrounded recalls equals `balanced_accuracy` within floating-point tolerance. Display rounding can produce a small difference if the displayed values are averaged. The new columns do not change the submission format, dataset, evaluation protocol, or primary ranking metric.

The supplied baseline still scores **78.64% balanced accuracy** (78.6355461315% before rounding), 57.1493214377% macro F1, and 76.9596026006% accuracy.

## Prediction format

The JSON must be an OSL object with version `2.0`, task `action_classification`, and a `data` array. Each sample must include its unique test `id` and an action label at `labels.action.label`. The example below shows the required structure; optional OSL fields such as prediction confidence may also be included.

```json
{
  "version": "2.0",
  "task": "action_classification",
  "dataset_name": "soccernet_gar_2027_predictions",
  "metadata": { "type": "predictions" },
  "data": [
    {
      "id": "test_000000",
      "labels": { "action": { "label": "PASS" } }
    }
  ]
}
```

The prediction IDs must cover every sample in the [OSL test ground truth](../data/annotations_test.json) exactly once, with no extra IDs. Labels must match one of the ten class names in that file, including capitalization and spaces. The scorer compares the two JSON files by ID, so record order does not affect the score.

## Package and submit

[CodaBench's submission guide](https://docs.codabench.org/latest/Participants/User_Participating-in-a-Competition/) specifies a ZIP upload for results submissions. Uploading `predictions.json` alone is not supported by this competition. The ZIP must contain **one file at its root**, with no enclosing folder:

```text
submission.zip
└── predictions.json
```

From the directory containing your `predictions.json`, create and inspect the archive:

```bash
python -m zipfile -c submission.zip predictions.json
python -m zipfile -l submission.zip
```

The listing should show only `predictions.json`. Upload `submission.zip` under **My Submissions** on the [GAR CodaBench page](https://www.codabench.org/competitions/18326/). This repository also provides a [ready-to-upload baseline ZIP](../baselines/gar-submission.zip) with that exact structure.

## Local scoring

Install the dependencies listed in the [Dockerfile](Dockerfile), then score an OSL prediction file against the OSL test ground truth:

```bash
python evaluation/scoring.py \
  --ground-truth data/annotations_test.json \
  --predictions baselines/predictions.json
```

The script prints all thirteen metrics as JSON. To also write `scores.json`, pass `--output output`. Run the regression checks with `python -m unittest discover -s evaluation -p 'test_*.py'`.

## CodaBench scoring

CodaBench provides the OSL ground truth as `input/ref/reference.json` and unpacks the team's OSL prediction file to `input/res/predictions.json`. The scoring program compares those files and writes `output/scores.json`. Only the scoring environment needs the reference file.

Submitted code supports reproducibility review under the [rules](../RULES.md).

Previously evaluated submissions retain their existing aggregate scores. Adding leaderboard columns does not compute recalls retroactively: submissions scored by the old program require evaluation with the updated scorer to populate those columns. Organizers should preserve existing submissions, tasks, and scoring resources; update the scoring program on the existing GAR task and add columns to the existing leaderboard. Back up historical scores and outputs before re-evaluation. Do not create a replacement competition or delete historical results to apply this update.
