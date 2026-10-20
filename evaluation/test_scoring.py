"""Regression checks for aggregate scores, class mapping, and OSL validation."""
import copy
import math
from pathlib import Path
import unittest

from scoring import RECALL_CLASSES, load_json, score


ROOT = Path(__file__).resolve().parents[1]


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.truth = load_json(ROOT / "data/annotations_test.json")
        cls.predictions = load_json(ROOT / "baselines/predictions.json")

    def assert_recall_mean(self, scores):
        self.assertEqual(len(scores), 13)
        self.assertTrue(all(math.isfinite(v) and 0 <= v <= 100
                            for v in scores.values()))
        self.assertTrue(math.isclose(
            sum(scores[key] for key in RECALL_CLASSES) / 10,
            scores["balanced_accuracy"], rel_tol=1e-12, abs_tol=1e-12,
        ))

    def test_baseline(self):
        scores = score(self.truth, self.predictions)
        for key, expected in {
            "balanced_accuracy": 78.63554613151219,
            "macro_f1": 57.14932143771274,
            "accuracy": 76.95960260062824,
        }.items():
            self.assertAlmostEqual(scores[key], expected, places=10)
        self.assertEqual(f'{scores["balanced_accuracy"]:.2f}', "78.64")
        self.assert_recall_mean(scores)
        # Independent class-name counts catch incorrect metric-to-class mapping.
        predicted = {r["id"]: r["labels"]["action"]["label"]
                     for r in self.predictions["data"]}
        for key, label in RECALL_CLASSES.items():
            rows = [r for r in self.truth["data"]
                    if r["labels"]["action"]["label"] == label]
            expected = 100 * sum(predicted[r["id"]] == label for r in rows) / len(rows)
            self.assertAlmostEqual(scores[key], expected, places=12)

    def test_mapping_and_order_independence(self):
        # Deliberately asymmetric recalls, shuffled schema and prediction records.
        classes = list(reversed(RECALL_CLASSES.values()))
        truth = {"version": "2.0", "task": "action_classification",
                 "labels": {"action": {"type": "single_label", "labels": classes}},
                 "data": []}
        predictions = {"version": "2.0", "task": "action_classification", "data": []}
        for index, label in enumerate(RECALL_CLASSES.values()):
            for n in range(10):
                sample_id = f"class_{index}_{n}"
                truth["data"].append({"id": sample_id, "labels": {"action": {"label": label}}})
                prediction = label if n <= index else classes[(classes.index(label) + 1) % 10]
                predictions["data"].append({"id": sample_id,
                                            "labels": {"action": {"label": prediction}}})
        predictions["data"].reverse()
        scores = score(truth, predictions)
        for index, key in enumerate(RECALL_CLASSES):
            self.assertAlmostEqual(scores[key], 10 * (index + 1))
        self.assert_recall_mean(scores)

    def test_perfect_and_no_predictions_for_nine_classes(self):
        perfect = score(self.truth, self.truth)
        self.assertTrue(all(value == 100 for value in perfect.values()))
        constant = copy.deepcopy(self.predictions)
        for row in constant["data"]:
            row["labels"]["action"]["label"] = "PASS"
        scores = score(self.truth, constant)
        self.assertEqual(scores["recall_pass"], 100)
        self.assertTrue(all(scores[key] == 0 for key in RECALL_CLASSES if key != "recall_pass"))
        self.assertEqual(scores["balanced_accuracy"], 10)
        self.assert_recall_mean(scores)

    def test_submission_validation_unchanged(self):
        for case in ("missing", "duplicate", "unknown"):
            predictions = copy.deepcopy(self.predictions)
            if case == "missing":
                predictions["data"].pop()
            elif case == "duplicate":
                predictions["data"].append(predictions["data"][0])
            else:
                predictions["data"][0]["labels"]["action"]["label"] = "UNKNOWN"
            with self.subTest(case=case), self.assertRaises(ValueError):
                score(self.truth, predictions)


if __name__ == "__main__":
    unittest.main()
