
import unittest
import numpy as np

from src.evaluation import CODES, matrix, metrics, paired_bootstrap, scores


class EvaluationTests(unittest.TestCase):
    def test_all_eight_classes_perfect(self):
        rows = [{"reference": c, "prediction": c} for c in CODES]
        self.assertEqual(metrics(rows)["accuracy"], 1.0)
        self.assertEqual(metrics(rows)["macro_f1"], 1.0)

    def test_invalid_prediction_counts_as_false_negative(self):
        rows = [{"reference": c, "prediction": c} for c in CODES]
        rows.append({"reference": "A01", "prediction": "invalid"})
        result = metrics(rows)
        self.assertEqual(result["invalid_outputs"], 1)
        self.assertAlmostEqual(result["accuracy"], 8 / 9)
        self.assertAlmostEqual(result["macro_f1"], (7 + 2 / 3) / 8)
        self.assertEqual(result["per_class"]["A01"]["recall"], 0.5)

    def test_absent_classes_are_in_macro_average(self):
        result = metrics([{"reference": "A01", "prediction": "A01"}])
        self.assertEqual(result["macro_f1"], 1 / 8)
        self.assertEqual(result["per_class"]["A08"]["f1"], 0.0)

    def test_unknown_reference_rejected(self):
        with self.assertRaises(ValueError):
            matrix([{"reference": "invalid", "prediction": "A01"}])

    def test_scores_accept_leading_dimensions(self):
        counts = np.pad(np.eye(8, dtype=int), ((0, 0), (0, 1)))
        np.testing.assert_array_equal(
            scores(np.stack([counts, counts])), np.ones((2, 2))
        )

    def test_shared_draws_cancel_identical_conditions(self):
        rng = np.random.default_rng(5)
        counts = rng.integers(0, 5, size=(2, 1, 3, 7, 8, 9))
        counts = np.repeat(counts, 2, axis=1)
        result = paired_bootstrap(counts, repetitions=129)
        for metric in result["metrics"].values():
            for contrast in metric.values():
                self.assertEqual(
                    contrast, {"estimate": 0.0, "ci95": [0.0, 0.0]}
                )

    def test_size_contrast_direction_and_percentage_points(self):
        good = np.pad(np.eye(8, dtype=int), ((0, 0), (0, 1)))
        bad = np.pad(
            np.roll(np.eye(8, dtype=int), 1, axis=1), ((0, 0), (0, 1))
        )
        counts = np.broadcast_to(good, (2, 2, 3, 4, 8, 9)).copy()
        counts[1, 0] = bad
        result = paired_bootstrap(counts, repetitions=129)
        for metric in result["metrics"].values():
            self.assertEqual(metric["effect_9B"]["estimate"], 0.0)
            self.assertEqual(
                metric["effect_2B"], {"estimate": 100.0, "ci95": [100.0, 100.0]}
            )
            self.assertEqual(metric["contrast_2B_minus_9B"]["estimate"], 100.0)

    def test_bootstrap_reproducible_across_partial_batch(self):
        counts = np.random.default_rng(9).integers(
            0, 4, size=(2, 2, 3, 7, 8, 9)
        )
        self.assertEqual(
            paired_bootstrap(counts, 129), paired_bootstrap(counts, 129)
        )


if __name__ == "__main__":
    unittest.main()
