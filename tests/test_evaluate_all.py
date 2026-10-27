import unittest

import pandas as pd

from evaluate_all import validate_and_align_predictions


class PredictionValidationTests(unittest.TestCase):
    def setUp(self):
        self.splits = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "diagnosis": ["B", "M", "B", "M"],
            "target": [0, 1, 0, 1],
            "split": ["development", "development", "test", "test"],
            "cv_fold": [0, 1, -1, -1],
        })

    def test_aligns_valid_oof_predictions_to_official_id_order(self):
        predictions = pd.DataFrame({
            "id": [2, 1],
            "y_true": [1, 0],
            "model_name": ["Model_A", "Model_A"],
            "fold": [1, 0],
            "y_prob": [0.8, 0.2],
        })

        aligned = validate_and_align_predictions(
            predictions, self.splits, "development"
        )["Model_A"]

        self.assertEqual(aligned["id"].tolist(), [1, 2])
        self.assertEqual(aligned["y_prob"].tolist(), [0.2, 0.8])

    def test_rejects_duplicate_or_missing_ids(self):
        predictions = pd.DataFrame({
            "id": [1, 1],
            "y_true": [0, 0],
            "model_name": ["Model_A", "Model_A"],
            "fold": [0, 0],
            "y_prob": [0.2, 0.2],
        })

        with self.assertRaisesRegex(ValueError, "IDs must be present and unique"):
            validate_and_align_predictions(predictions, self.splits, "development")

    def test_rejects_missing_model_name(self):
        predictions = pd.DataFrame({
            "id": [1, 2],
            "y_true": [0, 1],
            "model_name": ["Model_A", None],
            "fold": [0, 1],
            "y_prob": [0.2, 0.8],
        })

        with self.assertRaisesRegex(ValueError, "non-empty model_name"):
            validate_and_align_predictions(predictions, self.splits, "development")

    def test_rejects_incorrect_fold_or_target(self):
        predictions = pd.DataFrame({
            "id": [1, 2],
            "y_true": [0, 0],
            "model_name": ["Model_A", "Model_A"],
            "fold": [1, 0],
            "y_prob": [0.2, 0.8],
        })

        with self.assertRaisesRegex(ValueError, "y_true"):
            validate_and_align_predictions(predictions, self.splits, "development")

    def test_rejects_invalid_test_probabilities(self):
        predictions = pd.DataFrame({
            "id": [3, 4],
            "y_true": [0, 1],
            "model_name": ["Model_A", "Model_A"],
            "split": ["test", "test"],
            "y_prob": [0.2, 1.2],
        })

        with self.assertRaisesRegex(ValueError, r"within \[0, 1\]"):
            validate_and_align_predictions(predictions, self.splits, "test")


if __name__ == "__main__":
    unittest.main()
