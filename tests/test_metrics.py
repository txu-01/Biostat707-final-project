import unittest

import numpy as np

from metrics import (
    bootstrap_confidence_intervals,
    select_threshold_target_sensitivity,
    summarize_paired_differences,
)


class MetricsTests(unittest.TestCase):
    def test_selects_highest_threshold_meeting_sensitivity_target(self):
        y_true = np.array([1, 1, 1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])

        threshold, sensitivity = select_threshold_target_sensitivity(
            y_true, y_prob, target_sens=0.75
        )

        self.assertAlmostEqual(threshold, 0.7)
        self.assertAlmostEqual(sensitivity, 0.75)

    def test_paired_bootstrap_differences_use_matching_resamples(self):
        y_true = np.array([0, 0, 1, 1])
        probabilities = {
            "Strong_Model": np.array([0.1, 0.2, 0.8, 0.9]),
            "Constant_Model": np.full(4, 0.5),
        }

        _, boot_stats = bootstrap_confidence_intervals(
            y_true, probabilities, n_bootstraps=100, seed=707
        )
        rows = summarize_paired_differences(
            boot_stats, comparator="Constant_Model"
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["model_name"], "Strong_Model")
        self.assertEqual(rows[0]["comparator"], "Constant_Model")
        self.assertGreater(rows[0]["auroc_difference_ci_lower"], 0)
        self.assertLess(rows[0]["brier_difference_ci_upper"], 0)


if __name__ == "__main__":
    unittest.main()
