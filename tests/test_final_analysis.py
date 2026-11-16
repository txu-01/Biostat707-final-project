import unittest

import numpy as np
import pandas as pd

from final_analysis import (
    FEATURE_GROUPS,
    make_model,
    summarize_prediction_rows,
)


class FinalAnalysisTests(unittest.TestCase):
    def test_feature_groups_partition_the_thirty_predictors(self):
        self.assertEqual(len(FEATURE_GROUPS["mean"]), 10)
        self.assertEqual(len(FEATURE_GROUPS["se"]), 10)
        self.assertEqual(len(FEATURE_GROUPS["worst"]), 10)
        self.assertEqual(len(FEATURE_GROUPS["all"]), 30)
        combined = (
            FEATURE_GROUPS["mean"]
            + FEATURE_GROUPS["se"]
            + FEATURE_GROUPS["worst"]
        )
        self.assertEqual(set(combined), set(FEATURE_GROUPS["all"]))

    def test_scaling_is_inside_linear_model_pipelines(self):
        logistic = make_model("Logistic_Regression", {}, 707)
        ridge = make_model("Ridge_Logistic", {}, 707)
        elastic = make_model(
            "Elastic_Net", {"C": 0.1, "l1_ratio": 0.5}, 707
        )
        linear_svm = make_model("Linear_SVM", {"C": 0.1}, 707)
        self.assertEqual(logistic.steps[0][0], "scale")
        self.assertEqual(ridge.steps[0][0], "scale")
        self.assertEqual(elastic.steps[0][0], "scale")
        self.assertEqual(linear_svm.estimator.steps[0][0], "scale")

    def test_repeat_summary_uses_each_rows_training_selected_threshold(self):
        rows = pd.DataFrame(
            {
                "y_true": [0, 0, 1, 1],
                "y_prob": [0.1, 0.4, 0.6, 0.9],
                "threshold": [0.5, 0.5, 0.7, 0.7],
            }
        )
        result = summarize_prediction_rows(rows)
        self.assertEqual(result["TN"], 2)
        self.assertEqual(result["TP"], 1)
        self.assertEqual(result["FN"], 1)
        self.assertEqual(result["FP"], 0)
        self.assertAlmostEqual(result["sensitivity"], 0.5)
        self.assertAlmostEqual(result["specificity"], 1.0)


if __name__ == "__main__":
    unittest.main()
