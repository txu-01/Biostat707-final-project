"""Focused adversarial checks, using the pilot artifacts without retraining."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from final_analysis import summarize_prediction_rows
from metrics import select_threshold_target_sensitivity, compute_calibration_curve
from yan.random_forest.audit import ROOT, OUT, data_and_splits, align_reference
from yan.random_forest.train import positive_probability, validate_fold, load_checkpoint


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.outer, cls.inner = data_and_splits()
        cls.wrapper = json.loads((OUT / "checkpoints/repeat_01_fold_0.json").read_text())
        cls.ref = pd.read_csv(ROOT / "results/final_nested/predictions/Ridge_Logistic.csv")

    def test_ties_and_boundaries_select_highest_qualifying_threshold(self):
        y = np.array([1]*20 + [0, 0, 0])
        p = np.array([.1]+[.8]*19+[0, .8, 1.0])
        t, sensitivity = select_threshold_target_sensitivity(y, p, .95)
        self.assertEqual(t, .8)
        self.assertEqual(sensitivity, .95)
        self.assertEqual(select_threshold_target_sensitivity([0, 1], [1., 1.], .95), (1., 1.))
        self.assertEqual(select_threshold_target_sensitivity([0, 1], [0., 0.], .95), (0., 1.))

    def test_quantile_boundary_ties_preserve_all_records(self):
        y = np.array([0]*12+[1]*8)
        p = np.array([0.]*12+[.5]*4+[1.]*4)
        result = compute_calibration_curve(y, p)
        self.assertEqual(sum(result["bin_counts"]), len(y))
        constant = compute_calibration_curve(y, np.ones(len(y))*.2)
        self.assertEqual(constant["bin_counts"], [20])
        self.assertAlmostEqual(constant["bin_true_prop"][0], .4)

    def test_probability_column_is_looked_up_not_assumed(self):
        class Reverse:
            classes_ = np.array([1, 0])
            def predict_proba(self, x):
                return np.array([[.7, .3], [.2, .8]])
        np.testing.assert_array_equal(positive_probability(Reverse(), None), [.7, .2])

    def test_same_row_count_can_hide_duplicate_reference_ids(self):
        bad = self.ref.copy()
        bad.iloc[1] = bad.iloc[0]
        with self.assertRaises(ValueError):
            align_reference(bad, self.outer, "Ridge_Logistic")

    def test_same_fold_counts_do_not_prove_identical_record_assignment(self):
        bad = self.ref.copy()
        indices = bad.index[(bad["repeat"] == 1) & bad.fold.isin([0, 1])]
        a = bad.index[(bad["repeat"] == 1) & (bad.fold == 0)][0]
        b = bad.index[(bad["repeat"] == 1) & (bad.fold == 1)][0]
        bad.loc[a, "fold"], bad.loc[b, "fold"] = 1, 0
        with self.assertRaises(AssertionError):
            align_reference(bad, self.outer, "Ridge_Logistic")

    def test_wrong_label_is_rejected(self):
        bad = self.ref.copy()
        bad.loc[0, "y_true"] = 1-int(bad.loc[0, "y_true"])
        with self.assertRaises(AssertionError):
            align_reference(bad, self.outer, "Ridge_Logistic")

    def test_complete_pilot_passes_then_duplicate_predictions_fail(self):
        good = copy.deepcopy(self.wrapper["payload"])
        validate_fold(good, 1, 0, self.data, self.outer, self.inner)
        good["predictions"][1] = copy.deepcopy(good["predictions"][0])
        with self.assertRaises(AssertionError):
            validate_fold(good, 1, 0, self.data, self.outer, self.inner)

    def test_inner_oof_cannot_include_outer_validation_id(self):
        bad = copy.deepcopy(self.wrapper["payload"])
        bad["selected_inner_oof"][0]["id"] = bad["predictions"][0]["id"]
        with self.assertRaises(AssertionError):
            validate_fold(bad, 1, 0, self.data, self.outer, self.inner)

    def test_checkpoint_from_another_run_is_rejected(self):
        manifest = json.loads((OUT / "run_manifest.json").read_text())
        bad = copy.deepcopy(self.wrapper)
        bad["run_id"] = "a-different-run"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            path.write_text(json.dumps(bad))
            with self.assertRaises(AssertionError):
                load_checkpoint(path, manifest, 1, 0, self.data, self.outer, self.inner)

    def test_row_thresholds_and_undefined_predictive_values(self):
        rows = pd.DataFrame(dict(y_true=[0, 0, 1, 1], y_prob=[.1, .4, .6, .9],
                                 threshold=[.5, .5, .7, .7]))
        result = summarize_prediction_rows(rows)
        self.assertEqual((result["TN"], result["FP"], result["FN"], result["TP"]), (2, 0, 1, 1))
        rows["threshold"] = 1.
        self.assertTrue(np.isnan(summarize_prediction_rows(rows)["ppv"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
