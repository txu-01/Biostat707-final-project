"""RF dispatch and three-file export contract tests. No other model is fitted."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import final_analysis as common
from yan.random_forest import integration as bridge
from yan.random_forest.audit import data_and_splits, write_json


class RoutingTests(unittest.TestCase):
    def test_formal_defaults_are_ten_repeats_and_five_inner_outer_folds(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(bridge, "run_shared_random_forest") as owner:
            common.run_primary_nested(common.load_data(), Path(tmp), ["Random_Forest"])
            self.assertEqual(owner.call_count, 1)
            self.assertEqual(owner.call_args.args[2:], (10, 5, 5))
            self.assertEqual(owner.call_args.kwargs, {"resume": True})

    def test_common_rf_route_calls_yan_once_and_never_old_search(self):
        data = common.load_data()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with patch.object(bridge, "run_shared_random_forest") as owner, \
                 patch.object(common, "tune_inside_outer_training", side_effect=AssertionError("old RF search")), \
                 patch.object(common, "make_model", side_effect=AssertionError("old RF fit")):
                common.run_primary_nested(data, out, ["Random_Forest"], 1, 5, 5, resume=False)
            owner.assert_called_once()
            args, kwargs = owner.call_args
            pd.testing.assert_frame_equal(args[0], data)
            self.assertEqual(args[1:], (out, 1, 5, 5))
            self.assertEqual(kwargs, {"resume": False})

    def test_duplicate_rf_request_fails_before_training(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(bridge, "run_shared_random_forest") as owner:
            with self.assertRaises(ValueError):
                common.run_primary_nested(common.load_data(), Path(tmp),
                    ["Random_Forest", "Random_Forest"], 1, 5, 5)
            owner.assert_not_called()

    def test_non_rf_call_signatures_are_preserved_without_real_model_fits(self):
        data = common.load_data()
        dummy_search = ({}, np.full(455, .5), .3, 1., 0, 0, [])
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(bridge, "run_shared_random_forest") as owner, \
             patch.object(common, "tune_inside_outer_training", return_value=dummy_search) as tune, \
             patch.object(common, "make_model", return_value=object()) as factory, \
             patch.object(common, "fit_probability",
                          side_effect=lambda model, x, y, v: (np.full(len(v), .5), 0)) as fit:
            common.run_primary_nested(data, Path(tmp), ["Ridge_Logistic"], 1, 5, 5, resume=False)
            self.assertEqual(tune.call_count, 5)
            self.assertEqual(factory.call_count, 5)
            self.assertEqual(fit.call_count, 5)
            self.assertTrue(all(c.args[0] == "Ridge_Logistic" for c in factory.call_args_list))
            self.assertTrue(all(c.args[2] == "Ridge_Logistic" for c in tune.call_args_list))
            owner.assert_not_called()

    def test_rf_refuses_nonproposal_fold_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                bridge.run_shared_random_forest(common.load_data(), Path(tmp), 10, 3, 5)


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.outer, cls.inner = data_and_splits()
        root = Path(common.ROOT) / "results/yan/random_forest"
        cls.fixture = {
            name: pd.read_csv(root / name).query("repeat == 1").copy()
            for name in ["oof_predictions.csv", "fold_results.csv", "inner_cv_results.csv"]}

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name)
        self.native = self.output / bridge.DETAIL / "runs" / "test-fixture"
        self.native.mkdir(parents=True)
        for name, frame in self.fixture.items():
            frame.to_csv(self.native / name, index=False)
        # An IO fixture only, not a training/resume identity.
        self.manifest = dict(run_id="test-fixture", fingerprint="fixture-only",
                             config={"repeats": 1}, status="complete_verified")
        write_json(self.native / "run_manifest.json", self.manifest)
        write_json(self.output / bridge.DETAIL / bridge.POINTER,
            dict(run_id="test-fixture", run_directory=str(self.native.relative_to(self.output))))

    def tearDown(self):
        self.temp.cleanup()

    def publish(self):
        bridge.publish(self.output, self.native, self.manifest, self.data, self.outer)

    def test_all_three_files_are_compatible_and_summary_has_one_rf(self):
        self.publish()
        marker = bridge.validate_shared_export(self.output)
        self.assertEqual((marker["prediction_rows"], marker["outer_records"], marker["candidate_records"]), (569, 5, 80))
        self.assertFalse(marker["formal_ten_repeat_analysis"])
        fold = pd.read_csv(self.output / bridge.SHARED_FILES[1])
        search = pd.read_csv(self.output / bridge.SHARED_FILES[2])
        np.testing.assert_allclose(fold.seconds, fold.total_seconds)
        self.assertTrue(fold.convergence_warnings.eq(0).all())
        self.assertTrue(search.convergence_warnings.eq(0).all())
        repeats, summary = common.summarize_primary(self.output)
        self.assertEqual(repeats.model.tolist(), ["Random_Forest"])
        self.assertEqual(summary.model.tolist(), ["Random_Forest"])
        committed = json.loads((self.output / bridge.MARKER).read_text())
        self.assertEqual(committed["summary_status"], "complete")
        self.assertEqual(committed["summary_rf_run_id"], "test-fixture")

    def test_old_rf_slots_are_backed_up_then_wholly_replaced(self):
        for name in bridge.SHARED_FILES:
            p = self.output / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("old,incompatible,rows\n")
        self.publish()
        for name in bridge.SHARED_FILES:
            self.assertEqual((self.native / "previous_shared_files" / name).read_text(),
                             "old,incompatible,rows\n")
        marker = bridge.validate_shared_export(self.output)
        self.assertEqual(marker["prediction_rows"], 569)

    def test_mixing_only_search_file_is_rejected(self):
        self.publish()
        path = self.output / bridge.SHARED_FILES[2]
        path.write_text(path.read_text()+"\n")
        with self.assertRaises(ValueError):
            common.summarize_primary(self.output)

    def test_partial_publish_without_commit_marker_cannot_be_summarized(self):
        self.publish()
        (self.output / bridge.MARKER).unlink()
        with self.assertRaises(ValueError):
            common.summarize_primary(self.output)

    def test_run_id_mismatch_is_rejected(self):
        self.publish()
        marker = json.loads((self.output / bridge.MARKER).read_text())
        marker["run_id"] = "foreign-run"
        write_json(self.output / bridge.MARKER, marker)
        with self.assertRaises(ValueError):
            bridge.validate_shared_export(self.output)

    def test_legacy_archive_without_yan_manifest_is_still_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(bridge.validate_shared_export(Path(tmp)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
