"""Small IO/alignment tests only. Fixtures are temporary; no estimator is fitted."""
from contextlib import ExitStack
from pathlib import Path
import copy
import hashlib
import importlib
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import final_analysis as common
from metrics import compute_classification_at_threshold
from yan.random_forest import train
from yan.random_forest.audit import OUT, canonical, data_and_splits, digest, write_json

verifier = importlib.import_module("yan.random_forest.verify")


class ThreeFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.outer, cls.inner = data_and_splits()
        cls.checkpoint = OUT / "checkpoints/repeat_01_fold_0.json"
        cls.wrapper = json.loads(cls.checkpoint.read_text())
        cls.manifest = json.loads((OUT / "run_manifest.json").read_text())

    def setUp(self):
        self.guards = ExitStack()
        self.addCleanup(self.guards.close)
        for module, name in [(train, "fit_model"), (train, "run_fold"), (common, "make_model")]:
            self.guards.enter_context(patch.object(module, name, side_effect=AssertionError("No model fits allowed")))

    @staticmethod
    def snapshot(folder):
        return {str(p.relative_to(folder)): (p.read_bytes(), p.stat().st_mtime_ns)
                for p in folder.rglob("*") if p.is_file()}

    def partial_export(self, folder):
        path = train.checkpoint_path(folder, 1, 0)
        path.parent.mkdir()
        path.write_bytes(self.checkpoint.read_bytes())
        for key, name in [("predictions", "oof_predictions.csv"), ("fold_results", "fold_results.csv"),
                          ("inner_cv_results", "inner_cv_results.csv"), ("selected_inner_oof", "selected_inner_oof.csv"),
                          ("feature_importances", "feature_importances.csv")]:
            pd.DataFrame(self.wrapper["payload"][key]).to_csv(folder/name, index=False)

    def test_existing_exports_are_checked_without_any_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.partial_export(folder)
            before = self.snapshot(folder)
            with patch.object(pd.DataFrame, "to_csv", side_effect=AssertionError("CSV write")), \
                 patch.object(train, "write_json", side_effect=AssertionError("JSON write")):
                checks = train.export_and_verify(folder, self.manifest, complete=False, write_outputs=False)
            self.assertEqual(checks["completed_outer_folds"], 1)
            self.assertEqual(before, self.snapshot(folder))

    def test_corrupt_export_fails_and_is_not_repaired(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.partial_export(folder)
            path = folder/"oof_predictions.csv"
            p = pd.read_csv(path)
            p.loc[0, "y_prob"] = .123456789
            p.to_csv(path, index=False)
            before = self.snapshot(folder)
            with self.assertRaises(AssertionError):
                train.export_and_verify(folder, self.manifest, complete=False, write_outputs=False)
            self.assertEqual(before, self.snapshot(folder))

    def test_verify_cli_bypasses_initialization_and_directory_creation(self):
        folder = OUT/"never_created_three_fix_test"
        self.assertFalse(folder.exists())
        with patch("sys.argv", ["run.py", "verify", "--output", str(folder)]), \
             patch.object(verifier, "verify") as verify, \
             patch.object(train, "initialize", side_effect=AssertionError("initialize called")), \
             patch.object(train, "audit", side_effect=AssertionError("audit writes files")), \
             patch.object(Path, "mkdir", side_effect=AssertionError("directory created")):
            train.main()
        verify.assert_called_once_with(folder.resolve())

    def test_standalone_verify_requests_read_only_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/"run_manifest.json").write_text(json.dumps(dict(status="complete_verified",
                audit=dict(existing_rf_results_used=False, input_sha256={}))))
            before = self.snapshot(folder)
            with patch.object(verifier, "TRAINING_FILES", ()), \
                 patch.object(verifier, "export_and_verify", side_effect=ValueError("bad CSV")) as check:
                with self.assertRaisesRegex(ValueError, "bad CSV"):
                    verifier.verify(folder)
            check.assert_called_once_with(folder, unittest.mock.ANY, complete=True, write_outputs=False)
            self.assertEqual(before, self.snapshot(folder))

    def predictions(self, model, repeats=(1,)):
        p = self.outer.loc[self.outer["repeat"].isin(repeats)].copy()
        p["model"] = model
        p["y_prob"] = .1 + .8*p.y_true
        p["threshold"] = .5
        return p

    def test_valid_matching_models_are_summarized_despite_row_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); (folder/"predictions").mkdir()
            for model in ["Random_Forest", "Ridge_Logistic"]:
                self.predictions(model).iloc[::-1].to_csv(folder/f"predictions/{model}.csv", index=False)
            repeats, summary = common.summarize_primary(folder)
            self.assertEqual(len(repeats), 2)
            self.assertTrue(repeats.N.eq(569).all())
            self.assertEqual(len(summary), 2)

    def test_invalid_model_maps_fail_before_any_summary_file_is_changed(self):
        for problem in ["id", "label", "fold", "both_wrong_fold", "repeat_set", "missing_record", "duplicate", "missing_fold", "fractional_label", "model_name"]:
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp); (folder/"predictions").mkdir()
                a = self.predictions("Random_Forest", (1, 2))
                b = self.predictions("Ridge_Logistic", (1, 2))
                if problem == "id": b.loc[0, "id"] = -1
                elif problem == "label": b.loc[0, "y_true"] = 1-int(b.loc[0, "y_true"])
                elif problem in ["fold", "both_wrong_fold"]:
                    i = b.index[b.fold.eq(0)][0]; j = b.index[b.fold.eq(1)][0]
                    b.loc[i, "fold"], b.loc[j, "fold"] = 1, 0
                    if problem == "both_wrong_fold": a["fold"] = b.fold
                elif problem == "repeat_set": b = b.loc[b["repeat"].eq(1)]
                elif problem == "missing_record": b = b.iloc[1:]
                elif problem == "duplicate": b.iloc[1] = b.iloc[0]
                elif problem == "missing_fold": b = b.drop(columns="fold")
                elif problem == "fractional_label":
                    b["y_true"] = b.y_true.astype(float); b.loc[0, "y_true"] = .25
                elif problem == "model_name": b["model"] = "Random_Forest"
                a.to_csv(folder/"predictions/Random_Forest.csv", index=False)
                b.to_csv(folder/"predictions/Ridge_Logistic.csv", index=False)
                for name in ["outer_predictions_all_models.csv", "repeat_metrics.csv", "model_summary.csv"]:
                    (folder/name).write_text("existing summary must survive\n")
                before = self.snapshot(folder)
                with self.assertRaises(ValueError): common.summarize_primary(folder)
                self.assertEqual(before, self.snapshot(folder))

    def test_undefined_predictive_values_survive_json_csv_and_checkpoint_validation(self):
        for metric, probability in [("ppv", 0.), ("npv", 1.)]:
            with self.subTest(metric=metric), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                payload = copy.deepcopy(self.wrapper["payload"])
                for row in payload["predictions"]:
                    row["y_prob"] = probability
                    row["predicted_class"] = int(probability >= row["threshold"])
                values = common.summarize_prediction_rows(pd.DataFrame(payload["predictions"]))
                self.assertTrue(np.isnan(values[metric]))
                record = payload["fold_results"][0]
                record.update(values)
                record["validation_auroc"] = values["auroc"]
                record["training_validation_gap"] = record["train_resubstitution_auroc"]-values["auroc"]
                wrapper = dict(run_id=self.manifest["run_id"], fingerprint=self.manifest["fingerprint"],
                               payload_sha256=digest(payload), payload=payload)
                path = folder/"checkpoint.json"
                write_json(path, wrapper)
                restored = json.loads(path.read_text(), parse_constant=lambda value: self.fail("Nonstandard JSON "+value))
                self.assertIsNone(restored["payload"]["fold_results"][0][metric])
                loaded = train.load_checkpoint(path, self.manifest, 1, 0, self.data, self.outer, self.inner)
                frame = pd.DataFrame(loaded["fold_results"])
                csv = folder/"fold_results.csv"
                train._check_or_write_csv(frame, csv, True)
                self.assertTrue(pd.isna(pd.read_csv(csv).loc[0, metric]))
                train._check_or_write_csv(frame, csv, False)
                generic = compute_classification_at_threshold([0, 1], [probability]*2, .5)
                self.assertTrue(np.isnan(generic[metric]))

    def test_shared_summary_preserves_undefined_values_as_blank(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); (folder/"predictions").mkdir()
            for model, threshold in [("Random_Forest", 1.), ("Ridge_Logistic", 0.)]:
                p = self.predictions(model); p["y_prob"] = .5; p["threshold"] = threshold
                p.to_csv(folder/f"predictions/{model}.csv", index=False)
            common.summarize_primary(folder)
            r = pd.read_csv(folder/"repeat_metrics.csv").set_index("model")
            s = pd.read_csv(folder/"model_summary.csv").set_index("model")
            self.assertTrue(pd.isna(r.loc["Random_Forest", "ppv"]))
            self.assertTrue(pd.isna(r.loc["Ridge_Logistic", "npv"]))
            self.assertTrue(pd.isna(s.loc["Random_Forest", "ppv_mean"]))
            self.assertTrue(pd.isna(s.loc["Ridge_Logistic", "npv_mean"]))

    def test_null_cannot_hide_a_defined_metric_or_invalid_probability(self):
        payload = copy.deepcopy(self.wrapper["payload"])
        payload["fold_results"][0]["ppv"] = None
        with self.assertRaises(AssertionError):
            train.validate_fold(payload, 1, 0, self.data, self.outer, self.inner)
        for value in [{"y_prob": np.nan}, {"ppv": np.inf}, {"npv": -np.inf}]:
            with self.assertRaises(ValueError): canonical(value)
        self.assertEqual(digest(self.wrapper["payload"]), self.wrapper["payload_sha256"])

    def test_corrupt_metric_csv_is_rejected_without_rewriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); path = folder/"repeat_metrics.csv"
            expected = pd.DataFrame(dict(repeat=[1], ppv=[np.nan], npv=[.8]))
            expected.fillna(0).to_csv(path, index=False)
            before = self.snapshot(folder)
            with self.assertRaises(AssertionError): train._check_or_write_csv(expected, path, False)
            self.assertEqual(before, self.snapshot(folder))


if __name__ == "__main__":
    unittest.main(verbosity=2)
