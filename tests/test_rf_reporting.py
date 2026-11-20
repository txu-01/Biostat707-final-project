"""Report-only checks: pooling, alignment, source boundaries and immutable inputs."""
from pathlib import Path
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from yan.random_forest import reporting as report


class ReportingChecks(unittest.TestCase):
    def test_repeat_auc_is_pooled_and_locked_thresholds_are_used(self):
        # Each fold has perfect AUC; their pooled ranking has AUC .75.
        rows=pd.DataFrame(dict(model=["Random_Forest"]*4,repeat=[1]*4,
            fold=[0,0,1,1],y_true=[0,1,0,1],y_prob=[.49,.50,.01,.02],
            threshold=[.495,.495,.015,.015]))
        result=report.pooled_repeat_metrics(rows).iloc[0]
        self.assertAlmostEqual(result.auroc,.75)
        self.assertEqual((result.FN,result.FP),(0,0))

    def test_tied_calibration_edges_cover_every_record_once(self):
        rows=pd.DataFrame(dict(repeat=[1]*10,y_prob=[0]*6+[.5]*2+[1]*2,
                               y_true=[0,0,0,0,0,1,0,1,1,1]))
        bins=report.expanded_calibration(rows)
        self.assertEqual(bins.n.sum(),10)
        self.assertLess(len(bins),5)
        self.assertTrue(np.isfinite(bins.mean_probability).all())

    def test_changed_fold_or_label_is_rejected(self):
        p=pd.read_csv(report.ROOT/"results/yan/random_forest/oof_predictions.csv")
        expected=p[["id","y_true","repeat","fold"]].copy()
        for field in ["fold","y_true"]:
            bad=p.copy()
            bad.loc[0,field]=1-int(bad.loc[0,field])
            with self.assertRaises(AssertionError):
                report.validate_prediction_keys(bad,expected,"Random_Forest")

    def test_missing_or_unconfirmed_owner_never_uses_shared_model_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            with patch.object(pd,"read_csv",side_effect=AssertionError("unexpected result read")):
                refs,paired,status=report.owner_comparisons(out,pd.DataFrame(),pd.DataFrame())
                self.assertTrue(refs.empty and paired.empty)
                (out/"comparison_provenance.json").write_text(json.dumps(
                    {"svm":{"eligible":False,"owner":"unknown"}}))
                refs,paired,status=report.owner_comparisons(out,pd.DataFrame(),pd.DataFrame())
                self.assertTrue(refs.empty and paired.empty)
                self.assertIn("pending",status["svm"])

    def test_complete_report_never_fits_or_changes_primary_files(self):
        source=report.ROOT/"results/yan/random_forest"
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            for name in [*report.PRIMARY,"run_manifest.json","comparison_provenance.json"]:
                shutil.copy2(source/name,out/name)
            shutil.copytree(source/"comparison_sources",out/"comparison_sources")
            before={name:(out/name).read_bytes() for name in report.PRIMARY}
            read=pd.read_csv
            def guarded_read(path,*args,**kwargs):
                name=str(path)
                if "results/final_nested" in name or "model_summary" in name:
                    raise AssertionError("Shared all-model output is not an allowed report input")
                return read(path,*args,**kwargs)
            with patch.object(RandomForestClassifier,"fit",side_effect=AssertionError("RF fitted")), \
                 patch.object(LogisticRegression,"fit",side_effect=AssertionError("regression fitted")), \
                 patch.object(SVC,"fit",side_effect=AssertionError("SVM fitted")), \
                 patch.object(pd,"read_csv",side_effect=guarded_read):
                checked=report.build_report(out)
            self.assertTrue(checked["primary_files_unchanged"])
            self.assertEqual(before,{name:(out/name).read_bytes() for name in report.PRIMARY})
            self.assertEqual(len(read(out/"report_repeat_metrics.csv")),10)
            self.assertIn("verified owner-specific",checked["paired_sources"]["svm"])
            self.assertFalse(checked["regression_numeric_comparison_included"])


if __name__=="__main__":
    unittest.main()
