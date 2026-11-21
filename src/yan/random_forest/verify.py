"""Independent exported-CSV and provenance checks; no fitting and no threshold retuning."""
from pathlib import Path
import argparse
import io
import json
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from metrics import select_threshold_target_sensitivity, compute_calibration_curve
from yan.random_forest.audit import (ROOT, OUT, MODEL, sha, data_and_splits,
                                     align_reference, TRAINING_FILES)
from yan.random_forest.train import load_checkpoint, checkpoint_path, export_and_verify


def verify(out=OUT):
    out = Path(out)
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["status"] == "complete_verified"
    assert manifest["audit"]["existing_rf_results_used"] is False
    historical_sources = {}
    def check_recorded_file(path, checksum):
        current = ROOT / path
        if current.exists() and sha(current) == checksum:
            return
        # A later dispatcher/README integration does not relabel the original
        # completed analysis as a newly fitted run. Verify its exact archived
        # source instead. Raw/processed data must still match the live inputs.
        snapshot = out / "provenance/training_source" / path
        if path.startswith("data/") or not snapshot.exists() or sha(snapshot) != checksum:
            raise AssertionError(f"Recorded input/source is unavailable or changed: {path}")
        historical_sources[path] = "Exact historical source snapshot verified; current checkout differs"
    for path, checksum in manifest["audit"]["input_sha256"].items():
        check_recorded_file(path, checksum)
    for path in TRAINING_FILES:
        check_recorded_file(path, manifest["audit"]["training_source_sha256"][path])
    checks = export_and_verify(out, manifest, complete=True, write_outputs=False)
    data, outer, inner = data_and_splits()
    p = pd.read_csv(out / "oof_predictions.csv")
    folds = pd.read_csv(out / "fold_results.csv")
    search = pd.read_csv(out / "inner_cv_results.csv")
    selected = pd.read_csv(out / "selected_inner_oof.csv")
    importance = pd.read_csv(out / "feature_importances.csv")
    exp = pd.read_csv(out / "outer_fold_assignments.csv")
    exp_inner = pd.read_csv(out / "inner_fold_assignments.csv")
    pd.testing.assert_frame_equal(outer, exp, check_dtype=False)
    pd.testing.assert_frame_equal(inner, exp_inner, check_dtype=False)
    assert len(folds) == 50 and not folds.duplicated(["repeat", "fold"]).any()
    assert len(search) == 800 and search.groupby(["repeat", "fold"]).size().eq(16).all()
    assert len(importance) == 1500 and importance.groupby(["repeat", "fold"]).size().eq(30).all()
    assert importance.groupby(["repeat", "fold"]).importance.sum().sub(1).abs().lt(1e-12).all()
    assert len(selected) == len(inner) == 22760
    assert not selected.duplicated(["repeat", "fold", "id"]).any()
    assert folds.n_estimators.eq(500).all() and search.n_estimators.eq(500).all()
    assert folds.inner_forest_fits.sum() == 4000 and folds.final_forest_fits.sum() == 50
    assert folds.n_jobs.eq(1).all()
    assert folds.threshold_additional_model_fit_seconds.eq(0).all()
    checked_scores = 0
    for repeat in range(1, 11):
        for fold in range(5):
            payload = load_checkpoint(checkpoint_path(out, repeat, fold), manifest,
                                      repeat, fold, data, outer, inner)
            r = folds.query("repeat == @repeat and fold == @fold").iloc[0]
            s = search.query("repeat == @repeat and fold == @fold").sort_values("candidate")
            ordered = s.sort_values(["mean_inner_auroc", "candidate"], ascending=[False, True])
            assert ordered["rank"].tolist() == list(range(1, 17))
            assert r.selected_candidate == ordered.candidate.iloc[0]
            assert r.inner_seed == 1707 + 100*(repeat-1) + fold
            assert r.outer_seed == 707 + 10000*(repeat-1) + 100*fold
            for row in s.to_dict("records"):
                expected_seeds = [int(r.inner_seed)+100*row["candidate"]+k for k in range(5)]
                assert json.loads(row["inner_model_seeds_json"]) == expected_seeds
                assert len(json.loads(row["inner_fit_seconds_json"])) == 5
                assert all(v > 0 for v in json.loads(row["inner_fit_seconds_json"]))
                assert row["seconds"] > 0
            g = selected.query("repeat == @repeat and fold == @fold")
            chosen = s.loc[s.candidate.eq(r.selected_candidate)].iloc[0]
            for k, group in g.groupby("inner_fold"):
                auc = roc_auc_score(group.y_true, group.y_prob)
                assert np.isclose(auc, chosen[f"inner_fold_{k}_auroc"], atol=1e-12)
                checked_scores += 1
            assert np.isclose(roc_auc_score(g.y_true, g.y_prob), chosen.pooled_inner_auroc, atol=1e-12)
            t, sens = select_threshold_target_sensitivity(g.y_true, g.y_prob, .95)
            assert np.isclose(t, r.threshold, atol=1e-14, rtol=0)
            assert np.isclose(sens, r.inner_sensitivity, atol=1e-14)
            # Discrete brute-force oracle verifies the highest qualifying candidate.
            valid = [v for v in np.unique(g.y_prob)
                     if ((g.y_prob >= v) & g.y_true.eq(1)).sum()/g.y_true.sum() >= .95]
            assert np.isclose(t, max(valid), atol=1e-14, rtol=0)
            assert np.array_equal(g.predicted_class, g.y_prob.ge(g.threshold).astype(int))
            assert not set(g.id) & set(p.query("repeat == @repeat and fold == @fold").id)
            for key, filename in [("predictions", "oof_predictions.csv"),
                                  ("inner_cv_results", "inner_cv_results.csv"),
                                  ("feature_importances", "feature_importances.csv")]:
                exported = {"predictions": p, "inner_cv_results": search,
                            "feature_importances": importance}[key]
                exported = exported.query("repeat == @repeat and fold == @fold")
                expected = pd.DataFrame(payload[key])
                order = {"predictions": ["id"], "inner_cv_results": ["candidate"],
                         "feature_importances": ["feature"]}[key]
                pd.testing.assert_frame_equal(exported.sort_values(order).reset_index(drop=True),
                    expected.sort_values(order).reset_index(drop=True), check_dtype=False,
                    check_exact=False, atol=1e-12, rtol=1e-12)
    calibration = pd.read_csv(out / "calibration_points.csv")
    for repeat, group in p.groupby("repeat"):
        c = compute_calibration_curve(group.y_true, group.y_prob, n_bins=5)
        actual = calibration.loc[calibration["repeat"].eq(repeat)].sort_values("bin")
        assert actual.n.sum() == 569 and np.array_equal(actual.n, c["bin_counts"])
        np.testing.assert_allclose(actual.mean_probability, c["bin_pred_mean"], atol=1e-14)
        np.testing.assert_allclose(actual.observed_fraction, c["bin_true_prop"], atol=1e-14)

    # Read actual branch data/predictions too, when the cloned refs are available.
    branch_checks = {}
    for ref in ["origin/Haiyi-branches", "origin/asher/svm-analysis"]:
        try:
            blob = subprocess.check_output(["git", "-C", str(ROOT), "show",
                ref + ":data/processed/wdbc_clean.csv"])
        except subprocess.CalledProcessError:
            branch_checks[ref] = "not available; not checked"
            continue
        branch_data = pd.read_csv(io.BytesIO(blob)).sort_values("id").reset_index(drop=True)
        pd.testing.assert_frame_equal(branch_data, data, check_exact=False, atol=1e-14, rtol=1e-14)
        branch_checks[ref] = dict(data_values_features_labels_identical=True,
                                 data_blob_sha256=__import__("hashlib").sha256(blob).hexdigest())
    try:
        blob = subprocess.check_output(["git", "-C", str(ROOT), "show",
            "origin/asher/svm-analysis:results/svm/nested/outer_predictions.csv"])
        svm = pd.read_csv(io.BytesIO(blob))
        for name in ["Linear_SVM", "RBF_SVM"]:
            actual = align_reference(svm.loc[svm.model.eq(name)], outer, name)
            main = align_reference(pd.read_csv(ROOT / f"results/final_nested/predictions/{name}.csv"),
                                   outer, name)
            np.testing.assert_allclose(actual.y_prob, main.y_prob, rtol=0, atol=1e-14)
            np.testing.assert_allclose(actual.threshold, main.threshold, rtol=0, atol=1e-14)
        branch_checks["svm_archived_predictions_match_final_main"] = True
    except subprocess.CalledProcessError:
        branch_checks["svm_archive"] = "not available; not checked"
    # Core source in Haiyi's branch is available and identical, or disclosed.
    core = {}
    for path in ["src/final_analysis.py", "src/data_prep.py", "src/metrics.py"]:
        try:
            blob = subprocess.check_output(["git", "-C", str(ROOT), "show", "origin/Haiyi-branches:" + path])
            core[path] = blob == (ROOT / path).read_bytes()
        except subprocess.CalledProcessError:
            core[path] = "not available; not checked"
    checks.update(historical_source_snapshots_used=historical_sources,
        selected_inner_fold_auroc_recomputed=checked_scores,
        checkpoint_csv_roundtrip=True, class_1_probability_lookup=True,
        all_candidate_seed_records_checked=True, threshold_bruteforce_oracle_passed=True,
        raw_clean_agreement=True, actual_branch_checks=branch_checks,
        haiyi_core_identical=core, completed_inner_forest_fits=4000, completed_final_forest_fits=50,
        feature_importance_rows=1500, selected_inner_oof_rows=len(selected),
        report_sources="All statistics recomputable from exported CSV; no model refits during verification")
    print(json.dumps(checks, indent=2))
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    verify(parser.parse_args().output.resolve())
