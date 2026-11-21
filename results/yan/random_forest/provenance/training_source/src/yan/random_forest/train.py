"""Fresh, audited 10x5 RF nested CV using the unchanged shared design.

No existing Random Forest result is read. No other model is trained.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import json
import sys
import time
import uuid
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from final_analysis import (FEATURE_NAMES, MODEL_GRIDS, _outer_split, load_data,
                            make_model, params_json, summarize_prediction_rows)
from metrics import select_threshold_target_sensitivity, compute_calibration_curve
from .audit import (ROOT, OUT, MODEL, audit, data_and_splits, canonical, digest, sha,
                    write_json, align_reference)


def utc():
    return datetime.now(timezone.utc).isoformat()


def positive_probability(model, x):
    columns = np.flatnonzero(np.asarray(model.classes_) == 1)
    if len(columns) != 1 or set(model.classes_) != {0, 1}:
        raise ValueError("Cannot identify malignant class probability")
    p = model.predict_proba(x)[:, int(columns[0])]
    if not np.isfinite(p).all() or not ((0 <= p) & (p <= 1)).all():
        raise ValueError("Invalid probabilities")
    return p


def fit_model(params, seed, x, y):
    model = make_model(MODEL, params, seed).set_params(n_jobs=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x, y)
    if len(model.estimators_) != 500 or model.n_features_in_ != 30:
        raise AssertionError("Forest size or feature count changed")
    assert list(model.feature_names_in_) == FEATURE_NAMES
    return model, [f"{w.category.__name__}: {w.message}" for w in caught]


def run_fold(repeat, fold):
    """Training receives only outer-training data until final validation scoring."""
    with threadpool_limits(limits=1):
        data = load_data()
        tr, va = list(_outer_split(data, repeat - 1, 5))[fold]
        assert not np.intersect1d(tr, va).size
        x = data.iloc[tr][FEATURE_NAMES]
        y = data.target.to_numpy()[tr]
        inner_seed = 1707 + 100 * (repeat - 1) + fold
        outer_seed = 707 + 10000 * (repeat - 1) + 100 * fold
        cv = list(StratifiedKFold(5, shuffle=True, random_state=inner_seed).split(x, y))
        start = time.perf_counter()
        search, candidates_oof, notes = [], [], []
        inner_assignment = np.full(len(tr), -1, dtype=int)
        for k, (a, b) in enumerate(cv):
            assert not np.intersect1d(tr[a], tr[b]).size
            assert not np.intersect1d(tr[b], va).size
            inner_assignment[b] = k
        for candidate, params in enumerate(MODEL_GRIDS[MODEL]):
            tick = time.perf_counter()
            oof = np.full(len(tr), np.nan)
            scores, seeds, fit_times = [], [], []
            candidate_warnings = []
            for k, (a, b) in enumerate(cv):
                seed = inner_seed + 100 * candidate + k
                tick_fit = time.perf_counter()
                model, messages = fit_model(params, seed, x.iloc[a], y[a])
                fit_times.append(time.perf_counter() - tick_fit)
                oof[b] = positive_probability(model, x.iloc[b])
                scores.append(float(roc_auc_score(y[b], oof[b])))
                seeds.append(seed)
                candidate_warnings.extend(messages)
            assert np.isfinite(oof).all()
            search.append(dict(repeat=repeat, fold=fold, model=MODEL, candidate=candidate,
                params_json=params_json(params), mean_inner_auroc=float(np.mean(scores)),
                inner_auroc_sd=float(np.std(scores, ddof=1)),
                pooled_inner_auroc=float(roc_auc_score(y, oof)),
                inner_model_seeds_json=canonical(seeds), inner_fit_seconds_json=canonical(fit_times),
                seconds=time.perf_counter() - tick, warnings_json=canonical(candidate_warnings),
                n_estimators=500, **{f"inner_fold_{k}_auroc": s for k, s in enumerate(scores)}))
            candidates_oof.append(oof)
            notes.extend(candidate_warnings)
        # Same shared tie rule: highest mean; first ParameterGrid candidate on exact ties.
        selected = max(range(16), key=lambda i: (search[i]["mean_inner_auroc"], -i))
        order = sorted(range(16), key=lambda i: (-search[i]["mean_inner_auroc"], i))
        for rank, i in enumerate(order, 1):
            search[i]["rank"] = rank
        tuning_seconds = time.perf_counter() - start
        selected_oof = candidates_oof[selected]
        tick = time.perf_counter()
        threshold, inner_sensitivity = select_threshold_target_sensitivity(y, selected_oof, .95)
        threshold_seconds = time.perf_counter() - tick
        assert inner_sensitivity >= .95
        params = MODEL_GRIDS[MODEL][selected]
        tick = time.perf_counter()
        fitted, messages = fit_model(params, outer_seed, x, y)
        final_fit_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        train_probability = positive_probability(fitted, x)
        # Validation features are used only after parameters and threshold are locked.
        probability = positive_probability(fitted, data.iloc[va][FEATURE_NAMES])
        prediction_seconds = time.perf_counter() - tick
        predictions = [dict(id=int(data.id.iloc[i]), y_true=int(data.target.iloc[i]),
            model=MODEL, repeat=repeat, fold=fold, y_prob=float(p), threshold=float(threshold),
            predicted_class=int(p >= threshold)) for i, p in zip(va, probability)]
        validation = summarize_prediction_rows(pd.DataFrame(predictions))
        inner_predictions = [dict(repeat=repeat, fold=fold, inner_fold=int(k),
            id=int(data.id.iloc[i]), y_true=int(v), y_prob=float(p), threshold=float(threshold),
            predicted_class=int(p >= threshold))
            for i, v, p, k in zip(tr, y, selected_oof, inner_assignment)]
        importance = [dict(repeat=repeat, fold=fold, feature=f, importance=float(v))
                      for f, v in zip(FEATURE_NAMES, fitted.feature_importances_)]
        total_seconds = time.perf_counter() - start
        record = dict(repeat=repeat, fold=fold, model=MODEL, n_train=len(tr), n_validation=len(va),
            n_train_malignant=int(y.sum()), n_train_benign=int((y == 0).sum()),
            n_validation_malignant=int(data.target.iloc[va].sum()),
            n_validation_benign=int((data.target.iloc[va] == 0).sum()),
            params_json=params_json(params), selected_candidate=selected,
            best_mean_inner_auroc=search[selected]["mean_inner_auroc"],
            threshold=float(threshold), inner_sensitivity=float(inner_sensitivity),
            train_resubstitution_auroc=float(roc_auc_score(y, train_probability)),
            validation_auroc=float(validation["auroc"]),
            training_validation_gap=float(roc_auc_score(y, train_probability) - validation["auroc"]),
            tuning_seconds=tuning_seconds, threshold_selection_seconds=threshold_seconds,
            threshold_additional_model_fit_seconds=0.0, final_fit_seconds=final_fit_seconds,
            prediction_seconds=prediction_seconds, total_seconds=total_seconds,
            inner_seed=inner_seed, outer_seed=outer_seed, n_jobs=1, n_estimators=500,
            inner_forest_fits=80, final_forest_fits=1,
            warnings_json=canonical(notes + messages), exception_json="[]", **validation)
        return dict(predictions=predictions, fold_results=[record], inner_cv_results=search,
                    selected_inner_oof=inner_predictions, feature_importances=importance)


def validate_fold(payload, repeat, fold, data, outer, inner):
    """Reject incomplete/misaligned checkpoints before resuming or exporting."""
    p = pd.DataFrame(payload["predictions"])
    expect = outer.query("repeat == @repeat and fold == @fold").sort_values("id")
    assert len(p) == len(expect) and not p.id.duplicated().any()
    pd.testing.assert_frame_equal(p.sort_values("id")[["id", "y_true", "repeat", "fold"]].reset_index(drop=True),
        expect[["id", "y_true", "repeat", "fold"]].reset_index(drop=True), check_dtype=False)
    assert set(p.model) == {MODEL}
    assert np.isfinite(p[["y_prob", "threshold"]]).all().all()
    assert p.y_prob.between(0, 1).all() and p.threshold.between(0, 1).all()
    assert p.threshold.nunique() == 1
    assert np.array_equal(p.predicted_class, p.y_prob.ge(p.threshold).astype(int))
    s = pd.DataFrame(payload["inner_cv_results"])
    assert len(s) == 16 and set(s.candidate) == set(range(16))
    assert not s.duplicated(["repeat", "fold", "candidate"]).any()
    assert set(s["repeat"]) == {repeat} and set(s.fold) == {fold}
    assert set(s.model) == {MODEL} and s.n_estimators.eq(500).all()
    for r in s.to_dict("records"):
        assert r["params_json"] == params_json(MODEL_GRIDS[MODEL][r["candidate"]])
        scores = [r[f"inner_fold_{k}_auroc"] for k in range(5)]
        assert all(np.isfinite(scores)) and np.isclose(np.mean(scores), r["mean_inner_auroc"], atol=1e-14)
        assert np.isclose(np.std(scores, ddof=1), r["inner_auroc_sd"], atol=1e-14)
    chosen = s.sort_values(["mean_inner_auroc", "candidate"], ascending=[False, True]).iloc[0]
    i = pd.DataFrame(payload["selected_inner_oof"])
    exp_i = inner.query("repeat == @repeat and fold == @fold").sort_values("id")
    assert len(i) == len(exp_i) and not i.id.duplicated().any() and not set(i.id) & set(p.id)
    pd.testing.assert_frame_equal(i.sort_values("id")[["id", "y_true", "inner_fold", "repeat", "fold"]].reset_index(drop=True),
        exp_i[["id", "y_true", "inner_fold", "repeat", "fold"]].reset_index(drop=True), check_dtype=False)
    assert np.isfinite(i.y_prob).all() and i.y_prob.between(0, 1).all()
    assert set(i.id) == set(data.id) - set(p.id)
    threshold, sens = select_threshold_target_sensitivity(i.y_true, i.y_prob, .95)
    assert np.isclose(threshold, p.threshold.iloc[0], atol=1e-14, rtol=0)
    assert i.threshold.eq(threshold).all() and sens >= .95
    assert np.array_equal(i.predicted_class, i.y_prob.ge(i.threshold).astype(int))
    assert len(payload["fold_results"]) == 1
    r = payload["fold_results"][0]
    assert (r["repeat"], r["fold"], r["model"]) == (repeat, fold, MODEL)
    assert r["selected_candidate"] == int(chosen.candidate) and r["params_json"] == chosen.params_json
    assert np.isclose(r["threshold"], threshold, atol=1e-14, rtol=0)
    assert np.isclose(r["inner_sensitivity"], sens, atol=1e-14)
    assert np.isclose(r["best_mean_inner_auroc"], chosen.mean_inner_auroc, atol=1e-14)
    assert r["n_train"] == len(i) and r["n_validation"] == len(p)
    assert r["n_train_malignant"] == int(i.y_true.sum())
    assert r["n_validation_malignant"] == int(p.y_true.sum())
    assert r["n_train_benign"] == int(i.y_true.eq(0).sum())
    assert r["n_validation_benign"] == int(p.y_true.eq(0).sum())
    assert r["inner_forest_fits"] == 80 and r["final_forest_fits"] == 1 and r["n_estimators"] == 500
    assert np.isclose(r["pooled_inner_auroc"] if "pooled_inner_auroc" in r else roc_auc_score(i.y_true, i.y_prob),
                      roc_auc_score(i.y_true, i.y_prob))
    for metric, value in summarize_prediction_rows(p).items():
        assert np.isclose(r[metric], value, equal_nan=True, atol=1e-12)
    assert np.isclose(r["validation_auroc"], r["auroc"])
    assert np.isclose(r["training_validation_gap"], r["train_resubstitution_auroc"] - r["auroc"])
    assert 0 <= r["train_resubstitution_auroc"] <= 1
    importance = pd.DataFrame(payload["feature_importances"])
    assert len(importance) == 30 and set(importance.feature) == set(FEATURE_NAMES)
    assert not importance.feature.duplicated().any() and importance.importance.ge(0).all()
    assert set(importance["repeat"]) == {repeat} and set(importance.fold) == {fold}
    assert np.isclose(importance.importance.sum(), 1)


def checkpoint_path(out, repeat, fold):
    return out / "checkpoints" / f"repeat_{repeat:02d}_fold_{fold}.json"


def load_checkpoint(path, manifest, repeat, fold, data, outer, inner):
    wrapper = json.loads(path.read_text())
    assert wrapper["run_id"] == manifest["run_id"] and wrapper["fingerprint"] == manifest["fingerprint"]
    assert wrapper["payload_sha256"] == digest(wrapper["payload"])
    validate_fold(wrapper["payload"], repeat, fold, data, outer, inner)
    return wrapper["payload"]


def export_and_verify(out, manifest, complete):
    data, outer, inner = data_and_splits()
    payloads = []
    for repeat in range(1, 11):
        for fold in range(5):
            path = checkpoint_path(out, repeat, fold)
            if path.exists():
                payloads.append(load_checkpoint(path, manifest, repeat, fold, data, outer, inner))
    for key, filename in [
        ("predictions", "oof_predictions.csv"), ("fold_results", "fold_results.csv"),
        ("inner_cv_results", "inner_cv_results.csv"), ("selected_inner_oof", "selected_inner_oof.csv"),
        ("feature_importances", "feature_importances.csv")]:
        pd.DataFrame([r for p in payloads for r in p[key]]).to_csv(out / filename, index=False)
    p = pd.read_csv(out / "oof_predictions.csv")
    assert not p.duplicated(["model", "repeat", "id"]).any()
    checks = dict(completed_outer_folds=len(payloads), expected_outer_folds=50,
                  prediction_rows=len(p), complete=complete,
                  checkpoint_integrity_and_full_fold_alignment=True,
                  selected_inner_oof_thresholds_recomputed=True,
                  old_rf_outputs_consumed=False)
    if complete:
        assert len(payloads) == 50 and len(p) == 5690
        align_reference(p, outer, MODEL)
        assert p.groupby("repeat").size().eq(569).all()
        reps, calibration = [], []
        for repeat, g in p.groupby("repeat"):
            reps.append(dict(repeat=int(repeat), model=MODEL, **summarize_prediction_rows(g)))
            cal = compute_calibration_curve(g.y_true, g.y_prob, n_bins=5)
            assert sum(cal["bin_counts"]) == 569
            calibration.extend(dict(repeat=int(repeat), bin=b, mean_probability=v,
                                    observed_fraction=y, n=n)
                for b, (v, y, n) in enumerate(zip(cal["bin_pred_mean"], cal["bin_true_prop"], cal["bin_counts"]), 1))
        reps = pd.DataFrame(reps)
        reps.to_csv(out / "repeat_metrics.csv", index=False)
        metrics = [c for c in reps if c not in ["repeat", "model", "N"]]
        summary = reps[metrics].agg(["mean", "std", "min", "max"]).T.reset_index(names="metric")
        summary.to_csv(out / "summary_metrics.csv", index=False)
        pd.DataFrame(calibration).to_csv(out / "calibration_points.csv", index=False)
        # Independent formulas verify the CSV metrics rather than reusing the shared helper alone.
        from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, confusion_matrix
        for r in reps.to_dict("records"):
            g = p.loc[p["repeat"].eq(r["repeat"])]
            y, prob = g.y_true.to_numpy(), g.y_prob.to_numpy()
            tn, fp, fn, tp = confusion_matrix(y, prob >= g.threshold.to_numpy(), labels=[0, 1]).ravel()
            independent = dict(auroc=roc_auc_score(y, prob), auprc=average_precision_score(y, prob),
                brier_score=brier_score_loss(y, prob), log_loss=log_loss(y, np.clip(prob, 1e-15, 1-1e-15)),
                sensitivity=tp/(tp+fn), specificity=tn/(tn+fp), ppv=tp/(tp+fp) if tp+fp else np.nan,
                npv=tn/(tn+fn) if tn+fn else np.nan, TN=tn, FP=fp, FN=fn, TP=tp,
                mean_predicted=prob.mean(), observed_prevalence=y.mean())
            for key, val in independent.items():
                assert np.isclose(r[key], val, equal_nan=True, atol=1e-12)
        checks.update(repeat_rows=len(reps), candidate_rows=sum(len(v["inner_cv_results"]) for v in payloads),
                      independent_metric_recalculation=True, all_required_checks_passed=True)
    write_json(out / "verification.json", checks)
    return checks


def initialize(out, workers):
    info = audit(out)
    config = dict(model=MODEL, repeats=10, outer_folds=5, inner_folds=5, grid=MODEL_GRIDS[MODEL],
        n_estimators=500, n_jobs=1, outer_task_workers=workers, target_sensitivity=.95,
        outer_seed="707 + zero_based_repeat", inner_seed="1707 + 100*zero_based_repeat + fold",
        candidate_seed="inner_seed + 100*candidate + inner_fold",
        final_seed="707 + 10000*zero_based_repeat + 100*fold",
        tie_rule="highest mean inner AUROC; lowest candidate index on exact tie",
        probability_column="lookup classes_==1", preprocessing="none",
        threshold_oof="selected candidate same inner OOF; no third layer",
        log_loss_clip=[1e-15, 1-1e-15])
    fingerprint = digest(dict(audit=info, config=config))
    path = out / "run_manifest.json"
    if path.exists():
        m = json.loads(path.read_text())
        if m["fingerprint"] != fingerprint:
            raise RuntimeError("Data/code/config/environment changed. Use a new personal output directory.")
    else:
        from final_analysis import make_model
        defaults = make_model(MODEL, MODEL_GRIDS[MODEL][0], 707).set_params(n_jobs=1).get_params()
        m = dict(run_id=str(uuid.uuid4()), fingerprint=fingerprint, started_utc=utc(),
            status="initialized", audit=info, config=config, forest_example_parameters=defaults,
            commands=[], sessions=[], git_state="local generated files only; no commit/push/PR",
            source_changes="New src/yan/random_forest and scripts/yan files; shared files unchanged")
        write_json(path, m)
    return m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["audit", "pilot", "run", "verify", "report"])
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to((ROOT / "results/yan/random_forest").resolve()):
        parser.error("Output must stay inside results/yan/random_forest; shared results are protected.")
    if args.workers not in (1, 2):
        parser.error("Use one or two outer workers; every forest uses n_jobs=1.")
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "audit":
        print(json.dumps(audit(out), indent=2)); return
    if args.stage == "report":
        from .report import build_report
        build_report(out); return
    m = initialize(out, args.workers)
    data, outer, inner = data_and_splits()
    if args.stage == "verify":
        print(export_and_verify(out, m, complete=True)); return
    tick = time.perf_counter()
    m["commands"].append(dict(utc=utc(), argv=sys.argv))
    m["status"] = "running"
    write_json(out / "run_manifest.json", m)
    targets = [(1, 0)] if args.stage == "pilot" else [(r, f) for r in range(1, 11) for f in range(5)]
    pending = []
    for repeat, fold in targets:
        path = checkpoint_path(out, repeat, fold)
        if path.exists():
            load_checkpoint(path, m, repeat, fold, data, outer, inner)
        else:
            pending.append((repeat, fold))
    if args.stage == "run":
        # Full analysis follows only after an audited complete 16-candidate pilot.
        assert checkpoint_path(out, 1, 0).exists(), "Run pilot first."
        load_checkpoint(checkpoint_path(out, 1, 0), m, 1, 0, data, outer, inner)
    def save(payload, r, f):
        validate_fold(payload, r, f, data, outer, inner)
        write_json(checkpoint_path(out, r, f), dict(run_id=m["run_id"],
            fingerprint=m["fingerprint"], payload_sha256=digest(payload), payload=payload))
        print(f"Saved repeat {r}/10 fold {f}/4; candidates=16, fits=81; "
              f"seconds={payload['fold_results'][0]['total_seconds']:.1f}", flush=True)
    try:
        if args.stage == "pilot" or args.workers == 1:
            for r, f in pending:
                save(run_fold(r, f), r, f)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(run_fold, r, f): (r, f) for r, f in pending}
                for future in as_completed(futures):
                    r, f = futures[future]
                    save(future.result(), r, f)
        checks = export_and_verify(out, m, complete=args.stage == "run")
        m["status"] = "complete_verified" if args.stage == "run" else "pilot_verified"
        m["verification"] = checks
    except BaseException as error:
        m["status"] = "interrupted_or_failed"
        m["last_error"] = repr(error)
        raise
    finally:
        m["sessions"].append(dict(stage=args.stage, finished_utc=utc(),
            wall_seconds=time.perf_counter()-tick, newly_fitted_outer_folds=len(pending)))
        m["updated_utc"] = utc()
        write_json(out / "run_manifest.json", m)

