"""Asher Xu's prespecified SVM analysis; development and test are separate stages.

Run from the repository root: python src/train_svm.py develop, then finalize.
No outputs from a prior development lock are silently overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data_prep import FEATURE_NAMES
from evaluate_all import validate_and_align_predictions
from metrics import evaluate_predictions, select_threshold_target_sensitivity

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'svm'
PRED = ROOT / 'results' / 'predictions'
MODELS = ROOT / 'models' / 'svm'
SEED = 707
GRIDS = {
    'Linear_SVM': {'C': [0.01, 0.1, 1, 10]},
    'RBF_SVM': {'C': [0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1]},
}
TRACKED_INPUTS = [
    'data/processed/wdbc_clean.csv', 'data/processed/splits.csv',
    'src/train_svm.py', 'scripts/run_svm.py', 'src/metrics.py', 'src/data_prep.py', 'src/evaluate_all.py',
]


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def hashes():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in TRACKED_INPUTS}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')


def environment():
    packages = ['numpy', 'pandas', 'scikit-learn', 'scipy', 'matplotlib', 'joblib', 'tabulate']
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': {name: importlib.metadata.version(name) for name in packages}}


def make_model(name, params):
    # The scaler must be INSIDE the estimator wrapped by calibration. This also
    # prevents its fit from seeing the internal calibration validation subsets.
    base = Pipeline([
        ('scale', StandardScaler()),
        ('svc', SVC(kernel='linear' if name == 'Linear_SVM' else 'rbf',
                    class_weight=None, random_state=SEED,
                    cache_size=256, tol=1e-3, **params)),
    ])
    return CalibratedClassifierCV(
        estimator=base, method='sigmoid',
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
        ensemble=False, n_jobs=1,
    )


def load_data():
    data = pd.read_csv(ROOT / 'data/processed/wdbc_clean.csv')
    splits = pd.read_csv(ROOT / 'data/processed/splits.csv')
    if data.id.duplicated().any() or splits.id.duplicated().any():
        raise ValueError('IDs must be unique.')
    if set(data.id) != set(splits.id):
        raise ValueError('Data and official split IDs differ.')
    if [c for c in data.columns if c not in ['id', 'diagnosis', 'target']] != FEATURE_NAMES:
        raise ValueError('Expected exactly the original 30 predictors in their official order.')
    # Only split assignments, not duplicated label columns, are merged.
    data = data.merge(splits[['id', 'split', 'cv_fold']], on='id', validate='one_to_one')
    dev = data.loc[data.split.eq('development')].copy().reset_index(drop=True)
    if len(dev) != 455 or sorted(dev.cv_fold.unique()) != list(range(5)):
        raise ValueError('Development cohort or official CV folds changed.')
    if dev.target.sum() != 170 or not dev.groupby('cv_fold').size().eq(91).all():
        raise ValueError('Unexpected development labels or fold sizes.')
    if not np.isfinite(dev[FEATURE_NAMES].to_numpy()).all():
        raise ValueError('Missing/nonfinite development values; review protocol before proceeding.')
    return data, dev, splits


def check_lock():
    lock = json.loads((OUT / 'development_lock.json').read_text(encoding='utf8'))
    if lock['input_sha256'] != hashes():
        raise ValueError('Data/code changed after development lock. Do not retune using test results.')
    if lock['environment'] != environment():
        raise ValueError('Environment differs from the recorded development environment.')
    for rel, expected in lock['oof_sha256'].items():
        if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != expected:
            raise ValueError('Locked OOF predictions changed.')
    return lock


def develop():
    if (OUT / 'development_lock.json').exists():
        raise FileExistsError('Development is already locked; use verify or finalize, not retuning.')
    started = time.perf_counter()
    before_hashes = hashes()
    _, dev, splits = load_data()
    X, y = dev[FEATURE_NAMES], dev.target.to_numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    cv_rows, fold_rows, importance_rows, model_locks = [], [], [], {}
    for name, grid in GRIDS.items():
        best = None
        search_start = time.perf_counter()
        for candidate, params in enumerate(ParameterGrid(grid)):
            probs = np.full(len(dev), np.nan)
            estimators, scores, fit_seconds, predict_seconds = [], [], [], []
            for fold in range(5):
                train_idx = np.flatnonzero(dev.cv_fold.to_numpy() != fold)
                val_idx = np.flatnonzero(dev.cv_fold.to_numpy() == fold)
                model = make_model(name, params)
                tick = time.perf_counter()
                model.fit(X.iloc[train_idx], y[train_idx])
                fit_seconds.append(time.perf_counter() - tick)
                tick = time.perf_counter()
                probs[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
                predict_seconds.append(time.perf_counter() - tick)
                scores.append(roc_auc_score(y[val_idx], probs[val_idx]))
                estimators.append((model, train_idx, val_idx))
            mean_auc = float(np.mean(scores))
            row = {'model_name': name, 'candidate': candidate, **params,
                   'mean_cv_auroc': mean_auc, 'sd_cv_auroc': float(np.std(scores, ddof=1)),
                   'pooled_oof_auroc': float(roc_auc_score(y, probs)),
                   'fit_seconds': sum(fit_seconds), 'predict_seconds': sum(predict_seconds)}
            row.update({f'fold_{k}_auroc': score for k, score in enumerate(scores)})
            cv_rows.append(row)
            print(f'{name} {candidate + 1}/{len(ParameterGrid(grid))}: {params}; CV AUROC={mean_auc:.6f}', flush=True)
            # Predeclared deterministic tie rule: first ParameterGrid entry wins.
            if best is None or mean_auc > best['mean_auc'] + 1e-12:
                best = {'params': params, 'mean_auc': mean_auc, 'probs': probs,
                        'estimators': estimators, 'fit_seconds': fit_seconds,
                        'predict_seconds': predict_seconds, 'candidate': candidate}
        search_seconds = time.perf_counter() - search_start
        oof = pd.DataFrame({'id': dev.id, 'y_true': y, 'model_name': name,
                            'fold': dev.cv_fold, 'y_prob': best['probs']})
        validate_and_align_predictions(oof, splits, 'development')
        oof.to_csv(PRED / f'oof_{name}.csv', index=False)
        threshold, sensitivity = select_threshold_target_sensitivity(y, best['probs'], 0.95)
        best_metrics = evaluate_predictions(y, best['probs'], fixed_threshold=threshold)
        save_json(OUT / f'{name}_development_metrics.json', best_metrics)
        for fold, (model, train_idx, val_idx) in enumerate(best['estimators']):
            res = evaluate_predictions(y[val_idx], best['probs'][val_idx], fixed_threshold=threshold)
            fold_rows.append({'model_name': name, 'fold': fold, 'n_train': len(train_idx),
                              'n_validation': len(val_idx), 'auroc': res['auroc'],
                              'brier_score': res['brier_score'], 'log_loss': res['log_loss'],
                              'fit_seconds': best['fit_seconds'][fold],
                              'predict_seconds': best['predict_seconds'][fold]})
            imp = permutation_importance(model, X.iloc[val_idx], y[val_idx],
                                         scoring='roc_auc', n_repeats=10,
                                         random_state=SEED + fold, n_jobs=1)
            for feature, values in zip(FEATURE_NAMES, imp.importances):
                importance_rows.append({'model_name': name, 'fold': fold, 'feature': feature,
                                        'auroc_drop_mean': float(np.mean(values)),
                                        'auroc_drop_sd': float(np.std(values))})
        model_locks[name] = {'params': best['params'], 'selected_candidate': best['candidate'],
                             'mean_cv_auroc': best['mean_auc'], 'threshold': threshold,
                             'development_sensitivity': sensitivity, 'search_seconds': search_seconds}
    pd.DataFrame(cv_rows).to_csv(OUT / 'cv_search_results.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / 'selected_model_fold_metrics.csv', index=False)
    importance = pd.DataFrame(importance_rows)
    importance.to_csv(OUT / 'development_permutation_importance_by_fold.csv', index=False)
    importance.groupby(['model_name', 'feature']).auroc_drop_mean.agg(['mean', 'std']).reset_index().to_csv(
        OUT / 'development_permutation_importance.csv', index=False)
    if hashes() != before_hashes:
        raise RuntimeError('An input changed during development.')
    lock = {'created_utc': timestamp(), 'seed': SEED, 'feature_names': FEATURE_NAMES,
            'selection_metric': 'mean shared-five-fold AUROC',
            'tie_rule': 'first ParameterGrid entry within absolute tolerance 1e-12',
            'probability_calibration': '3-fold sigmoid; ensemble=False; scaling inside base estimator',
            'oof_limitation': 'Hyperparameters selected on these same folds; post-selection OOF is optimistic.',
            'preferred_svm_before_test': max(model_locks, key=lambda n: model_locks[n]['mean_cv_auroc']),
            'models': model_locks, 'environment': environment(), 'input_sha256': before_hashes,
            'oof_sha256': {str((PRED / f'oof_{n}.csv').relative_to(ROOT)).replace('\\', '/'):
                           hashlib.sha256((PRED / f'oof_{n}.csv').read_bytes()).hexdigest() for n in GRIDS},
            'development_total_seconds': time.perf_counter() - started}
    save_json(OUT / 'development_lock.json', lock)
    print('Development parameters and thresholds LOCKED. No held-out predictions evaluated.', flush=True)


def finalize():
    if (OUT / 'test_complete.json').exists():
        raise FileExistsError('Final test already completed. Use verify; do not retune or repeatedly score.')
    lock = check_lock()
    data, dev, splits = load_data()
    MODELS.mkdir(parents=True, exist_ok=True)
    # Finish ALL fits, and save evidence, before touching held-out features/labels.
    fitted, times = {}, {}
    for name, specification in lock['models'].items():
        tick = time.perf_counter()
        fitted[name] = make_model(name, specification['params']).fit(dev[FEATURE_NAMES], dev.target)
        times[name] = {'final_fit_seconds': time.perf_counter() - tick}
        joblib.dump(fitted[name], MODELS / f'{name}.joblib')
    save_json(OUT / 'final_models_fitted.json', {'created_utc': timestamp(), 'models': list(fitted)})
    test = data.loc[data.split.eq('test')].copy().reset_index(drop=True)
    if len(test) != 114 or not np.isfinite(test[FEATURE_NAMES].to_numpy()).all():
        raise ValueError('Unexpected held-out cohort or nonfinite values.')
    if not np.array_equal(test.target.to_numpy(), (test.diagnosis == 'M').astype(int).to_numpy()):
        raise ValueError('Invalid held-out labels.')
    for name, model in fitted.items():
        tick = time.perf_counter()
        probs = model.predict_proba(test[FEATURE_NAMES])[:, 1]
        times[name]['test_predict_seconds'] = time.perf_counter() - tick
        pred = pd.DataFrame({'id': test.id, 'y_true': test.target, 'model_name': name,
                             'split': 'test', 'y_prob': probs})
        validate_and_align_predictions(pred, splits, 'test')
        pred.to_csv(PRED / f'test_{name}.csv', index=False)
        res = evaluate_predictions(test.target, probs, fixed_threshold=lock['models'][name]['threshold'])
        save_json(OUT / f'{name}_test_metrics.json', res)
        print(f'{name}: test AUROC={res["auroc"]:.6f}; Brier={res["brier_score"]:.6f}', flush=True)
    # Coefficients describe the final standardized linear decision score, not log odds.
    linear = fitted['Linear_SVM'].calibrated_classifiers_[0].estimator
    pd.DataFrame({'feature': FEATURE_NAMES, 'standardized_margin_coefficient':
                  linear.named_steps['svc'].coef_[0]}).to_csv(OUT / 'linear_svm_coefficients.csv', index=False)
    save_json(OUT / 'runtime_summary.json', times)
    save_json(OUT / 'test_complete.json', {
        'created_utc': timestamp(), 'development_lock_sha256': hashlib.sha256(
            (OUT / 'development_lock.json').read_bytes()).hexdigest(),
        'test_evaluation_count': 1, 'no_test_based_changes': True,
        'prediction_sha256': {n: hashlib.sha256((PRED / f'test_{n}.csv').read_bytes()).hexdigest() for n in GRIDS},
    })


def verify():
    lock = check_lock()
    _, dev, splits = load_data()
    for name in GRIDS:
        oof = pd.read_csv(PRED / f'oof_{name}.csv')
        test = pd.read_csv(PRED / f'test_{name}.csv')
        validate_and_align_predictions(oof, splits, 'development')
        validate_and_align_predictions(test, splits, 'test')
        threshold, _ = select_threshold_target_sensitivity(oof.y_true, oof.y_prob, 0.95)
        if not np.isclose(threshold, lock['models'][name]['threshold'], rtol=0, atol=1e-14):
            raise AssertionError('OOF threshold roundtrip failed.')
        if len(oof) != len(dev) or len(test) != 114:
            raise AssertionError('Prediction coverage failed.')
        if not oof.id.is_unique or not test.id.is_unique or set(oof.id) & set(test.id):
            raise AssertionError('Prediction overlap/duplicate detected.')
        if not (MODELS / f'{name}.joblib').exists():
            raise AssertionError('Saved final model missing.')
        complete = json.loads((OUT / 'test_complete.json').read_text(encoding='utf8'))
        if hashlib.sha256((PRED / f'test_{name}.csv').read_bytes()).hexdigest() != complete['prediction_sha256'][name]:
            raise AssertionError('Test predictions changed after finalization.')
    save_json(OUT / 'verification.json', {'created_utc': timestamp(), 'passed': True,
               'checks': ['input and OOF hashes', '455 OOF and 114 test IDs per model',
                          'official labels and folds', 'no ID overlap', 'finite [0,1] probabilities',
                          'threshold CSV roundtrip', 'test output hashes', 'saved model presence'],
               'note': 'No models refitted and no new test predictions generated during verification.'})
    print('All SVM deliverable checks PASSED.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['develop', 'finalize', 'verify'])
    args = parser.parse_args()
    {'develop': develop, 'finalize': finalize, 'verify': verify}[args.stage]()
