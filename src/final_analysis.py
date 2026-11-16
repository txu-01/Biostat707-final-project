"""Unified analyses requested in the preliminary BIOSTAT 707 feedback.

The primary analysis is repeated nested cross-validation on all 569 records.
The original 80/20 split remains a secondary, one-shot sensitivity analysis.

This module deliberately centralizes the validation protocol while keeping the
prespecified model families and tuning grids from the proposal.  It also
provides two secondary analyses requested by the instructor:

* learning curves at training sizes 50, 100, 200, and 455; and
* a mean/SE/worst/all feature-set comparison using Elastic Net.

Run from the repository root.  Outputs are checkpointed by repeat so an
interrupted long run can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_prep import FEATURE_NAMES
from metrics import compute_calibration_curve, select_threshold_target_sensitivity


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "wdbc_clean.csv"
DEFAULT_OUT = ROOT / "results" / "final_nested"
SEED = 707

MODEL_GRIDS = {
    "Logistic_Regression": [{}],
    "Ridge_Logistic": [{}],
    "Elastic_Net": list(
        ParameterGrid(
            {"C": [0.01, 0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.5, 0.9, 1.0]}
        )
    ),
    "Random_Forest": list(
        ParameterGrid(
            {
                "max_depth": [3, None],
                "min_samples_leaf": [1, 3, 5, 10],
                "max_features": ["sqrt", 0.5],
            }
        )
    ),
    "Decision_Tree": list(
        ParameterGrid(
            {"max_depth": [3, 5, 7, None], "min_samples_leaf": [5, 10, 20]}
        )
    ),
    "Linear_SVM": list(ParameterGrid({"C": [0.01, 0.1, 1.0, 10.0]})),
    "RBF_SVM": list(
        ParameterGrid(
            {"C": [0.01, 0.1, 1.0, 10.0], "gamma": [0.001, 0.01, 0.1, 1.0]}
        )
    ),
}

PRIMARY_MODELS = [
    "Constant_Baseline",
    "Logistic_Regression",
    "Ridge_Logistic",
    "Elastic_Net",
    "Random_Forest",
    "Decision_Tree",
    "Linear_SVM",
    "RBF_SVM",
]

FEATURE_GROUPS = {
    "mean": [name for name in FEATURE_NAMES if name.endswith("_mean")],
    "se": [name for name in FEATURE_NAMES if name.endswith("_se")],
    "worst": [name for name in FEATURE_NAMES if name.endswith("_worst")],
    "all": list(FEATURE_NAMES),
}


def params_json(params: dict) -> str:
    """Stable serialization for result tables and deterministic tie handling."""

    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH).sort_values("id").reset_index(drop=True)
    predictors = [c for c in data.columns if c not in {"id", "diagnosis", "target"}]
    if len(data) != 569 or data["id"].duplicated().any():
        raise ValueError("Expected 569 records with unique IDs.")
    if int(data["target"].sum()) != 212 or predictors != list(FEATURE_NAMES):
        raise ValueError("Outcome counts or predictor order differs from the locked data.")
    if not np.isfinite(data[FEATURE_NAMES].to_numpy()).all():
        raise ValueError("Predictors contain missing or nonfinite values.")
    return data


def make_model(model_name: str, params: dict, random_state: int):
    """Construct a prespecified model with fold-local preprocessing."""

    if model_name == "Logistic_Regression":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty=None,
                        solver="lbfgs",
                        max_iter=20_000,
                        tol=1e-7,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if model_name == "Ridge_Logistic":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        l1_ratio=0.0,
                        solver="lbfgs",
                        max_iter=20_000,
                        tol=1e-7,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if model_name == "Elastic_Net":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=params["C"],
                        l1_ratio=params["l1_ratio"],
                        solver="saga",
                        max_iter=20_000,
                        tol=1e-4,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if model_name == "Random_Forest":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            class_weight=None,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "Decision_Tree":
        return DecisionTreeClassifier(
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight=None,
            random_state=random_state,
        )
    if model_name in {"Linear_SVM", "RBF_SVM"}:
        base = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="linear" if model_name == "Linear_SVM" else "rbf",
                        C=params["C"],
                        gamma=params.get("gamma", "scale"),
                        cache_size=512,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        return CalibratedClassifierCV(
            estimator=base,
            method="sigmoid",
            cv=StratifiedKFold(3, shuffle=True, random_state=random_state),
            ensemble=False,
            n_jobs=1,
        )
    raise KeyError(f"Unknown model: {model_name}")


def fit_probability(model, x_train, y_train, x_eval):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train)
    convergence_warnings = sum(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )
    probability = model.predict_proba(x_eval)[:, 1]
    if not np.isfinite(probability).all() or not ((0 <= probability) & (probability <= 1)).all():
        raise ValueError("Model returned invalid probabilities.")
    return probability, convergence_warnings


def tune_inside_outer_training(
    x: pd.DataFrame,
    y: np.ndarray,
    model_name: str,
    inner_splits: int,
    random_state: int,
):
    """Tune entirely within an outer-training set and return OOF probabilities."""

    cv = list(
        StratifiedKFold(
            n_splits=inner_splits, shuffle=True, random_state=random_state
        ).split(x, y)
    )
    best = None
    search_rows = []
    for candidate, params in enumerate(MODEL_GRIDS[model_name]):
        oof = np.full(len(y), np.nan)
        fold_scores = []
        convergence_warnings = 0
        started = time.perf_counter()
        for fold, (train_idx, validation_idx) in enumerate(cv):
            model = make_model(model_name, params, random_state + 100 * candidate + fold)
            probability, n_warnings = fit_probability(
                model,
                x.iloc[train_idx],
                y[train_idx],
                x.iloc[validation_idx],
            )
            oof[validation_idx] = probability
            convergence_warnings += n_warnings
            fold_scores.append(roc_auc_score(y[validation_idx], probability))
        if not np.isfinite(oof).all():
            raise AssertionError("Inner OOF predictions are incomplete.")
        mean_score = float(np.mean(fold_scores))
        pooled_score = float(roc_auc_score(y, oof))
        search_rows.append(
            {
                "candidate": candidate,
                "params_json": params_json(params),
                "mean_inner_auroc": mean_score,
                "pooled_inner_auroc": pooled_score,
                "inner_auroc_sd": float(np.std(fold_scores, ddof=1)),
                "convergence_warnings": convergence_warnings,
                "seconds": time.perf_counter() - started,
            }
        )
        tie_key = params_json(params)
        if best is None or (mean_score, -candidate, tie_key) > (
            best[0],
            -best[1],
            best[2],
        ):
            best = (mean_score, candidate, tie_key, params, oof, convergence_warnings)
    _, candidate, _, params, oof, convergence_warnings = best
    threshold, sensitivity = select_threshold_target_sensitivity(y, oof, 0.95)
    if sensitivity < 0.95 - 1e-12:
        raise AssertionError("Training-only threshold failed the sensitivity target.")
    return params, oof, threshold, sensitivity, candidate, convergence_warnings, search_rows


def summarize_prediction_rows(rows: pd.DataFrame) -> dict:
    """Compute repeat-level metrics using pooled outer predictions."""

    y = rows["y_true"].to_numpy(dtype=int)
    probability = rows["y_prob"].to_numpy(dtype=float)
    threshold = rows["threshold"].to_numpy(dtype=float)
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    calibration = compute_calibration_curve(y, probability, n_bins=5)
    return {
        "N": len(rows),
        "auroc": roc_auc_score(y, probability),
        "auprc": average_precision_score(y, probability),
        "brier_score": brier_score_loss(y, probability),
        "log_loss": log_loss(y, np.clip(probability, 1e-15, 1 - 1e-15)),
        "ece": calibration["ece"],
        "mean_predicted": float(probability.mean()),
        "observed_prevalence": float(y.mean()),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "ppv": tp / (tp + fp) if tp + fp else math.nan,
        "npv": tn / (tn + fn) if tn + fn else math.nan,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }


def _outer_split(data: pd.DataFrame, repeat: int, folds: int):
    return StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=SEED + repeat
    ).split(data[FEATURE_NAMES], data["target"])


def _write_checkpoint(frame: pd.DataFrame, path: Path, keys: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = frame.drop_duplicates(keys, keep="last").sort_values(keys)
    frame.to_csv(path, index=False)


def run_primary_nested(
    data: pd.DataFrame,
    output: Path,
    models: list[str],
    repeats: int = 10,
    outer_folds: int = 5,
    inner_folds: int = 5,
    resume: bool = True,
):
    """Run repeated nested CV and checkpoint after every completed repeat."""

    output.mkdir(parents=True, exist_ok=True)
    prediction_dir = output / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    x_all = data[FEATURE_NAMES]
    y_all = data["target"].to_numpy(dtype=int)

    for model_name in models:
        if model_name in {"Linear_SVM", "RBF_SVM"}:
            continue
        pred_path = prediction_dir / f"{model_name}.csv"
        search_path = output / f"inner_search_{model_name}.csv"
        fold_path = output / f"outer_fold_metrics_{model_name}.csv"
        predictions = pd.read_csv(pred_path) if resume and pred_path.exists() else pd.DataFrame()
        searches = pd.read_csv(search_path) if resume and search_path.exists() else pd.DataFrame()
        fold_metrics = pd.read_csv(fold_path) if resume and fold_path.exists() else pd.DataFrame()
        completed = set()
        if not predictions.empty:
            counts = predictions.groupby("repeat")["id"].nunique()
            completed = set(counts[counts.eq(len(data))].index.astype(int))

        for repeat in range(repeats):
            repeat_number = repeat + 1
            if repeat_number in completed:
                print(f"{model_name}: repeat {repeat_number}/{repeats} already complete", flush=True)
                continue
            repeat_predictions = []
            repeat_searches = []
            repeat_folds = []
            for fold, (train_idx, validation_idx) in enumerate(
                _outer_split(data, repeat, outer_folds)
            ):
                x_train = x_all.iloc[train_idx]
                y_train = y_all[train_idx]
                x_validation = x_all.iloc[validation_idx]
                y_validation = y_all[validation_idx]
                model_seed = SEED + 10_000 * repeat + 100 * fold
                started = time.perf_counter()

                if model_name == "Constant_Baseline":
                    probability = np.full(len(validation_idx), y_train.mean())
                    threshold = float(y_train.mean())
                    inner_sensitivity = 1.0
                    params = {}
                    selected_candidate = 0
                    convergence_warnings = 0
                    search_rows = []
                else:
                    (
                        params,
                        _,
                        threshold,
                        inner_sensitivity,
                        selected_candidate,
                        convergence_warnings,
                        search_rows,
                    ) = tune_inside_outer_training(
                        x_train,
                        y_train,
                        model_name,
                        inner_folds,
                        1707 + 100 * repeat + fold,
                    )
                    model = make_model(model_name, params, model_seed)
                    probability, final_warnings = fit_probability(
                        model, x_train, y_train, x_validation
                    )
                    convergence_warnings += final_warnings
                    for row in search_rows:
                        row.update(
                            {
                                "repeat": repeat_number,
                                "fold": fold,
                                "model": model_name,
                            }
                        )
                    repeat_searches.extend(search_rows)

                for row_index, probability_value in zip(validation_idx, probability):
                    repeat_predictions.append(
                        {
                            "repeat": repeat_number,
                            "fold": fold,
                            "model": model_name,
                            "id": int(data.iloc[row_index]["id"]),
                            "y_true": int(y_all[row_index]),
                            "y_prob": float(probability_value),
                            "threshold": float(threshold),
                        }
                    )
                fold_summary = summarize_prediction_rows(pd.DataFrame(repeat_predictions).query("fold == @fold"))
                repeat_folds.append(
                    {
                        "repeat": repeat_number,
                        "fold": fold,
                        "model": model_name,
                        "n_train": len(train_idx),
                        "n_validation": len(validation_idx),
                        "params_json": params_json(params),
                        "selected_candidate": selected_candidate,
                        "inner_sensitivity": inner_sensitivity,
                        "threshold": threshold,
                        "convergence_warnings": convergence_warnings,
                        "seconds": time.perf_counter() - started,
                        **fold_summary,
                    }
                )
                print(
                    f"{model_name}: repeat {repeat_number}/{repeats}, "
                    f"outer fold {fold + 1}/{outer_folds}",
                    flush=True,
                )

            repeat_frame = pd.DataFrame(repeat_predictions)
            if len(repeat_frame) != len(data) or repeat_frame["id"].nunique() != len(data):
                raise AssertionError("A completed repeat must predict each record exactly once.")
            predictions = pd.concat([predictions, repeat_frame], ignore_index=True)
            searches = pd.concat(
                [searches, pd.DataFrame(repeat_searches)], ignore_index=True
            )
            fold_metrics = pd.concat(
                [fold_metrics, pd.DataFrame(repeat_folds)], ignore_index=True
            )
            _write_checkpoint(predictions, pred_path, ["repeat", "id"])
            if not searches.empty:
                _write_checkpoint(
                    searches,
                    search_path,
                    ["repeat", "fold", "candidate"],
                )
            _write_checkpoint(
                fold_metrics, fold_path, ["repeat", "fold"]
            )


def import_existing_svm_predictions(output: Path, repeats: int, outer_folds: int):
    """Reuse the already completed SVM nested run after verifying its splits."""

    source = ROOT / "results" / "svm" / "nested" / "outer_predictions.csv"
    if not source.exists() or repeats != 10 or outer_folds != 5:
        return
    rows = pd.read_csv(source)
    required = {"repeat", "fold", "model", "id", "y_true", "y_prob", "threshold"}
    if not required.issubset(rows.columns):
        raise ValueError("Existing SVM nested predictions have an unexpected schema.")
    data = load_data()
    expected = []
    for repeat in range(repeats):
        for fold, (_, validation_idx) in enumerate(_outer_split(data, repeat, outer_folds)):
            expected.extend(
                {
                    "repeat": repeat + 1,
                    "fold": fold,
                    "id": int(data.iloc[index]["id"]),
                }
                for index in validation_idx
            )
    expected = pd.DataFrame(expected).sort_values(["repeat", "fold", "id"]).reset_index(drop=True)
    actual = (
        rows.loc[rows.model.eq("Linear_SVM"), ["repeat", "fold", "id"]]
        .sort_values(["repeat", "fold", "id"])
        .reset_index(drop=True)
    )
    if not expected.equals(actual):
        raise ValueError("Existing SVM results do not use the required common outer splits.")
    prediction_dir = output / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for model_name in ["Linear_SVM", "RBF_SVM"]:
        model_rows = rows.loc[rows.model.eq(model_name), sorted(required)].copy()
        _write_checkpoint(model_rows, prediction_dir / f"{model_name}.csv", ["repeat", "id"])

    source_metrics = ROOT / "results" / "svm" / "nested" / "outer_metrics.csv"
    if source_metrics.exists():
        svm_metrics = pd.read_csv(source_metrics)
        for model_name in ["Linear_SVM", "RBF_SVM"]:
            model_rows = svm_metrics.loc[svm_metrics.model.eq(model_name)].copy()
            parameter_columns = [c for c in ["C", "gamma"] if c in model_rows]
            model_rows["params_json"] = model_rows[parameter_columns].apply(
                lambda row: params_json(
                    {
                        key: value
                        for key, value in row.items()
                        if pd.notna(value)
                    }
                ),
                axis=1,
            )
            _write_checkpoint(
                model_rows,
                output / f"outer_fold_metrics_{model_name}.csv",
                ["repeat", "fold"],
            )


def summarize_primary(output: Path):
    prediction_dir = output / "predictions"
    frames = []
    for model_name in PRIMARY_MODELS:
        path = prediction_dir / f"{model_name}.csv"
        if path.exists():
            frame = pd.read_csv(path)
            if "model" not in frame:
                frame["model"] = model_name
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("No repeated nested-CV prediction files were found.")
    predictions = pd.concat(frames, ignore_index=True)
    truth = load_data().set_index("id")["target"]
    if predictions.duplicated(["model", "repeat", "id"]).any():
        raise ValueError("Duplicate model/repeat/ID predictions were found.")
    expected_truth = predictions["id"].map(truth)
    if expected_truth.isna().any() or not np.array_equal(
        predictions["y_true"].to_numpy(dtype=int),
        expected_truth.to_numpy(dtype=int),
    ):
        raise ValueError("Outer predictions contain an unknown ID or incorrect label.")
    for column in ["y_prob", "threshold"]:
        values = predictions[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not ((0 <= values) & (values <= 1)).all():
            raise ValueError(f"{column} must be finite and within [0, 1].")
    predictions.to_csv(output / "outer_predictions_all_models.csv", index=False)

    repeat_rows = []
    for (repeat, model_name), group in predictions.groupby(["repeat", "model"]):
        if len(group) != 569 or group["id"].nunique() != 569:
            raise ValueError(f"Incomplete outer predictions for {model_name}, repeat {repeat}.")
        repeat_rows.append(
            {"repeat": int(repeat), "model": model_name, **summarize_prediction_rows(group)}
        )
    repeat_metrics = pd.DataFrame(repeat_rows).sort_values(["model", "repeat"])
    repeat_metrics.to_csv(output / "repeat_metrics.csv", index=False)

    metric_columns = [
        "auroc",
        "auprc",
        "brier_score",
        "log_loss",
        "ece",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "FN",
        "FP",
    ]
    summary = (
        repeat_metrics.groupby("model")[metric_columns]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        col if isinstance(col, str) else "_".join(part for part in col if part)
        for col in summary.columns
    ]
    summary.to_csv(output / "model_summary.csv", index=False)

    comparator_name = (
        "Ridge_Logistic"
        if repeat_metrics.model.eq("Ridge_Logistic").any()
        else "Logistic_Regression"
    )
    comparator = repeat_metrics.loc[
        repeat_metrics.model.eq(comparator_name), ["repeat", *metric_columns]
    ].set_index("repeat")
    difference_rows = []
    if not comparator.empty:
        for model_name, group in repeat_metrics.groupby("model"):
            if model_name == comparator_name:
                continue
            indexed = group.set_index("repeat")
            common = comparator.index.intersection(indexed.index)
            for repeat in common:
                row = {"repeat": repeat, "model": model_name, "comparator": comparator_name}
                row.update(
                    {
                        f"{metric}_difference": indexed.loc[repeat, metric]
                        - comparator.loc[repeat, metric]
                        for metric in metric_columns
                    }
                )
                difference_rows.append(row)
    differences = pd.DataFrame(difference_rows)
    differences.to_csv(output / "paired_repeat_differences_vs_stable_logistic.csv", index=False)
    if not differences.empty:
        value_columns = [c for c in differences if c.endswith("_difference")]
        difference_summary = (
            differences.groupby(["model", "comparator"])[value_columns]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
        )
        difference_summary.columns = [
            col if isinstance(col, str) else "_".join(part for part in col if part)
            for col in difference_summary.columns
        ]
        difference_summary.to_csv(
            output / "paired_difference_summary_vs_stable_logistic.csv", index=False
        )
    return repeat_metrics, summary


def consensus_parameters(output: Path, model_name: str) -> dict:
    if model_name == "Constant_Baseline":
        return {}
    path = output / f"outer_fold_metrics_{model_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing nested-CV fold results for {model_name}.")
    rows = pd.read_csv(path)
    if "params_json" not in rows:
        raise ValueError(f"No parameter record for {model_name}.")
    values = [value for value in rows["params_json"].dropna().astype(str) if value]
    if not values:
        return {}
    return json.loads(Counter(values).most_common(1)[0][0])


def fixed_parameter_oof_threshold(
    x: pd.DataFrame,
    y: np.ndarray,
    model_name: str,
    params: dict,
    random_state: int,
    folds: int = 5,
):
    if model_name == "Constant_Baseline":
        probability = np.full(len(y), y.mean())
    else:
        probability = np.full(len(y), np.nan)
        cv = StratifiedKFold(folds, shuffle=True, random_state=random_state)
        for fold, (train_idx, validation_idx) in enumerate(cv.split(x, y)):
            model = make_model(model_name, params, random_state + fold)
            probability[validation_idx], _ = fit_probability(
                model,
                x.iloc[train_idx],
                y[train_idx],
                x.iloc[validation_idx],
            )
    threshold, sensitivity = select_threshold_target_sensitivity(y, probability, 0.95)
    return threshold, sensitivity


def run_learning_curves(data: pd.DataFrame, output: Path, models: list[str]):
    """Learning curves on common outer folds without repeatedly opening the old test set."""

    rows = []
    y_all = data["target"].to_numpy(dtype=int)
    sizes = [50, 100, 200, 455]
    for model_name in models:
        params = consensus_parameters(output, model_name)
        for fold, (outer_train, outer_validation) in enumerate(_outer_split(data, 0, 5)):
            for size in sizes:
                if size > len(outer_train):
                    continue
                if size == len(outer_train):
                    selected = outer_train
                elif len(outer_train) - size < 2:
                    # One outer fold contains 456 rather than 455 training
                    # records.  A stratified one-record complement is
                    # impossible, so deterministically drop one record while
                    # retaining both outcome classes in the 455-record sample.
                    rng = np.random.RandomState(SEED + 100 * fold + size)
                    drop_position = rng.randint(len(outer_train))
                    selected = np.delete(outer_train, drop_position)
                else:
                    selected, _ = train_test_split(
                        outer_train,
                        train_size=size,
                        stratify=y_all[outer_train],
                        random_state=SEED + 100 * fold + size,
                    )
                x_train = data.iloc[selected][FEATURE_NAMES]
                y_train = y_all[selected]
                x_validation = data.iloc[outer_validation][FEATURE_NAMES]
                threshold, inner_sensitivity = fixed_parameter_oof_threshold(
                    x_train,
                    y_train,
                    model_name,
                    params,
                    2707 + 100 * fold + size,
                )
                if model_name == "Constant_Baseline":
                    probability = np.full(len(outer_validation), y_train.mean())
                else:
                    model = make_model(model_name, params, 3707 + 100 * fold + size)
                    probability, _ = fit_probability(
                        model, x_train, y_train, x_validation
                    )
                frame = pd.DataFrame(
                    {
                        "y_true": y_all[outer_validation],
                        "y_prob": probability,
                        "threshold": threshold,
                    }
                )
                rows.append(
                    {
                        "model": model_name,
                        "fold": fold,
                        "train_size": size,
                        "params_json": params_json(params),
                        "inner_sensitivity": inner_sensitivity,
                        **summarize_prediction_rows(frame),
                    }
                )
                print(
                    f"learning curve: {model_name}, size {size}, fold {fold + 1}/5",
                    flush=True,
                )
    result = pd.DataFrame(rows)
    result.to_csv(output / "learning_curve_fold_metrics.csv", index=False)
    metrics = ["auroc", "brier_score", "sensitivity", "specificity", "FN", "FP"]
    summary = result.groupby(["model", "train_size"])[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = [
        col if isinstance(col, str) else "_".join(part for part in col if part)
        for col in summary.columns
    ]
    summary.to_csv(output / "learning_curve_summary.csv", index=False)
    return result, summary


def run_feature_set_comparison(
    data: pd.DataFrame,
    output: Path,
    repeats: int = 10,
    outer_folds: int = 5,
    inner_folds: int = 5,
):
    """Isolate feature representation using the same tuned Elastic Net model."""

    rows = []
    selected_rows = []
    y_all = data["target"].to_numpy(dtype=int)
    for feature_set, features in FEATURE_GROUPS.items():
        for repeat in range(repeats):
            repeat_rows = []
            for fold, (train_idx, validation_idx) in enumerate(
                _outer_split(data, repeat, outer_folds)
            ):
                x_train = data.iloc[train_idx][features]
                y_train = y_all[train_idx]
                (
                    params,
                    _,
                    threshold,
                    inner_sensitivity,
                    candidate,
                    convergence_warnings,
                    _,
                ) = tune_inside_outer_training(
                    x_train,
                    y_train,
                    "Elastic_Net",
                    inner_folds,
                    4707 + 100 * repeat + fold,
                )
                model = make_model("Elastic_Net", params, 5707 + 100 * repeat + fold)
                probability, final_warnings = fit_probability(
                    model,
                    x_train,
                    y_train,
                    data.iloc[validation_idx][features],
                )
                selected_rows.append(
                    {
                        "feature_set": feature_set,
                        "repeat": repeat + 1,
                        "fold": fold,
                        "params_json": params_json(params),
                        "selected_candidate": candidate,
                        "threshold": threshold,
                        "inner_sensitivity": inner_sensitivity,
                        "convergence_warnings": convergence_warnings + final_warnings,
                    }
                )
                repeat_rows.extend(
                    {
                        "feature_set": feature_set,
                        "repeat": repeat + 1,
                        "fold": fold,
                        "id": int(data.iloc[index]["id"]),
                        "y_true": int(y_all[index]),
                        "y_prob": float(probability_value),
                        "threshold": float(threshold),
                    }
                    for index, probability_value in zip(validation_idx, probability)
                )
            frame = pd.DataFrame(repeat_rows)
            rows.extend(repeat_rows)
            print(
                f"feature sets: {feature_set}, repeat {repeat + 1}/{repeats}, "
                f"AUROC={roc_auc_score(frame.y_true, frame.y_prob):.4f}",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(output / "feature_set_outer_predictions.csv", index=False)
            pd.DataFrame(selected_rows).to_csv(output / "feature_set_selected_models.csv", index=False)

    predictions = pd.DataFrame(rows)
    repeat_metrics = []
    for (feature_set, repeat), group in predictions.groupby(["feature_set", "repeat"]):
        repeat_metrics.append(
            {
                "feature_set": feature_set,
                "repeat": repeat,
                **summarize_prediction_rows(group),
            }
        )
    repeat_metrics = pd.DataFrame(repeat_metrics)
    repeat_metrics.to_csv(output / "feature_set_repeat_metrics.csv", index=False)
    metrics = ["auroc", "auprc", "brier_score", "log_loss", "ece", "sensitivity", "specificity"]
    summary = repeat_metrics.groupby("feature_set")[metrics].agg(["mean", "std", "min", "max"]).reset_index()
    summary.columns = [
        col if isinstance(col, str) else "_".join(part for part in col if part)
        for col in summary.columns
    ]
    summary.to_csv(output / "feature_set_summary.csv", index=False)
    return repeat_metrics, summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["nested", "summarize", "learning-curves", "feature-sets", "all"],
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-import-svm", action="store_true")
    parser.add_argument("--models", nargs="+", choices=PRIMARY_MODELS)
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_data()
    args.output.mkdir(parents=True, exist_ok=True)
    models = args.models or PRIMARY_MODELS
    if args.stage in {"nested", "all"}:
        run_primary_nested(
            data,
            args.output,
            models,
            args.repeats,
            args.outer_folds,
            args.inner_folds,
            resume=not args.no_resume,
        )
        if not args.no_import_svm:
            import_existing_svm_predictions(args.output, args.repeats, args.outer_folds)
        summarize_primary(args.output)
    if args.stage == "summarize":
        if not args.no_import_svm:
            import_existing_svm_predictions(args.output, args.repeats, args.outer_folds)
        summarize_primary(args.output)
    if args.stage in {"learning-curves", "all"}:
        run_learning_curves(data, args.output, models)
    if args.stage in {"feature-sets", "all"}:
        run_feature_set_comparison(
            data,
            args.output,
            args.repeats,
            args.outer_folds,
            args.inner_folds,
        )


if __name__ == "__main__":
    main()
