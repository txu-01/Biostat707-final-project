
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_prep import FEATURE_NAMES
from metrics import (compute_calibration_curve, evaluate_predictions,
                     select_threshold_target_sensitivity)

OUTPUT = Path(__file__).resolve().parent
RESULTS = OUTPUT / "results"
FIGURES = OUTPUT / "figures"
SEED = 707


def metric_row(labels, probabilities, predicted):
    """Evaluate probabilities and predictions made with fold-specific thresholds."""
    scores = evaluate_predictions(labels, probabilities, fixed_threshold=0.5)
    counts = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    tn, fp, fn, tp = (int(value) for value in counts)
    ratio = lambda numerator, denominator: numerator / denominator if denominator else 0.0
    return {key: scores[key] for key in ("N", "prevalence", "mean_predicted", "auroc",
                                         "auprc", "brier_score", "log_loss", "ece")} | {
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "sensitivity": ratio(tp, tp + fn), "specificity": ratio(tn, tn + fp),
        "ppv": ratio(tp, tp + fp), "npv": ratio(tn, tn + fn),
        "missed_malignant_cases": fn, "unnecessary_escalations": fp,
    }


def make_figures(predictions, example_tree):
    sample = predictions.loc[predictions["repeat"] == 1]
    labels, probabilities = sample["target"], sample["probability"]
    fpr, tpr, _ = roc_curve(labels, probabilities)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUROC = {roc_auc_score(labels, probabilities):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="False positive rate", ylabel="Sensitivity", title="Outer-CV ROC — repeat 1")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "roc_repeat_1.png", dpi=200); plt.close(fig)

    matrix = confusion_matrix(labels, sample["prediction"], labels=[0, 1])
    fig, ax = plt.subplots()
    image = ax.imshow(matrix, cmap="Blues")
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(value), ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Benign", "Malignant"],
           yticklabels=["Benign", "Malignant"], xlabel="Predicted", ylabel="Actual",
           title="Outer-CV confusion matrix — repeat 1")
    fig.colorbar(image, ax=ax); fig.tight_layout()
    fig.savefig(FIGURES / "confusion_matrix_repeat_1.png", dpi=200); plt.close(fig)

    calibration = compute_calibration_curve(labels, probabilities, n_bins=5)
    fig, ax = plt.subplots()
    ax.plot(calibration["bin_pred_mean"], calibration["bin_true_prop"], "o-")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability",
           ylabel="Observed fraction malignant", title="Outer-CV calibration — repeat 1")
    fig.tight_layout(); fig.savefig(FIGURES / "calibration_repeat_1.png", dpi=200); plt.close(fig)
    pd.DataFrame({"mean_predicted": calibration["bin_pred_mean"],
                  "observed_fraction": calibration["bin_true_prop"],
                  "count": calibration["bin_counts"]}).to_csv(RESULTS / "calibration_repeat_1.csv", index=False)

    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(example_tree, feature_names=FEATURE_NAMES, class_names=["Benign", "Malignant"],
              filled=True, rounded=True, fontsize=6, ax=ax)
    ax.set_title("Example tree: repeat 1, outer fold 1 training data")
    fig.tight_layout(); fig.savefig(FIGURES / "tree_example.png", dpi=200); plt.close(fig)

    importance = pd.Series(example_tree.feature_importances_, index=FEATURE_NAMES).sort_values()
    fig, ax = plt.subplots(figsize=(8, 10))
    importance.plot.barh(ax=ax)
    ax.set(xlabel="Impurity-based importance", title="Example tree: repeat 1, outer fold 1")
    fig.tight_layout(); fig.savefig(FIGURES / "feature_importance_example.png", dpi=200); plt.close(fig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / "data" / "processed" / "wdbc_clean.csv")
    if (len(data) != 569 or data["id"].duplicated().any()
            or data["target"].value_counts().to_dict() != {0: 357, 1: 212}
            or not np.isfinite(data[FEATURE_NAMES].to_numpy(dtype=float)).all()):
        raise ValueError("Unexpected cohort, labels, IDs, or nonfinite predictors.")
    x, y = data[FEATURE_NAMES], data["target"].to_numpy()
    prediction_frames, fold_rows, searches, importances, comparisons = [], [], [], [], []
    example_tree = None

    for repeat_index in range(10):
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + repeat_index)
        for fold_index, (train_idx, validation_idx) in enumerate(outer.split(x, y)):
            repeat, fold = repeat_index + 1, fold_index + 1
            seed = SEED + 10_000 * repeat_index + 100 * fold_index
            x_train, y_train = x.iloc[train_idx], y[train_idx]
            inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            tree = DecisionTreeClassifier(random_state=seed)
            search = GridSearchCV(tree, {"max_depth": [3, 5, 7, None],
                                        "min_samples_leaf": [5, 10, 20]},
                                  scoring="roc_auc", cv=inner, refit=True, n_jobs=1,
                                  return_train_score=True, error_score="raise")
            start = time.perf_counter()
            search.fit(x_train, y_train)
            # The selected settings produce OOF probabilities only within outer training.
            train_oof = cross_val_predict(clone(search.best_estimator_), x_train, y_train,
                                          cv=inner, method="predict_proba", n_jobs=1)[:, 1]
            threshold, training_sensitivity = select_threshold_target_sensitivity(y_train, train_oof)
            probabilities = search.best_estimator_.predict_proba(x.iloc[validation_idx])[:, 1]
            predicted = (probabilities >= threshold).astype(int)
            elapsed = time.perf_counter() - start
            training_auroc = roc_auc_score(
                y_train, search.best_estimator_.predict_proba(x_train)[:, 1]
            )
            outer_auroc = roc_auc_score(y[validation_idx], probabilities)
            # Auxiliary comparison only: scaler and ridge fit see outer training data.
            reference = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=20000))
            reference.fit(x_train, y_train)
            coefficients = reference.named_steps["logisticregression"].coef_[0]
            tree_importance = search.best_estimator_.feature_importances_
            comparisons.append(pd.DataFrame({
                "repeat": repeat, "fold": fold, "feature": FEATURE_NAMES,
                "tree_impurity_importance": tree_importance,
                "ridge_standardized_coefficient": coefficients,
                "ridge_absolute_coefficient": np.abs(coefficients),
                "tree_rank": pd.Series(tree_importance).rank(ascending=False, method="min").to_numpy(),
                "ridge_absolute_rank": pd.Series(np.abs(coefficients)).rank(ascending=False, method="min").to_numpy(),
            }))
            frame = pd.DataFrame({"id": data.iloc[validation_idx]["id"].to_numpy(),
                                  "target": y[validation_idx], "model": "Decision_Tree",
                                  "repeat": repeat, "fold": fold, "probability": probabilities,
                                  "threshold": threshold, "prediction": predicted})
            prediction_frames.append(frame)
            fold_rows.append({"repeat": repeat, "fold": fold, "threshold": threshold,
                              "training_oof_sensitivity": training_sensitivity,
                              "max_depth": search.best_params_["max_depth"],
                              "min_samples_leaf": search.best_params_["min_samples_leaf"],
                              "inner_auroc": search.best_score_,
                              "training_fit_auroc": training_auroc,
                              "training_minus_inner_auroc": training_auroc - search.best_score_,
                              "training_minus_outer_auroc": training_auroc - outer_auroc,
                              "inner_minus_outer_auroc": search.best_score_ - outer_auroc,
                              "seconds": elapsed, **metric_row(y[validation_idx], probabilities, predicted)})
            candidates = pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score", "mean_train_score", "rank_test_score"]].copy()
            candidates.insert(0, "fold", fold); candidates.insert(0, "repeat", repeat)
            searches.append(candidates)
            importances.append(pd.DataFrame({"repeat": repeat, "fold": fold, "feature": FEATURE_NAMES,
                                             "importance": search.best_estimator_.feature_importances_}))
            if example_tree is None:
                example_tree = search.best_estimator_

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(RESULTS / "outer_predictions.csv", index=False)
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(RESULTS / "outer_fold_metrics.csv", index=False)
    overfitting_columns = ["repeat", "fold", "training_fit_auroc", "inner_auroc", "auroc",
                           "training_minus_inner_auroc", "training_minus_outer_auroc",
                           "inner_minus_outer_auroc"]
    overfitting = fold_metrics[overfitting_columns].rename(columns={"auroc": "outer_auroc"})
    overfitting.to_csv(RESULTS / "overfitting_by_fold.csv", index=False)
    overfitting.drop(columns=["repeat", "fold"]).agg(["mean", "std", "min", "max"]).T.to_csv(
        RESULTS / "overfitting_summary.csv", index_label="metric")
    pd.concat(searches, ignore_index=True).to_csv(RESULTS / "inner_search.csv", index=False)
    pd.concat(importances, ignore_index=True).to_csv(RESULTS / "feature_importances.csv", index=False)
    feature_comparison = pd.concat(comparisons, ignore_index=True)
    feature_comparison.to_csv(RESULTS / "feature_comparison_by_fold.csv", index=False)
    feature_comparison.groupby("feature", sort=False).agg(
        tree_importance_mean=("tree_impurity_importance", "mean"),
        tree_importance_std=("tree_impurity_importance", "std"),
        tree_nonzero_fraction=("tree_impurity_importance", lambda values: (values > 0).mean()),
        ridge_coefficient_mean=("ridge_standardized_coefficient", "mean"),
        ridge_coefficient_std=("ridge_standardized_coefficient", "std"),
        ridge_absolute_coefficient_mean=("ridge_absolute_coefficient", "mean"),
        tree_rank_mean=("tree_rank", "mean"),
        ridge_absolute_rank_mean=("ridge_absolute_rank", "mean"),
    ).to_csv(RESULTS / "feature_comparison_summary.csv")
    repeats = pd.DataFrame([{"repeat": repeat, **metric_row(group["target"], group["probability"], group["prediction"])}
                            for repeat, group in predictions.groupby("repeat", sort=True)])
    repeats.to_csv(RESULTS / "repeat_metrics.csv", index=False)
    repeats.drop(columns="repeat").agg(["mean", "std", "min", "max"]).T.to_csv(RESULTS / "repeat_summary.csv", index_label="metric")
    make_figures(predictions, example_tree)
    print(f"Finished. Results: {RESULTS}; figures: {FIGURES}")


if __name__ == "__main__":
    main()
