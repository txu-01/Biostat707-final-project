"""Create the logistic figures and calibration bins from the team's final results."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from metrics import compute_calibration_curve


MODELS = (
    ("Logistic_Regression", "Unpenalized", "#64748b"),
    ("Ridge_Logistic", "Ridge", "#176c66"),
    ("Elastic_Net", "Elastic Net", "#8b4e25"),
)


def calibration_bins(results):
    records = []
    for name, _, _ in MODELS:
        predictions = pd.read_csv(results / "predictions" / f"{name}.csv")
        for repeat, rows in predictions.groupby("repeat"):
            curve = compute_calibration_curve(rows["y_true"], rows["y_prob"])
            bins = zip(curve["bin_pred_mean"], curve["bin_true_prop"], curve["bin_counts"])
            for bin_number, (predicted, observed, count) in enumerate(bins, start=1):
                records.append({
                    "model": name,
                    "repeat": repeat,
                    "bin": bin_number,
                    "mean_predicted": predicted,
                    "observed_prevalence": observed,
                    "n_records": count,
                })
    return pd.DataFrame(records)


def plot_calibration(bins, path):
    fig, ax = plt.subplots(figsize=(5, 4.5), layout="constrained")
    ax.plot([0, 1], [0, 1], "--", color="#8995a3", label="Perfect calibration")
    for name, label, color in MODELS:
        rows = bins.loc[bins["repeat"].eq(1) & bins["model"].eq(name)]
        ax.plot(rows["mean_predicted"], rows["observed_prevalence"],
                color=color, marker="o", label=label)
    ax.set(xlabel="Mean predicted malignancy probability",
           ylabel="Observed malignant fraction", aspect="equal",
           xlim=(-0.025, 1.025), ylim=(-0.025, 1.025))
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_summary(results, path):
    repeat_metrics = pd.read_csv(results / "repeat_metrics.csv")
    learning = pd.read_csv(results / "learning_curve_fold_metrics.csv")
    feature = pd.read_csv(results / "feature_set_summary.csv").set_index("feature_set")
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.7), layout="constrained")
    for index, (name, label, color) in enumerate(MODELS):
        brier = repeat_metrics.loc[repeat_metrics["model"].eq(name), "brier_score"]
        axes[0].scatter(brier, np.full(len(brier), index), color=color, alpha=0.45)
        axes[0].scatter(brier.mean(), index, color=color, marker="D", edgecolor="white")
        curve = (learning.loc[learning["model"].eq(name)]
                 .groupby("train_size")["auroc"].agg(["mean", "min", "max"]))
        bounds = [curve["mean"] - curve["min"], curve["max"] - curve["mean"]]
        axes[1].errorbar(curve.index, curve["mean"], yerr=bounds, label=label,
                         color=color, marker="o", capsize=3)

    axes[0].set(title="Probability accuracy", xlabel="Outer Brier score (lower is better)")
    axes[0].set_yticks(range(len(MODELS)), [label for _, label, _ in MODELS])
    axes[0].invert_yaxis()
    axes[1].set(title="Learning curves", xlabel="Training records", ylabel="Outer AUROC")
    axes[1].set_xticks([50, 100, 200, 455])
    axes[1].legend(frameon=False, fontsize=8, loc="lower center")
    groups = ["mean", "se", "worst", "all"]
    axes[2].errorbar(feature.loc[groups, "auroc_mean"], range(len(groups)),
                     xerr=feature.loc[groups, "auroc_std"], fmt="o",
                     color=MODELS[-1][2], capsize=3)
    axes[2].set(title="Elastic Net feature sets", xlabel="Outer AUROC")
    axes[2].set_yticks(range(len(groups)), ["Mean", "SE", "Worst", "All 30"])
    axes[2].invert_yaxis()
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    results = ROOT / "results" / "final_nested"
    figures = results / "figures"
    figures.mkdir(exist_ok=True)
    bins = calibration_bins(results)
    bins.to_csv(results / "haiyi_logistic_calibration_bins.csv", index=False)
    plot_calibration(bins, figures / "haiyi_logistic_calibration.png")
    plot_summary(results, figures / "haiyi_logistic_summary.png")


if __name__ == "__main__":
    main()
