"""Build final tables, figures, and a report draft from the unified analyses."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "final_nested"
FIGURES = RESULTS / "figures"
DOCS = ROOT / "docs"

MODEL_LABELS = {
    "Constant_Baseline": "Constant baseline",
    "Logistic_Regression": "Logistic regression",
    "Ridge_Logistic": "Ridge logistic (C=1)",
    "Elastic_Net": "Elastic Net",
    "Decision_Tree": "Decision tree",
    "Random_Forest": "Random forest",
    "Linear_SVM": "Linear SVM",
    "RBF_SVM": "RBF SVM",
}


def save_model_figure(summary: pd.DataFrame, metric: str, title: str, filename: str):
    # Keep the non-informative baseline in the numeric table, but omit it here
    # so the near-ceiling differences among patient-specific models remain
    # visible rather than being compressed against a 0.50 AUROC point.
    ordered = summary.loc[~summary.model.eq("Constant_Baseline")].sort_values(
        f"{metric}_mean"
    )
    labels = [MODEL_LABELS.get(value, value) for value in ordered.model]
    values = ordered[f"{metric}_mean"].to_numpy()
    errors = ordered[f"{metric}_std"].fillna(0).to_numpy()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.errorbar(values, labels, xerr=errors, fmt="o", capsize=3, color="#275D78")
    ax.set_xlabel(f"Mean across 10 repeated outer-CV estimates (error bar = 1 SD)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=200)
    plt.close(fig)


def save_learning_curve(summary: pd.DataFrame):
    # The constant baseline and the separated unpenalized logistic fit are
    # documented in tables, but omitting them here keeps the learning behavior
    # of the stable patient-specific models legible.
    summary = summary.loc[
        ~summary.model.isin(["Constant_Baseline", "Logistic_Regression"])
    ].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    panels = [
        ("auroc_mean", "AUROC", True),
        ("brier_score_mean", "Brier score", False),
        ("sensitivity_mean", "Sensitivity at training-selected threshold", True),
    ]
    for model_name, group in summary.groupby("model"):
        group = group.sort_values("train_size")
        for ax, (column, label, _) in zip(axes, panels):
            ax.plot(
                group.train_size,
                group[column],
                marker="o",
                label=MODEL_LABELS.get(model_name, model_name),
            )
            sd_column = column.replace("_mean", "_std")
            if sd_column in group:
                lower = group[column] - group[sd_column].fillna(0)
                upper = group[column] + group[sd_column].fillna(0)
                ax.fill_between(group.train_size, lower, upper, alpha=0.08)
    for ax, (_, label, _) in zip(axes, panels):
        ax.set_xlabel("Outer-training sample size")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[-1].legend(fontsize=7, bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.suptitle("Learning curves on common outer folds")
    fig.tight_layout()
    fig.savefig(FIGURES / "learning_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_feature_sets(summary: pd.DataFrame):
    order = ["mean", "se", "worst", "all"]
    summary = summary.set_index("feature_set").loc[order].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, metric, label in [
        (axes[0], "auroc", "AUROC"),
        (axes[1], "brier_score", "Brier score"),
    ]:
        ax.errorbar(
            summary.feature_set,
            summary[f"{metric}_mean"],
            yerr=summary[f"{metric}_std"],
            fmt="o",
            capsize=4,
            color="#8B3A3A",
        )
        ax.set_xlabel("Feature set")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Elastic Net feature-representation sensitivity analysis")
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_set_comparison.png", dpi=200)
    plt.close(fig)


def concise_primary_table(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary[
        [
            "model",
            "auroc_mean",
            "auroc_std",
            "brier_score_mean",
            "brier_score_std",
            "sensitivity_mean",
            "specificity_mean",
        ]
    ].copy()
    table["model"] = table.model.map(lambda value: MODEL_LABELS.get(value, value))
    for metric in ["auroc", "brier_score"]:
        table[metric] = table.apply(
            lambda row: f"{row[f'{metric}_mean']:.4f} ± {row[f'{metric}_std']:.4f}",
            axis=1,
        )
    table["sensitivity"] = table.sensitivity_mean.map(lambda value: f"{value:.3f}")
    table["specificity"] = table.specificity_mean.map(lambda value: f"{value:.3f}")
    return table[["model", "auroc", "brier_score", "sensitivity", "specificity"]]


def build_report(primary: pd.DataFrame):
    table = concise_primary_table(primary)
    nonbaseline = primary.loc[~primary.model.eq("Constant_Baseline")]
    best_auc = nonbaseline.loc[nonbaseline.auroc_mean.idxmax()]
    best_brier = nonbaseline.loc[nonbaseline.brier_score_mean.idxmin()]
    paired_note = ""
    paired_path = RESULTS / "paired_difference_summary_vs_stable_logistic.csv"
    if paired_path.exists():
        paired = pd.read_csv(paired_path)
        rbf = paired.loc[paired.model.eq("RBF_SVM")]
        if not rbf.empty:
            rbf = rbf.iloc[0]
            paired_note = (
                f" Relative to ridge logistic, the mean paired RBF SVM AUROC difference "
                f"was {rbf.auroc_difference_mean:+.4f}, ranging from "
                f"{rbf.auroc_difference_min:+.4f} to {rbf.auroc_difference_max:+.4f} "
                "across repeats."
            )
    learning_note = "Learning-curve results are in `results/final_nested/learning_curve_summary.csv`."
    feature_note = "Feature-set results are in `results/final_nested/feature_set_summary.csv`."
    if (RESULTS / "learning_curve_summary.csv").exists():
        learning = pd.read_csv(RESULTS / "learning_curve_summary.csv")
        learning = learning.loc[
            ~learning.model.isin(["Constant_Baseline", "Logistic_Regression"])
        ]
        size_50 = learning.loc[learning.train_size.eq(50), "auroc_mean"].mean()
        size_455 = learning.loc[learning.train_size.eq(455), "auroc_mean"].mean()
        learning_note = (
            f"Across stable patient-specific model families, mean outer-fold AUROC increased from approximately "
            f"{size_50:.3f} at training size 50 to {size_455:.3f} at size 455. "
            "The curve is descriptive because the same records contribute to multiple fits."
        )
    if (RESULTS / "feature_set_summary.csv").exists():
        feature = pd.read_csv(RESULTS / "feature_set_summary.csv")
        best_feature = feature.loc[feature.auroc_mean.idxmax()]
        feature_note = (
            f"In the Elastic Net sensitivity analysis, the highest mean AUROC was obtained by "
            f"the `{best_feature.feature_set}` feature set ({best_feature.auroc_mean:.4f}). "
            "This comparison isolates feature representation and should not be read as a new "
            "post-hoc model-selection contest."
        )

    report = f"""# BIOSTAT 707 Final Project Report Draft

## Breast Mass Classification Model Comparison

### Research question

We compared prespecified linear, tree-based, and support-vector classifiers for distinguishing malignant from benign breast masses in the Wisconsin Diagnostic Breast Cancer dataset. The goal was a fair assessment of discrimination, calibration, and the false-negative/false-positive tradeoff, not deployment of a clinical diagnostic tool.

### Data

The analysis contains 569 records: 212 malignant and 357 benign. Each record has 30 continuous nuclear-image features: the mean, standard error (SE), and “worst” value of ten measurements. ID was never used as a predictor. The source file contained no missing predictor values or exact duplicate records. Unique record IDs do not independently prove unique patients, which remains a limitation.

### Models

We evaluated a constant-probability baseline, unpenalized logistic regression, Elastic Net logistic regression, a decision tree, a 500-tree random forest, a linear SVM, and an RBF SVM. Hyperparameter grids were fixed in the proposal. The unpenalized logistic model produced extremely large standardized coefficients and nearly all-zero/all-one outer-fold probabilities, consistent with the proposal's prespecified separation contingency. We therefore retained it as an unstable comparator and used the prespecified ridge logistic model (C=1) as the stable full-feature logistic comparator. Standardization was fitted only inside training folds for logistic and SVM models. SVM probabilities used training-only three-fold sigmoid calibration.

### Primary validation design

In response to preliminary feedback, the primary comparison used all 569 records in 10 repeats of stratified five-fold outer cross-validation. Hyperparameters were selected separately inside each outer-training set using stratified five-fold inner CV and mean inner-fold AUROC. Each outer validation record was therefore predicted by a model that had not trained or tuned on that record. The same outer folds were used for every model.

Within each outer-training set, we generated inner out-of-fold probabilities and selected the highest threshold achieving at least 95% training sensitivity. That threshold was then applied unchanged to the corresponding outer validation fold. Repeat-level metrics pooled the five outer validation folds, so every record contributed once per repeat.

The original stratified 80/20 split was retained only as a secondary one-shot check. Its bootstrap intervals are conditional on that single split and are not the primary evidence.

### Primary results

Values are mean ± SD across the ten repeated outer-CV estimates. Repeats share records and are not ten independent datasets; SD describes split sensitivity rather than population-level uncertainty.

{table.to_markdown(index=False)}

The highest mean AUROC was {MODEL_LABELS.get(best_auc.model, best_auc.model)} ({best_auc.auroc_mean:.4f}), while the lowest mean Brier score was {MODEL_LABELS.get(best_brier.model, best_brier.model)} ({best_brier.brier_score_mean:.4f}).{paired_note} Differences among the strongest classifiers were small relative to the already high performance of this heavily studied dataset. We therefore do not claim that a numerically leading model is clinically or statistically superior.

### Learning curves

{learning_note}

### Mean, SE, worst, and all-feature comparison

{feature_note}

### Interpretation

Near-ceiling AUROC means that model ranking is sensitive to a small number of records. Model choice should therefore consider calibration, threshold behavior, interpretability, and computational complexity in addition to AUROC. Logistic coefficients and permutation importance must also be interpreted cautiously because radius, perimeter, and area measurements—and their mean/worst summaries—are highly correlated.

### Limitations

- The dataset is small, old, and heavily studied; it does not represent external validation.
- Records are stratified at the record level because patient, site, and temporal group identifiers are unavailable.
- Repeated cross-validation reduces dependence on one split but does not create new independent patients.
- The 95% sensitivity target is exploratory and is not an established clinical operating point.
- Feature importance can be redistributed among correlated predictors.
- The original 114-record test analysis is secondary and its intervals condition on one development/test split.

### AI use disclosure — team must complete in first person

AI-assisted tools were used for parts of code implementation, debugging, validation checks, and preparation of draft tables and text. Each member must add their own accurate first-person account of what they used AI for, what it made possible, what required correction or verification, and how it affected both workload and understanding. The final personal reactions must be written and approved by the individual member rather than generated from this placeholder.

### Reproducibility

From the repository root:

```bash
python3 -m pip install -r requirements-svm.txt
PYTHONPATH=src python3 src/final_analysis.py nested
PYTHONPATH=src python3 src/final_analysis.py learning-curves
PYTHONPATH=src python3 src/final_analysis.py feature-sets
MPLBACKEND=Agg python3 scripts/build_final_outputs.py
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
"""
    (DOCS / "FINAL_REPORT_DRAFT.md").write_text(report, encoding="utf-8")


def build_presentation_outline():
    outline = """# BIOSTAT 707 Presentation Outline (7.5 minutes)

## Slide 1 — Question and data (45 seconds)

- 569 breast-mass records; 212 malignant and 357 benign.
- Thirty nuclear-image predictors: mean, SE, and worst summaries.
- Question: do nonlinear classifiers offer a meaningful advantage over interpretable linear models?

## Slide 2 — Why we changed the validation plan (60 seconds)

- Expected AUROC is near 0.99, so a 114-record test set cannot resolve tiny differences.
- Primary analysis: common 10×5 repeated nested CV on all 569 records.
- Original 80/20 split retained only as a secondary check.

## Slide 3 — Models and leakage prevention (60 seconds)

- Baseline, unpenalized and ridge logistic, Elastic Net, tree, random forest, linear SVM, RBF SVM.
- Tuning, scaling, SVM calibration, and threshold selection stayed inside training data.
- Every model used identical outer folds.

## Slide 4 — Primary comparison (90 seconds)

- Show `results/final_nested/figures/primary_auroc.png`.
- Emphasize effect sizes and split-to-split variability, not only the ranking.
- Avoid saying that a 0.001–0.003 AUROC advantage is meaningful without stronger evidence.

## Slide 5 — Calibration and 95% sensitivity threshold (75 seconds)

- Show Brier-score comparison and report achieved outer sensitivity/specificity.
- Explain that 95% was a training target, not a guaranteed clinical sensitivity.

## Slide 6 — Instructor-requested sensitivity analyses (75 seconds)

- Learning curves: size 50, 100, 200, 455.
- Mean vs SE vs worst vs all features using a common Elastic Net protocol.
- Explain what these analyses add beyond a model leaderboard.

## Slide 7 — Take-home message, limitations, AI use (75 seconds)

- Near-ceiling performance; simple models may be preferable when differences are tiny.
- No external/site-level validation and correlated predictors limit interpretation.
- State AI use factually; each member gives their own brief plus/minus reflection.

## Q&A preparation

- Why repeated nested CV instead of one train/test split?
- Were preprocessing and calibration kept inside folds?
- Why is the 95% sensitivity threshold exploratory?
- Does the best numerical AUROC justify a more complex model?
"""
    (DOCS / "PRESENTATION_OUTLINE.md").write_text(outline, encoding="utf-8")


def build_ai_template():
    template = """# Team AI Use Reflection Template

The instructor requires each person's actual reaction. Do not submit a shared AI-written reflection. Each member should replace the prompts below in their own words.

## [Member name]

- **What I used AI for:** [specific tasks you personally requested or reviewed]
- **What it helped me do:** [work that became possible, faster, or more thorough]
- **What did not work or needed correction:** [specific limitation, error, or overconfident suggestion]
- **How I verified the work:** [tests, manual checks, course concepts, teammate review]
- **Effect on my workload:** [your honest assessment]
- **Effect on my understanding:** [what you understand better or worse after using it]
- **One example I can explain without AI:** [nested CV, calibration, thresholding, model interpretation, etc.]
"""
    (DOCS / "AI_USE_REFLECTION_TEMPLATE.md").write_text(template, encoding="utf-8")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    primary = pd.read_csv(RESULTS / "model_summary.csv")
    save_model_figure(primary, "auroc", "Repeated nested-CV discrimination", "primary_auroc.png")
    save_model_figure(primary, "brier_score", "Repeated nested-CV probability accuracy", "primary_brier.png")
    if (RESULTS / "learning_curve_summary.csv").exists():
        save_learning_curve(pd.read_csv(RESULTS / "learning_curve_summary.csv"))
    if (RESULTS / "feature_set_summary.csv").exists():
        save_feature_sets(pd.read_csv(RESULTS / "feature_set_summary.csv"))
    build_report(primary)
    build_presentation_outline()
    build_ai_template()
    print("Built final figures and document drafts.")


if __name__ == "__main__":
    main()
