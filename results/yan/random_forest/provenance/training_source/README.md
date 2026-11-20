# BIOSTAT 707 Final Project

## Breast Mass Classification Model Comparison

This project compares linear, tree-based, and support-vector classifiers for distinguishing malignant from benign breast masses in the Wisconsin Diagnostic Breast Cancer dataset. The analysis focuses on discrimination, calibration, and the false-negative/false-positive tradeoff. It is an educational model comparison, not a clinical diagnostic tool.

## Final validation design

The final analysis follows the instructor's preliminary feedback:

- All 569 records are used in 10 repeats of stratified five-fold outer cross-validation.
- Hyperparameters are selected inside each outer-training set with stratified five-fold inner cross-validation.
- Every model uses the same outer folds.
- Scaling, SVM probability calibration, and threshold selection are fitted using training data only.
- Each record receives one out-of-fold prediction per model in every repeat.
- Learning curves use training sizes 50, 100, 200, and 455.
- Elastic Net is also compared using mean, SE, worst, and all 30 features.

All submitted results follow this repeated nested-CV protocol. Superseded split files, predictions, figures, tables, and model-specific output folders were removed.

## Models

- Constant-probability baseline
- Unpenalized logistic regression
- Ridge logistic regression (`C=1`)
- Elastic Net logistic regression
- Decision tree
- Random forest
- Linear SVM
- RBF SVM

The unpenalized logistic fit showed separation-like behavior, so it is reported as an unstable comparator.

## Main result

RBF SVM had the highest mean outer-CV AUROC (0.9943), followed closely by ridge logistic regression (0.9936). The paired RBF-minus-ridge difference averaged only +0.0007 and ranged from -0.0003 to +0.0014 across repeats, so the results do not support a strong superiority claim.

The complete write-up is in [docs/FINAL_REPORT_DRAFT.md](docs/FINAL_REPORT_DRAFT.md), and all submitted result files are under [results/final_nested](results/final_nested).

## Data

- Dataset: Breast Cancer Wisconsin (Diagnostic), or WDBC
- Source: UCI Machine Learning Repository, DOI [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B)
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Records: 569 total, including 212 malignant and 357 benign
- Predictors: 30 continuous nuclear-image features

The record ID is used only to align predictions and is never used as a predictor. The raw data audit found no missing predictor values, nonfinite values, or exact duplicate records.

## Repository organization

```text
.
|-- README.md
|-- requirements.txt
|-- data/
|   |-- raw/                         # Original UCI files and integrity audit
|   `-- processed/                  # Clean 569-record dataset and dictionary
|-- src/
|   |-- data_prep.py                 # Full-cohort cleaning and validation
|   |-- metrics.py                   # Shared metrics and threshold functions
|   `-- final_analysis.py            # Repeated nested CV and sensitivity analyses
|-- scripts/
|   |-- download_and_check_data.py   # Source download and audit
|   `-- build_final_outputs.py       # Final figures and document drafts
|-- docs/
|   |-- FINAL_REPORT_DRAFT.md
|   |-- PRESENTATION_OUTLINE.md
|   |-- AI_USE_REFLECTION_TEMPLATE.md
|   |-- data_and_cohort.md
|   |-- common_evaluation.md
|   `-- TEAM_WORKFLOW.md
|-- tests/
|   |-- test_final_analysis.py
|   `-- test_metrics.py
`-- results/
    `-- final_nested/                # The only submitted model-result set
```

## Reproduce the analysis

```bash
python3 -m pip install -r requirements.txt
python3 src/data_prep.py
PYTHONPATH=src python3 src/final_analysis.py nested
PYTHONPATH=src python3 src/final_analysis.py learning-curves
PYTHONPATH=src python3 src/final_analysis.py feature-sets
MPLBACKEND=Agg python3 scripts/build_final_outputs.py
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The repeated folds reuse the same 569 records. Their standard deviation and range describe sensitivity to data splitting; they should not be interpreted as uncertainty from ten independent datasets.

## Team responsibilities

| Team member | Main responsibility |
|---|---|
| Tianyi Lu | Data preparation, baseline, and common evaluation |
| Haiyi Xiong | Logistic regression and Elastic Net |
| Yan Yan | Random forest |
| Wenqing Zhao | Decision tree |
| Asher Xu | Linear and RBF SVM |

Each team member should review the final method and results and add an accurate first-person contribution and AI-use reflection before submission.
