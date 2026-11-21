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

The archived comparison in results/final_nested follows this repeated nested-CV protocol.
The completed 50-fold Yan analysis in results/yan/random_forest is retained as a
historical audited run. New common training commands dispatch Random_Forest to
src/yan/random_forest/train.py through the integration adapter; they do not use
the archived RF CSV as a training checkpoint.

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
|   |-- final_analysis.py            # Shared scheduling and evaluation
|   `-- yan/random_forest/           # RF training, checkpoints and export adapter
|-- scripts/
|   |-- download_and_check_data.py   # Source download and audit
|   |-- build_final_outputs.py       # Final figures and document drafts
|   `-- yan/run.py                   # Standalone RF audit/report entry
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
    |-- final_nested/                # Archived common comparison
    `-- yan/random_forest/           # Historical RF diagnostics/provenance
```

## Install and prepare shared data

```bash
python3 -m pip install -r requirements.txt -r src/yan/random_forest/requirements.txt
python3 src/data_prep.py
```

Use Python 3.12 and run from the repository root. Clone a revision containing the
new RF module and this README. The historical run's base commit, 06349f8, is
provenance; that older commit alone does not contain the later integration code.
On Windows, use python if python3 is not available.

## Check or summarize existing results without training

Verify the historical RF exports and their exact original source snapshots:

```bash
python3 src/yan/random_forest/verify.py
```

Recompute the archived common tables from saved predictions:

```bash
python3 src/final_analysis.py summarize --output results/final_nested --no-import-svm
```

Summarize does not train any model, but rewrites the summary CSVs in its selected
directory. The archived RF files remain historical until a new integrated RF run
is explicitly produced. Successful summarization is not a new training run.

## Fresh training through the common entry

For a complete Random Forest analysis (10 repeats, five outer folds and five
inner folds), use a new output directory:

```bash
python3 src/final_analysis.py nested --models Random_Forest --no-resume --no-import-svm --output results/reproduction_rf
```

The common scheduler calls yan.random_forest.train.run_fold via integration.py.
Each outer fit searches all 16 settings with 500 trees, selects a threshold from
the selected configuration's inner OOF, and fits the final forest once.
It does not fall through to the legacy RF training loop. Do not additionally run
the standalone RF pilot/run commands for the same common analysis.

To retrain the model families currently supported by the common training loop:

```bash
python3 src/final_analysis.py nested --models Constant_Baseline Logistic_Regression Ridge_Logistic Elastic_Net Random_Forest Decision_Tree --no-resume --no-import-svm --output results/reproduction_models
```

## Reproduce Asher's SVM analysis

The restored SVM owner entry trains from the cleaned data without the deleted
legacy import directory. Use an empty output directory and preserve the archive:

```bash
python3 scripts/asher/run.py --output results/asher/svm_reproduction --repeats 10 --verify-against results/final_nested
python3 scripts/asher/supplement.py
python3 scripts/asher/build_owner_report.py
```

The common entry now also accepts `--models Linear_SVM RBF_SVM` with a new
`--output` directory. All other models retain their current implementations.
SVM checkpoints require matching source, data, environment and configuration;
unattributed archived prediction files are never silently overwritten.
See [Asher's reproduction guide](docs/ASHER_SVM_REPRODUCIBILITY.md) and
[SVM report section](docs/ASHER_SVM_REPORT_SECTION.md). The full 100-fit rerun
matched all 11,380 archived probabilities and thresholds exactly at exported
precision in the recorded environment; this is not a clinical validation claim.

## Resume and shared RF output contract

Resume the same RF run without --no-resume:

```bash
python3 src/final_analysis.py nested --models Random_Forest --no-import-svm --output results/reproduction_rf
```

RF resumes only its own checkpoints after checking source, data, configuration,
environment, labels, assignments and linked fold records. Shared RF predictions
alone never cause a skip. --no-resume always creates a new RF run ID and ignores
old checkpoints. Prefer a new output directory; previous native runs and replaced
shared RF slots are retained as backup evidence.

For output directory OUT, the adapter explicitly converts and publishes:

- OUT/predictions/Random_Forest.csv
- OUT/outer_fold_metrics_Random_Forest.csv
- OUT/inner_search_Random_Forest.csv

Detailed checkpoints, selected inner OOF, feature importance and diagnostics stay
under OUT/yan/random_forest/runs/RUN_ID/. The shared export adds seconds and
convergence_warnings fields, retaining all detailed diagnostics. The model name
is Random_Forest, repeats are 1-10, folds are 0-4 and candidates are 0-15.

OUT/random_forest_export.json binds all three files to one run ID and their
SHA-256 hashes. Mixed files or incomplete publication block common summarization.
After summarization it records summary_status=complete and the RF run ID used by
the combined tables. Only one Random_Forest enters the comparison; detailed
personal files are not scanned as a second model.

To rebuild tables later without fitting:

```bash
python3 src/final_analysis.py summarize --output results/reproduction_rf --no-import-svm
```


## Integration checks and optional later analyses

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 src/yan/random_forest/test_checks.py
```

The real common-entry integration check used --repeats 1 with five complete
outer folds, the full 16-setting grid and 500 trees in a separate temporary
directory. It is an integration check, not the formal 10-repeat analysis.
Tests also cover RF-only dispatch, unchanged non-RF call signatures, complete
three-file publication, incomplete/mixed exports and one RF in summaries.

Learning curves and feature-set analyses remain separate stages. The existing
build_final_outputs.py also rewrites report drafts and the AI-reflection template;
protect members' edited documents before using it.

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
