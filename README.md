# BIOSTAT 707 Final Project

## Breast Mass Classification Model Comparison

We will compare logistic regression, regularized logistic regression, random forest, decision tree, and support vector machines for classifying breast masses as malignant or benign. The analysis will evaluate discrimination, probability calibration, and sensitivity–specificity tradeoffs using a common validation design.

The intended application is an educational investigation of classification after fine-needle aspiration and extraction of nuclear image features. The project does not predict future cancer onset, establish clinical utility, or provide a tool for clinical deployment.

## Team and responsibilities

| Team member | Primary responsibility |
|---|---|
| Tianyi Lu | Data preparation, shared splits, constant-probability baseline, and final evaluation |
| Haiyi Xiong | Logistic regression and Elastic Net logistic regression |
| Yan Yan | Random forest |
| Wenqing Zhao | Decision tree |
| Asher Xu | Linear-kernel and RBF-kernel SVM |

Each model owner will implement training and tuning, produce standardized prediction outputs, interpret their model, and write the corresponding report section. All members will contribute to code review, discussion, and presentation. Each member will commit their own work from their own GitHub account and submit an individual contribution statement as required by the course.

### Tianyi Lu: Data preparation, shared splits, constant baseline, and evaluation pipeline

Data preparation, stratified splitting, and the common evaluation framework are fully implemented and reproducible:
- **Unified splits:** Stratified 80% development ($N=455$) and 20% test ($N=114$) with random seed `707`. Stratified 5-fold cross-validation on the development set is saved in `data/processed/splits.csv`.
- **Baseline model:** Constant probability baseline predicting marginal prevalence ($\hat{p} \approx 0.3736$) implemented in `src/baseline.py`.
- **Shared evaluation:** Metric suite (`src/metrics.py`) providing AUROC, AUPRC, Brier score, log loss, 5-bin calibration curves, 95% sensitivity threshold locking on development OOF, 2,000 paired stratified bootstrap confidence intervals, and paired AUROC/Brier model differences.
- **Current scope:** The committed comparison tables contain the constant baseline only. Final cross-model tables will be generated after all model owners provide validated OOF and held-out-test predictions.
- **Team guide:** Detailed workflow and ingestion instructions are provided in `docs/TEAM_WORKFLOW.md`.

### Wenqing Zhao: Planned decision tree analysis

The decision tree will be tuned using the shared stratified five-fold cross-validation folds. We will test 12 combinations of `max_depth` (`3`, `5`, `7`, `None`) and `min_samples_leaf` (`5`, `10`, `20`), selecting settings by mean validation AUROC. We will check for overfitting, generate development out-of-fold probabilities, and select a fixed threshold targeting at least 95% development sensitivity. We will also examine feature importance and compare the decision tree with other models using discrimination, calibration, sensitivity, and specificity. The final test set will not be used for tuning or threshold selection.


## Proposal

The project proposal is located at the repository root as [707-proposal.md](707-proposal.md). It describes the research question, cohort and outcome, models, validation design, expected findings, and detailed division of responsibilities.

## Data source and license

- **Dataset:** Breast Cancer Wisconsin (Diagnostic), also known as WDBC.
- **Dataset source (UCI Machine Learning Repository):** [Breast Cancer Wisconsin (Diagnostic), DOI: 10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B).
- **License:** Creative Commons. Attribution 4.0 International (CC BY 4.0). https://creativecommons.org/licenses/by/4.0/.

The dataset license permits sharing and adaptation with appropriate credit, a license link, and an indication of modifications. This license applies to the UCI dataset; it does not automatically license our project code.

### Required dataset citation

Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B.

### Dataset structure and initial checks

The original data contain 569 records, with 212 malignant and 357 benign labels, and 30 numeric predictors derived from cell nuclei in digitized fine-needle aspiration images. The original file has no header: column 1 is the record ID, column 2 is the diagnosis (`M` or `B`), and columns 3–32 are predictors. The accompanying `wdbc.names` file explains feature order and definitions.

During proposal preparation, the source archive was downloaded and read. Checks found no missing predictor cells, invalid or nonfinite numeric predictor cells, or exact duplicate rows. There are 569 unique record IDs. These checks establish file-level feasibility, not independent verification of patient identity.

The supplied download script preserves the original `wdbc.data` and `wdbc.names` contents. It additionally creates a separate `data-audit.json` file with validation results and a SHA-256 checksum. Future transformed datasets and any changes will be documented separately. Malignancy will be encoded as 1 and benign status as 0 in analysis; ID will not be a predictor.

## Download and check the original data

Place `download-and-check-data.ps1` at the repository root alongside this README. On Windows, run the following from the repository root in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File ./download-and-check-data.ps1 -Destination ./data/raw
```

The script requires internet access to UCI. It downloads and extracts the source archive, checks the data, and writes `data/raw/data-audit.json`. It does not fit models or create the formal development/test split.

Alternatively, access the UCI dataset through the DOI link above, download the original archive, and extract it into `data/raw/`. The small original data files may also be included in this repository under the dataset license, retaining this attribution and the source documentation. The download script provides access to the original source even when raw data are not committed.

## Repository organization

The repository is organized to ensure end-to-end reproducibility, clean separation between development and sequestered test data, and seamless model submission for team members:

```text
.
|-- 707-proposal.md                  # Root-level formal project proposal
|-- README.md                        # Project documentation and instructions
|-- requirements.txt                 # Verified Python package versions
|-- download-and-check-data.ps1      # PowerShell UCI download and audit script
|-- scripts/
|   `-- download_and_check_data.py   # Cross-platform Python download and audit script
|-- data/
|   |-- raw/
|   |   |-- wdbc.data                # Original UCI data file (569 rows)
|   |   |-- wdbc.names               # UCI feature definitions
|   |   `-- data-audit.json          # Automated SHA-256 integrity and field checks
|   |-- processed/
|   |   |-- wdbc_clean.csv           # Clean data with standard headers and target (M=1, B=0)
|   |   |-- splits.csv               # Shared 80/20 train/test and 5-fold CV assignment
|   |   `-- data_dictionary.md       # Detailed data dictionary with definitions and ranges
|   `-- splits.csv                   # Convenience root copy of shared splits
|-- src/
|   |-- data_prep.py                 # Clean data and shared split generator (seed=707)
|   |-- eda_table1.py                # Development set EDA and Table 1 generation
|   |-- baseline.py                  # Constant-probability baseline model pipeline
|   |-- metrics.py                   # Common evaluation functions, calibration, and bootstrap CI
|   `-- evaluate_all.py              # Master multi-model evaluation and curve plotter
|-- docs/
|   |-- TEAM_WORKFLOW.md             # Guide for teammates to load splits and output predictions
|   |-- GROUP_HANDOFF_MESSAGE.md     # Copy-ready WeChat handoff messages
|   |-- data_and_cohort.md           # Draft of Data & Cohort final report section
|   `-- common_evaluation.md         # Draft of Common Evaluation final report section
|-- tests/                           # Automated metric and prediction-validation checks
`-- results/
    |-- figures/                     # EDA plots, calibration, ROC/PR, and clinical tradeoff curves
    |-- tables/                      # Table 1 baseline cohort, OOF metrics, and test comparison
    `-- predictions/                 # Standardized model prediction CSV outputs
```

The shared data, split, EDA, baseline, and evaluation components are implemented. Model-specific pipelines and predictions remain the responsibility of their assigned owners. The current aggregate results therefore contain only the constant baseline and are not the final team comparison.

## Planned analysis and reproducibility

We will reserve a stratified 20% test set and use the remaining 80% for development, with random seed 707 and shared stratified five-fold cross-validation. All preprocessing, tuning, and SVM probability calibration will occur within the relevant training subsets. Test labels will not be used for model selection, threshold selection, or recalibration.

Primary evaluation will use test AUROC. Secondary evaluation will include AUPRC, Brier score, log loss, calibration plots, and classification metrics at development-selected thresholds. A development sensitivity target of 95% is exploratory, not a clinical standard or a guarantee of test sensitivity. The proposal specifies uncertainty estimates and interpretation limits.

The shared pipeline was verified with Python 3.14.5 and the package versions pinned in `requirements.txt`. Install them with:

```bash
python3 -m pip install -r requirements.txt
```

From the repository root, reproduce and test the shared components with:

```bash
python3 src/data_prep.py
MPLBACKEND=Agg python3 src/eda_table1.py
PYTHONPATH=src python3 src/baseline.py
MPLBACKEND=Agg PYTHONPATH=src python3 src/evaluate_all.py
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The held-out comparison is provisional until the remaining model owners add their standardized OOF and test prediction files.
