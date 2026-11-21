# Asher SVM reproduction

Owner: Asher Xu. This supplement restores training for the existing Linear SVM and RBF SVM comparison. It does not replace the team's archived primary results.

## Run from the repository root

Install the scientific packages listed in requirements.txt. This verification run used Python 3.12.14, NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1 and SciPy 1.18.1. The root file pins pandas 3.1.0; the actual verification environment is recorded in each run manifest, and an exact dependency-lock reproduction is not claimed.

```text
python scripts/asher/run.py --output results/asher/svm_reproduction --repeats 10 --verify-against results/final_nested
python scripts/asher/supplement.py
```

The standalone entry automatically prepares the Windows DLL search path for a conda or pixi environment. No files from the old checkpoint project are used. The required input is data/processed/wdbc_clean.csv, and the remaining dependencies are the project's current feature dictionary and metric helpers.

The common entry also dispatches the two SVM names to the restored implementation:

```text
PYTHONPATH=src python src/final_analysis.py nested --models Linear_SVM RBF_SVM --output results/asher/common_entry_run --no-import-svm
```

On PowerShell, set `$env:PYTHONPATH='src'` first instead of using the POSIX prefix above. For a clean run always choose a new output directory. Never use results/final_nested as a fresh training destination. Existing unattributed prediction files are rejected, rather than overwritten or treated as completed training.

## Historical fitting specification

- Sort the complete 569-record cohort by ID; malignancy is positive.
- Ten repeats of stratified five-fold outer CV, seeds 707 through 716.
- Five-fold inner CV, seed 1707 + 5 × zero-based repeat + zero-based outer fold. These are the original SVM seeds; they need not equal other model owners' inner seeds. Outer folds are identical across models.
- C = 0.01, 0.1, 1, 10; RBF gamma = 0.001, 0.01, 0.1, 1.
- Select the largest mean inner-fold AUROC, retaining the first candidate when the improvement is no greater than 1e-12.
- StandardScaler is inside the SVC pipeline wrapped by CalibratedClassifierCV. Three-fold stratified sigmoid calibration uses seed 707 and ensemble=False. SVC uses seed 707, tol=1e-3, no class reweighting.
- Select the highest threshold achieving at least 95% sensitivity in the selected candidate's training-only inner OOF probabilities. Fit on all outer-training records and apply that threshold unchanged to outer validation.
- Pool five outer folds within each repeat before computing the primary metrics. Do not substitute the mean fold AUROC for the pooled-repeat AUROC.

## Outputs and validation

Each model has a prediction file, an outer-fold metric file, an inner-search file, and a manifest recording data/source hashes, environment and configuration. Each fold is checkpointed after fitting. A partial or changed checkpoint bundle is rejected; preserve the failed run for inspection and use a new output directory when needed. The complete run contains 11,380 predictions, 100 outer fits and 1,000 inner candidate records across both kernels.

archive_reproduction_check.csv aligns every new and archived prediction using model, repeat, fold, ID and label, and checks both probabilities and thresholds with absolute tolerance 1e-7. Comparison reads the archive only after fitting and never uses it to choose settings.

The supplement produces standardized linear margin coefficients across 50 outer-training fits and training-only permutation summaries. For permutation analysis, only repeat 1 is used: an inner validation subset is held out from fitting inside each outer-training set; ten shuffles per feature estimate AUROC decreases. The hyperparameters were already selected using the outer-training data, including these inner validation labels. Importance is therefore descriptive after selection, not an independent validation estimate. Outer-validation labels are never used in this interpretation step, and importance is never used for feature selection. Coefficients describe a standardized SVM margin, not logistic odds ratios or causal effects.

## Safe integration

The local change to src/final_analysis.py routes SVM training to src/asher/svm.py and prevents the legacy importer from overwriting a newly attributed SVM run. It does not change the random forest owner's adapter, the logistic or tree models, or the archived predictions. Test the complete shared suite after integration. New files and reports remain local until explicitly approved for upload.
