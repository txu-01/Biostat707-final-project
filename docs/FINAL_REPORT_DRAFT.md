# BIOSTAT 707 Final Project Report Draft

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

### Primary results

Values are mean ± SD across the ten repeated outer-CV estimates. Repeats share records and are not ten independent datasets; SD describes split sensitivity rather than population-level uncertainty.

| model                | auroc           | brier_score     |   sensitivity |   specificity |
|:---------------------|:----------------|:----------------|--------------:|--------------:|
| Constant baseline    | 0.4955 ± 0.0000 | 0.2338 ± 0.0000 |         1     |         0     |
| Decision tree        | 0.9644 ± 0.0037 | 0.0582 ± 0.0050 |         0.952 |         0.892 |
| Elastic Net          | 0.9919 ± 0.0025 | 0.0241 ± 0.0015 |         0.95  |         0.99  |
| Linear SVM           | 0.9930 ± 0.0011 | 0.0225 ± 0.0018 |         0.95  |         0.987 |
| Logistic regression  | 0.9624 ± 0.0037 | 0.0486 ± 0.0064 |         0.941 |         0.955 |
| RBF SVM              | 0.9943 ± 0.0009 | 0.0205 ± 0.0017 |         0.945 |         0.989 |
| Random forest        | 0.9887 ± 0.0008 | 0.0332 ± 0.0011 |         0.951 |         0.962 |
| Ridge logistic (C=1) | 0.9936 ± 0.0006 | 0.0214 ± 0.0010 |         0.95  |         0.989 |

The highest mean AUROC was RBF SVM (0.9943), while the lowest mean Brier score was RBF SVM (0.0205). Relative to ridge logistic, the mean paired RBF SVM AUROC difference was +0.0007, ranging from -0.0003 to +0.0014 across repeats. Differences among the strongest classifiers were small relative to the already high performance of this heavily studied dataset. We therefore do not claim that a numerically leading model is clinically or statistically superior.

### Learning curves

Across stable patient-specific model families, mean outer-fold AUROC increased from approximately 0.972 at training size 50 to 0.989 at size 455. The curve is descriptive because the same records contribute to multiple fits.

### Mean, SE, worst, and all-feature comparison

In the Elastic Net sensitivity analysis, the highest mean AUROC was obtained by the `all` feature set (0.9910). This comparison isolates feature representation and should not be read as a new post-hoc model-selection contest.

### Interpretation

Near-ceiling AUROC means that model ranking is sensitive to a small number of records. Model choice should therefore consider calibration, threshold behavior, interpretability, and computational complexity in addition to AUROC. Logistic coefficients and permutation importance must also be interpreted cautiously because radius, perimeter, and area measurements—and their mean/worst summaries—are highly correlated.

### Limitations

- The dataset is small, old, and heavily studied; it does not represent external validation.
- Records are stratified at the record level because patient, site, and temporal group identifiers are unavailable.
- Repeated cross-validation reduces dependence on one split but does not create new independent patients.
- The 95% sensitivity target is exploratory and is not an established clinical operating point.
- Feature importance can be redistributed among correlated predictors.

### AI use disclosure — team must complete in first person

AI-assisted tools were used for parts of code implementation, debugging, validation checks, and preparation of draft tables and text. Each member must add their own accurate first-person account of what they used AI for, what it made possible, what required correction or verification, and how it affected both workload and understanding. The final personal reactions must be written and approved by the individual member rather than generated from this placeholder.

### Reproducibility

From the repository root:

```bash
python3 -m pip install -r requirements.txt
PYTHONPATH=src python3 src/final_analysis.py nested
PYTHONPATH=src python3 src/final_analysis.py learning-curves
PYTHONPATH=src python3 src/final_analysis.py feature-sets
MPLBACKEND=Agg python3 scripts/build_final_outputs.py
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
