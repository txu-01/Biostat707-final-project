# Breast Mass Classification Model Comparison

BIOSTAT 707 Final Project Proposal

Research focus: Discrimination, calibration, and decision thresholds

Team: Tianyi Lu, Haiyi Xiong, Yan Yan, Wenqing Zhao, Asher Xu

## 1  Research question and intended application

We will investigate whether nonlinear classifiers improve prediction of malignant versus benign breast masses compared with logistic regression, and whether any improvement in discrimination is accompanied by reliable probability estimates and fewer false negatives at a common sensitivity target. Our emphasis is a fair, reproducible comparison rather than maximizing a single accuracy score.

The intended application is an educational prototype for supporting review of a breast mass after fine-needle aspiration (FNA) and extraction of image-derived nuclear features. A predicted malignancy probability could hypothetically help prioritize cases for prompt specialist review. It does not predict future cancer onset and is not a population screening tool. This project will not establish clinical utility or support deployment.

Our primary question is whether additional model complexity improves AUROC. Secondary questions concern calibration, sensitivity–specificity tradeoffs, and stability of feature interpretation. We will not infer causal effects from predictive associations.

## 2  Dataset  license  and completed feasibility check

We will use (the Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B), distributed under CC BY 4.0 https://creativecommons.org/licenses/by/4.0/. This permits public redistribution and adaptation with attribution. We will retain attribution, link the license, and identify modifications in the repository and report.

The original UCI archive has already been downloaded, extracted, and read during proposal preparation. Direct inspection of wdbc.data confirmed:

| Item | Verified result |
| --- | --- |
| Records | 569 |
| Outcome labels | 212 malignant; 357 benign |
| Predictors | 30 numeric features |
| Columns | ID, diagnosis, and 30 predictors |
| Missing predictor cells | 0 |
| Invalid/nonfinite numeric predictor cells | 0 |
| Unique record IDs | 569 |
| Exact duplicate rows | 0 |

The predictors summarize ten nuclear characteristics using their mean, standard error, and “worst” values. We will retain all records with valid labels and features; the initial check identified no exclusions. Malignancy will be coded as 1 and benign status as 0. ID will be used only for tracking predictions, never as a predictor. A unique record ID does not independently prove that every record is from a distinct patient; we will document this limitation.

These are cross-sectional diagnostic measurements, not longitudinal observations. No additional data source, protected information, or restricted-access database is required.

## 3  Exploratory analysis and preprocessing

We will describe class balance, predictor distributions, outliers, and correlations, including correlations between summaries of the same nuclear characteristic. Model selection and evaluation will be separated through nested cross-validation.

We will not remove plausible extreme measurements solely because they are extreme. There are no observed missing values, so imputation is unnecessary. We will document this explicitly rather than introduce artificial missingness into the main analysis. If a revised source reveals missing values, median imputation will be fitted exclusively within the training folds.

Logistic regression and SVM pipelines will standardize features using training-fold means and standard deviations. Tree models will use the original numeric scale. Any later feature selection must occur inside training folds. We will not use oversampling in the primary analysis; sensitivity will instead be examined through training-selected decision thresholds.

## 4  Models and simple baseline

All candidate models will use the same 30 predictors and identical repeated outer cross-validation folds.

### Tianyi Lu Constant probability baseline

Implementation and tuning: Predict the training-set malignancy proportion

Purpose: Check improvement over a model without patient-specific predictors

### Haiyi Xiong Unpenalized logistic regression  Elastic Net logistic regression

Implementation and tuning: Use scikit-learn pipelines; tune Elastic Net C in {0.01, 0.1, 1, 10} and l1_ratio in {0.1, 0.5, 0.9, 1.0}

Purpose: Compare an interpretable linear model with regularized estimation

### Yan Yan Random forest

Implementation and tuning: Fix 500 trees; assess 16 combinations of max_depth in {3, None}, min_samples_leaf in {1, 3, 5, 10}, and max_features in {sqrt, 0.5}

Purpose: Assess nonlinear effects and interactions through bagged trees

### Wenqing Zhao Decision Tree

Implementation and tuning: Test 12 combinations of max_depth in {3, 5, 7, None} and min_samples_leaf in {5, 10, 20} using cross-validation.

Purpose: Evaluate how well a single decision tree performs compared with more complex models.

### Asher Xu Linear kernel SVM  RBF kernel SVM

Implementation and tuning: Linear: C in {0.01, 0.1, 1, 10}; RBF: cross these values with gamma in {0.001, 0.01, 0.1, 1}

Purpose: Test whether a nonlinear margin improves on a linear margin

For SVM probability estimates, we will use three-fold sigmoid calibration within each model-fitting training subset. During tuning, calibration will not see the fold used to evaluate that candidate.

Unpenalized logistic regression may be unstable if classes are separable. We will record convergence and coefficient behavior. If convergence remains inadequate after increasing the iteration limit, it will be reported as an unstable comparator and replaced for probability comparisons by a prespecified ridge logistic model (C=1). Elastic Net remains the required regularized model regardless of this contingency. We will describe the actual final model list transparently.

## 5  Validation design and leakage prevention

Use repeated nested cross-validation: evaluate all 569 records with ten repeats of stratified five-fold outer CV. Every model uses identical outer folds. Within each outer-training set, select hyperparameters by mean AUROC from stratified five-fold inner CV. Every transformation and SVM calibration step is fitted only within the relevant training subset.

Choose thresholds without validation labels: within each outer-training set, generate training-only out-of-fold probabilities and select the highest threshold achieving at least 95% sensitivity. This is a prespecified exploratory sensitivity target, not an established clinical standard or a guarantee of future sensitivity. Apply the selected threshold unchanged to the corresponding outer validation fold.

Pool the five outer validation folds within each repeat so every record contributes exactly one out-of-fold prediction per model and repeat. Compare models using paired repeat-level differences rather than revising models to improve observed rankings.

Record-level stratification is appropriate for the available cross-sectional structure, subject to the unresolved patient-identity limitation. This dataset does not support temporal or hospital-level external validation.

## 6  Final evaluation and interpretation

Primary metric: repeated outer-CV AUROC. We will report paired differences between nonlinear models and the stable full-feature logistic comparator, rather than focus solely on a leaderboard.

Secondary metrics: AUPRC, Brier score, log loss, and a calibration plot using approximately five equal-frequency bins. We will also report mean predicted versus observed malignancy frequency. The constant-probability baseline will be evaluated using the same appropriate probability metrics.

Threshold evaluation: at each training-selected sensitivity threshold, report the outer-validation confusion matrix, sensitivity, specificity, positive predictive value, and negative predictive value. We will report achieved validation sensitivity even if it falls below 95%. We will discuss false negatives as missed malignant cases and false positives as potentially unnecessary escalations, without assigning unsupported clinical costs.

Stability: summarize the mean, standard deviation, minimum, and maximum across the ten outer-CV repeats, and report paired repeat-level differences against ridge logistic regression. Because the repeats reuse the same records, these summaries describe sensitivity to the fold assignments and are not treated as estimates from ten independent datasets.

Interpretability: examine standardized logistic coefficients and training-only model summaries without using outer-validation labels for feature selection. Compare consistency of broad feature families, recognizing that correlated predictors can redistribute apparent importance. Each model owner will prepare their own interpretation; the team will synthesize agreements and disagreements.

We will interpret small or uncertain differences cautiously and discuss performance alongside calibration, complexity, runtime, and interpretability. The study is not powered or designed to establish equivalence or noninferiority.

## 7  Detailed division of responsibilities

### Tianyi Lu   Data preparation and final evaluation

Analysis: maintain the source download script and data dictionary; verify coding, IDs, numerical validity, and exclusions; generate shared outer folds; implement shared metric functions and the constant baseline; and assemble the repeated nested-CV comparison tables and plots.

Application and follow-up: explain what the dataset represents, translate confusion matrices into missed malignant cases and unnecessary escalations, and document the limits of applying the models to a different setting. Write the data/cohort and common evaluation sections.

Boundary: this member will not tune, repair, or write the methods for all other models. Model owners must submit tested code and standardized outputs.

### Haiyi Xiong   Logistic regression and Elastic Net

Analysis: implement scaling pipelines, fit both regression variants, complete the specified inner searches, diagnose convergence/separation, and examine coefficient direction and stability across outer folds.

Application and follow-up: assess whether a relatively interpretable probability model is sufficient for the proposed review-prioritization setting; discuss how regularization changes stability, calibration, and errors. Write this model group's methods, results, and limitations.

### Yan Yan   Random forest

Analysis: implement and tune the forest, inspect training–validation gaps, summarize the effect of leaf size and depth, and provide outer-fold predictions and runtime summaries.

Application and follow-up: examine whether capturing interactions reduces false negatives or unnecessary escalations relative to the linear comparator, and whether probability reliability offsets any discrimination gain. Write the forest's methods, results, and limitations.

### Wenqing Zhao   Decision tree

Analysis: implement and tune the decision tree, check for overfitting, generate outer-fold predictions, and compare feature behavior with other models.

Application and follow-up: evaluate how well the decision tree identifies malignant cases while limiting false positives, considering sensitivity, specificity, and probability calibration. Write the model's methods, results, and limitations.

### Asher Xu   Linear and RBF SVM

Analysis: implement scaling and training-only probability calibration, tune both kernels, compare their outer-CV performance and runtime, and verify that probability fitting remains inside the training pipeline.

Application and follow-up: assess whether a nonlinear margin improves case prioritization sufficiently to justify reduced interpretability and probability-calibration complexity. Write the SVM methods, results, and limitations.

### Shared deliverables and workload control

Each model owner will review the shared implementation, selected settings, runtime, outer-fold predictions, thresholds, and model-specific interpretation. Prediction tables contain record ID, true label, model name, repeat, fold, predicted probability, and the training-selected threshold.

All five members will write and present their own sections, review another member's code, and jointly write the discussion and conclusions. More extensive data/evaluation work is balanced against Tianyi Lu having no complex model search. The team will review effort after the first working pipeline and redistribute supporting documentation or plotting tasks if needed.

## 8  Expected findings and limitations

We hypothesize that linear or regularized models may already discriminate well, leaving limited gains for more complex classifiers. Nonlinear methods might improve discrimination without improving calibration, and the model with the largest AUROC may not offer the best operating-point tradeoff. These are hypotheses, not results; no predictive model has been fitted for this proposal.

Limitations include the small historical dataset, correlated predictors, incomplete information for verifying patient independence, and no independent contemporary validation cohort. Available fields do not support a meaningful demographic fairness audit. We cannot estimate treatment benefit, prospective workflow effects, or performance in a screening population. A real application would require external validation, clinically justified thresholds, and prospective evaluation.

## 9  Reproducibility and submission

We will create one new GitHub repository for this final project, add all five members as collaborators, and add the instructor and teaching assistants if it is private. Each member will commit from their own account and submit an individual contribution statement. This proposal will be placed at the repository root as 707-proposal.md, and the repository URL will be submitted in Canvas.

The repository contains the original-source download script, README with attribution and license, data checks, shared nested-CV pipeline, evaluation code, figures, and a versioned environment file. Fixed seeds and shared configurations allow the analysis to be rerun from the source data.

## References

Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B.

Creative Commons. Attribution 4.0 International (CC BY 4.0). https://creativecommons.org/licenses/by/4.0/.
