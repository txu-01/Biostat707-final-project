# Common Evaluation Method

## Metrics

We report AUROC as the main discrimination metric and AUPRC as a prevalence-sensitive summary of precision and recall. Probability quality is assessed with Brier score, log loss, and five-bin expected calibration error (ECE).

At an exploratory operating point, each model selects the highest threshold that achieves at least 95% sensitivity in training-only out-of-fold predictions. That threshold is transferred unchanged to the relevant outer validation fold. We then report sensitivity, specificity, positive predictive value, negative predictive value, and the false-negative and false-positive counts. The 95% target is not an established clinical standard and does not guarantee 95% validation sensitivity.

## Repeated nested cross-validation

The comparison uses ten repeats of stratified five-fold outer CV on all 569 records. Hyperparameters are chosen by mean AUROC in five-fold inner CV. Preprocessing and SVM calibration occur inside the fitting data. Identical outer folds are used for all eight models.

For each repeat, the five outer validation folds are pooled so every record contributes exactly once. Model differences are paired by repeat against ridge logistic regression, the stable full-feature logistic comparator. Because the ten repeats contain the same records, their SD and range describe sensitivity to the random fold assignments rather than ten independent estimates of population uncertainty.

## Constant-probability baseline

The baseline predicts the malignancy proportion in each outer-training set. It has no patient-specific discrimination, so its AUROC is approximately 0.5 and its AUPRC is close to the cohort prevalence. At a threshold low enough to target 95% training sensitivity, it labels every validation record positive, giving 100% sensitivity and 0% specificity. This provides a useful reference for whether the fitted models add patient-level information.

## Interpretation

False negatives are malignant records below the training-selected threshold, while false positives are benign records above it. These counts describe classification behavior in this dataset; they are not demonstrated clinical outcomes. Near-ceiling AUROC also means that very small numerical differences should be interpreted alongside calibration, complexity, and stability rather than as proof that one model is clinically superior.

The complete repeat-level metrics are stored in `results/final_nested/repeat_metrics.csv`, and the combined model summary is in `results/final_nested/model_summary.csv`.
