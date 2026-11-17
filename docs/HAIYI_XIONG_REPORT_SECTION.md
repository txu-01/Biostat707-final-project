# Logistic regression and Elastic Net

Haiyi Xiong · BIOSTAT 707 final project

## Methods

We compared unpenalized logistic regression, ridge logistic regression with the prespecified setting `C = 1`, and Elastic Net logistic regression. The analysis included 569 Wisconsin Diagnostic Breast Cancer records, of which 212 were malignant and 357 were benign. The three models used the same 30 features derived from nuclear images. These features summarize ten measurements by their mean, standard error (SE), and worst value. Malignancy was the positive class. We used record IDs to match the supplied fold assignments and excluded IDs and diagnosis labels from the predictor matrix.

The team's final analysis used the supplied ten repeats of stratified five-fold outer cross-validation. Each record received one validation prediction per repeat per model. Standardization was fitted inside each training split. For Elastic Net, the shared pipeline selected the highest mean five-fold inner AUROC over all 16 combinations of `C ∈ {0.01, 0.1, 1, 10}` and `l1_ratio ∈ {0.1, 0.5, 0.9, 1.0}`. The ratio of 1 corresponds to an L1-only penalty. The selected pipeline was refitted on the entire outer-training fold. For each outer fold, the highest threshold giving at least 95% sensitivity on inner out-of-fold predictions was applied unchanged to outer validation. Outer sensitivities and specificities summarize 50 thresholds per model. The five validation folds were pooled within each repeat before computing the primary metrics. [Shared analysis code](../src/final_analysis.py)

We examined the team's convergence records and measured the share of outer predictions within 10⁻⁶ of 0 or 1. The nested Elastic Net feature comparison used the same outer folds for the 10 mean, 10 SE, 10 worst, and all 30 predictors. The team's descriptive learning curves used the five folds of repeat 1 at training sizes 50, 100, 200, and 455. These curves used each model's most frequently selected settings from the primary analysis and estimated the threshold within each sampled training subset.

## Primary results

AUROC is reported as the mean ± SD across ten repeats. Other values are repeat means. FN and FP are the mean counts among 569 validation predictions per repeat.

| Model | AUROC | AUPRC | Brier | Log loss | Sensitivity | Specificity | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unpenalized logistic | 0.9624 ± 0.0037 | 0.9174 | 0.0486 | 1.5896 | 94.10% | 95.55% | 12.5 | 15.9 |
| Ridge logistic | 0.9936 ± 0.0006 | 0.9924 | 0.0214 | 0.0829 | 94.95% | 98.88% | 10.7 | 4.0 |
| Elastic Net | 0.9919 ± 0.0025 | 0.9911 | 0.0241 | 0.0968 | 95.00% | 99.02% | 10.6 | 3.5 |

Ridge had the best mean AUROC, Brier score, and log loss among the three logistic models. The paired Elastic Net minus ridge AUROC difference averaged −0.00167 and ranged from −0.00473 to +0.00122. At the training-selected operating point, ridge and Elastic Net had similar sensitivity and missed about 11 malignant records per repeat. Elastic Net averaged 3.5 false positives per repeat, and ridge averaged 4.0. The most frequently selected Elastic Net setting was `C = 0.1, l1_ratio = 0.1` in 20 of 50 outer fits.

The median recorded time per outer fold was 0.023 seconds for unpenalized logistic, 0.022 seconds for ridge, and 5.362 seconds for Elastic Net. These timings include inner fitting, the outer refit, and validation scoring in the team's environment. Elastic Net evaluated 16 candidate settings per fold. [Elastic Net candidate timings](../results/final_nested/inner_search_Elastic_Net.csv)

The observed malignancy prevalence was 37.26%. Mean predicted probabilities were 37.97% for unpenalized logistic, 37.13% for ridge, and 37.04% for Elastic Net. Mean expected calibration errors across the ten repeats were 0.0405, 0.0038, and 0.0160, respectively. The reliability plot displays the first outer repeat using up to five quantile bins, with tied boundaries combined. [Team model summary](../results/final_nested/model_summary.csv) · [Calibration plot](../results/final_nested/figures/haiyi_logistic_calibration.png) · [Calibration bins](../results/final_nested/haiyi_logistic_calibration_bins.csv)

## Convergence and probability stability

The selected inner models and outer refits recorded zero solver convergence warnings. Yet 99.46% of unpenalized validation probabilities were within 10⁻⁶ of zero or one, compared with 11.00% for ridge and 6.59% for Elastic Net. The near-boundary probabilities, large standardized coefficients documented by the team, and poor log loss are consistent with separation or near separation. The unpenalized fit is retained as an unstable comparator. [Team report](FINAL_REPORT_DRAFT.md) · [Unpenalized fold diagnostics](../results/final_nested/outer_fold_metrics_Logistic_Regression.csv)

## Coefficient direction and stability

We refitted each of the 50 outer-training folds per model using the archived training IDs, selected hyperparameters, and outer-fit seeds. Each standardized coefficient measures the change in fitted log odds associated with a one-training-SD increase in that feature, holding the other features fixed. We recorded the median, interquartile range, and counts of positive, negative, and zero coefficients across folds. Outer-validation outcomes played no role in fitting these coefficients.

| Model | Predictors with both positive and negative coefficients across folds | Median largest absolute coefficient per fold |
| --- | ---: | ---: |
| Unpenalized logistic | 25 of 30 | 1,424.99 |
| Ridge logistic | 9 of 30 | 1.29 |
| Elastic Net | 6 of 30 | 1.30 |

Worst texture, worst radius, and worst area had positive coefficients in all 50 folds of both penalized models. In Elastic Net, their median standardized coefficients were 0.98, 0.94, and 0.87, respectively. Radius SE was positive in every ridge fold and in 49 of 50 Elastic Net folds, with the remaining estimate shrunk to zero. Elastic Net retained a median of 26.5 nonzero coefficients across folds. The shared directions for worst radius and worst area are consistent with a role for nuclear size in the model scores. Their related measurements make individual coefficients sensitive to how size is represented.

The coefficient refits used the team's pinned NumPy, pandas, SciPy, and scikit-learn versions. All 5,690 refit probabilities per penalized model agreed with the archived predictions within an absolute difference of 10⁻⁷. In the unpenalized model, 5,687 of 5,690 probabilities met that tolerance and the largest difference was 1.00, further illustrating the instability of that fit. The primary performance table uses the team's archived predictions. [Fold coefficients](../results/final_nested/haiyi_logistic_coefficients_by_fold.csv) · [Stability table](../results/final_nested/haiyi_logistic_coefficient_stability.csv) · [Refit verification](../results/final_nested/haiyi_logistic_coefficient_replay.json)

## Sensitivity analyses

| Elastic Net predictors | AUROC, mean ± SD | Mean Brier score |
| --- | ---: | ---: |
| Mean (10) | 0.9806 ± 0.0031 | 0.0492 |
| SE (10) | 0.9503 ± 0.0034 | 0.0803 |
| Worst (10) | 0.9906 ± 0.0009 | 0.0224 |
| All (30) | 0.9910 ± 0.0029 | 0.0243 |

The ten worst-value predictors achieved nearly the same mean AUROC as all 30, with a lower mean Brier score. The paired all-minus-worst AUROC difference averaged +0.00038 and ranged from −0.00489 to +0.00349 over the ten repeats. This small and variable gap leaves the estimated discrimination of the two feature sets close.

Mean outer AUROC increased from 0.9911 to 0.9932 for ridge and from 0.9893 to 0.9938 for Elastic Net between training sizes 50 and 455. Unpenalized logistic regression fell from 0.9865 to 0.9579. These means come from five folds of one repeat with overlapping training data. Since the hyperparameter settings were summarized from the primary analysis, the curves describe performance at those settings within this cohort. [Team learning curves](../results/final_nested/learning_curve_summary.csv) · [Team feature-set results](../results/final_nested/feature_set_summary.csv)

The learning-curve bars span the observed range across the five outer folds of repeat 1. Feature-set bars show SD across the ten repeats. Each point in the Brier-score panel represents a repeat, and the diamond marks its model mean.

## Limitations and interpretation

Standardized coefficients describe conditional associations within the model. Radius, perimeter, and area encode related measurements, and mean and worst values may also be correlated. Penalties can redistribute coefficients among these predictors, so individual weights and signs are sensitive to the feature representation. The ten primary repeats reuse the same records, so their SD and paired ranges describe split variability. The exploratory training sensitivity target and the observed validation error counts describe behavior in this dataset. The models require external validation before any clinical use.

Ridge provides a stable linear reference for the team's comparison with more complex classifiers. A modest improvement in AUROC would have to be weighed against any changes in probability accuracy and errors at the chosen operating point.

[Shared fold assignments](https://github.com/txu-01/Biostat707-final-project/blob/asher/svm-analysis/results/svm/nested/outer_splits.csv) · [Team evaluation protocol](common_evaluation.md) · [Summary figure](../results/final_nested/figures/haiyi_logistic_summary.png)
