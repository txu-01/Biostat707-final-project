# BIOSTAT 707 Presentation Outline

## Slide 1 — Research question and data

- Question: do nonlinear classifiers improve on logistic regression, and does higher AUROC mean fewer missed malignant cases?
- 569 breast-mass records: 212 malignant and 357 benign.
- Thirty nuclear-image predictors: mean, standard error (SE), and worst summaries of ten measurements.
- Intended application: hypothetical case-review prioritization after fine-needle aspiration, not prediction of future cancer onset or population screening.
- Include the project title on this slide; do not add a separate title slide.
- Suggested visual: a simple data overview highlighting the record count, malignant count, and number of predictors.

## Slide 2 — Models and validation

- Models: constant-probability baseline, unpenalized logistic regression, ridge logistic regression, Elastic Net, decision tree, random forest, linear SVM, and RBF SVM.
- Ten repeats of stratified five-fold outer CV use all 569 records. All models share the same outer folds.
- Five-fold inner CV selects hyperparameters; outer CV evaluates held-out predictions.
- Scaling, SVM probability calibration, and threshold selection remain inside training data to prevent leakage.
- Select the highest threshold achieving at least 95% sensitivity on training-only inner out-of-fold probabilities, then apply it unchanged to outer validation.
- Suggested visual: one diagram showing inner tuning inside outer evaluation.

## Slide 3 — Main results and error tradeoffs

- Compare all models using `results/final_nested/figures/primary_auroc.png`.
- RBF SVM had the highest mean pooled-repeat AUROC (0.9943), followed closely by ridge logistic regression (0.9936).
- The paired RBF-minus-ridge AUROC difference averaged approximately +0.0007, with a repeat range from -0.0003 to +0.0014. RBF did not outperform ridge in every repeat.
- RBF had the lowest mean Brier score (0.0205), compared with 0.0214 for ridge; lower is better.
- At training-selected thresholds, RBF averaged 11.7 false negatives and 3.8 false positives per repeat; ridge averaged 10.7 and 4.0.
- Error counts are means among 569 validation predictions per repeat. Achieved validation sensitivities differ, so this is not a comparison at exactly matched sensitivity.
- Take-home message: the highest AUROC does not necessarily mean fewer missed malignant cases. Do not claim a substantial or statistically established advantage from these small differences.
- Suggested visual: one AUROC comparison figure plus a small ridge-versus-RBF error table.

## Slide 4 — Sample size and feature representation

- Learning curves compare training sizes 50, 100, 200, and 455. Decision tree performance improved substantially with more records; regularized linear models and SVMs already performed well at smaller sizes.
- The learning curves use one outer repeat and settings summarized from the primary analysis. Present them as exploratory post-selection descriptions, not fully independent nested learning-curve estimates.
- In the separate Elastic Net feature-group analysis, worst-only predictors achieved mean AUROC 0.9906, close to 0.9910 for all 30 predictors. SE-only predictors were weaker (0.9503).
- Worst-only predictors also had a lower mean Brier score than all predictors (0.0224 versus 0.0243).
- These feature-group findings apply to the Elastic Net sensitivity analysis and should not be generalized to every model. Its all-feature result comes from a separate run from the primary Elastic Net comparison.
- Suggested visuals: `results/final_nested/figures/learning_curves.png` and `results/final_nested/figures/feature_set_comparison.png` as two compact panels.

## Slide 5 — Conclusions, limitations, and AI use

- Main answer: RBF SVM led numerically, but stable regularized linear models remained highly competitive. More complex models did not show a clear, consistent overall advantage.
- Model choice should consider probability quality, false negatives, false positives, and interpretability alongside AUROC.
- Main limitations: a small historical dataset, correlated predictors, unresolved patient independence, and no external validation. Repeated CV does not create new independent cases.
- The 95% training sensitivity target does not guarantee 95% sensitivity in new records, and clinical utility has not been established.
- Briefly disclose actual AI uses. Each member should supply their own genuine assessment of benefits, limitations, workload effects, and effects on understanding for the final slide and speaker notes.
- Suggested layout: three short conclusions and two main limitations, with a concise AI-use disclosure.

## Q&A preparation

- Why use repeated nested CV for a small dataset?
- How were scaling, calibration, tuning, and threshold selection kept inside training data?
- Why did the highest-AUROC model not have the fewest false negatives?
- Why are the learning curves exploratory, and what does the feature-group comparison add?
- Why does a small numerical AUROC advantage not establish clinical or statistical superiority?
