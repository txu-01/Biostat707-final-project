# BIOSTAT 707 Presentation Outline (7.5 minutes)

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
