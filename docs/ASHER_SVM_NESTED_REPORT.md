# Asher Xu Repeated Nested CV Supplement

## Methods

Both SVMs were compared on all 569 WDBC records using 10 repeats of stratified five-fold outer CV. Records were sorted by ID. Outer seeds were 707–716. Within each outer training set, both SVMs used the same stratified five-fold inner CV (seed 1707 + 5 × zero-based repeat + zero-based outer fold). The existing parameter grids and the training-only three-fold sigmoid calibration were retained. The scaler sits inside the calibrated base estimator. Parameters maximize mean inner-fold AUROC. The highest threshold achieving at least 95% sensitivity on selected inner OOF probabilities was locked before prediction on each outer validation fold.

## Results

Mean ± SD across the 10 repeat estimates. Each repeat is an unweighted mean of its five outer-fold metrics. AUPRC is computed as average precision.

| Outer validation metric | Linear SVM | RBF SVM |
|---|---:|---:|
| auroc | 0.9940 ± 0.0015 | 0.9951 ± 0.0010 |
| auprc | 0.9928 ± 0.0023 | 0.9943 ± 0.0010 |
| brier_score | 0.0225 ± 0.0018 | 0.0205 ± 0.0017 |
| log_loss | 0.0881 ± 0.0084 | 0.0781 ± 0.0047 |
| ece | 0.0211 ± 0.0052 | 0.0190 ± 0.0033 |
| sensitivity | 0.9496 ± 0.0096 | 0.9449 ± 0.0093 |
| specificity | 0.9871 ± 0.0050 | 0.9894 ± 0.0032 |
| ppv | 0.9783 ± 0.0081 | 0.9820 ± 0.0057 |
| npv | 0.9711 ± 0.0053 | 0.9685 ± 0.0053 |

Paired AUROC differences (RBF minus linear): mean +0.00115, range -0.00080 to +0.00408 across repeats. This is a descriptive range, not a confidence interval. Repeats share the same records and do not provide ten independent datasets.

The original 80/20 results and their paired bootstrap intervals remain a secondary check. Because the full-cohort repeated CV includes the original test records, the two analyses are not independent confirmations. The original holdout settings and predictions were not changed. This analysis compares only the two SVMs and cannot answer whether either beats logistic regression.

## Reproduction and handoff

Run python scripts/run_svm_nested.py, then python scripts/summarize_asher_nested.py. Outer split IDs, inner search, outer predictions, fold metrics, calibration bins, repeat summaries and paired differences are under results/svm/nested. The outer split file can be shared, but the team must confirm a common protocol before combining other model families. Keep these files separate from the original 80/20 evaluator inputs.
