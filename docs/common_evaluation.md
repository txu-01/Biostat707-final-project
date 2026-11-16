# Section 6: Common Evaluation Methodology, Baseline Benchmark, and Clinical Tradeoffs

**Author:** Tianyi Lu  
**Course:** BIOSTAT 707: Statistical Methods for Learning and Discovery  
**Project:** Breast Mass Classification Model Comparison

---

## 1. Unified Metric Suite

To avoid relying on an arbitrary threshold or an uncalibrated accuracy score, all models are evaluated across four complementary dimensions:

### 1.1 Discrimination
- **Area Under the Receiver Operating Characteristic (AUROC)**: Primary metric assessing the probability that a randomly chosen malignant sample receives a higher predicted risk than a randomly chosen benign sample.
- **Area Under the Precision-Recall Curve (AUPRC)**: Secondary discrimination metric sensitive to positive predictive value across recall levels, particularly relevant given the 37.4% malignancy prevalence.

### 1.2 Calibration and Probability Quality
- **Brier Score**: Quadratic scoring rule quantifying mean squared difference between predicted probability $\hat{p}_i \in [0, 1]$ and true outcome $y_i \in \{0, 1\}$:
  $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$$
- **Log Loss (Binary Cross-Entropy)**: Evaluates probabilistic likelihood with severe penalties for confident misclassifications.
- **Quantile Calibration Curves**: Assessed using 5 equal-frequency bins. Expected Calibration Error (ECE) summarizes the frequency-weighted absolute deviation between bin mean prediction and observed malignancy frequency.

### 1.3 Clinical Decision Thresholds
Rather than assuming a default cutoff of $p = 0.5$ (which implicitly assigns equal utility to false positives and false negatives), we select a **pre-specified exploratory operating point targeting $\ge 95\%$ sensitivity using predictions generated entirely inside each outer-training set**. This target is not an established clinical standard.
- Within every repeated nested-CV outer fold, the threshold is selected from inner out-of-fold predictions and transferred without modification to the corresponding outer validation fold.
- The original development-selected threshold is also transferred unchanged to the secondary held-out test analysis.
- Classification metrics include: True Positives (TP), False Positives (FP), False Negatives (FN), True Negatives (TN), Sensitivity, Specificity, Positive Predictive Value (PPV), and Negative Predictive Value (NPV).

### 1.4 Primary repeated nested cross-validation
- The primary comparison uses all 569 records in **10 repeats of stratified five-fold outer CV**. Hyperparameters are selected using five-fold inner CV. Every model receives the same outer splits.
- Repeat-level metrics pool the five outer validation folds so that each record contributes exactly once per repeat.
- Paired differences are computed within repeat against the stable ridge logistic comparator. Because repeats reuse the same records, their SD and range describe split sensitivity and are not treated as confidence intervals from ten independent samples.
- The original 114-record held-out analysis and its 2,000 paired class-stratified bootstrap resamples remain a secondary check. Those bootstrap intervals condition on one development/test split and do not capture training or model-selection variability.

---

## 2. Constant-Probability Baseline Benchmark

The constant baseline model predicts the malignancy proportion of the relevant training subset for every outer validation record. In the original development split this is:
$$\hat{p}_{\text{baseline}} = \bar{y}_{\text{train}} = \frac{170}{455} \approx 0.3736$$

### Theoretical and Empirical Performance:
- **AUROC**: Degenerate at 0.5000 (identically ranked predictions).
- **AUPRC**: Exactly matches cohort prevalence (0.3736).
- **Brier Score**: $0.3736 \times (1 - 0.3736) = 0.2340$.
- **Log Loss**: $- [0.3736 \ln(0.3736) + (1-0.3736) \ln(1-0.3736)] = 0.6609$.
- **Calibration ECE**: 0.0000 (perfect marginal calibration, but zero case-level sharpness).
- **Decision Threshold Tradeoff**:
  To achieve the exploratory $\ge 95\%$ sensitivity target, the constant baseline must set threshold $\le 0.3736$, thereby classifying **all 455 development records as positive**.
  - Achieved Sensitivity: 100.0% (0 missed cancers).
  - Achieved Specificity: 0.0% (285 unnecessary escalations among 285 benign lesions).

This baseline shows what patient-specific cytological features would need to improve upon: an informative classifier should reduce positive classifications among benign records while preserving high sensitivity.

---

## 3. Clinical Translation and Operating Dilemmas

In a hypothetical post-FNA review-prioritization setting, false negatives and false positives would have different potential consequences. The dataset alone does not establish clinical utility or a treatment pathway.

1. **False Negative (Missed Malignancy, $y=1, \hat{y}=0$)**:
   - Represents a malignant record assigned a probability below the selected threshold.
   - In a future validated workflow, this could correspond to a potentially missed or deprioritized malignant case.
   - The exploratory $\ge 95\%$ development-sensitivity target is intended to limit such errors, but it does not guarantee the same sensitivity on new data.

2. **False Positive (Unnecessary Escalation, $y=0, \hat{y}=1$)**:
   - Represents a benign record assigned a probability at or above the selected threshold. The dataset does not specify the benign subtype.
   - In a future validated workflow, this could correspond to a potentially unnecessary escalation for additional review or testing.
   - Excessive false-positive rates could reduce practical usefulness, but downstream actions and costs cannot be inferred from this dataset.

The primary counts are stored in `results/final_nested/repeat_metrics.csv`; the secondary held-out visualization remains in `results/figures/clinical_threshold_tradeoffs.png`. These counts support model comparison but should not be interpreted as demonstrated patient outcomes or clinical procedures.
