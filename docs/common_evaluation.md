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
Rather than assuming a default cutoff of $p = 0.5$ (which implicitly assigns equal utility to false positives and false negatives), we select a **pre-specified exploratory operating point targeting $\ge 95\%$ sensitivity on development out-of-fold predictions**. This target is not an established clinical standard.
- Once determined on development OOF, this numerical threshold is **strictly locked** and transferred without modification to the held-out test cohort.
- Classification metrics evaluated at this threshold include: True Positives (TP), False Positives (FP), False Negatives (FN), True Negatives (TN), Sensitivity, Specificity, Positive Predictive Value (PPV), and Negative Predictive Value (NPV).

### 1.4 Statistical Uncertainty
- To account for sampling variability without inflating confidence, we employ **2,000 paired, class-stratified bootstrap resamples** of the held-out evaluation set to estimate 95% percentile confidence intervals for AUROC, Brier score, and paired model contrasts. When the full-feature logistic model is available, it is the prespecified comparator; before other model predictions arrive, no pairwise contrast can be estimated.

---

## 2. Constant-Probability Baseline Benchmark

The constant baseline model predicts the marginal training-set malignancy proportion for every record:
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

Our unified evaluation pipeline visualizes counts of malignant records below the threshold and benign records above it (`results/figures/clinical_threshold_tradeoffs.png`). These counts support model comparison but should not be interpreted as demonstrated patient outcomes or clinical procedures.
