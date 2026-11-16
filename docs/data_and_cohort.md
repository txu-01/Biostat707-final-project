# Data, Cohort, and Validation Design

## Data source

We used the Breast Cancer Wisconsin (Diagnostic) dataset from the UCI Machine Learning Repository (DOI: [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B), CC BY 4.0). It contains measurements from digitized fine-needle aspirate images of breast masses.

The dataset has 569 records: 212 malignant and 357 benign. For each record, ten nuclear characteristics are summarized by their mean, standard error (SE), and worst value, giving 30 continuous predictors. Malignancy is coded as 1 and benign status as 0. Record ID is kept only for alignment and is never used as a predictor.

## Data checks

The committed audit and cleaning scripts confirmed:

- 569 rows and 569 unique record IDs;
- 212 malignant and 357 benign labels;
- no missing or nonfinite predictor values;
- no exact duplicate rows; and
- 30 predictors in the expected UCI order.

Unique IDs do not prove that all rows represent unrelated patients because patient, site, and time identifiers are unavailable. This remains a limitation.

## Final validation design

All 569 records enter the same repeated nested cross-validation analysis. The outer loop uses ten repeats of stratified five-fold CV. The inner loop uses stratified five-fold CV within each outer-training set to select hyperparameters. All models use identical outer folds.

Standardization, SVM probability calibration, hyperparameter selection, and the exploratory 95% sensitivity threshold are fitted using training data only. The selected model and threshold are then evaluated on the corresponding outer validation fold. Pooling the five outer folds gives one out-of-fold prediction per record, model, and repeat.

This design avoids basing the conclusion on one random partition and makes the model comparisons paired at the record and repeat levels. Repeated folds still reuse the same records, so their standard deviation describes sensitivity to splitting rather than uncertainty from independent datasets.

## Additional analyses

Learning curves evaluate training sizes of 50, 100, 200, and 455 using the common outer folds. A separate Elastic Net comparison uses the mean, SE, worst, and all-feature representations. These analyses address sample-size behavior and the value of different feature summaries without changing the primary model-comparison protocol.
