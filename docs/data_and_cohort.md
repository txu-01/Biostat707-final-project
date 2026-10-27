# Section 2 & 3: Data Source, Cohort Characterization, and Validation Design

**Author:** Tianyi Lu  
**Course:** BIOSTAT 707: Statistical Methods for Learning and Discovery  
**Project:** Breast Mass Classification Model Comparison

---

## 1. Dataset Origin and Attribution

We utilize the Breast Cancer Wisconsin (Diagnostic) dataset (WDBC), contributed by Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian (1993, 1995) at the University of Wisconsin. The dataset is archived at the UCI Machine Learning Repository under DOI [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B) and licensed under the Creative Commons Attribution 4.0 International license (CC BY 4.0).

### Clinical Provenance and Image Acquisition
The dataset comprises 569 fine-needle aspirate (FNA) specimens of human breast masses. Cytological smears were digitized, and nuclear boundaries of individual epithelial cell nuclei were traced using interactive curve-fitting algorithms (XCONF software). For each digitized image, ten morphological characteristics of cell nuclei were measured:
1. **Radius**: Mean distance from nuclear center to contour boundary points.
2. **Texture**: Standard deviation of gray-scale intensity values.
3. **Perimeter**: Nuclear contour length.
4. **Area**: Nuclear cross-sectional area.
5. **Smoothness**: Local variation in radius lengths.
6. **Compactness**: $\frac{\text{perimeter}^2}{\text{area}} - 1.0$.
7. **Concavity**: Severity and depth of concave contour indentations.
8. **Concave points**: Number of concave indentations along the contour.
9. **Symmetry**: Relative length of nuclear perimeter chords.
10. **Fractal dimension**: Coastline boundary approximation minus 1.

For each specimen, the mean, standard error (SE), and "worst" (mean of the three largest values across all nuclei in the image) were computed for each of the 10 characteristics, yielding exactly 30 real-valued continuous predictors.

---

## 2. Completed Data Feasibility and Integrity Audit

Prior to model development, an automated Python data audit (`scripts/download_and_check_data.py`) was executed on the raw data file (`data/raw/wdbc.data`). The audit verified:
- **Total records**: 569 instances.
- **Diagnosis labels**: 357 benign (62.74%) and 212 malignant (37.26%).
- **Columns**: 32 columns (unique record ID, categorical diagnosis, and 30 continuous features).
- **Missing / non-finite values**: Exactly 0 missing cells, 0 invalid numeric values (`NaN` or `Inf`).
- **Uniqueness**: 569 unique record IDs; 0 duplicate records.
- **SHA-256 Checksum**: `D606AF411F3E5BE8A317A5A8B652B425AAF0FF38CA683D5327FFFF94C3695F4A`.

Target encoding was designated as Malignant = 1 (positive class) and Benign = 0 (negative class). The specimen ID column is strictly preserved for tracking prediction records and is never provided as an input feature to any model.

---

## 3. Strict Validation Design and Leakage Prevention

To reduce data-leakage risk and preserve an untouched final evaluation set:
1. **Stratified Development/Test Partition**:
   - A single, locked 80% / 20% stratified split was performed with `random_state = 707`.
   - **Development set**: 455 records (285 benign, 170 malignant; prevalence = 37.36%).
   - **Test set**: 114 records (72 benign, 42 malignant; prevalence = 36.84%).
   - The test set is completely sequestered: no exploratory data analysis, transformation fitting, hyperparameter selection, or threshold tuning has access to the test records.
2. **Stratified 5-Fold Cross-Validation on Development Cohort**:
   - Within the 455 development samples, a 5-fold cross-validation scheme was partitioned with `random_state = 707`.
   - Each fold contains exactly 91 samples: 57 benign and 34 malignant cases, preserving the exact 37.36% marginal prevalence across every fold.
   - All feature standardizations (`StandardScaler`) and SVM probability calibration pipelines are required to fit strictly on the 4 training folds ($\sim 364$ samples) and transform the validation fold ($\sim 91$ samples) out-of-fold.
   - Record identifiers and their assigned splits and folds are persisted in `data/processed/splits.csv` for uniform consumption by all modeling pipelines.

---

## 4. Development Cohort Baseline Characteristics (Table 1)

All exploratory analysis was conducted strictly within the 455 development records. Table 1 summarizes the distribution of all 30 features across diagnosis classes, evaluated using non-parametric two-sided Mann-Whitney U tests:

- **Nuclear Size Metrics**: Malignant nuclei exhibit significantly larger dimensions across mean, SE, and worst summaries ($p < 0.001$). For example, `area_mean` in malignant cases has a median of 929.85 [IQR: 712.93, 1200.75] in the dataset's recorded units, compared with 458.40 [IQR: 380.30, 551.10] in benign cases. The UCI documentation does not specify a physical measurement unit for these recoded feature values.
- **Nuclear Irregularity**: Markers of architectural disorder—`concave_points_mean`, `concavity_mean`, and `compactness_mean`—display marked elevation in malignant lesions ($p < 0.001$). Malignant `concave_points_mean` has a median of 0.087 [IQR: 0.065, 0.105], nearly four times that of benign lesions (median 0.023 [IQR: 0.015, 0.033]).
- **Severe Collinearity**: Correlation analysis (`results/figures/feature_correlation_heatmap.png`) reveals very strong collinearity between `radius`, `perimeter`, and `area` (Pearson $r$ approximately 0.978–0.998 across the mean and worst subsets). This geometric dependency motivates evaluation of regularized regression (Elastic Net) and tree-based ensemble methods for stability.
