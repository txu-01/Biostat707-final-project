# WDBC Dataset Data Dictionary

Source: UCI Machine Learning Repository — Breast Cancer Wisconsin (Diagnostic)
DOI: 10.24432/C5DW2B | License: CC BY 4.0

| Column Name | Type | Description | Values / Range |
|---|---|---|---|
| `id` | Integer | Unique specimen identifier | Unique per record (569 total) |
| `diagnosis` | Categorical | Clinical cytology diagnosis | 'M' (Malignant), 'B' (Benign) |
| `target` | Binary | Model target indicator | 1 = Malignant (positive), 0 = Benign (negative) |
| `radius_mean` | Float | Mean of distances from center to points on the contour perimeter (Mean value) | [6.9810, 28.1100] |
| `texture_mean` | Float | Standard deviation of gray-scale values (Mean value) | [9.7100, 39.2800] |
| `perimeter_mean` | Float | Nuclear contour perimeter (Mean value) | [43.7900, 188.5000] |
| `area_mean` | Float | Nuclear contour area (Mean value) | [143.5000, 2501.0000] |
| `smoothness_mean` | Float | Local variation in radius lengths (Mean value) | [0.0526, 0.1634] |
| `compactness_mean` | Float | Perimeter^2 / area - 1.0 (Mean value) | [0.0194, 0.3454] |
| `concavity_mean` | Float | Severity of concave portions of the contour (Mean value) | [0.0000, 0.4268] |
| `concave_points_mean` | Float | Number of concave portions of the contour (Mean value) | [0.0000, 0.2012] |
| `symmetry_mean` | Float | Nuclear symmetry measurement (Mean value) | [0.1060, 0.3040] |
| `fractal_dimension_mean` | Float | 'Coastline approximation' - 1 (Mean value) | [0.0500, 0.0974] |
| `radius_se` | Float | Mean of distances from center to points on the contour perimeter (Standard error) | [0.1115, 2.8730] |
| `texture_se` | Float | Standard deviation of gray-scale values (Standard error) | [0.3602, 4.8850] |
| `perimeter_se` | Float | Nuclear contour perimeter (Standard error) | [0.7570, 21.9800] |
| `area_se` | Float | Nuclear contour area (Standard error) | [6.8020, 542.2000] |
| `smoothness_se` | Float | Local variation in radius lengths (Standard error) | [0.0017, 0.0311] |
| `compactness_se` | Float | Perimeter^2 / area - 1.0 (Standard error) | [0.0023, 0.1354] |
| `concavity_se` | Float | Severity of concave portions of the contour (Standard error) | [0.0000, 0.3960] |
| `concave_points_se` | Float | Number of concave portions of the contour (Standard error) | [0.0000, 0.0528] |
| `symmetry_se` | Float | Nuclear symmetry measurement (Standard error) | [0.0079, 0.0790] |
| `fractal_dimension_se` | Float | 'Coastline approximation' - 1 (Standard error) | [0.0009, 0.0298] |
| `radius_worst` | Float | Mean of distances from center to points on the contour perimeter (Largest ('worst') value (mean of 3 largest)) | [7.9300, 36.0400] |
| `texture_worst` | Float | Standard deviation of gray-scale values (Largest ('worst') value (mean of 3 largest)) | [12.0200, 49.5400] |
| `perimeter_worst` | Float | Nuclear contour perimeter (Largest ('worst') value (mean of 3 largest)) | [50.4100, 251.2000] |
| `area_worst` | Float | Nuclear contour area (Largest ('worst') value (mean of 3 largest)) | [185.2000, 4254.0000] |
| `smoothness_worst` | Float | Local variation in radius lengths (Largest ('worst') value (mean of 3 largest)) | [0.0712, 0.2226] |
| `compactness_worst` | Float | Perimeter^2 / area - 1.0 (Largest ('worst') value (mean of 3 largest)) | [0.0273, 1.0580] |
| `concavity_worst` | Float | Severity of concave portions of the contour (Largest ('worst') value (mean of 3 largest)) | [0.0000, 1.2520] |
| `concave_points_worst` | Float | Number of concave portions of the contour (Largest ('worst') value (mean of 3 largest)) | [0.0000, 0.2910] |
| `symmetry_worst` | Float | Nuclear symmetry measurement (Largest ('worst') value (mean of 3 largest)) | [0.1565, 0.6638] |
| `fractal_dimension_worst` | Float | 'Coastline approximation' - 1 (Largest ('worst') value (mean of 3 largest)) | [0.0550, 0.2075] |
