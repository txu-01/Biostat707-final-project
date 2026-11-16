"""
data_prep.py
Prepares the clean full-cohort dataset used by repeated nested cross-validation.
BIOSTAT 707 Final Project.
"""

import os
import pandas as pd

BASE_FEATURES = [
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave_points",
    "symmetry",
    "fractal_dimension"
]

FEATURE_NAMES = (
    [f"{feat}_mean" for feat in BASE_FEATURES] +
    [f"{feat}_se" for feat in BASE_FEATURES] +
    [f"{feat}_worst" for feat in BASE_FEATURES]
)

ALL_COLUMNS = ["id", "diagnosis"] + FEATURE_NAMES

def load_and_clean_raw(raw_path="data/raw/wdbc.data"):
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at {raw_path}")

    df = pd.read_csv(raw_path, header=None, names=ALL_COLUMNS)
    
    # Target encoding: Malignant (M) = 1, Benign (B) = 0
    df["target"] = (df["diagnosis"] == "M").astype(int)

    # Reorder columns: id, diagnosis, target, followed by 30 features
    ordered_cols = ["id", "diagnosis", "target"] + FEATURE_NAMES
    df = df[ordered_cols]
    
    return df

def generate_data_dictionary(df, output_path="data/processed/data_dictionary.md"):
    lines = [
        "# WDBC Dataset Data Dictionary",
        "",
        "Source: UCI Machine Learning Repository — Breast Cancer Wisconsin (Diagnostic)",
        "DOI: 10.24432/C5DW2B | License: CC BY 4.0",
        "",
        "| Column Name | Type | Description | Values / Range |",
        "|---|---|---|---|"
    ]
    lines.append("| `id` | Integer | Unique specimen identifier | Unique per record (569 total) |")
    lines.append("| `diagnosis` | Categorical | Clinical cytology diagnosis | 'M' (Malignant), 'B' (Benign) |")
    lines.append("| `target` | Binary | Model target indicator | 1 = Malignant (positive), 0 = Benign (negative) |")
    
    feature_desc = {
        "radius": "Mean of distances from center to points on the contour perimeter",
        "texture": "Standard deviation of gray-scale values",
        "perimeter": "Nuclear contour perimeter",
        "area": "Nuclear contour area",
        "smoothness": "Local variation in radius lengths",
        "compactness": "Perimeter^2 / area - 1.0",
        "concavity": "Severity of concave portions of the contour",
        "concave_points": "Number of concave portions of the contour",
        "symmetry": "Nuclear symmetry measurement",
        "fractal_dimension": "'Coastline approximation' - 1"
    }

    for col in FEATURE_NAMES:
        base = col.rsplit("_", 1)[0]
        stat = col.rsplit("_", 1)[1]
        stat_label = {"mean": "Mean value", "se": "Standard error", "worst": "Largest ('worst') value (mean of 3 largest)"}[stat]
        desc = f"{feature_desc[base]} ({stat_label})"
        col_min = df[col].min()
        col_max = df[col].max()
        lines.append(f"| `{col}` | Float | {desc} | [{col_min:.4f}, {col_max:.4f}] |")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Data dictionary written to {output_path}")

def run_pipeline():
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    print("Step 1: Loading and formatting raw data...")
    clean_df = load_and_clean_raw()
    clean_csv_path = os.path.join(processed_dir, "wdbc_clean.csv")
    clean_df.to_csv(clean_csv_path, index=False)
    print(f"Saved clean dataset to {clean_csv_path} (shape: {clean_df.shape})")

    print("\nStep 2: Verifying the full analysis cohort...")
    if len(clean_df) != 569 or clean_df["id"].duplicated().any():
        raise ValueError("Expected 569 rows with unique record IDs.")
    if clean_df["target"].value_counts().to_dict() != {0: 357, 1: 212}:
        raise ValueError("Unexpected target counts.")
    if not clean_df[FEATURE_NAMES].notna().all().all():
        raise ValueError("Predictors contain missing values.")
    print("Verified 569 records: 357 benign and 212 malignant.")

    print("\nStep 3: Generating data dictionary...")
    generate_data_dictionary(clean_df)

    print("\n[SUCCESS] Full-cohort data preparation complete.")

if __name__ == "__main__":
    run_pipeline()
