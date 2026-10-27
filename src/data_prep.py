"""
data_prep.py
Prepares clean dataset and generates fixed, stratified development/test and 5-fold CV splits.
BIOSTAT 707 Final Project.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

# Fixed project seed
RANDOM_SEED = 707

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

def generate_splits(df, seed=RANDOM_SEED):
    """
    Generate:
    - 80% development set / 20% test set (stratified by diagnosis, seed=707)
    - Stratified 5-fold CV on the development set (seed=707)
    """
    dev_idx, test_idx = train_test_split(
        df.index,
        test_size=0.20,
        random_state=seed,
        stratify=df["target"]
    )

    splits_df = df[["id", "diagnosis", "target"]].copy()
    splits_df["split"] = ""
    splits_df["cv_fold"] = -1

    splits_df.loc[dev_idx, "split"] = "development"
    splits_df.loc[test_idx, "split"] = "test"

    # Stratified 5-fold CV on development set
    dev_subset = splits_df.loc[dev_idx]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for fold, (_, val_pos) in enumerate(skf.split(dev_subset, dev_subset["target"])):
        val_orig_idx = dev_subset.index[val_pos]
        splits_df.loc[val_orig_idx, "cv_fold"] = fold

    return splits_df

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

    print("\nStep 2: Generating stratified development/test and 5-fold CV splits...")
    splits_df = generate_splits(clean_df, seed=RANDOM_SEED)
    splits_csv_path = os.path.join(processed_dir, "splits.csv")
    splits_df.to_csv(splits_csv_path, index=False)
    
    # Also save at data/splits.csv for convenience
    splits_df.to_csv("data/splits.csv", index=False)
    print(f"Saved splits to {splits_csv_path} and data/splits.csv")

    print("\nSplit verification:")
    split_summary = splits_df.groupby(["split", "diagnosis"]).size().unstack(fill_value=0)
    split_summary["Total"] = split_summary.sum(axis=1)
    split_summary["% Malignant"] = (split_summary["M"] / split_summary["Total"] * 100).round(2)
    print(split_summary)

    print("\nDevelopment 5-Fold CV verification:")
    dev_folds = splits_df[splits_df["split"] == "development"].groupby(["cv_fold", "diagnosis"]).size().unstack()
    dev_folds["Total"] = dev_folds.sum(axis=1)
    print(dev_folds)

    print("\nStep 3: Generating data dictionary...")
    generate_data_dictionary(clean_df)

    print("\n[SUCCESS] Data preparation and shared split generation complete.")

if __name__ == "__main__":
    run_pipeline()
