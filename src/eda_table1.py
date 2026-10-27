"""
eda_table1.py
Performs Exploratory Data Analysis (EDA) and generates Table 1 baseline summary.
CRITICAL LEAKAGE-PREVENTION RULE:
All EDA and descriptive statistics are performed STRICTLY on the development subset (455 records).
BIOSTAT 707 Final Project.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def compute_table_1(dev_df, feature_cols):
    """
    Computes baseline cohort characteristics:
    - Overall (N=455)
    - Benign (N=285)
    - Malignant (N=170)
    - Mean (SD) and Median [Q1, Q3]
    - Two-sample Wilcoxon rank-sum / Mann-Whitney U test p-value
    """
    table_rows = []
    
    benign_df = dev_df[dev_df["target"] == 0]
    malignant_df = dev_df[dev_df["target"] == 1]
    
    n_total = len(dev_df)
    n_benign = len(benign_df)
    n_malignant = len(malignant_df)
    
    for feat in feature_cols:
        all_vals = dev_df[feat]
        b_vals = benign_df[feat]
        m_vals = malignant_df[feat]
        
        # Stats
        all_mean, all_sd = all_vals.mean(), all_vals.std()
        all_med, all_q1, all_q3 = all_vals.median(), all_vals.quantile(0.25), all_vals.quantile(0.75)
        
        b_mean, b_sd = b_vals.mean(), b_vals.std()
        b_med, b_q1, b_q3 = b_vals.median(), b_vals.quantile(0.25), b_vals.quantile(0.75)
        
        m_mean, m_sd = m_vals.mean(), m_vals.std()
        m_med, m_q1, m_q3 = m_vals.median(), m_vals.quantile(0.25), m_vals.quantile(0.75)
        
        # Mann-Whitney U test (non-parametric comparison)
        _, p_val = stats.mannwhitneyu(b_vals, m_vals, alternative="two-sided")
        
        table_rows.append({
            "Feature": feat,
            "Overall Mean (SD)": f"{all_mean:.3f} ({all_sd:.3f})",
            "Overall Median [IQR]": f"{all_med:.3f} [{all_q1:.3f}, {all_q3:.3f}]",
            f"Benign (N={n_benign}) Mean (SD)": f"{b_mean:.3f} ({b_sd:.3f})",
            f"Benign Median [IQR]": f"{b_med:.3f} [{b_q1:.3f}, {b_q3:.3f}]",
            f"Malignant (N={n_malignant}) Mean (SD)": f"{m_mean:.3f} ({m_sd:.3f})",
            f"Malignant Median [IQR]": f"{m_med:.3f} [{m_q1:.3f}, {m_q3:.3f}]",
            "p-value (MW U)": "<0.001" if p_val < 0.001 else f"{p_val:.4f}"
        })
        
    return pd.DataFrame(table_rows)

def plot_class_balance(dev_df, output_path="results/figures/class_balance.png"):
    counts = dev_df["diagnosis"].value_counts()
    percentages = (counts / len(dev_df) * 100).round(1)
    
    plt.figure(figsize=(6, 4.5))
    bars = plt.bar(["Benign (0)", "Malignant (1)"], [counts["B"], counts["M"]], color=["#2b5c8f", "#d95f02"], width=0.5)
    plt.ylabel("Number of Samples (Development Set)", fontsize=11)
    plt.title("Class Balance in BIOSTAT 707 Development Set (N=455)", fontsize=12, fontweight="bold")
    plt.ylim(0, 340)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    for bar, pct in zip(bars, [percentages["B"], percentages["M"]]):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 6,
                 f"{int(height)} ({pct}%)", ha="center", va="bottom", fontsize=11, fontweight="bold")
                 
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved class balance plot to {output_path}")

def plot_correlation_heatmap(dev_df, feature_cols, output_path="results/figures/feature_correlation_heatmap.png"):
    corr = dev_df[feature_cols].corr()
    
    plt.figure(figsize=(12, 10))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Pearson Correlation Coefficient", fraction=0.046, pad=0.04)
    plt.xticks(range(len(feature_cols)), feature_cols, rotation=90, fontsize=8)
    plt.yticks(range(len(feature_cols)), feature_cols, fontsize=8)
    plt.title("Correlation Matrix of 30 Nuclear Features (Development Set N=455)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved correlation heatmap to {output_path}")

def plot_top_features_boxplot(dev_df, output_path="results/figures/top_features_boxplots.png"):
    # Select top discriminating features across nuclear characteristics
    top_features = [
        "concave_points_mean", "perimeter_worst", "radius_worst",
        "area_worst", "concavity_mean", "texture_worst"
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    for i, feat in enumerate(top_features):
        ax = axes[i]
        b_data = dev_df[dev_df["target"] == 0][feat]
        m_data = dev_df[dev_df["target"] == 1][feat]
        
        box = ax.boxplot([b_data, m_data], tick_labels=["Benign", "Malignant"], patch_artist=True)
        box['boxes'][0].set_facecolor("#a6bddb")
        box['boxes'][1].set_facecolor("#fc9272")
        
        ax.set_title(feat, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        
    plt.suptitle("Distribution of Key Nuclear Features by Diagnosis (Development Set N=455)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved top features boxplots to {output_path}")

def run_eda():
    figures_dir = "results/figures"
    tables_dir = "results/tables"
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    
    clean_path = "data/processed/wdbc_clean.csv"
    splits_path = "data/processed/splits.csv"
    
    data_df = pd.read_csv(clean_path)
    splits_df = pd.read_csv(splits_path)
    
    # Merge split info
    merged_df = data_df.merge(splits_df[["id", "split", "cv_fold"]], on="id")
    
    # Filter STRICTLY to development set
    dev_df = merged_df[merged_df["split"] == "development"].copy()
    feature_cols = [c for c in data_df.columns if c not in ["id", "diagnosis", "target", "split", "cv_fold"]]
    
    print(f"Running EDA on Development cohort: N={len(dev_df)} (Malignant={sum(dev_df['target']==1)}, Benign={sum(dev_df['target']==0)})")
    
    # 1. Class balance plot
    plot_class_balance(dev_df)
    
    # 2. Correlation heatmap
    plot_correlation_heatmap(dev_df, feature_cols)
    
    # 3. Top features boxplots
    plot_top_features_boxplot(dev_df)
    
    # 4. Table 1
    table1 = compute_table_1(dev_df, feature_cols)
    table1_csv = os.path.join(tables_dir, "table1_cohort_summary.csv")
    table1_md = os.path.join(tables_dir, "table1_cohort_summary.md")
    
    table1.to_csv(table1_csv, index=False)
    with open(table1_md, "w", encoding="utf-8") as f:
        f.write("# Table 1: Baseline Characteristics of Development Cohort (N=455)\n\n")
        f.write(table1.to_markdown(index=False))
        f.write("\n\n*Note: All values computed on development set only to prevent data leakage. P-values from two-sided Mann-Whitney U test.*\n")
        
    print(f"Saved Table 1 to {table1_csv} and {table1_md}")
    print("\n[SUCCESS] Development set EDA and Table 1 generated successfully.")

if __name__ == "__main__":
    run_eda()
