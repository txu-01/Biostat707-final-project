"""
evaluate_all.py
Master evaluation and synthesis pipeline for BIOSTAT 707 Final Project.
Author: Tianyi Lu

Aggregates predictions from all models:
- Constant Baseline (Tianyi Lu)
- Logistic Regression & Elastic Net (Haiyi Xiong)
- Random Forest (Yan Yan)
- Decision Tree (Wenqing Zhao)
- Linear & RBF SVM (Asher Xu)

Strictly enforces:
1. Hyperparameters and decision thresholds (target >= 95% sensitivity) locked on development OOF.
2. One-shot test set evaluation with paired bootstrap 95% confidence intervals.
3. Translation of confusion matrices into missed malignant cases and unnecessary escalations.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from metrics import (
    evaluate_predictions,
    select_threshold_target_sensitivity,
    compute_calibration_curve,
    bootstrap_confidence_intervals,
    summarize_paired_differences,
)

OOF_REQUIRED_COLUMNS = {"id", "y_true", "model_name", "fold", "y_prob"}
TEST_REQUIRED_COLUMNS = {"id", "y_true", "model_name", "split", "y_prob"}
PAIRWISE_COLUMNS = [
    "model_name",
    "comparator",
    "auroc_difference",
    "auroc_difference_95_ci",
    "brier_difference",
    "brier_difference_95_ci",
]

MODEL_PALETTE = {
    "Constant_Baseline": "#7f7f7f",
    "Logistic_Regression": "#1f77b4",
    "Elastic_Net": "#aec7e8",
    "Decision_Tree": "#ff7f0e",
    "Random_Forest": "#2ca02c",
    "Linear_SVM": "#9467bd",
    "RBF_SVM": "#d62728"
}

def load_all_predictions(pred_dir="results/predictions", prefix="oof_"):
    """
    Loads all prediction files matching prefix ('oof_' or 'test_').
    Expected columns: id, y_true, model_name, y_prob (and fold or split).
    """
    files = glob.glob(os.path.join(pred_dir, f"{prefix}*.csv"))
    if not files:
        print(f"No prediction files found matching {pred_dir}/{prefix}*.csv")
        return pd.DataFrame()
        
    dfs = [pd.read_csv(f) for f in sorted(files)]
    combined = pd.concat(dfs, ignore_index=True)
    return combined


def validate_and_align_predictions(pred_df, splits_df, expected_split):
    """Validate prediction files and align every model to the official IDs.

    This prevents silent corruption of paired comparisons when a prediction
    file contains missing, duplicated, mislabeled, or out-of-order records.
    """
    if expected_split not in {"development", "test"}:
        raise ValueError("expected_split must be 'development' or 'test'.")

    required = (
        OOF_REQUIRED_COLUMNS
        if expected_split == "development"
        else TEST_REQUIRED_COLUMNS
    )
    missing_columns = sorted(required - set(pred_df.columns))
    if missing_columns:
        raise ValueError(
            f"{expected_split} predictions are missing required columns: {missing_columns}"
        )
    missing_model_names = pred_df["model_name"].isna()
    blank_model_names = pred_df["model_name"].astype(str).str.strip().eq("")
    if missing_model_names.any() or blank_model_names.any():
        raise ValueError("Every prediction row must have a non-empty model_name.")

    reference_columns = ["id", "target"]
    if expected_split == "development":
        reference_columns.append("cv_fold")
    reference = (
        splits_df.loc[splits_df["split"] == expected_split, reference_columns]
        .sort_values("id")
        .reset_index(drop=True)
    )
    if reference["id"].duplicated().any():
        raise ValueError("Official splits contain duplicate IDs.")

    expected_ids = set(reference["id"])
    aligned = {}
    for model_name, raw_sub in pred_df.groupby("model_name", sort=False):
        sub = raw_sub.copy()
        if sub["id"].isna().any() or sub["id"].duplicated().any():
            raise ValueError(f"{model_name}: IDs must be present and unique.")

        actual_ids = set(sub["id"])
        missing_ids = list(expected_ids - actual_ids)
        extra_ids = list(actual_ids - expected_ids)
        if missing_ids or extra_ids:
            raise ValueError(
                f"{model_name}: prediction IDs do not match the official {expected_split} set "
                f"(missing={missing_ids[:5]}, extra={extra_ids[:5]})."
            )

        sub["y_true"] = pd.to_numeric(sub["y_true"], errors="raise")
        sub["y_prob"] = pd.to_numeric(sub["y_prob"], errors="raise")
        if not np.isfinite(sub["y_prob"]).all() or not sub["y_prob"].between(0, 1).all():
            raise ValueError(f"{model_name}: y_prob must be finite and within [0, 1].")

        merged = reference.merge(sub, on="id", how="left", validate="one_to_one")
        if not np.array_equal(merged["target"].to_numpy(), merged["y_true"].to_numpy()):
            raise ValueError(f"{model_name}: y_true does not match the official target labels.")

        if expected_split == "development":
            merged["fold"] = pd.to_numeric(merged["fold"], errors="raise")
            if not np.array_equal(merged["cv_fold"].to_numpy(), merged["fold"].to_numpy()):
                raise ValueError(f"{model_name}: fold assignments do not match splits.csv.")
        elif not sub["split"].eq("test").all():
            raise ValueError(f"{model_name}: all final prediction rows must have split='test'.")

        aligned[str(model_name)] = merged

    if not aligned:
        raise ValueError(f"No valid {expected_split} model predictions were found.")
    return aligned

def plot_curves(models_dict, y_true, output_prefix="results/figures/"):
    """
    Plots ROC, PR, and Calibration curves for all available models.
    """
    os.makedirs(output_prefix, exist_ok=True)
    
    # 1. ROC Curves
    plt.figure(figsize=(7, 6))
    for name, probs in models_dict.items():
        color = MODEL_PALETTE.get(name, None)
        fpr, tpr, _ = roc_curve(y_true, probs)
        try:
            auc = roc_auc_score(y_true, probs)
        except ValueError:
            auc = 0.5
        plt.plot(fpr, tpr, label=f"{name.replace('_', ' ')} (AUROC = {auc:.3f})", lw=2, color=color)
        
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Chance / Baseline")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.title("ROC Curves Comparison", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_prefix, "roc_curves_comparison.png"), dpi=300)
    plt.close()
    
    # 2. Precision-Recall Curves
    plt.figure(figsize=(7, 6))
    baseline_prev = np.mean(y_true)
    for name, probs in models_dict.items():
        color = MODEL_PALETTE.get(name, None)
        precision, recall, _ = precision_recall_curve(y_true, probs)
        ap = average_precision_score(y_true, probs)
        plt.plot(recall, precision, label=f"{name.replace('_', ' ')} (AUPRC = {ap:.3f})", lw=2, color=color)
        
    plt.axhline(baseline_prev, color="k", linestyle="--", lw=1.5, alpha=0.7, label=f"Prevalence ({baseline_prev:.3f})")
    plt.xlabel("Recall (Sensitivity)", fontsize=11)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11)
    plt.title("Precision-Recall Curves Comparison", fontsize=12, fontweight="bold")
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_prefix, "pr_curves_comparison.png"), dpi=300)
    plt.close()
    
    # 3. Calibration Curves
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Perfect Calibration")
    for name, probs in models_dict.items():
        color = MODEL_PALETTE.get(name, None)
        cal = compute_calibration_curve(y_true, probs, n_bins=5)
        plt.plot(cal["bin_pred_mean"], cal["bin_true_prop"], "s-",
                 label=f"{name.replace('_', ' ')} (ECE = {cal['ece']:.3f})", lw=2, color=color)
                 
    plt.xlabel("Mean Predicted Probability (5 Quantile Bins)", fontsize=11)
    plt.ylabel("Observed Fraction of Malignant Cases", fontsize=11)
    plt.title("Calibration Curves Comparison (5 Equal-Frequency Bins)", fontsize=12, fontweight="bold")
    plt.legend(loc="upper left", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_prefix, "calibration_curves_comparison.png"), dpi=300)
    plt.close()

def plot_clinical_tradeoffs(records, output_path="results/figures/clinical_threshold_tradeoffs.png"):
    """
    Plots Missed Malignant Cases (FN) vs Unnecessary Escalations (FP) at selected thresholds.
    """
    df = pd.DataFrame(records)
    if df.empty:
        return
        
    plt.figure(figsize=(8, 5.5))
    for _, row in df.iterrows():
        name = row["model_name"]
        color = MODEL_PALETTE.get(name, "#333333")
        plt.scatter(row["unnecessary_escalations"], row["missed_malignant_cases"],
                    s=120, color=color, zorder=5, label=name.replace("_", " "))
        plt.annotate(
            f"{name.replace('_', ' ')}\n(Thresh={row['threshold']:.2f})",
            (row["unnecessary_escalations"] + 0.5, row["missed_malignant_cases"] + 0.1),
            fontsize=8
        )
        
    plt.xlabel("Unnecessary Escalations / Biopsies (False Positives)", fontsize=11)
    plt.ylabel("Missed Malignant Cases (False Negatives)", fontsize=11)
    plt.title("Clinical Decision Tradeoffs at >=95% Sensitivity Threshold", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def run_evaluation():
    tables_dir = "results/tables"
    figures_dir = "results/figures"
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    splits_df = pd.read_csv("data/processed/splits.csv")
    
    # -------------------------------------------------------------
    # 1. Process Development Out-of-Fold Predictions
    # -------------------------------------------------------------
    oof_df = load_all_predictions(prefix="oof_")
    locked_thresholds = {}
    oof_summary_rows = []
    
    if not oof_df.empty:
        aligned_oof = validate_and_align_predictions(oof_df, splits_df, "development")
        models = list(aligned_oof)
        print(f"Loaded and validated OOF predictions for {len(models)} model(s): {models}")
        
        models_prob_dict = {}
        first_y_true = None
        
        for m, sub in aligned_oof.items():
            y_true = sub["target"].values
            y_prob = sub["y_prob"].values
            if first_y_true is None:
                first_y_true = y_true
            models_prob_dict[m] = y_prob
            
            # Select threshold for >=95% sensitivity
            thresh, sens_achieved = select_threshold_target_sensitivity(y_true, y_prob, target_sens=0.95)
            locked_thresholds[m] = thresh
            
            res = evaluate_predictions(y_true, y_prob, fixed_threshold=thresh)
            t95 = res["threshold_target_95"]
            t05 = res["threshold_default_05"]
            
            oof_summary_rows.append({
                "model_name": m,
                "auroc": round(res["auroc"], 4),
                "auprc": round(res["auprc"], 4),
                "brier_score": round(res["brier_score"], 4),
                "log_loss": round(res["log_loss"], 4),
                "ece": round(res["ece"], 4),
                "locked_threshold_95": round(thresh, 4),
                "sens_at_locked": round(t95["sensitivity"], 4),
                "spec_at_locked": round(t95["specificity"], 4),
                "ppv_at_locked": round(t95["ppv"], 4),
                "npv_at_locked": round(t95["npv"], 4),
                "fn_missed_cancer_at_locked": t95["missed_malignant_cases"],
                "fp_unnecessary_escalation_at_locked": t95["unnecessary_escalations"],
                "sens_at_0.5": round(t05["sensitivity"], 4),
                "spec_at_0.5": round(t05["specificity"], 4)
            })
            
        oof_summary_df = pd.DataFrame(oof_summary_rows)
        oof_summary_df.to_csv(os.path.join(tables_dir, "oof_model_comparison.csv"), index=False)
        with open(os.path.join(tables_dir, "oof_model_comparison.md"), "w", encoding="utf-8") as f:
            f.write("# Development Set Out-of-Fold Model Comparison\n\n")
            f.write(oof_summary_df.to_markdown(index=False))
            f.write("\n\n*Thresholds selected exclusively on development OOF predictions targeting >=95% sensitivity.*\n")
            
        print(f"Saved OOF model comparison to {tables_dir}/oof_model_comparison.csv and .md")
        
        # Plot curves for development OOF
        if first_y_true is not None:
            plot_curves(models_prob_dict, first_y_true, output_prefix=figures_dir)
            
    # -------------------------------------------------------------
    # 2. Process Final Test Predictions (If available)
    # -------------------------------------------------------------
    test_df = load_all_predictions(prefix="test_")
    if not test_df.empty:
        aligned_test = validate_and_align_predictions(test_df, splits_df, "test")
        models = list(aligned_test)
        missing_oof = sorted(set(models) - set(locked_thresholds))
        if missing_oof:
            raise ValueError(
                "Final test predictions have no validated development OOF threshold for: "
                f"{missing_oof}"
            )
        print(f"\nLoaded and validated Test predictions for {len(models)} model(s): {models}")
        
        test_summary_rows = []
        test_tradeoff_records = []
        test_prob_dict = {}
        first_test_true = None
        
        for m, sub in aligned_test.items():
            y_true = sub["target"].values
            y_prob = sub["y_prob"].values
            if first_test_true is None:
                first_test_true = y_true
            test_prob_dict[m] = y_prob
            
            # Use threshold locked on development OOF
            locked_thresh = locked_thresholds[m]
            res = evaluate_predictions(y_true, y_prob, fixed_threshold=locked_thresh)
            t_lock = res["threshold_target_95"]
            t05 = res["threshold_default_05"]
            
            test_summary_rows.append({
                "model_name": m,
                "test_N": res["N"],
                "auroc": round(res["auroc"], 4),
                "auprc": round(res["auprc"], 4),
                "brier_score": round(res["brier_score"], 4),
                "log_loss": round(res["log_loss"], 4),
                "ece": round(res["ece"], 4),
                "applied_threshold": round(locked_thresh, 4),
                "test_sensitivity": round(t_lock["sensitivity"], 4),
                "test_specificity": round(t_lock["specificity"], 4),
                "test_ppv": round(t_lock["ppv"], 4),
                "test_npv": round(t_lock["npv"], 4),
                "missed_malignant_FN": t_lock["missed_malignant_cases"],
                "unnecessary_escalations_FP": t_lock["unnecessary_escalations"],
                "sens_at_0.5": round(t05["sensitivity"], 4),
                "spec_at_0.5": round(t05["specificity"], 4)
            })
            
            test_tradeoff_records.append({
                "model_name": m,
                "threshold": locked_thresh,
                "missed_malignant_cases": t_lock["missed_malignant_cases"],
                "unnecessary_escalations": t_lock["unnecessary_escalations"]
            })
            
        test_summary_df = pd.DataFrame(test_summary_rows)
        test_summary_df.to_csv(os.path.join(tables_dir, "test_model_comparison.csv"), index=False)
        with open(os.path.join(tables_dir, "test_model_comparison.md"), "w", encoding="utf-8") as f:
            f.write("# Final Test Set Model Comparison (One-Shot Held-Out Evaluation)\n\n")
            f.write(test_summary_df.to_markdown(index=False))
            f.write("\n\n*Note: Applied thresholds were locked strictly on development OOF predictions without seeing test labels.*\n")
            
        print(f"Saved Test model comparison to {tables_dir}/test_model_comparison.csv and .md")
        
        # Bootstrap 95% CIs
        if first_test_true is not None and len(test_prob_dict) > 0:
            print("Computing 2,000 paired, stratified bootstrap resamples for 95% CIs...")
            boot_summary, boot_stats = bootstrap_confidence_intervals(
                first_test_true, test_prob_dict, n_bootstraps=2000, seed=707
            )
            boot_rows = []
            for m, s in boot_summary.items():
                boot_rows.append({
                    "model_name": m,
                    "auroc_mean": round(s["auroc_mean"], 4),
                    "auroc_95_ci": f"[{s['auroc_ci_lower']:.4f}, {s['auroc_ci_upper']:.4f}]",
                    "brier_mean": round(s["brier_mean"], 4),
                    "brier_95_ci": f"[{s['brier_ci_lower']:.4f}, {s['brier_ci_upper']:.4f}]"
                })
            boot_df = pd.DataFrame(boot_rows)
            boot_df.to_csv(os.path.join(tables_dir, "test_bootstrap_confidence_intervals.csv"), index=False)
            print(f"Saved bootstrap CIs to {tables_dir}/test_bootstrap_confidence_intervals.csv")

            comparator = "Logistic_Regression" if "Logistic_Regression" in test_prob_dict else None
            paired_rows = summarize_paired_differences(boot_stats, comparator=comparator)
            formatted_rows = []
            for row in paired_rows:
                model_name = row["model_name"]
                comparator_name = row["comparator"]
                observed_auc_difference = (
                    roc_auc_score(first_test_true, test_prob_dict[model_name])
                    - roc_auc_score(first_test_true, test_prob_dict[comparator_name])
                )
                observed_brier_difference = (
                    brier_score_loss(first_test_true, test_prob_dict[model_name])
                    - brier_score_loss(first_test_true, test_prob_dict[comparator_name])
                )
                formatted_rows.append({
                    "model_name": model_name,
                    "comparator": comparator_name,
                    "auroc_difference": round(observed_auc_difference, 4),
                    "auroc_difference_95_ci": (
                        f"[{row['auroc_difference_ci_lower']:.4f}, "
                        f"{row['auroc_difference_ci_upper']:.4f}]"
                    ),
                    "brier_difference": round(observed_brier_difference, 4),
                    "brier_difference_95_ci": (
                        f"[{row['brier_difference_ci_lower']:.4f}, "
                        f"{row['brier_difference_ci_upper']:.4f}]"
                    ),
                })
            paired_df = pd.DataFrame(formatted_rows, columns=PAIRWISE_COLUMNS)
            paired_path = os.path.join(tables_dir, "test_bootstrap_pairwise_differences.csv")
            paired_df.to_csv(paired_path, index=False)
            paired_md_path = os.path.join(tables_dir, "test_bootstrap_pairwise_differences.md")
            if paired_df.empty:
                print("Pairwise bootstrap table is empty because only one test model is available.")
                with open(paired_md_path, "w", encoding="utf-8") as f:
                    f.write("# Paired Bootstrap Model Differences\n\n")
                    f.write(
                        "No paired difference is estimable yet because only one validated "
                        "test model is available.\n"
                    )
            else:
                direction_note = (
                    "Differences are model minus comparator; positive AUROC and negative Brier "
                    "favor the named model."
                )
                with open(paired_md_path, "w", encoding="utf-8") as f:
                    f.write("# Paired Bootstrap Model Differences\n\n")
                    f.write(paired_df.to_markdown(index=False))
                    f.write(f"\n\n*{direction_note}*\n")
            print(f"Saved paired bootstrap differences to {paired_path}")
            
        # Clinical tradeoff plot
        plot_clinical_tradeoffs(test_tradeoff_records)
        
    print("\n[SUCCESS] Master evaluation pipeline executed.")

if __name__ == "__main__":
    run_evaluation()
