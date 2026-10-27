"""
baseline.py
Implements and evaluates the Constant-Probability Baseline Model for BIOSTAT 707.
Author: Tianyi Lu

The constant baseline predicts the marginal training malignancy proportion.
It serves as the reference benchmark: all patient-specific classifiers must
demonstrate substantive improvement over this non-informative model.
"""

import os
import pandas as pd
import numpy as np
from metrics import evaluate_predictions

class ConstantProbabilityBaseline:
    def __init__(self):
        self.marginal_prob_ = None
        
    def fit(self, y_train):
        y_arr = np.asarray(y_train)
        self.marginal_prob_ = float(np.mean(y_arr))
        return self
        
    def predict_proba(self, X):
        n_samples = len(X)
        return np.full(n_samples, self.marginal_prob_)
        
    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

def run_baseline():
    os.makedirs("results/predictions", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)
    
    clean_df = pd.read_csv("data/processed/wdbc_clean.csv")
    splits_df = pd.read_csv("data/processed/splits.csv")
    merged = clean_df.merge(splits_df[["id", "split", "cv_fold"]], on="id")
    
    # -------------------------------------------------------------
    # 1. 5-Fold Cross-Validation on Development Set (Out-of-Fold)
    # -------------------------------------------------------------
    dev_df = merged[merged["split"] == "development"].copy()
    oof_predictions = []
    
    for fold in range(5):
        train_fold = dev_df[dev_df["cv_fold"] != fold]
        val_fold = dev_df[dev_df["cv_fold"] == fold]
        
        baseline = ConstantProbabilityBaseline()
        baseline.fit(train_fold["target"])
        val_probs = baseline.predict_proba(val_fold)
        
        for rec_id, y_true, prob in zip(val_fold["id"], val_fold["target"], val_probs):
            oof_predictions.append({
                "id": rec_id,
                "y_true": y_true,
                "model_name": "Constant_Baseline",
                "fold": fold,
                "y_prob": prob
            })
            
    oof_df = pd.DataFrame(oof_predictions)
    oof_csv_path = "results/predictions/oof_baseline.csv"
    oof_df.to_csv(oof_csv_path, index=False)
    print(f"Saved baseline OOF predictions to {oof_csv_path}")
    
    # Evaluate OOF performance
    dev_eval = evaluate_predictions(oof_df["y_true"], oof_df["y_prob"], target_sens=0.95)
    print("\n--- Baseline Development OOF Metrics ---")
    print(f"Marginal Prevalence:  {dev_eval['prevalence']:.4f}")
    print(f"Mean Predicted Prob:  {dev_eval['mean_predicted']:.4f}")
    print(f"AUROC:                {dev_eval['auroc']:.4f}")
    print(f"AUPRC:                {dev_eval['auprc']:.4f}")
    print(f"Brier Score:          {dev_eval['brier_score']:.4f}")
    print(f"Log Loss:             {dev_eval['log_loss']:.4f}")
    print(f"Calibration ECE:      {dev_eval['ece']:.4f}")
    print(f"Selected Threshold (>=95% sens): {dev_eval['threshold_target_95']['threshold']:.4f}")
    print(f"Achieved Sens / Spec: {dev_eval['threshold_target_95']['sensitivity']:.4f} / {dev_eval['threshold_target_95']['specificity']:.4f}")
    print(f"Missed Malignant (FN) / Unnecessary Escalations (FP): {dev_eval['threshold_target_95']['missed_malignant_cases']} / {dev_eval['threshold_target_95']['unnecessary_escalations']}")

    # -------------------------------------------------------------
    # 2. Test Set Prediction (Conditioned on Full Development Set)
    # -------------------------------------------------------------
    test_df = merged[merged["split"] == "test"].copy()
    full_dev_baseline = ConstantProbabilityBaseline()
    full_dev_baseline.fit(dev_df["target"])
    test_probs = full_dev_baseline.predict_proba(test_df)
    
    test_pred_df = pd.DataFrame({
        "id": test_df["id"],
        "y_true": test_df["target"],
        "model_name": "Constant_Baseline",
        "split": "test",
        "y_prob": test_probs
    })
    test_csv_path = "results/predictions/test_baseline.csv"
    test_pred_df.to_csv(test_csv_path, index=False)
    print(f"\nSaved baseline test predictions to {test_csv_path}")
    
    # Save baseline metrics summary
    baseline_metrics = {
        "model_name": "Constant Baseline",
        "split": "development_oof",
        "auroc": dev_eval["auroc"],
        "auprc": dev_eval["auprc"],
        "brier_score": dev_eval["brier_score"],
        "log_loss": dev_eval["log_loss"],
        "ece": dev_eval["ece"],
        "threshold": dev_eval["threshold_target_95"]["threshold"],
        "sensitivity": dev_eval["threshold_target_95"]["sensitivity"],
        "specificity": dev_eval["threshold_target_95"]["specificity"],
        "ppv": dev_eval["threshold_target_95"]["ppv"],
        "npv": dev_eval["threshold_target_95"]["npv"],
        "missed_malignant_FN": dev_eval["threshold_target_95"]["missed_malignant_cases"],
        "unnecessary_escalation_FP": dev_eval["threshold_target_95"]["unnecessary_escalations"]
    }
    pd.DataFrame([baseline_metrics]).to_csv("results/tables/baseline_metrics.csv", index=False)
    print("Saved baseline metrics to results/tables/baseline_metrics.csv")
    print("\n[SUCCESS] Constant baseline model complete.")

if __name__ == "__main__":
    run_baseline()
