# Team Modeling and Evaluation Workflow Guide

**BIOSTAT 707 Final Project**  
**Author / Data Lead:** Tianyi Lu (`tianyilu666-netizen`)  
**Target Audience:** Haiyi Xiong (Logistic Regression / Elastic Net), Yan Yan (Random Forest), Wenqing Zhao (Decision Tree), Asher Xu (SVM)

---

## 1. Quick Start: Loading Data & Splits

All models must train on the identical split and 5-fold cross-validation folds. Do **not** create your own train/test split or shuffle seeds.

```python
import pandas as pd

# Load clean features and target
df = pd.read_csv("data/processed/wdbc_clean.csv")

# Load unified splits
splits = pd.read_csv("data/processed/splits.csv")

# Merge splits with data
data = df.merge(splits[["id", "split", "cv_fold"]], on="id")

# 1. Development set (N=455) for training & CV tuning
dev_data = data[data["split"] == "development"].copy()

# 2. Test set (N=114) — LOCKED until hyperparameter tuning & threshold selection are complete
test_data = data[data["split"] == "test"].copy()

feature_cols = [c for c in df.columns if c not in ["id", "diagnosis", "target"]]
```

---

## 2. 5-Fold Cross-Validation Workflow

Train your model inside each fold ($k \in \{0, 1, 2, 3, 4\}$). Preprocessing (e.g., `StandardScaler`, Platt calibration) **must be fitted only on the training fold**.

```python
oof_predictions = []

for fold in range(5):
    train_fold = dev_data[dev_data["cv_fold"] != fold]
    val_fold = dev_data[dev_data["cv_fold"] == fold]
    
    X_train, y_train = train_fold[feature_cols], train_fold["target"]
    X_val, y_val = val_fold[feature_cols], val_fold["target"]
    
    # Example: Fit your model / pipeline here
    # pipeline.fit(X_train, y_train)
    # val_probs = pipeline.predict_proba(X_val)[:, 1]
    
    for rec_id, y_true, prob in zip(val_fold["id"], y_val, val_probs):
        oof_predictions.append({
            "id": rec_id,
            "y_true": y_true,
            "model_name": "YOUR_MODEL_NAME",  # e.g., 'Random_Forest'
            "fold": fold,
            "y_prob": prob
        })

oof_df = pd.DataFrame(oof_predictions)
oof_df.to_csv("results/predictions/oof_YOUR_MODEL_NAME.csv", index=False)
```

---

## 3. Threshold Selection on Development OOF

After generating all 455 out-of-fold predictions, lock your decision threshold targeting $\ge 95\%$ sensitivity:

```python
import sys
sys.path.append(".")
from src.metrics import select_threshold_target_sensitivity, evaluate_predictions

threshold, achieved_sens = select_threshold_target_sensitivity(
    oof_df["y_true"], oof_df["y_prob"], target_sens=0.95
)
print(f"Locked threshold: {threshold:.4f} (Achieved OOF Sensitivity: {achieved_sens:.4f})")
```

---

## 4. Final Held-Out Test Prediction (One-Shot)

Once your hyperparameters and decision threshold are finalized:
1. Retrain your final tuned pipeline on the **entire development set** (`dev_data`).
2. Generate predicted probabilities on the **test set** (`test_data`).
3. Save predictions to `results/predictions/test_YOUR_MODEL_NAME.csv`:

```python
test_pred_df = pd.DataFrame({
    "id": test_data["id"],
    "y_true": test_data["target"],
    "model_name": "YOUR_MODEL_NAME",
    "split": "test",
    "y_prob": test_probs
})
test_pred_df.to_csv("results/predictions/test_YOUR_MODEL_NAME.csv", index=False)
```

---

## 5. Master Evaluation

Once your prediction files are saved in `results/predictions/`, run:
```bash
python3 src/evaluate_all.py
```
This automatically updates the comparison tables in `results/tables/` and comparative curves in `results/figures/`.

Before evaluation, the script now validates every prediction file against
`data/processed/splits.csv`. It stops with an error if a model has missing or
duplicate IDs, incorrect labels or folds, non-finite/out-of-range probabilities,
or test predictions without a development-OOF threshold. Do not bypass these
checks or manually reorder rows.

When two or more test models are available, the evaluation also writes paired
bootstrap differences to
`results/tables/test_bootstrap_pairwise_differences.csv`. If
`Logistic_Regression` is available it is used as the prespecified comparator;
otherwise all available pairs are reported. Differences are defined as model
minus comparator, so positive AUROC differences and negative Brier-score
differences favor the named model.

---

## 6. Assigned Models and Required Names

Use these exact `model_name` values in prediction files so that the master
evaluation script can recognize and compare models consistently.

| Owner | Model | Required `model_name` | Prespecified search |
|---|---|---|---|
| Haiyi Xiong | Unpenalized logistic regression | `Logistic_Regression` | Full-feature logistic regression; document convergence and separation checks |
| Haiyi Xiong | Elastic Net logistic regression | `Elastic_Net` | `C` in `{0.01, 0.1, 1, 10}` and `l1_ratio` in `{0.1, 0.5, 0.9, 1.0}` |
| Yan Yan | Random forest | `Random_Forest` | 500 trees; `max_depth` in `{3, None}`, `min_samples_leaf` in `{1, 3, 5, 10}`, and `max_features` in `{sqrt, 0.5}` |
| Wenqing Zhao | Decision tree | `Decision_Tree` | `max_depth` in `{3, 5, 7, None}` and `min_samples_leaf` in `{5, 10, 20}` |
| Asher Xu | Linear SVM | `Linear_SVM` | `C` in `{0.01, 0.1, 1, 10}` |
| Asher Xu | RBF SVM | `RBF_SVM` | Linear-SVM `C` grid crossed with `gamma` in `{0.001, 0.01, 0.1, 1}` |

For logistic regression and SVM models, fit standardization only inside the
current training fold. SVM probability calibration must also remain inside the
training pipeline and must not see the validation fold used to score a
candidate configuration.

---

## 7. Test-Set Lock and Required Deliverables

The held-out test set is not permanently forbidden; it is embargoed until the
model specification, hyperparameters, preprocessing, and development-OOF
threshold are locked. After those decisions are final:

1. Refit the locked pipeline once on all 455 development records.
2. Generate probabilities once for all 114 test records without further
   tuning, feature selection, threshold adjustment, or recalibration.
3. Submit all of the following:
   - executable model-training code;
   - selected hyperparameters and development-CV selection results;
   - runtime summary;
   - `results/predictions/oof_<MODEL_NAME>.csv` with exactly 455 unique IDs;
   - `results/predictions/test_<MODEL_NAME>.csv` with exactly 114 unique IDs;
   - the locked development-OOF threshold;
   - model-specific methods, results, interpretation, and limitations text.

Each model owner is responsible for testing and explaining their own model.
The data/evaluation lead will validate and combine standardized outputs, not
retune or repair every model.

---

## 8. GitHub Handoff

After the shared infrastructure PR is merged, update the team repository with:

```bash
git switch main
git pull origin main
python3 -m pip install -r requirements.txt
```

Create a feature branch for your assigned model, commit from your own GitHub
account, push the branch, and open a pull request to the team repository. Do
not push generated changes directly over another member's branch. Include the
model name, completed checks, output filenames, and any unresolved issue in the
pull-request description.
