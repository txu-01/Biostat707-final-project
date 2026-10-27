"""
metrics.py
Shared evaluation module for BIOSTAT 707 Final Project.
Includes:
- Discrimination: AUROC, AUPRC
- Calibration: Brier score, Log Loss, 5-bin calibration curve, ECE, Mean Pred vs Observed
- Decision thresholds: Training/OOF-selected threshold targeting >=95% sensitivity
- Classification metrics: Confusion matrix (TP, FP, FN, TN), Sensitivity, Specificity, PPV, NPV
- Clinical translation: Missed malignant cases (FN), unnecessary escalations (FP)
- Uncertainty: 2,000 paired, class-stratified bootstrap resamples for 95% CIs
"""

from itertools import combinations

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
)

def select_threshold_target_sensitivity(y_true, y_prob, target_sens=0.95):
    """
    Selects the highest threshold that achieves >= target_sens on development/OOF predictions.
    Avoids using test labels!
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Candidate thresholds: sorted unique predicted probabilities
    candidates = np.sort(np.unique(y_prob))
    best_thresh = 0.0
    best_sens = 0.0

    for thresh in candidates:
        y_pred = (y_prob >= thresh).astype(int)
        # Sensitivity = TP / (TP + FN)
        positives = (y_true == 1)
        if np.sum(positives) == 0:
            continue
        sens = np.sum((y_pred == 1) & positives) / np.sum(positives)
        if sens >= target_sens:
            if thresh >= best_thresh:
                best_thresh = float(thresh)
                best_sens = float(sens)
        else:
            # Since thresholds are sorted ascending, once sens drops below target, higher thresholds will too
            break

    # If even threshold 0 does not satisfy, fallback to 0.0
    return best_thresh, best_sens

def compute_classification_at_threshold(y_true, y_prob, threshold):
    """
    Computes confusion matrix, sensitivity, specificity, PPV, NPV at a fixed threshold.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "missed_malignant_cases": int(fn),
        "unnecessary_escalations": int(fp)
    }

def compute_calibration_curve(y_true, y_prob, n_bins=5):
    """
    Computes quantile-based (equal-frequency) calibration curve using ~5 bins.
    Returns bin midpoints, observed frequencies, and bin counts.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    # Use quantile bins to ensure equal frequency
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(y_prob, quantiles)
    bin_edges = np.unique(bin_edges)
    
    if len(bin_edges) < 2:
        return {
            "bin_pred_mean": [float(np.mean(y_prob))],
            "bin_true_prop": [float(np.mean(y_true))],
            "bin_counts": [len(y_prob)],
            "ece": float(abs(np.mean(y_prob) - np.mean(y_true)))
        }

    bin_pred_mean = []
    bin_true_prop = []
    bin_counts = []
    ece = 0.0

    for i in range(len(bin_edges) - 1):
        low, high = bin_edges[i], bin_edges[i+1]
        if i == len(bin_edges) - 2:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)
            
        n_in_bin = np.sum(mask)
        if n_in_bin > 0:
            pred_m = np.mean(y_prob[mask])
            true_m = np.mean(y_true[mask])
            bin_pred_mean.append(float(pred_m))
            bin_true_prop.append(float(true_m))
            bin_counts.append(int(n_in_bin))
            ece += (n_in_bin / len(y_prob)) * abs(pred_m - true_m)

    return {
        "bin_pred_mean": bin_pred_mean,
        "bin_true_prop": bin_true_prop,
        "bin_counts": bin_counts,
        "ece": float(ece)
    }

def evaluate_predictions(y_true, y_prob, fixed_threshold=None, target_sens=0.95):
    """
    Comprehensive evaluation of predictions on a dataset.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # 1. Discrimination
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auroc = 0.5  # Degenerate case (e.g. constant baseline)

    auprc = float(average_precision_score(y_true, y_prob))

    # 2. Probability calibration & loss
    # Clip prob to prevent log loss explosion
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    brier = float(brier_score_loss(y_true, y_prob))
    logloss = float(log_loss(y_true, y_prob_clipped))
    mean_pred = float(np.mean(y_prob))
    observed_prevalence = float(np.mean(y_true))
    calib = compute_calibration_curve(y_true, y_prob, n_bins=5)

    # 3. Decision threshold evaluation
    if fixed_threshold is None:
        thresh, achieved_sens = select_threshold_target_sensitivity(y_true, y_prob, target_sens=target_sens)
    else:
        thresh = fixed_threshold

    perf_thresh = compute_classification_at_threshold(y_true, y_prob, thresh)
    perf_half = compute_classification_at_threshold(y_true, y_prob, 0.5)

    return {
        "N": len(y_true),
        "prevalence": observed_prevalence,
        "mean_predicted": mean_pred,
        "auroc": auroc,
        "auprc": auprc,
        "brier_score": brier,
        "log_loss": logloss,
        "ece": calib["ece"],
        "calibration_curve": calib,
        "threshold_target_95": perf_thresh,
        "threshold_default_05": perf_half
    }

def bootstrap_confidence_intervals(y_true, y_prob_dict, n_bootstraps=2000, seed=707):
    """
    Computes 2,000 paired, class-stratified bootstrap resamples of records.
    Estimates 95% CIs for AUROC and Brier score for each model. The same
    class-stratified record indices are used for every model so that
    ``boot_stats`` can be passed to :func:`summarize_paired_differences`.
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    
    models = list(y_prob_dict.keys())
    boot_stats = {m: {"auroc": [], "brier": []} for m in models}
    
    for _ in range(n_bootstraps):
        # Stratified resample
        boot_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        boot_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        boot_idx = np.concatenate([boot_pos, boot_neg])
        
        y_boot_true = y_true[boot_idx]
        
        for m in models:
            y_boot_prob = np.asarray(y_prob_dict[m])[boot_idx]
            try:
                auc = roc_auc_score(y_boot_true, y_boot_prob)
            except ValueError:
                auc = 0.5
            br = brier_score_loss(y_boot_true, y_boot_prob)
            boot_stats[m]["auroc"].append(auc)
            boot_stats[m]["brier"].append(br)

    summary = {}
    for m in models:
        auc_arr = np.array(boot_stats[m]["auroc"])
        br_arr = np.array(boot_stats[m]["brier"])
        
        summary[m] = {
            "auroc_mean": float(np.mean(auc_arr)),
            "auroc_ci_lower": float(np.percentile(auc_arr, 2.5)),
            "auroc_ci_upper": float(np.percentile(auc_arr, 97.5)),
            "brier_mean": float(np.mean(br_arr)),
            "brier_ci_lower": float(np.percentile(br_arr, 2.5)),
            "brier_ci_upper": float(np.percentile(br_arr, 97.5)),
        }
        
    return summary, boot_stats


def summarize_paired_differences(boot_stats, comparator=None):
    """Summarize paired bootstrap differences between model performances.

    Parameters
    ----------
    boot_stats : dict
        Per-model bootstrap samples returned by
        :func:`bootstrap_confidence_intervals`.
    comparator : str or None
        When supplied, compare every other model against this comparator.
        Otherwise, report every unordered pair.

    Returns
    -------
    list of dict
        Difference direction is ``model - comparator``. Positive AUROC
        differences and negative Brier differences favor ``model``.
    """
    models = list(boot_stats)
    if comparator is not None:
        if comparator not in boot_stats:
            raise ValueError(f"Comparator {comparator!r} is not present in bootstrap results.")
        pairs = [(model, comparator) for model in models if model != comparator]
    else:
        pairs = list(combinations(models, 2))

    rows = []
    for model, reference in pairs:
        model_auc = np.asarray(boot_stats[model]["auroc"], dtype=float)
        ref_auc = np.asarray(boot_stats[reference]["auroc"], dtype=float)
        model_brier = np.asarray(boot_stats[model]["brier"], dtype=float)
        ref_brier = np.asarray(boot_stats[reference]["brier"], dtype=float)

        if not (
            len(model_auc) == len(ref_auc) == len(model_brier) == len(ref_brier)
        ):
            raise ValueError("Paired bootstrap arrays must have identical lengths.")

        auc_diff = model_auc - ref_auc
        brier_diff = model_brier - ref_brier
        rows.append({
            "model_name": model,
            "comparator": reference,
            "auroc_difference_mean": float(np.mean(auc_diff)),
            "auroc_difference_ci_lower": float(np.percentile(auc_diff, 2.5)),
            "auroc_difference_ci_upper": float(np.percentile(auc_diff, 97.5)),
            "brier_difference_mean": float(np.mean(brier_diff)),
            "brier_difference_ci_lower": float(np.percentile(brier_diff, 2.5)),
            "brier_difference_ci_upper": float(np.percentile(brier_diff, 97.5)),
        })

    return rows
