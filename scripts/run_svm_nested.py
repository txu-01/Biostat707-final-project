"""Asher-only repeated nested CV. Run from repository root with Python."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
handles = []
dlls = Path(sys.prefix) / "Library" / "bin"
if os.name == "nt" and dlls.is_dir():
    handles.append(os.add_dll_directory(str(dlls)))
    os.environ["PATH"] = str(dlls) + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import roc_auc_score
from train_svm import GRIDS, make_model, FEATURE_NAMES
from metrics import evaluate_predictions, select_threshold_target_sensitivity

def main():
    out = ROOT / "results/svm/nested"
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / "data/processed/wdbc_clean.csv").sort_values("id").reset_index(drop=True)
    X, y = data[FEATURE_NAMES], data.target.to_numpy()
    assert len(data) == 569 and data.id.is_unique and y.sum() == 212
    predictions, metrics, splits, searches, calibration = [], [], [], [], []
    for repeat in range(10):
        outer = StratifiedKFold(5, shuffle=True, random_state=707 + repeat)
        for fold, (tr, va) in enumerate(outer.split(X, y)):
            assert not np.intersect1d(tr, va).size
            inner = list(StratifiedKFold(5, shuffle=True, random_state=1707 + 5*repeat + fold).split(X.iloc[tr], y[tr]))
            splits.extend(dict(repeat=repeat+1, fold=fold, id=data.id.iloc[i]) for i in va)
            for name, grid in GRIDS.items():
                best = None
                for params in ParameterGrid(grid):
                    oof = np.full(len(tr), np.nan)
                    scores = []
                    for a,b in inner:
                        assert not np.intersect1d(a,b).size
                        clf = make_model(name, params).fit(X.iloc[tr[a]], y[tr[a]])
                        oof[b] = clf.predict_proba(X.iloc[tr[b]])[:,1]
                        scores.append(roc_auc_score(y[tr[b]], oof[b]))
                    score = float(np.mean(scores))
                    searches.append(dict(repeat=repeat+1, fold=fold, model=name, **params, inner_auroc=score))
                    if best is None or score > best[0] + 1e-12:
                        best = (score, params, oof)
                score, params, oof = best
                assert np.isfinite(oof).all()
                threshold, sens = select_threshold_target_sensitivity(y[tr], oof, .95)
                assert sens >= .95
                prob = make_model(name, params).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:,1]
                assert np.isfinite(prob).all() and ((0 <= prob) & (prob <= 1)).all()
                m = evaluate_predictions(y[va], prob, fixed_threshold=threshold)
                key = dict(repeat=repeat+1, fold=fold, model=name)
                t = m["threshold_target_95"]
                metric = {k:m[k] for k in ("auroc","auprc","brier_score","log_loss","ece")}
                metrics.append(dict(**key, **params, **metric, threshold=threshold, inner_sensitivity=sens,
                                    **{k:t[k] for k in ("sensitivity","specificity","ppv","npv","FN","FP","TP","TN")}))
                cal = m["calibration_curve"]
                calibration.extend(dict(**key, bin=j, mean_probability=p, observed_fraction=o, n=n)
                                   for j,(p,o,n) in enumerate(zip(cal["bin_pred_mean"],cal["bin_true_prop"],cal["bin_counts"]),1))
                predictions.extend(dict(**key, id=data.id.iloc[i], y_true=y[i], y_prob=p, threshold=threshold)
                                   for i,p in zip(va,prob))
        print(f"Completed repeat {repeat+1}/10", flush=True)
    pred = pd.DataFrame(predictions)
    assert len(pred)==11380 and pred.groupby(["repeat","model"]).id.nunique().eq(569).all()
    assert not pred.duplicated(["repeat","model","id"]).any()
    table = pd.DataFrame(metrics)
    for name, records in dict(outer_predictions=pred, outer_metrics=table, outer_splits=pd.DataFrame(splits),
                              inner_search=pd.DataFrame(searches), outer_calibration=pd.DataFrame(calibration)).items():
        records.to_csv(out / f"{name}.csv", index=False)
    cols = ["auroc","auprc","brier_score","log_loss","ece","sensitivity","specificity","ppv","npv"]
    reps = table.groupby(["repeat","model"])[cols].mean()
    reps.to_csv(out / "repeat_metrics.csv")
    delta = reps.xs("RBF_SVM",level="model") - reps.xs("Linear_SVM",level="model")
    delta.to_csv(out / "paired_repeat_differences.csv")
    reps.groupby("model").agg(["mean","std"]).to_csv(out / "summary.csv")
    print(reps.groupby("model").mean().to_string(), flush=True)
    print(delta.agg(["mean","min","max"]).to_string(), flush=True)

if __name__ == "__main__":
    main()
