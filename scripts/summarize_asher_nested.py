import csv, hashlib, json, statistics as s
from pathlib import Path
root=Path(__file__).resolve().parents[1]
d=root/"results/svm/nested"
def read(name):
    with (d/name).open(newline="") as f: return list(csv.DictReader(f))
lock=json.loads((root/"results/svm/test_complete.json").read_text())
for name, expected in lock["prediction_sha256"].items():
    assert hashlib.sha256((root/f"results/predictions/test_{name}.csv").read_bytes()).hexdigest()==expected
dev_lock=json.loads((root/"results/svm/development_lock.json").read_text())
for path, expected in dev_lock["oof_sha256"].items():
    assert hashlib.sha256((root/path).read_bytes()).hexdigest()==expected
p=read("outer_predictions.csv")
fold=read("outer_splits.csv")
rows=read("repeat_metrics.csv")
search=read("inner_search.csv")
with (root/"data/processed/wdbc_clean.csv").open(newline="") as f:
    labels={r["id"]:r["target"] for r in csv.DictReader(f)}
assert len(labels)==569 and all(r["y_true"]==labels[r["id"]] and 0<=float(r["y_prob"])<=1 for r in p)
assert len(p)==11380 and len(fold)==5690 and len(rows)==20 and len(search)==1000
foldmap={(r["repeat"],r["id"]):r["fold"] for r in fold}
assert len(foldmap)==5690 and all(foldmap[r["repeat"],r["id"]]==r["fold"] for r in p)
for name in ("Linear_SVM","RBF_SVM"):
    assert len({(r["repeat"],r["id"]) for r in p if r["model"]==name})==5690
cols=["auroc","auprc","brier_score","log_loss","ece","sensitivity","specificity","ppv","npv"]
def values(name,key): return [float(r[key]) for r in rows if r["model"]==name]
lines=["# Asher Xu Repeated Nested CV Supplement","","## Methods","",
"Both SVMs were compared on all 569 WDBC records using 10 repeats of stratified five-fold outer CV. Records were sorted by ID. Outer seeds were 707–716. Within each outer training set, both SVMs used the same stratified five-fold inner CV (seed 1707 + 5 × zero-based repeat + zero-based outer fold). The existing parameter grids and the training-only three-fold sigmoid calibration were retained. The scaler sits inside the calibrated base estimator. Parameters maximize mean inner-fold AUROC. The highest threshold achieving at least 95% sensitivity on selected inner OOF probabilities was locked before prediction on each outer validation fold.","",
"## Results","","Mean ± SD across the 10 repeat estimates. Each repeat is an unweighted mean of its five outer-fold metrics. AUPRC is computed as average precision.","",
"| Outer validation metric | Linear SVM | RBF SVM |","|---|---:|---:|"]
for k in cols:
    a,b=(values(n,k) for n in ("Linear_SVM","RBF_SVM"))
    lines.append(f"| {k} | {s.mean(a):.4f} ± {s.stdev(a):.4f} | {s.mean(b):.4f} ± {s.stdev(b):.4f} |")
delta=[float(x["auroc"]) for x in read("paired_repeat_differences.csv")]
lines+=["",f"Paired AUROC differences (RBF minus linear): mean {s.mean(delta):+.5f}, range {min(delta):+.5f} to {max(delta):+.5f} across repeats. This is a descriptive range, not a confidence interval. Repeats share the same records and do not provide ten independent datasets.","",
"The original 80/20 results and their paired bootstrap intervals remain a secondary check. Because the full-cohort repeated CV includes the original test records, the two analyses are not independent confirmations. The original holdout settings and predictions were not changed. This analysis compares only the two SVMs and cannot answer whether either beats logistic regression.","",
"## Reproduction and handoff","","Run python scripts/run_svm_nested.py, then python scripts/summarize_asher_nested.py. Outer split IDs, inner search, outer predictions, fold metrics, calibration bins, repeat summaries and paired differences are under results/svm/nested. The outer split file can be shared, but the team must confirm a common protocol before combining other model families. Keep these files separate from the original 80/20 evaluator inputs."]
(root/"docs/ASHER_SVM_NESTED_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf8")
print("\n".join(lines[8:24])); print("Prediction coverage, fold alignment, and original holdout checksums passed.")
