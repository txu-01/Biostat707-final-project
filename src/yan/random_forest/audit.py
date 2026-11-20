"""Read-only audits of shared inputs, actual record assignments and source versions."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import subprocess

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from final_analysis import load_data, _outer_split, FEATURE_NAMES, MODEL_GRIDS
from data_prep import load_and_clean_raw

ROOT = Path(__file__).resolve().parents[3]
MODEL = "Random_Forest"
OUT = ROOT / "results/yan/random_forest"
REFERENCES = ("Logistic_Regression", "Ridge_Logistic", "Elastic_Net", "Linear_SVM", "RBF_SVM")
TRAINING_FILES = (
    "src/yan/random_forest/train.py", "src/yan/random_forest/audit.py",
    "src/yan/random_forest/__init__.py", "scripts/yan/run.py",
    "src/final_analysis.py", "src/data_prep.py", "src/metrics.py", "requirements.txt",
)
INPUTS = ("data/raw/wdbc.data", "data/processed/wdbc_clean.csv", "707-proposal.md",
          "docs/TEAM_WORKFLOW.md", "docs/common_evaluation.md", "README.md")
PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "tabulate",
            "joblib", "threadpoolctl")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_metrics(value):
    """Encode undefined PPV/NPV as JSON null; reject other nonfinite values."""
    if isinstance(value, dict):
        return {key: (None if key in ("ppv", "npv") and isinstance(item, (float, np.floating))
                      and np.isnan(item) else _json_metrics(item))
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_metrics(item) for item in value]
    return value


def canonical(value):
    return json.dumps(_json_metrics(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_json_metrics(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def data_and_splits():
    data = load_data()
    assert data.columns.tolist() == ["id", "diagnosis", "target", *FEATURE_NAMES]
    assert len(data) == 569 and data.id.is_unique
    assert data.diagnosis.isin(["M", "B"]).all()
    assert np.array_equal(data.target, data.diagnosis.eq("M").astype(int))
    assert data.target.value_counts().to_dict() == {0: 357, 1: 212}
    assert len(FEATURE_NAMES) == 30 and len(set(FEATURE_NAMES)) == 30
    assert np.isfinite(data[FEATURE_NAMES].to_numpy()).all()
    assert data[FEATURE_NAMES].notna().all().all()
    raw = load_and_clean_raw(ROOT / "data/raw/wdbc.data").sort_values("id").reset_index(drop=True)
    pd.testing.assert_frame_equal(data, raw, check_exact=False, rtol=1e-14, atol=1e-14)
    assert not raw.duplicated().any()
    outer, inner = [], []
    for repeat in range(1, 11):
        for fold, (tr, va) in enumerate(_outer_split(data, repeat - 1, 5)):
            assert not np.intersect1d(tr, va).size
            assert len(tr) + len(va) == len(data)
            outer.extend(dict(repeat=repeat, fold=fold, id=int(data.id.iloc[i]),
                              y_true=int(data.target.iloc[i])) for i in va)
            cv = StratifiedKFold(5, shuffle=True, random_state=1707 + 100 * (repeat - 1) + fold)
            for infold, (a, b) in enumerate(cv.split(data.iloc[tr][FEATURE_NAMES], data.target.iloc[tr])):
                assert not np.intersect1d(tr[a], tr[b]).size
                assert not np.intersect1d(tr[b], va).size
                assert np.array_equal(np.sort(np.r_[tr[a], tr[b]]), np.sort(tr))
                inner.extend(dict(repeat=repeat, fold=fold, inner_fold=infold,
                                  id=int(data.id.iloc[i]), y_true=int(data.target.iloc[i])) for i in tr[b])
    return data, pd.DataFrame(outer), pd.DataFrame(inner)


def align_reference(reference, expected, name):
    """Require the complete key set and labels, not just equal fold counts."""
    keys = ["repeat", "id"]
    need = {*keys, "fold", "y_true", "y_prob", "threshold", "model"}
    if not need.issubset(reference) or set(reference.model) != {name}:
        raise ValueError(f"{name}: wrong schema/model")
    if len(reference) != 5690 or reference.duplicated(keys).any():
        raise ValueError(f"{name}: incomplete or duplicate rows")
    got = reference.sort_values(keys).reset_index(drop=True)
    exp = expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(got[["repeat", "id", "fold", "y_true"]],
                                  exp[["repeat", "id", "fold", "y_true"]], check_dtype=False)
    for col in ["y_prob", "threshold"]:
        if not np.isfinite(got[col]).all() or not got[col].between(0, 1).all():
            raise ValueError(f"{name}: invalid {col}")
    if not got.groupby(["repeat", "fold"]).threshold.nunique().eq(1).all():
        raise ValueError(f"{name}: threshold varies within fold")
    return got


def environment():
    return dict(python=platform.python_version(), platform=platform.platform(),
                packages={p: importlib.metadata.version(p) for p in PACKAGES})


def audit(out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    data, outer, inner = data_and_splits()
    refs = {}
    for name in REFERENCES:
        path = ROOT / f"results/final_nested/predictions/{name}.csv"
        if path.exists():
            align_reference(pd.read_csv(path), outer, name)
            refs[name] = dict(status="matched all 5690 ID/label/repeat/fold keys",
                              path=str(path.relative_to(ROOT)), sha256=sha(path))
        else:
            refs[name] = dict(status="not available; not checked")
    sources = {}
    for ref, path in [
        ("origin/Haiyi-branches", "src/final_analysis.py"),
        ("origin/Haiyi-branches", "scripts/build_haiyi_logistic_coefficients.py"),
        ("origin/Haiyi-branches", "docs/HAIYI_XIONG_REPORT_SECTION.md"),
        ("origin/asher/svm-analysis", "scripts/run_svm_nested.py"),
        ("origin/asher/svm-analysis", "src/train_svm.py"),
        ("origin/asher/svm-analysis", "src/data_prep.py"),
    ]:
        try:
            contents = git("show", f"{ref}:{path}")
            sources[f"{ref}:{path}"] = dict(commit=git("rev-parse", ref),
                sha256=hashlib.sha256(contents.encode()).hexdigest())
        except subprocess.CalledProcessError:
            sources[f"{ref}:{path}"] = dict(status="not available; not checked")
    rows = [
        ("Data/features", "Shared load_data sorts by ID; 30 ordered features", "Reuse, compare raw/clean and M/B labels", "yes", "none"),
        ("Outer CV", "Shared _outer_split; seed 707+r; 1-based repeat/0-based fold", "Reuse and match every ID against available regression/SVM files", "yes", "none"),
        ("Inner CV", "Regression/shared: 1707+100*r+f; Asher: 1707+5*r+f", "Use shared/regression seed rule; retain model-specific seed evidence", "yes", "SVM inner partitions differ in repeats 2-10; do not claim identical inner folds"),
        ("Threshold", "Selected candidate inner OOF; highest p with sensitivity>=.95", "Reuse select_threshold_target_sensitivity; save selected OOF", "yes", "OOF used for selection is not independent generalization performance"),
        ("Metrics", "Shared summarize_prediction_rows pools five folds; row thresholds", "Reuse, independently recompute exported metrics", "yes", "Generic evaluate_predictions(None) retunes; never call it here"),
        ("SVM legacy summary", "Asher run_svm_nested.py averages fold metrics", "Use final main pooled-record summary rule", "yes", "Legacy SVM branch repeat tables conflict with pooled-AUROC rule; final main recomputes from predictions"),
        ("Undefined ratios", "Generic metrics.py fills undefined ratios with 0; final summary uses NaN for PPV/NPV", "Use final summary; export undefined as blank", "yes", "Generic helper is not used for outer evaluation"),
        ("RF", "Shared make_model; 500 trees, 16 settings, raw scale", "Reuse RF factory with n_jobs=1; parallelize outer tasks only", "yes", "none"),
        ("Resume", "Shared resume counts unique IDs only", "New run fingerprint and full per-fold checkpoint validation", "yes", "Existing RF outputs/checkpoints never consumed"),
        ("Personal directory", "TEAM_WORKFLOW discourages separate formal model sets", "User-authorized yan review artifacts pending integration", "yes", "Team must replace Random_Forest once, not add duplicate model"),
        ("Ridge comparator", "common_evaluation and final report document unstable unpenalized fit", "Compare only aligned archived Ridge_Logistic predictions", "yes", "Team should explain numerical/separation fallback; no solver warnings alone establish failure"),
    ]
    pd.DataFrame(rows, columns=["check", "existing", "execution", "proposal", "open_issue"]).to_csv(
        out / "consistency_checks.csv", index=False)
    outer.to_csv(out / "outer_fold_assignments.csv", index=False)
    inner.to_csv(out / "inner_fold_assignments.csv", index=False)
    info = dict(base_commit=git("rev-parse", "HEAD"), environment=environment(),
        data=dict(n=569, malignant=212, benign=357, n_features=30, features=FEATURE_NAMES,
                  ordering="numeric ID ascending", raw_clean_agreement=True, exclusions=0),
        input_sha256={p: sha(ROOT / p) for p in INPUTS},
        training_source_sha256={p: sha(ROOT / p) for p in TRAINING_FILES},
        outer_mapping_sha256=sha(out / "outer_fold_assignments.csv"),
        inner_mapping_sha256=sha(out / "inner_fold_assignments.csv"),
        other_model_audit=refs, inspected_branch_sources=sources,
        rf_grid=MODEL_GRIDS[MODEL], existing_rf_results_used=False)
    write_json(out / "input_audit.json", info)
    return info
