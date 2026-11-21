# Independent decision tree analysis

From the repository root, create a Python 3.9 environment and install the
versions in `wenqing_decision_tree/requirements.txt`. The saved results were
generated with Python 3.9.10 and these direct dependency versions; they differ
from the repository-root requirements. For example:

```text
python -m venv .venv
```

Activate the environment using the command for your operating system, then run:

```text
python -m pip install -r wenqing_decision_tree/requirements.txt
python wenqing_decision_tree/decision_tree_analysis.py
python wenqing_decision_tree/verify_outputs.py
```

The main script uses paths relative to its own location and depends on the
repository's `data/processed/wdbc_clean.csv`, `src/data_prep.py`, and
`src/metrics.py`. It writes into this folder and **overwrites its existing
analysis CSVs and five original figures**. Numerical details and runtime can
vary across environments. The verification script checks saved artifacts and
outer fold assignments; it does not independently refit threshold models.

`plot_mean_feature_importance.py` creates the additional top-10 plot from the
saved importances without fitting a model. It refuses to overwrite an existing
`figures/mean_feature_importance.png`; move or rename that figure first if you
want to recreate it.

`overfitting_by_fold.csv` contains the fitted outer-training AUROC, the selected
candidate's mean inner-validation AUROC, and outer-validation AUROC, along with
their differences. `overfitting_summary.csv` summarizes those fold values. The
training score is in-sample, and the inner score was used for selection; the
outer score is evaluated on held-out records. Fold summaries reuse records.

No per-feature fitted coefficients or importances from another team model were
found in the existing results. For a feature comparison, the script separately
fits a standardized ridge logistic regression (C=1) on each corresponding
outer-training set. This auxiliary fit is not the team's saved fitted model.
The comparison CSVs contain signed and absolute standardized coefficients,
tree impurity importances, ranks, and summaries across folds. The two measures
have different meanings and scales; ranks and nonzero frequency are descriptive.
No outer-validation labels enter these feature summaries.
