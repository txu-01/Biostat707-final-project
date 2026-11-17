"""Audit standardized coefficients on the shared logistic outer-training folds."""

from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from final_analysis import FEATURE_NAMES, SEED, fit_probability, load_data, make_model


MODELS = ("Logistic_Regression", "Ridge_Logistic", "Elastic_Net")
FOLDS = {(repeat, fold) for repeat in range(1, 11) for fold in range(5)}
PREFIX = "haiyi_logistic_coefficient"


def replay_outer_fits(results):
    data = load_data().set_index("id")
    expected_ids = set(data.index)
    coefficients = []
    replay = []
    common_assignments = None
    for name in MODELS:
        predictions = pd.read_csv(results / "predictions" / f"{name}.csv")
        assignments = predictions[["repeat", "fold", "id"]].sort_values(
            ["repeat", "id"], ignore_index=True
        )
        if common_assignments is None:
            common_assignments = assignments
        elif not common_assignments.equals(assignments):
            raise ValueError(f"{name} has different outer fold assignments.")
        if set(assignments["id"]) != expected_ids or assignments.duplicated(["repeat", "id"]).any():
            raise ValueError(f"{name} has unknown or duplicate record IDs.")
        if not assignments.groupby("repeat").size().eq(len(data)).all():
            raise ValueError(f"{name} is missing records in at least one repeat.")

        selected = pd.read_csv(results / f"outer_fold_metrics_{name}.csv").set_index(
            ["repeat", "fold"]
        )
        groups = predictions.groupby(["repeat", "fold"])
        if set(groups.groups) != FOLDS or set(selected.index) != FOLDS or not selected.index.is_unique:
            raise ValueError(f"{name} is missing or has duplicate outer-fold settings.")

        for (repeat, fold), validation in groups:
            held_out = validation.sort_values("id").set_index("id")
            train = data.loc[~data.index.isin(held_out.index)]
            validation_data = data.loc[held_out.index]
            if not np.array_equal(validation_data["target"], held_out["y_true"]):
                raise ValueError(f"{name}, repeat {repeat}, fold {fold} has invalid labels.")
            params_json = selected.loc[(repeat, fold), "params_json"]
            model = make_model(name, json.loads(params_json), SEED + (repeat - 1) * 10_000 + fold * 100)
            simulated, warnings = fit_probability(
                model, train[FEATURE_NAMES], train["target"].to_numpy(), validation_data[FEATURE_NAMES]
            )
            error = np.abs(simulated - held_out["y_prob"].to_numpy())
            replay.append({
                "model": name,
                "repeat": repeat,
                "fold": fold,
                "max_abs_probability_difference": float(error.max()),
                "probabilities_within_1e_7": int((error <= 1e-7).sum()),
                "validation_records": len(held_out),
                "refit_convergence_warnings": warnings,
            })
            weights = model.named_steps["model"].coef_[0]
            if not np.isfinite(weights).all():
                raise ValueError(f"{name}, repeat {repeat}, fold {fold} has invalid coefficients.")
            coefficients.extend({
                "model": name, "repeat": repeat, "fold": fold,
                "feature": feature, "standardized_coefficient": float(weight),
            } for feature, weight in zip(FEATURE_NAMES, weights))
    return pd.DataFrame(coefficients), pd.DataFrame(replay)


def write_outputs(results, coefficients, replay):
    summary = (coefficients.groupby(["model", "feature"])["standardized_coefficient"]
               .agg(median="median",
                    q25=lambda values: values.quantile(0.25),
                    q75=lambda values: values.quantile(0.75),
                    positive_folds=lambda values: values.gt(0).sum(),
                    negative_folds=lambda values: values.lt(0).sum(),
                    zero_folds=lambda values: values.eq(0).sum())
               .reset_index())
    agreement = (replay.groupby("model")
                 .agg(outer_fits=("fold", "size"),
                      predicted_records=("validation_records", "sum"),
                      probabilities_within_1e_7=("probabilities_within_1e_7", "sum"),
                      max_abs_probability_difference=("max_abs_probability_difference", "max"),
                      convergence_warnings=("refit_convergence_warnings", "sum")))
    provenance = {
        "software": {
            "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
        },
        "training": "Same outer-training IDs, selected parameters and outer-fit seeds as the archived team analysis.",
        "prediction_check": agreement.to_dict(orient="index"),
    }
    coefficients.to_csv(results / f"{PREFIX}s_by_fold.csv", index=False)
    summary.to_csv(results / f"{PREFIX}_stability.csv", index=False)
    (results / f"{PREFIX}_replay.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


def main():
    results = ROOT / "results" / "final_nested"
    coefficients, replay = replay_outer_fits(results)
    write_outputs(results, coefficients, replay)


if __name__ == "__main__":
    main()
