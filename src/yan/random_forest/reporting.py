"""Read-only statistical audit and RF report writing from approved sources.

No estimator is fitted. Primary RF predictions/records are never rewritten.
No shared all-model result table or alternative RF run is a report input.
"""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             log_loss, confusion_matrix)
from sklearn.model_selection import StratifiedKFold
from final_analysis import (load_data, _outer_split, FEATURE_NAMES, MODEL_GRIDS,
                            params_json, summarize_prediction_rows)
from metrics import select_threshold_target_sensitivity, compute_calibration_curve
from .audit import ROOT, write_json, sha

PRIMARY = ("oof_predictions.csv", "fold_results.csv", "inner_cv_results.csv",
           "selected_inner_oof.csv", "feature_importances.csv", "calibration_points.csv",
           "repeat_metrics.csv", "summary_metrics.csv")
METRICS = ["auroc", "auprc", "brier_score", "log_loss", "sensitivity", "specificity",
           "ppv", "npv", "FN", "FP", "TP", "TN", "mean_predicted", "observed_prevalence", "ece"]
MAIN = METRICS[:10]
LABELS = {"auroc":"AUROC", "auprc":"AUPRC (average precision)", "brier_score":"Brier score",
          "log_loss":"Log loss", "sensitivity":"Sensitivity", "specificity":"Specificity",
          "ppv":"PPV", "npv":"NPV", "FN":"False negatives per repeat", "FP":"False positives per repeat"}


def evaluate_rows(rows):
    """Use row-specific locked thresholds; audit shared metric definitions."""
    y = rows.y_true.to_numpy(dtype=int)
    p = rows.y_prob.to_numpy(dtype=float)
    t = rows.threshold.to_numpy(dtype=float)
    tn, fp, fn, tp = confusion_matrix(y, p >= t, labels=[0, 1]).ravel()
    direct = dict(auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p),
        brier_score=brier_score_loss(y, p), log_loss=log_loss(y, np.clip(p, 1e-15, 1-1e-15)),
        sensitivity=tp/(tp+fn) if tp+fn else np.nan,
        specificity=tn/(tn+fp) if tn+fp else np.nan,
        ppv=tp/(tp+fp) if tp+fp else np.nan, npv=tn/(tn+fn) if tn+fn else np.nan,
        FN=int(fn), FP=int(fp), TP=int(tp), TN=int(tn),
        mean_predicted=float(p.mean()), observed_prevalence=float(y.mean()))
    shared = summarize_prediction_rows(rows)
    for key, value in direct.items():
        if not np.isclose(shared[key], value, equal_nan=True, atol=1e-12):
            raise AssertionError(f"Shared metric definition mismatch: {key}")
    return shared


def pooled_repeat_metrics(predictions):
    return pd.DataFrame([dict(model=name, repeat=int(repeat), **evaluate_rows(g))
        for (name, repeat), g in predictions.groupby(["model", "repeat"])])


def validate_prediction_keys(predictions, expected, model):
    if len(predictions) != 5690 or set(predictions.model) != {model}:
        raise ValueError(f"{model}: incomplete predictions or wrong model")
    if predictions.duplicated(["repeat", "id"]).any():
        raise ValueError(f"{model}: duplicate record key")
    keys = ["id", "y_true", "repeat", "fold"]
    pd.testing.assert_frame_equal(
        predictions.sort_values(["repeat", "id"])[keys].reset_index(drop=True),
        expected.sort_values(["repeat", "id"])[keys].reset_index(drop=True), check_dtype=False)
    for name in ["y_prob", "threshold"]:
        if not np.isfinite(predictions[name]).all() or not predictions[name].between(0, 1).all():
            raise ValueError(f"{model}: invalid {name}")
    if not predictions.groupby(["repeat", "fold"]).threshold.nunique().eq(1).all():
        raise ValueError(f"{model}: inconsistent locked threshold")


def expanded_calibration(predictions):
    result = []
    for repeat, g in predictions.groupby("repeat"):
        p, y = g.y_prob.to_numpy(), g.y_true.to_numpy()
        edges = np.unique(np.quantile(p, np.linspace(0, 1, 6)))
        masks = [(p >= edges[k]) & ((p <= edges[k+1]) if k == len(edges)-2 else (p < edges[k+1]))
                 for k in range(len(edges)-1)] if len(edges)>1 else [np.ones(len(p), bool)]
        seen = np.zeros(len(p), int)
        for k, mask in enumerate(masks):
            seen += mask.astype(int)
            if not mask.any():
                continue
            low = edges[k] if len(edges)>1 else edges[0]
            high = edges[k+1] if len(edges)>1 else edges[0]
            mean_p, observed = float(p[mask].mean()), float(y[mask].mean())
            result.append(dict(repeat=int(repeat), bin=k+1, lower_edge=float(low), upper_edge=float(high),
                mean_probability=mean_p, observed_fraction=observed, n=int(mask.sum()),
                predicted_minus_observed=mean_p-observed))
        if not np.all(seen == 1):
            raise AssertionError("Calibration quantiles omit or double-count records")
        shared = compute_calibration_curve(y, p, n_bins=5)
        actual = [r for r in result if r["repeat"] == repeat]
        np.testing.assert_allclose([r["mean_probability"] for r in actual], shared["bin_pred_mean"])
        np.testing.assert_allclose([r["observed_fraction"] for r in actual], shared["bin_true_prop"])
        assert [r["n"] for r in actual] == shared["bin_counts"]
    return pd.DataFrame(result)


def audit_rf(out):
    data = load_data()
    assert len(data) == 569 and data.target.value_counts().to_dict() == {0:357, 1:212}
    assert np.array_equal(data.target, data.diagnosis.eq("M").astype(int))
    assert len(FEATURE_NAMES) == 30 and np.isfinite(data[FEATURE_NAMES]).all().all()
    expected = []
    for repeat in range(1, 11):
        for fold, (tr, va) in enumerate(_outer_split(data, repeat-1, 5)):
            assert not set(tr) & set(va)
            expected.extend(dict(id=int(data.id.iloc[i]),y_true=int(data.target.iloc[i]),
                                 repeat=repeat,fold=fold) for i in va)
    expected = pd.DataFrame(expected)
    p = pd.read_csv(out / "oof_predictions.csv")
    folds = pd.read_csv(out / "fold_results.csv")
    search = pd.read_csv(out / "inner_cv_results.csv")
    inner = pd.read_csv(out / "selected_inner_oof.csv")
    importance = pd.read_csv(out / "feature_importances.csv")
    validate_prediction_keys(p, expected, "Random_Forest")
    assert np.array_equal(p.predicted_class, p.y_prob.ge(p.threshold).astype(int))
    assert len(folds)==50 and not folds.duplicated(["repeat","fold"]).any()
    assert len(search)==800 and not search.duplicated(["repeat","fold","candidate"]).any()
    assert folds.n_estimators.eq(500).all() and search.n_estimators.eq(500).all()
    assert len(importance)==1500 and importance.groupby(["repeat","fold"]).size().eq(30).all()
    assert importance.groupby(["repeat","fold"]).importance.sum().sub(1).abs().lt(1e-12).all()
    truth=data.set_index("id").target
    for row in folds.to_dict("records"):
        r,f=row["repeat"],row["fold"]
        g=p.loc[p["repeat"].eq(r)&p.fold.eq(f)]
        training=inner.loc[inner["repeat"].eq(r)&inner.fold.eq(f)].copy()
        assert training.id.is_unique and set(training.id)==set(data.id)-set(g.id)
        assert np.array_equal(training.y_true,training.id.map(truth))
        tr=np.flatnonzero(~data.id.isin(g.id))
        cv=StratifiedKFold(5,shuffle=True,random_state=1707+100*(r-1)+f)
        for k,(_,v) in enumerate(cv.split(data.iloc[tr][FEATURE_NAMES],data.target.iloc[tr])):
            assert set(training.loc[training.inner_fold.eq(k),"id"])==set(data.id.iloc[tr[v]])
        s=search.loc[search["repeat"].eq(r)&search.fold.eq(f)]
        assert len(s)==16 and set(s.candidate)==set(range(16))
        for candidate in s.to_dict("records"):
            assert candidate["params_json"]==params_json(MODEL_GRIDS["Random_Forest"][candidate["candidate"]])
            assert np.isclose(candidate["mean_inner_auroc"],
                              np.mean([candidate[f"inner_fold_{k}_auroc"] for k in range(5)]))
        best=s.sort_values(["mean_inner_auroc","candidate"],ascending=[False,True]).iloc[0]
        assert row["selected_candidate"]==best.candidate and row["params_json"]==best.params_json
        t,sensitivity=select_threshold_target_sensitivity(training.y_true,training.y_prob,.95)
        assert sensitivity>=.95 and np.isclose(t,row["threshold"],atol=1e-14,rtol=0)
        assert np.isclose(sensitivity,row["inner_sensitivity"])
        assert g.threshold.sub(t).abs().lt(1e-14).all()
        assert np.isclose(evaluate_rows(g)["auroc"],row["validation_auroc"],atol=1e-12)
        assert np.isclose(row["training_validation_gap"],
                          row["train_resubstitution_auroc"]-row["validation_auroc"])
    repeats=pooled_repeat_metrics(p).sort_values("repeat")
    recorded=pd.read_csv(out/"repeat_metrics.csv").sort_values("repeat")
    np.testing.assert_allclose(repeats[METRICS],recorded[METRICS],atol=1e-12,equal_nan=True)
    supplement=repeats[METRICS].agg(["mean","std","min","max"]).T.reset_index(names="metric")
    old=pd.read_csv(out/"summary_metrics.csv").set_index("metric")
    np.testing.assert_allclose(supplement.set_index("metric").loc[METRICS,["mean","std","min","max"]],
                              old.loc[METRICS,["mean","std","min","max"]],atol=1e-12,equal_nan=True)
    calibration=expanded_calibration(p)
    old_cal=pd.read_csv(out/"calibration_points.csv").sort_values(["repeat","bin"])
    np.testing.assert_allclose(calibration[["repeat","bin","mean_probability","observed_fraction","n"]],
                              old_cal[["repeat","bin","mean_probability","observed_fraction","n"]],atol=1e-12)
    return data,expected,p,folds,search,importance,repeats,supplement,calibration


def owner_comparisons(out, expected, rf_repeats):
    path=out/"comparison_provenance.json"
    if not path.exists():
        return pd.DataFrame(),pd.DataFrame(),{"svm":"pending source attribution","regression":"pending source attribution"}
    meta=json.loads(path.read_text())
    status={"svm":"pending","regression":"pending owner-attributed compatible predictions"}
    records=[]
    try:
        source=meta["svm"]
        if not source.get("eligible") or source["owner"]!="Asher Xu":
            raise ValueError("SVM owner attribution is not confirmed")
        folder=out/"comparison_sources/asher"
        for name,item in source["files"].items():
            if sha(folder/name)!=item["sha256"]:
                raise ValueError(f"Owner source changed: {name}")
        owner_data=pd.read_csv(folder/"input_data.csv").sort_values("id").reset_index(drop=True)
        pd.testing.assert_frame_equal(owner_data,load_data(),check_exact=False,rtol=1e-14,atol=1e-14)
        pred=pd.read_csv(folder/"outer_predictions.csv")
        if set(pred.model)!={"Linear_SVM","RBF_SVM"}:
            raise ValueError("Unexpected owner model set")
        for name in source["models"]:
            g=pred.loc[pred.model.eq(name)].copy()
            validate_prediction_keys(g,expected,name)
            records.append(pooled_repeat_metrics(g))
        status["svm"]="verified owner-specific nested predictions; metrics recomputed from pooled outer records"
    except (OSError,KeyError,ValueError,AssertionError) as error:
        records=[]
        status["svm"]="pending: "+str(error)
    refs=pd.concat(records,ignore_index=True) if records else pd.DataFrame()
    differences=[]
    if not refs.empty:
        for name,g in refs.groupby("model"):
            merged=rf_repeats.merge(g,on="repeat",suffixes=("_rf","_reference"),validate="one_to_one")
            for _,row in merged.iterrows():
                differences.append(dict(repeat=int(row["repeat"]),model="Random_Forest",comparator=name,
                    **{m+"_difference":row[m+"_rf"]-row[m+"_reference"] for m in METRICS}))
    return refs,pd.DataFrame(differences),status


def descriptive_tables(out,folds,search,importance,calibration,p):
    params=folds.params_json.map(json.loads)
    selected=pd.DataFrame([dict(repeat=r["repeat"],fold=r["fold"],max_depth="unrestricted" if v["max_depth"] is None else str(v["max_depth"]),
        min_samples_leaf=v["min_samples_leaf"],max_features=str(v["max_features"]),params_json=r["params_json"])
        for r,v in zip(folds.to_dict("records"),params)])
    selection=selected.groupby("params_json").size().sort_values(ascending=False).reset_index(name="selected_folds")
    selection.to_csv(out/"report_parameter_selection.csv",index=False)
    marginals=[]
    for field in ["max_depth","min_samples_leaf","max_features"]:
        levels={"max_depth":["3","unrestricted"],"min_samples_leaf":[1,3,5,10],"max_features":["sqrt","0.5"]}[field]
        for level in levels:
            marginals.append(dict(parameter=field,value=level,selected_folds=int(selected[field].eq(level).sum())))
    pd.DataFrame(marginals).to_csv(out/"report_parameter_marginals.csv",index=False)
    full=search.copy()
    full["depth"]=full.params_json.map(lambda s:"unrestricted" if json.loads(s)["max_depth"] is None else str(json.loads(s)["max_depth"]))
    full["leaf"]=full.params_json.map(lambda s:json.loads(s)["min_samples_leaf"])
    full["features"]=full.params_json.map(lambda s:str(json.loads(s)["max_features"]))
    contrasts=[]
    for feature in ["sqrt","0.5"]:
        sub=full.loc[full.features.eq(feature)]
        for depth in ["3","unrestricted"]:
            fixed=sub.loc[sub.depth.eq(depth)]
            d=fixed.loc[fixed.leaf.eq(1)].set_index(["repeat","fold"]).mean_inner_auroc-fixed.loc[fixed.leaf.eq(10)].set_index(["repeat","fold"]).mean_inner_auroc
            contrasts.append(dict(contrast="leaf_1_minus_10",max_features=feature,fixed_depth=depth,fixed_leaf="",mean=d.mean(),std=d.std(),min=d.min(),max=d.max()))
        for leaf in [1,3,5,10]:
            fixed=sub.loc[sub.leaf.eq(leaf)]
            d=fixed.loc[fixed.depth.eq("unrestricted")].set_index(["repeat","fold"]).mean_inner_auroc-fixed.loc[fixed.depth.eq("3")].set_index(["repeat","fold"]).mean_inner_auroc
            contrasts.append(dict(contrast="unrestricted_minus_depth_3",max_features=feature,fixed_depth="",fixed_leaf=leaf,mean=d.mean(),std=d.std(),min=d.min(),max=d.max()))
    contrasts=pd.DataFrame(contrasts)
    contrasts.to_csv(out/"report_parameter_contrasts.csv",index=False)
    imp=importance.copy()
    imp["rank"]=imp.groupby(["repeat","fold"]).importance.rank(ascending=False,method="min")
    ranks=imp.groupby("feature").agg(importance_mean=("importance","mean"),importance_sd=("importance","std"),
        importance_min=("importance","min"),importance_max=("importance","max"),mean_rank=("rank","mean"),
        best_rank=("rank","min"),worst_rank=("rank","max"),top5_folds=("rank",lambda x:int(x.le(5).sum())),
        top10_folds=("rank",lambda x:int(x.le(10).sum()))).sort_values("importance_mean",ascending=False).reset_index()
    ranks.to_csv(out/"report_feature_stability.csv",index=False)
    imp["family"]=imp.feature.str.rsplit("_",n=1).str[0]
    family=imp.groupby(["repeat","fold","family"]).importance.sum().groupby("family").agg(["mean","std","min","max"]).sort_values("mean",ascending=False).reset_index()
    family.to_csv(out/"report_feature_families.csv",index=False)
    cb=calibration.groupby("bin").agg(mean_pred_min=("mean_probability","min"),mean_pred_max=("mean_probability","max"),
        observed_min=("observed_fraction","min"),observed_max=("observed_fraction","max"),
        overestimate_repeats=("predicted_minus_observed",lambda x:int(x.gt(1e-12).sum())),
        underestimate_repeats=("predicted_minus_observed",lambda x:int(x.lt(-1e-12).sum())),
        min_bin_n=("n","min"),max_bin_n=("n","max")).reset_index()
    cb.to_csv(out/"report_calibration_pattern.csv",index=False)
    extremes=p.assign(zero_probability=p.y_prob.eq(0),one_probability=p.y_prob.eq(1),
        malignant_with_zero=p.y_prob.eq(0)&p.y_true.eq(1),benign_with_one=p.y_prob.eq(1)&p.y_true.eq(0))
    extremes=extremes.groupby("repeat")[["zero_probability","one_probability","malignant_with_zero","benign_with_one"]].sum().reset_index()
    extremes.to_csv(out/"report_probability_extremes.csv",index=False)
    return selection,pd.DataFrame(marginals),contrasts,ranks,family,cb,extremes


def build_report(out):
    out=Path(out).resolve()
    before={name:sha(out/name) for name in PRIMARY}
    data,expected,p,folds,search,importance,repeats,supplement,calibration=audit_rf(out)
    refs,paired,status=owner_comparisons(out,expected,repeats)
    repeats.to_csv(out/"report_repeat_metrics.csv",index=False)
    supplement.to_csv(out/"report_supplementary_metrics.csv",index=False)
    calibration.to_csv(out/"report_calibration_bins.csv",index=False)
    selection,marginals,contrasts,ranks,family,cb,extremes=descriptive_tables(out,folds,search,importance,calibration,p)
    mean=repeats[METRICS].mean()
    sd=repeats[METRICS].std()
    main=[]
    for metric in MAIN:
        rate=metric in ["sensitivity","specificity","ppv","npv"]
        display=(f"{mean[metric]:.4f} ± {sd[metric]:.4f}" if metric=="auroc" else
                 f"{100*mean[metric]:.2f}%" if rate else f"{mean[metric]:.1f}" if metric in ["FN","FP"] else f"{mean[metric]:.4f}")
        main.append(dict(metric=metric,label=LABELS[metric],mean=float(mean[metric]),
            auroc_sd=float(sd[metric]) if metric=="auroc" else np.nan,display=display,
            unit="proportion" if rate else "records per 569-record repeat" if metric in ["FN","FP"] else "score",
            source_csv="repeat_metrics.csv",source_column=metric))
    main=pd.DataFrame(main);main.to_csv(out/"report_main_results.csv",index=False)
    combined=pd.concat([repeats,refs],ignore_index=True) if not refs.empty else repeats
    comparison=combined.groupby("model")[METRICS].agg(["mean","std","min","max"])
    comparison.columns=["_".join(c) for c in comparison.columns]
    comparison.reset_index().to_csv(out/"report_model_comparison.csv",index=False)
    if not refs.empty:
        refs.to_csv(out/"owner_comparator_repeat_metrics.csv",index=False)
        paired.to_csv(out/"owner_paired_repeat_differences.csv",index=False)
        dcols=[c for c in paired if c.endswith("_difference")]
        ds=paired.groupby("comparator")[dcols].agg(["mean","std","min","max"])
        ds.columns=["_".join(c) for c in ds.columns]
        ds.reset_index().to_csv(out/"owner_paired_difference_summary.csv",index=False)
    else:
        # A formerly eligible source can become unavailable. Do not leave its
        # derived comparison tables looking like part of the current report.
        for name in ["owner_comparator_repeat_metrics.csv","owner_paired_repeat_differences.csv",
                     "owner_paired_difference_summary.csv"]:
            path=out/name
            if path.exists():
                archive=out/"provenance/superseded_comparisons"
                archive.mkdir(parents=True,exist_ok=True)
                path.replace(archive/(sha(path)[:12]+"_"+name))
    manifest=json.loads((out/"run_manifest.json").read_text())
    wall=sum(s["wall_seconds"] for s in manifest["sessions"] if s["stage"] in ["pilot","run"])
    runtime=pd.DataFrame([dict(quantity="wall_seconds",value=wall,source="run_manifest.json:sessions"),
        dict(quantity="outer_task_seconds_sum",value=float(folds.total_seconds.sum()),source="fold_results.csv:total_seconds"),
        dict(quantity="median_outer_task_seconds",value=float(folds.total_seconds.median()),source="fold_results.csv:total_seconds"),
        dict(quantity="additional_threshold_model_fit_seconds",value=float(folds.threshold_additional_model_fit_seconds.sum()),source="fold_results.csv:threshold_additional_model_fit_seconds")])
    runtime.to_csv(out/"report_runtime.csv",index=False)
    context=dict(data=data,p=p,folds=folds,repeats=repeats,mean=mean,sd=sd,main=main,supplement=supplement,
        calibration=calibration,cal_pattern=cb,selection=selection,marginals=marginals,contrasts=contrasts,
        ranks=ranks,family=family,extremes=extremes,refs=refs,paired=paired,status=status,wall=wall)
    from .report_text import write_text
    write_text(out,context)
    plot_calibration(out,calibration)
    after={name:sha(out/name) for name in PRIMARY}
    if before!=after:
        raise AssertionError("A primary RF prediction/record file changed during report generation")
    check=dict(primary_file_sha256_before=before,primary_file_sha256_after=after,
        primary_files_unchanged=True,no_models_fitted=True,rf_repeat_count=10,rf_prediction_count=5690,
        candidate_records=800,outer_records=50,feature_importance_records=1500,
        paired_sources=status,regression_numeric_comparison_included=False,
        source_policy="RF-owned primary records, revised proposal, audited shared infrastructure, and verified owner-specific SVM nested predictions only",
        report_source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
                             "src/yan/random_forest/report_text.py":sha(ROOT/"src/yan/random_forest/report_text.py")})
    write_json(out/"report_verification.json",check)
    print(json.dumps({"primary_files_unchanged":True,"comparisons":status,"formal_report":"report_section.md"},indent=2))
    return check


def plot_calibration(out,calibration):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures=out/"figures";figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for repeat,g in calibration.groupby("repeat"):
        color=plt.get_cmap("tab10")(repeat-1)
        axes[0].plot(g.mean_probability,g.observed_fraction,"o-",color=color,alpha=.65,markersize=3)
        axes[1].plot(g.bin,g.predicted_minus_observed,"o-",color=color,alpha=.7,markersize=3,label=str(repeat))
    axes[0].plot([0,1],[0,1],"k--",lw=1)
    axes[0].set(xlabel="Mean predicted probability",ylabel="Observed malignant fraction",
                title="Separate quantile bins within each repeat",xlim=(-.02,1.02),ylim=(-.02,1.02))
    axes[1].axhline(0,color="black",ls="--",lw=1)
    axes[1].set(xlabel="Ordered quantile group within repeat",ylabel="Predicted minus observed fraction",
                title="Positive = overestimation; negative = underestimation",xticks=range(1,6))
    axes[1].legend(title="Repeat",ncol=5,fontsize=7)
    fig.tight_layout();fig.savefig(figures/"report_calibration.png",dpi=180,bbox_inches="tight");plt.close(fig)
