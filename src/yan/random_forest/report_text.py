import json
import pandas as pd


def write_text(out,c):
    m,sd=c["mean"],c["sd"]
    f,r=c["folds"],c["repeats"]
    cal=c["cal_pattern"].set_index("bin")
    third,fourth,fifth=cal.loc[3],cal.loc[4],cal.loc[5]
    threshold_min,threshold_max=f.threshold.min(),f.threshold.max()
    below=int(r.sensitivity.lt(.95).sum())
    zero_repeats=int(c["extremes"].malignant_with_zero.gt(0).sum())
    selection=c["selection"].iloc[0]
    most=json.loads(selection.params_json)
    margins=c["marginals"]
    def count(parameter,value):
        return int(margins.loc[margins.parameter.eq(parameter)&margins.value.astype(str).eq(str(value)),"selected_folds"].iloc[0])
    unrestricted=count("max_depth","unrestricted")
    sqrt_count=count("max_features","sqrt")
    leaf_counts=[count("min_samples_leaf",v) for v in [1,3,5,10]]
    leaf=c["contrasts"].query("contrast == 'leaf_1_minus_10' and max_features == 'sqrt' and fixed_depth == 'unrestricted'")["mean"].iloc[0]
    depth=c["contrasts"].loc[c["contrasts"].contrast.eq("unrestricted_minus_depth_3")&
        c["contrasts"].max_features.eq("sqrt")&c["contrasts"].fixed_leaf.eq(1),"mean"].iloc[0]
    top=c["ranks"].head(5)
    feature_names=", ".join(top.feature)
    first=top.iloc[0]
    # The primary table intentionally avoids technical version/interface metadata.
    table=c["main"][["label","display"]].rename(columns={"label":"Metric","display":"RF repeat summary"}).to_markdown(index=False)
    refs,paired=c["refs"],c["paired"]
    compare_primary="The planned regression and SVM comparisons are pending compatible owner-provided predictions."
    compare_threshold=""
    compare_interpret="Cross-model conclusions remain pending verified owner-provided predictions."
    compare_source="S7: No eligible owner prediction file is available; numerical cross-model comparisons are pending."
    owner_audit="- No eligible owner-specific SVM predictions are available to this report run."
    if not refs.empty:
        ref=refs.groupby("model").mean(numeric_only=True)
        ds=paired.groupby("comparator").mean(numeric_only=True)
        linear,rbf=ref.loc["Linear_SVM"],ref.loc["RBF_SVM"]
        dl,dr=ds.loc["Linear_SVM"],ds.loc["RBF_SVM"]
        compare_primary=(
            f"Using compatible owner-provided nested predictions, the mean paired RF-minus-linear-SVM "
            f"and RF-minus-RBF-SVM AUROC differences were {dl.auroc_difference:+.6f} and "
            f"{dr.auroc_difference:+.6f}. The corresponding Brier differences were "
            f"{dl.brier_score_difference:+.6f} and {dr.brier_score_difference:+.6f}; RF also had higher "
            f"mean log loss. These are descriptive paired differences across repeats, not independent "
            f"study estimates. The planned regression comparison remains pending owner-attributed "
            f"compatible prediction outputs. [S7]")
        compare_threshold=(
            f"Linear SVM and RBF SVM averaged {linear.FN:.1f} and {rbf.FN:.1f} false negatives, "
            f"and {linear.FP:.1f} and {rbf.FP:.1f} false positives. Their achieved sensitivities were "
            f"{linear.sensitivity*100:.2f}% and {rbf.sensitivity*100:.2f}%, compared with "
            f"{m.sensitivity*100:.2f}% for RF. Thus, RF's slightly fewer false negatives came with "
            f"more false positives and a higher achieved sensitivity. The models shared a training "
            f"target, not a strictly matched validation sensitivity. [S7]")
        compare_interpret=(
            "RF did not show a numerical advantage over the two SVMs in discrimination or probability "
            "quality in this analysis. Its threshold tradeoff should be considered alongside the "
            "different achieved sensitivities. The primary comparison with a stable full-feature "
            "logistic model is still pending; no regression conclusion is drawn.")
        compare_source=("S7: owner_comparator_repeat_metrics.csv, owner_paired_repeat_differences.csv and "
            "owner_paired_difference_summary.csv, recomputed from the owner-specific nested predictions "
            "recorded in comparison_provenance.json. Regression numerical comparison is pending.")
        owner_audit=("- Owner-specific Asher nested predictions were used directly. The 80/20 results and "
            "legacy fold-averaged nested tables were excluded. Metrics were recomputed by model and repeat "
            "after pooling five folds, using stored per-row thresholds.\n"
            "- RF and SVM inner seed schedules differ, while actual outer ID/label/repeat/fold mappings agree. "
            "Three-fold sigmoid SVM calibration is a prescribed model difference.")
    section=f"""# Random Forest — Yan Yan

## 1. Methods

We fitted Random Forest models to all 569 WDBC records, including 212 malignant and 357
benign records. Malignancy was coded as 1. All 30 nuclear-image features were retained on
their original scale; record ID and outcome fields were excluded from the predictors.
No records were removed, and no imputation, oversampling, class reweighting, feature
selection or extra probability calibration was used.

The analysis followed ten repeats of stratified five-fold outer cross-validation, using
the shared outer assignments. In every outer-training set, stratified five-fold inner CV
selected the largest mean AUROC among 16 settings: max_depth in {{3, None}},
min_samples_leaf in {{1, 3, 5, 10}}, and max_features in {{sqrt, 0.5}}. Each forest had
500 trees. We selected the highest threshold giving at least 95% sensitivity in the
selected setting's training-only inner OOF probabilities. The selected model was refitted
on the complete outer-training set, and the threshold was applied unchanged to outer
validation. Inner OOF results had also been used for setting selection and were not treated
as independent generalization estimates. [S1]

Each repeat pooled its five validation folds before calculating metrics. AUPRC denotes
average precision. AUROC is reported as mean ± sample SD across repeats; other primary
values are repeat means. The repeats reuse the same records and are not ten independent
studies. The 5,690 predictions therefore do not represent 5,690 independent cases. [S2]

## 2. Primary results

{table}

FN and FP are counts among 569 validation predictions in each repeat; their repeat means
can be fractional. The observed malignant fraction was {m.observed_prevalence*100:.2f}%.
Full mean, SD, minimum and maximum values are provided in the supplementary table. [S2]

{compare_primary}

## 3. Calibration and threshold tradeoffs

Mean predicted malignancy was {m.mean_predicted*100:.2f}%, close to the observed
{m.observed_prevalence*100:.2f}%. This aggregate agreement did not imply good calibration
throughout the probability range. In the third quantile group, mean probabilities ranged
from {third.mean_pred_min*100:.1f}% to {third.mean_pred_max*100:.1f}%, while observed fractions
ranged from {third.observed_min*100:.1f}% to {third.observed_max*100:.1f}%; all ten repeats
showed overestimation. The fourth group showed underestimation in all repeats: mean
probabilities were {fourth.mean_pred_min*100:.1f}%–{fourth.mean_pred_max*100:.1f}%, versus
observed fractions of {fourth.observed_min*100:.1f}%–{fourth.observed_max*100:.1f}%.
The highest group also showed slight underestimation, with mean predictions of
{fifth.mean_pred_min*100:.1f}%–{fifth.mean_pred_max*100:.1f}% and an observed fraction of 100%.
The lowest two groups had less consistent bias directions. These are within-repeat
quantile groups, not fixed probability intervals. Approximately five equal-frequency
groups are coarse, and tied probabilities affected boundaries and group sizes. [S3]

Brier score and log loss assess probability quality, including more than pure calibration
error. A probability of zero was assigned to a malignant validation record in
{zero_repeats} repeats; it must not be interpreted as proof that malignancy is impossible.
Log loss used a temporary probability copy clipped to [1e-15, 1-1e-15], leaving the
exported probabilities unchanged. [S3]

The 50 training-selected thresholds ranged from {threshold_min:.4f} to {threshold_max:.4f}.
Pooled validation sensitivity ranged from {r.sensitivity.min()*100:.2f}% to
{r.sensitivity.max()*100:.2f}% and fell below 95% in {below} of ten repeats.
The target was a training criterion, not a validation guarantee. FN denotes malignant
records classified negative; FP denotes benign records classified malignant. A false
negative could hypothetically delay review and a false positive could trigger additional
review. Downstream clinical consequences were not measured. [S2, S3]

{compare_threshold}

## 4. Parameter selection and overfitting diagnostics

The most frequent setting was max_depth={most['max_depth']}, min_samples_leaf={most['min_samples_leaf']} and max_features={most['max_features']},
selected in {int(selection.selected_folds)} of 50 fits. Unrestricted depth was selected
{unrestricted} times, sqrt {sqrt_count} times, and leaf sizes 1, 3, 5 and 10 were selected {leaf_counts[0]}, {leaf_counts[1]}, {leaf_counts[2]} and {leaf_counts[3]} times.
{len(c['selection'])} settings were selected overall: the broad preference was consistent, while the exact
configuration was not fixed across splits. Holding max_features=sqrt and depth unrestricted,
the mean paired inner-AUROC difference for leaf size 1 minus 10 was {leaf:+.6f}.
Holding max_features=sqrt and leaf size 1, unrestricted depth minus depth 3 averaged
{depth:+.6f}. Candidate-specific random seeds also differed. These conditional grid patterns
do not establish causal parameter effects. [S4]

Mean training resubstitution AUROC was {f.train_resubstitution_auroc.mean():.4f}, compared
with a mean outer-fold AUROC of {f.validation_auroc.mean():.4f}; the mean within-fit gap
was {f.training_validation_gap.mean():.4f}. Training performance is optimistic because those
records were used in fitting. This fold-averaged diagnostic is a different summary from
the primary pooled-repeat AUROC. It was not used to revise the model. The complete
analysis took {c['wall']/60:.2f} minutes of recorded wall time, with up to two outer tasks
in parallel and one thread per forest; runtime depends on the computing environment. [S5]

## 5. Feature importance

The leading features were {feature_names}.
These same five features occupied the top five in every outer-training model, but their
relative order and weights changed. For example, {first.feature} had mean importance
{first.importance_mean:.4f} with fold SD {first.importance_sd:.4f}, and ranked between
{int(first.best_rank)} and {int(first.worst_rank)}. These features represent size and contour
irregularity information. Impurity-based importance can favor variables with more split
opportunities, and correlated radius, perimeter, area and mean/worst summaries can share
or substitute importance. The values do not give effect directions, independent
contributions, causal effects or evidence for a specific interaction. No feature was
removed or selected using this analysis. [S6]

## 6. Interpretation and limitations

RF discriminated malignant from benign records well, but its local calibration patterns
and threshold errors remain relevant to any proposed review-prioritization use.
{compare_interpret}

The dataset is small, historical and contains correlated predictors. Unique record IDs
do not independently establish unique patients. Repeated CV describes sensitivity to
splitting the available records; it does not supply new independent patients or external
validation. Five-bin calibration is approximate, and the exploratory 95% sensitivity
target is not a clinically validated operating point. These diagnostic measurements do
not predict future cancer onset in the general population. The analysis establishes
neither deployment readiness nor demonstrated clinical utility. Small differences do
not prove equivalence or noninferiority.

Source notes: S1: fold_results.csv, inner_cv_results.csv, selected_inner_oof.csv and
oof_predictions.csv. S2: report_main_results.csv, repeat_metrics.csv and
report_supplementary_metrics.csv. S3: calibration_points.csv, report_calibration_bins.csv,
report_calibration_pattern.csv and report_probability_extremes.csv. S4: inner_cv_results.csv,
report_parameter_selection.csv, report_parameter_marginals.csv and report_parameter_contrasts.csv.
S5: fold_results.csv and report_runtime.csv (wall time from run_manifest.json sessions).
S6: feature_importances.csv, report_feature_stability.csv and report_feature_families.csv.
{compare_source}
"""
    (out/"report_section.md").write_text(section,encoding="utf-8")
    full_table=c["supplement"].set_index("metric")[["mean","std","min","max"]].to_markdown(floatfmt=".6f")
    feature_table=top[["feature","importance_mean","importance_sd","best_rank","worst_rank","top5_folds"]].to_markdown(index=False,floatfmt=".4f")
    detailed=section.replace("# Random Forest — Yan Yan","# Random Forest analysis — Yan Yan",1)+f"""

## Supplementary RF summaries

These tables describe the same RF run; they are not additional experiments.

### Repeat-level ranges

{full_table}

Source: report_supplementary_metrics.csv, recomputed and checked against the original
repeat_metrics.csv and summary_metrics.csv without modifying those files.

### Feature stability across training models

{feature_table}

Source: feature_importances.csv and report_feature_stability.csv.
The 50 models use overlapping training data; SD is descriptive split variability.

### Calibration and diagnostics

![Separate per-repeat calibration and signed bin gaps](figures/report_calibration.png)

![Repeat-level RF performance](figures/repeat_metrics.png)

![Training and outer-validation diagnostic](figures/training_validation_gap.png)

![Selection counts](figures/parameter_selection.png)

![Conditional inner-CV complexity patterns](figures/inner_parameter_results.png)

![Training-model feature importance](figures/feature_importance.png)

## Data attribution

Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993).
Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository.
https://doi.org/10.24432/C5DW2B . CC BY 4.0:
https://creativecommons.org/licenses/by/4.0/ .
Modifications include binary outcome coding, ID sorting, and derived cross-validated
predictions, diagnostics and summaries for this educational study.
"""
    (out/"model_report.md").write_text(detailed,encoding="utf-8")
    coverage=[
        ("500 trees / 16 settings","complete","fold_results.csv; inner_cv_results.csv"),
        ("10 repeated stratified 5-fold outer / 5-fold inner","complete","oof_predictions.csv; selected_inner_oof.csv; run_manifest.json"),
        ("Training-only OOF 95% target; locked outer thresholds","complete","selected_inner_oof.csv; fold_results.csv"),
        ("Pooled repeat discrimination and probability quality","complete","report_main_results.csv; report_supplementary_metrics.csv"),
        ("Calibration pattern and zero-probability caution","completed explanation","report_calibration_bins.csv; report_probability_extremes.csv"),
        ("Training-validation gap and controlled grid contrasts","completed explanation","fold_results.csv; report_parameter_contrasts.csv"),
        ("Feature importance and stability","completed explanation","feature_importances.csv; report_feature_stability.csv"),
        ("Runtime","complete","report_runtime.csv"),
        ("Methods / interpretation / application / limitations","completed writing","report_section.md"),
        ("Owner-specific SVM paired comparison",c["status"]["svm"],"comparison_provenance.json; owner_* CSV"),
        ("Owner-attributed regression comparison","pending","Await Haiyi confirmation/export; no shared-table substitution"),
    ]
    pd.DataFrame(coverage,columns=["proposal_responsibility","status","evidence"]).to_csv(out/"report_coverage.csv",index=False)
    sources=[
        ("S1","Design and threshold execution","fold_results.csv; inner_cv_results.csv; selected_inner_oof.csv; oof_predictions.csv","50 outer fits; 16 settings each; locked training-only OOF thresholds"),
        ("S2","Primary and supplementary metrics","repeat_metrics.csv; report_main_results.csv; report_supplementary_metrics.csv","Pool 569 validation predictions per repeat; mean/sample SD/min/max over 10 repeats"),
        ("S3","Calibration and probability extremes","calibration_points.csv; report_calibration_bins.csv; report_calibration_pattern.csv; report_probability_extremes.csv","Separate quantile bins per repeat; signed group mean minus observed fraction"),
        ("S4","Parameter selection and conditional contrasts","inner_cv_results.csv; report_parameter_selection.csv; report_parameter_marginals.csv; report_parameter_contrasts.csv","Selection counts; within-outer-split candidate differences with other grid parameters held fixed"),
        ("S5","Training gap and runtime","fold_results.csv; run_manifest.json; report_runtime.csv","Mean within-fit AUROC gap; sum pilot/run session wall seconds separately from task seconds"),
        ("S6","Feature importance and stability","feature_importances.csv; report_feature_stability.csv; report_feature_families.csv","50 model-specific rankings; mean/sample SD/range and top-five frequency; descriptive family sums"),
        ("S7","Verified owner comparison","comparison_provenance.json; comparison_sources/asher/outer_predictions.csv; owner_* CSV",c["status"]["svm"]+"; paired RF-minus-comparator differences by repeat; regression pending"),
    ]
    pd.DataFrame(sources,columns=["source_note","report_content","csv_or_record","aggregation"]).to_csv(out/"report_sources.csv",index=False)
    audit=f"""# RF report source audit and handover

This revision concerns Yan Yan's proposal-assigned Random Forest analysis. It did not
train any model, alter the grid, add calibration/feature-set/learning-curve experiments,
or change primary predictions.

Research standard: current repository 707-proposal.md, blob
55e39287b36359ed8ab83449f6ad63d963439dd7. Shared loader, outer splitter, metric and
threshold functions were inspected and checked against direct calculations.
RF input numbers came only from this run's PRIMARY files listed in report_verification.json.

## Comparison source decisions

- SVM: {c['status']['svm']}.
- Regression: pending owner-attributed compatible per-record predictions.
{owner_audit}
- Haiyi's branch contains a report and coefficient replay. Its available per-record
  prediction files trace to inherited shared import history; that does not establish
  owner-attributed delivery of those predictions. No regression numerical conclusion
  remains in the current report. This is a source limitation, not a judgment about
  the member's effort.
- Prior unsupported regression comparator CSVs are excluded from the current result
  directory and retained outside the publishable report as superseded audit records.
  If an earlier personal package was uploaded, remove its comparator_repeat_metrics.csv,
  paired_repeat_differences.csv and paired_difference_summary.csv from the active report
  directory; do not use them as comparison evidence.

## Outputs for integration

- report_section.md: six-section English research text, with source notes.
- report_main_results.csv: compact RF table.
- report_supplementary_metrics.csv: complete repeat mean/SD/min/max.
- model_report.md: detailed scientific report, tables and figures.
- comparison_provenance.json: exact owner source attribution/version and exclusions.
- report_coverage.csv / report_verification.json: completeness and unchanged-input checks.

## Remaining questions

1. Haiyi should provide or explicitly confirm compatible owner-attributed 5690-row
   nested predictions per regression model, source/config version and training-only
   selection provenance.
2. Confirm the stable logistic comparator and any prespecified fallback reason with
   the regression owner, using that owner's evidence.
3. The two SVMs' achieved sensitivities differ from RF. No claim of strictly
   matched achieved sensitivity or clinical utility is made.
4. Existing cross-model orchestration/SVM-entry limitations remain an operational
   issue outside this writing task; no member's code/report was changed.

The earlier RF model training actually reused shared infrastructure and was AI-assisted.
Those facts remain recorded. No independent handwriting, personal understanding or
personal reaction is invented for Yan. Operational version/upload records are kept
outside the formal report section.
"""
    (out/"report_audit.md").write_text(audit,encoding="utf-8")
