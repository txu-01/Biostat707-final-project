"""Research prose and Chinese explanation from audited RF-owned tables."""
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
    chinese_comparison="目前没有通过来源与键值核对的模型负责人预测；跨模型数值比较待整合。"
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
        chinese_comparison=("Asher个人nested预测已从其明确版本直接取得并核对。旧80/20结果和按折平均的nested汇总没有用于比较；"
            "重新合并每轮569条预测，计算同定义指标，且未重训SVM。可用结果和RF减去各SVM的配对差在owner_* CSV及report_model_comparison.csv。"
            f"RF的平均AUROC分别低{abs(dl.auroc_difference):.4f}和{abs(dr.auroc_difference):.4f}；"
            f"Brier分别高{dl.brier_score_difference:.4f}和{dr.brier_score_difference:.4f}，log loss也较高。"
            f"RF每轮FN平均{m.FN:.1f}，两种SVM分别{linear.FN:.1f}和{rbf.FN:.1f}；"
            f"RF的FP平均{m.FP:.1f}，SVM分别{linear.FP:.1f}和{rbf.FP:.1f}。"
            "RF较少漏判伴随较多误报，各模型实际敏感度不同，不能说严格相同敏感度下比较。"
            "这些是配对的描述性差异；没有显著性检验，也不证明等效或临床效用。")
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
    chinese=f"""# Yan Yan 的 Random Forest：结果解释与 presentation 准备

## 1. 本次分析做了什么

这是按707-proposal.md完成的Random Forest分析。
使用全部569条记录，其中212条恶性、357条良性，30个原始尺度特征。
500棵树和16组参数固定；10次重复、每次5折外层CV，
每个外层训练集内部再做5折调参。阈值只由训练内部OOF选择。
本次报告修订只读取已完成的输出，没有训练任何模型或改变参数。
代码复用了经检查的共同数据读取、CV划分、森林构造和评价函数；
实际执行包含AI辅助，不等于本人独立手写、运行或审核过每一步。

## 2. 主结果怎么讲

{table}

AUROC {m.auroc:.4f}表示在这批记录上的排序表现较好，但不证明概率都可靠。
AUPRC用average precision，不是另一种积分方式的PR面积。
每轮先合并5个验证折的569条记录，再计算指标；之后总结10轮。
FN均值{m.FN:.1f}、FP均值{m.FP:.1f}不是“半个人”，
而是10轮中整数错误计数的平均值。
10轮重复使用同一批记录，5690条预测不能说成5690名独立患者。
来源：repeat_metrics.csv、report_main_results.csv。

## 3. 校准图的实际发现

平均预测恶性概率{m.mean_predicted*100:.2f}%，实际恶性比例
{m.observed_prevalence*100:.2f}%，总体上接近，但局部偏差可能相互抵消。

- 第三个分位组在全部10轮都高估：组内平均预测
  {third.mean_pred_min*100:.1f}%–{third.mean_pred_max*100:.1f}%，实际
  {third.observed_min*100:.1f}%–{third.observed_max*100:.1f}%。
- 第四组在全部10轮都低估：平均预测
  {fourth.mean_pred_min*100:.1f}%–{fourth.mean_pred_max*100:.1f}%，实际
  {fourth.observed_min*100:.1f}%–{fourth.observed_max*100:.1f}%。
- 最高组也略低估，平均预测约{fifth.mean_pred_min*100:.1f}%–{fifth.mean_pred_max*100:.1f}%，
  本数据组内实际比例100%；这不保证其他人群也如此。
- 最低两个组方向不稳定，不能概括成“一直高估”或“一直低估”。

这些数字是各组的均值范围，不是固定概率区间的上下限。
分组每轮单独形成；概率并列使人数不完全相等，约5组也比较粗。
有{zero_repeats}轮出现恶性验证记录被给出0概率；
模型输出0不表示现实中的恶性概率不可能大于0。
Brier和log loss综合反映概率质量，不能把它们都解释成纯校准误差。
log loss为计算稳定性临时截断概率，原始导出概率没有改。
来源：calibration_points.csv、report_calibration_bins.csv、
report_calibration_pattern.csv、report_probability_extremes.csv。

## 4. 95%阈值不是外层保证

每个外层训练集使用各自的内层OOF概率，选达到至少95%敏感度的最高阈值。
某条OOF记录的预测模型没有用它拟合，但这些OOF同时用于了选参数，
因此不能把内层成绩说成独立验证结果。
参数和阈值确定后才评价外层记录，阈值没有重新调整。
实际50个阈值范围{threshold_min:.4f}–{threshold_max:.4f}。
合并后的外层敏感度范围{r.sensitivity.min()*100:.2f}%–{r.sensitivity.max()*100:.2f}%，
有{below}轮低于95%，照实报告。
FN是恶性记录被判成阴性；FP是良性记录被判成恶性。
延迟复核或增加复核只能作为假设性讨论；本研究没有测量后续临床后果。
来源：fold_results.csv、selected_inner_oof.csv、repeat_metrics.csv。

## 5. 选参数是否稳定

最常选中max_depth={most['max_depth']}、min_samples_leaf={most['min_samples_leaf']}、max_features={most['max_features']}，
占{int(selection.selected_folds)}/50。无限制深度{unrestricted}/50，sqrt为{sqrt_count}/50；
叶节点最小样本数1、3、5、10分别选中{leaf_counts[0]}、{leaf_counts[1]}、{leaf_counts[2]}、{leaf_counts[3]}次。
总共出现{len(c['selection'])}种配置，因此“大方向偏好一致”不等于“每次都选同一配置”。
固定sqrt和无限制深度时，leaf=1相对leaf=10的平均内层AUROC差为{leaf:+.6f}；
固定sqrt和leaf=1时，None相对深度3的差为{depth:+.6f}。
其他设置已控制，但候选模型种子也不同，不能解释成因果效应。
没有看外层分数后再改网格。
来源：inner_cv_results.csv、report_parameter_selection.csv、report_parameter_contrasts.csv。

## 6. 为什么有两个验证AUROC数字

训练集resubstitution AUROC平均{f.train_resubstitution_auroc.mean():.4f}，
50个外层折AUROC的平均{f.validation_auroc.mean():.4f}，
平均差{f.training_validation_gap.mean():.4f}。
训练记录已用于拟合，训练成绩偏乐观，这只是诊断。
正式主结果{m.auroc:.4f}是每轮合并569条OOF后计算AUROC，再平均10轮。
“先每折算再平均”与“合并预测再算”不同，不能拿它们互相替代。
没有根据这个差距重新调参。来源：fold_results.csv、repeat_metrics.csv。

## 7. 特征重要性如何解释

主要特征是{feature_names}。
这五个特征在全部50个训练模型都位于前五，但先后顺序和权重变化。
大小信息包括半径、周长、面积；轮廓不规则性包括concave_points等。
重要性是训练模型的impurity-based importance，不是回归系数。
相关特征可能分摊或替代重要性，不能从中推出方向、独立贡献、因果或具体交互。
本次没有依据重要性筛选特征或重新拟合。
来源：feature_importances.csv、report_feature_stability.csv、report_feature_families.csv。

## 8. 运行时间

完整分析记录的wall time为{c['wall']/60:.2f}分钟。
各外层任务耗时相加为{f.total_seconds.sum()/60:.2f}分钟，因为有并行，
两者不矛盾，也不能把任务时间简单说成等待时间。
阈值复用了搜索时的OOF，没有虚构额外一轮OOF训练时间。
来源：fold_results.csv、run_manifest.json、report_runtime.csv。

## 9. 跨模型比较的边界

{chinese_comparison}

海怡分支有报告和系数审核材料，但目前不能将现有逐条回归预测明确归属为她提供的兼容输出，
因此回归数值比较暂缓，不能用共同全模型表补位。
这不判断她是否完成工作，只说明当前可获得的比较输入证据不足。
需要她提供或明确确认预测版本、方法及稳定回归比较对象。
来源与版本说明在comparison_provenance.json和report_audit.md。

## 10. 可以怎样做简短presentation

“我负责Random Forest分析。我们按照方案使用500棵树和16组参数，
在共同的10次重复nested CV中评价，阈值只用训练内部OOF选定。
平均AUROC约{m.auroc:.4f}，但总体平均概率接近实际比例并不等于局部校准良好。
阈值下平均每轮漏判{m.FN:.1f}条恶性记录，误报{m.FP:.1f}条良性记录。
主要特征集中在大小和轮廓信息，但重要性不能解释因果或效应方向。
这些是历史数据中的诊断分类结果，不证明临床可部署或能预测未来患癌。”

## 11. 老师可能追问

- 为什么nested CV？内层选设置，外层评价整个选择流程。
- OOF是否完全独立？每条不参与该次拟合，但OOF参加过选参数；最终独立评价来自外层。
- 为什么验证低于95%？95%只是训练内部目标，外层保持原阈值。
- 为什么不看到校准偏差就加校准？本次主分析按事先方案，不看结果新增调整。
- 为什么不说RF优于回归？目前缺少可归属到回归负责人的兼容逐条输出。
- 为什么不能说严格相同敏感度下比错误？各模型达到的外层敏感度不同。
- 为什么不是5690名患者？同569条记录每轮预测一次，重复10轮。
- 特征重要性有正负方向吗？没有，MDI不是效应系数。
- 你本人贡献是什么？按实际做过的理解、核查、修改和整合填写，AI辅助记录另附。

## 12. 交给报告整合组员什么

交report_section.md和report_main_results.csv作为正式RF章节和简表。
report_supplementary_metrics.csv保留完整mean、SD、min、max；
model_report.md保留详细科学说明，comparison_provenance.json保留比较来源。
上传步骤、代码版本及接口核查放交接/审核材料，不复制进正式研究正文。
"""
    (out/"results_explained_zh.md").write_text(chinese,encoding="utf-8")
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
- results_explained_zh.md: interpretation and presentation preparation.
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
