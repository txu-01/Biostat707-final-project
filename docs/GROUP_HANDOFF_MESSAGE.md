# WeChat Group Handoff Messages

Send these messages only after the shared infrastructure pull request has been
merged into the team repository's `main` branch. Replace `[PR link]` with the
merged pull-request URL if the group would benefit from a change summary.

## Message 1 — Shared infrastructure is ready

各位，我负责的项目基础框架已经完成并合并到团队仓库，自动测试 7/7 通过（PR：[PR link]）：

1. 完成 WDBC 数据清洗与完整性检查；
2. 使用 seed=707 锁定统一的 455 例开发集、114 例测试集，以及开发集内部统一的 5 折 CV；
3. 完成开发集 Table 1、类别分布、相关性热图和主要特征图；
4. 完成常数概率基线；
5. 完成公共评估管线，包括 AUROC、AUPRC、Brier、Log Loss、校准曲线、ECE、开发集 95% 敏感度阈值、混淆矩阵，以及 2,000 次 paired stratified bootstrap；
6. 总评估脚本会自动检查 ID、标签、fold 和概率是否合法，避免不同模型的结果错位；
7. 完整使用说明在 `docs/TEAM_WORKFLOW.md`。

请先运行：

```bash
git switch main
git pull origin main
python3 -m pip install -r requirements.txt
```

目前汇总表中只有 Constant Baseline。等大家的模型输出加入后，才会生成正式的跨模型比较结果。

## Message 2 — Each owner's next steps

大家接下来请按照自己的分工完成模型：

- Haiyi：Logistic Regression + Elastic Net
- Yan：Random Forest
- Wenqing：Decision Tree
- Asher：Linear SVM + RBF SVM

请严格使用 `data/processed/splits.csv`，不要重新随机切分：

1. 只在 `split == "development"` 的 455 条记录上，使用指定的 `cv_fold=0–4` 完成调参和 OOF 预测。所有标准化、特征处理及 SVM 校准都必须在训练 fold 内部拟合。
2. 锁定最佳参数后，生成完整的 455 条 OOF 预测，并在 OOF 结果上确定达到至少 95% 敏感度的最高阈值。
3. 参数和阈值完全锁定后，才可以在完整开发集上重新训练一次，并对 114 条 test 记录生成一次最终概率。Test 集不能用于调参、特征选择、阈值选择或重新校准。
4. 每个模型需要提交：可运行的训练代码、最佳参数与搜索结果、运行时间、OOF CSV、test CSV、锁定阈值，以及模型方法、结果、解释和局限性草稿。
5. 请使用统一模型名：`Logistic_Regression`、`Elastic_Net`、`Random_Forest`、`Decision_Tree`、`Linear_SVM`、`RBF_SVM`。
6. 请从自己的 GitHub 账号提交 commit，并通过 feature branch 创建 PR，不要直接覆盖别人的文件。

提交后请在群里发送 PR 链接。我会运行统一验证和总评估脚本，生成最终比较表、ROC/PR/校准曲线、阈值结果及成对 bootstrap 置信区间。
