# 随机森林共同入口整合：交接与审阅

## 这次完成了什么

这次接入了共同训练入口，Random_Forest只调度到Yan模块并训练一次。
共同命令仍是 src/final_analysis.py nested；当模型列表包含 Random_Forest 时：

1. run_primary_nested 调用 yan.random_forest.integration.run_shared_random_forest。
2. 适配器调用原个人模块 train.run_fold，执行完整内层搜索、训练内部OOF阈值、
   外层重拟合、预测及训练模型诊断。
3. 每个外层划分只执行一次。首次完整外层检查本身就是正式计划中的第一折。
4. 完成所请求的全部外层折后，核验详细检查点，生成原生详细CSV。
5. 明确转换并成套发布共同格式的预测、外层记录和内层搜索记录。
6. 共同 summarize_primary 读取三份共同文件，生成一个 Random_Forest 的汇总。

src/final_analysis.py 只增加21行：RF调度/重复请求保护、共同汇总前后核验，
以及直接启动脚本时的共享模块别名。其他模型构造器、调参核心、CV划分、
指标函数和SVM导入函数均未改变。源码AST比对与测试记录已保存。

## 新增文件与替换文件

相对本次基础 main commit 06349f89853e55d9b46b98304e68d6580abafa48：

- **需要替换：README.md。** 已区分核查/汇总已有结果、从头训练及续跑。
- **需要替换：src/final_analysis.py。** 共同训练入口正式调用个人RF实现。
- **新增：src/yan/random_forest/。** 含原个人模块及 integration.py；
  verify.py 可验证精确历史源代码快照，不能把历史结果冒充本次整合后的新训练。
- **新增：scripts/yan/run.py。** 保留独立审核用途，共同训练不要求再运行它。
- **新增：tests/test_rf_integration.py。** 检验调度、兼容导出、防混用和单一RF汇总。
- **新增：tests/artifacts/random_forest_integration/。** 实际整合测试记录与证据压缩包，
  全部明确为测试产物，不是正式10-repeat结果。
- **新增：docs/RANDOM_FOREST_INTEGRATION.md。** 即本说明。
- **新增：results/yan/random_forest/。** 历史完整50折分析、报告和可核验的原始源码快照。

若你已经上传过旧个人包，同一路径的个人文件也应用本包版本覆盖。
不要把新旧包的 verify.py、integration.py 和共同文件拆开混用。
逐文件清单见同目录中的 integration_changes.csv。

## 共享数据、调参和阈值没有改变

共同读取全部569条记录，M=1、B=0，使用原来排序和全部30个特征。
适配器逐值检查传入数据与个人模块读取结果相同，并使用共同的外层划分。
内层为5折，16组配置，每个森林500棵树；按平均内层AUROC选配置。
阈值来自选中配置的训练内部OOF，按 probability >= threshold 分类。
没有额外校准、重加权、过采样、特征筛选或外层标签选阈值。
其他模型预测只用于折归属核对，不参与RF方案选择。

正式默认仍为10次重复×5个外层折。--repeats 1只用于独立整合测试，
会记录 formal_ten_repeat_analysis=false，不能作为正式10轮分析提交。

## 三份共同文件如何生成

以 --output results/reproduction_rf 为例：

```text
results/reproduction_rf/
├── predictions/Random_Forest.csv
├── outer_fold_metrics_Random_Forest.csv
├── inner_search_Random_Forest.csv
├── random_forest_export.json
├── outer_predictions_all_models.csv
├── repeat_metrics.csv
├── model_summary.csv
└── yan/random_forest/
    ├── active_run.json
    └── runs/RUN_ID/
        ├── run_manifest.json
        ├── checkpoints/
        ├── oof_predictions.csv
        ├── fold_results.csv
        ├── inner_cv_results.csv
        ├── selected_inner_oof.csv
        ├── feature_importances.csv
        ├── shared_export/
        └── previous_shared_files/
```

模型名称始终为 Random_Forest；repeat为1–10，fold为0–4，candidate为0–15。
预测字段含 id、y_true、model、repeat、fold、y_prob、threshold、predicted_class。
转换还添加共同格式所需的 seconds 和 convergence_warnings。
不是只复制预测CSV，而是一起核对、转换和发布三份相互关联的记录。

random_forest_export.json记录运行编号和三文件SHA-256。
若某次发布中断、只换了一份文件、记录编号不匹配或文件被改动，
共同汇总会报错，不会静默把新旧RF混在一起。
完成共同汇总后，summary_status为complete，
summary_rf_run_id应与RF运行编号一致；model_summary只有一行RF。

详细目录是审核证据，不是另一个模型；共同汇总不扫描它为第二份RF。

## 老师怎样从头运行

克隆包含本次整合代码的最终仓库版本，在项目根目录运行：

```bash
python -m pip install -r requirements.txt -r src/yan/random_forest/requirements.txt
python src/data_prep.py
python src/final_analysis.py nested --models Random_Forest --no-resume --no-import-svm --output results/reproduction_rf
```

用Python 3.12。这个共同命令自动完成50个外层划分并生成共同汇总，
不需要再分别运行个人pilot/run，也不会再进入原共同RF循环。
已有RF预测文件不会使这个从头命令跳过训练。
推荐用一个新的输出目录；本次未删除或改写 results/final_nested。

若中断，在相同代码、数据、配置和环境中续跑：

```bash
python src/final_analysis.py nested --models Random_Forest --no-import-svm --output results/reproduction_rf
```

续跑检查运行编号、完整关联记录、重复键、标签、折归属、概率和阈值，
不只检查唯一ID数量。不兼容的旧检查点会被拒绝。
只有Git提交编号或README文字变化、实际训练代码和配置不变时，
不会因此把同一RF运行误判为另一份训练。实际版本与文档仍会记录。

加 --no-resume 时创建新的运行编号，旧原生运行保留，
被替换的共同RF文件先备份。测试也实际验证了这一点。

只重算已有共同表，不训练：

```bash
python src/final_analysis.py summarize --output results/reproduction_rf --no-import-svm
```

## 这次实际怎么测试

测试目录是 /workspace/work/shared_rf_checks/live，独立于正式分析目录。
实际命令：

```bash
python src/final_analysis.py nested --models Random_Forest --repeats 1 --no-resume --no-import-svm --output /workspace/work/shared_rf_checks/live
python src/final_analysis.py nested --models Random_Forest --repeats 1 --no-import-svm --output /workspace/work/shared_rf_checks/live
python src/final_analysis.py nested --models Random_Forest --repeats 1 --no-resume --no-import-svm --output /workspace/work/shared_rf_checks/live
```

第一次：实际完成5个外层划分，共400个内层森林、5个最终森林，
得到569条预测、5条外层记录、80条候选记录，共同汇总读取成功。
第二次：同一运行编号，核验5个检查点，新增拟合0。
第三次：已有完整RF结果仍创建新运行编号，再实际拟合5个外层划分。
旧文件备份通过哈希核验；共同汇总始终只有一个RF。
每个森林均为500棵树，没有为测试缩小网格、内层折数或树数。

单元测试命令：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python src/yan/random_forest/test_checks.py
```

共同测试16项、个人审核测试10项均通过。非RF调用检查使用mock，
本次没有实际训练回归、决策树或SVM。
源码比对显示构造器、调参函数、划分函数、评价函数和SVM函数保持不变。

原来完整10轮RF仍是之前那次已审核分析。本次没有宣称把正式50折重新训练了一遍；
新的真实运行是为验证共同入口、续跑、强制重训和汇总衔接。
测试证据见 tests/artifacts/random_forest_integration/test_report.json。

## 你如何上传和审阅

1. 本次使用新的“integrated”ZIP，解压后按根目录相对路径合并。
2. 在GitHub自己的 yan/random-forest-review 分支上操作。
3. 先替换README.md和src/final_analysis.py，再上传新增src/yan、
   scripts/yan、tests文件及docs说明。
4. 结果文件较多，网页每批控制在100个文件以内。
   可以先用Create new file创建 results/yan/random_forest/README.md作为目录说明，
   然后进入该目录分批上传根文件、checkpoints、figures、provenance子目录。
   上传前核对最终路径，避免多套一层ZIP名称。
5. 只复制新包中提供的文件，不删除仓库中未出现在包内的其他成员文件。
6. 创建PR让组员审阅：base为main，compare为你的分支；不要直接合并main。
7. PR重点说明：共同RF已改为个人实现，完整三文件导出，强制重训和严格续跑通过测试；
   其他模型训练逻辑未改，SVM问题未处理。

变更清单记录相对基础main的新增/替换动作。如果之前已上传旧个人包，
请统一使用本包同路径文件，不同时保留两套入口版本。

## 未解决事项和Git状态

SVM从头重训入口仍不完整；本次未修改或修复它。
README明确可训练模型范围，不宣称全部8个模型都可从头重训。
共同绘图/报告生成脚本仍可能重写全组文档，未自动调用。
若今后共享数据、特征、CV或训练实现改变，需要新运行并重新核验，不能混用。

## 报告写作修订

完整RF分析无需因写报告而重训。正式英文章节是
results/yan/random_forest/report_section.md，精简表是report_main_results.csv，
中文讲解是results_explained_zh.md。报告只从本次RF原始记录和经确认的
模型负责人输出计算数字；Asher个人nested预测已核对并按repeat重算，
回归比较仍待海怡确认来源。共同全模型表不作为报告输入或正确性依据。
详情见report_audit.md和comparison_provenance.json。

本次只在云端本地修改，未commit、未push、未创建PR、未改GitHub权限。
本地分支：yan/random-forest-review。正式 results/final_nested 未修改。
AI操作记录与个人贡献反思仍应如实填写，不虚构本人独立完成的经历。
