# Random Forest：交接与复现

## 当前交付状态

基础commit为06349f89853e55d9b46b98304e68d6580abafa48，分支yan/random-forest-review。
完整50个外层划分的RF主分析已完成。本次只修订结果解释、核对来源和导出报告，
没有训练模型，没有改原始预测、参数网格或主分析记录。
文件仍只在云端本地工作区，未commit、push或创建PR，未改GitHub主分支。

之前的共同入口整合保留：共同文件需要替换README.md和src/final_analysis.py；
详细说明见docs/RANDOM_FOREST_INTEGRATION.md。本次报告任务未再改变这两个文件。
个人目录是待整合审核产物；共同汇总只应纳入一个Random_Forest。

## 1. 下载后先读什么

1. report_section.pdf或report_section.md：可以交给报告整合组员的英文六节正文。
2. report_main_results.csv：精简主结果表；report_supplementary_metrics.csv：完整范围。
3. results_explained_zh.pdf或.md：中文结果解释、presentation示例和老师可能追问的问题。
4. model_report.pdf或.md：详细科学报告及图表。
5. report_audit.md、comparison_provenance.json和report_verification.json：来源与本次核查。
6. oof_predictions.csv等主分析CSV不可手工修改。report_*是读取它们生成的派生结果。

## 2. 本次报告补充

- 解释校准图的实际区间表现及跨repeat稳定性，而不只解释指标定义。
- 补充控制其他网格参数的内层AUROC对比、选择次数和特征排名稳定性。
- 区分训练/验证折均值与正式pooled-repeat AUROC。
- 所有外层评价保持每行对应的训练选定阈值，不重新挑全局阈值。
- Asher个人nested预测已直接核对，并从逐条记录按repeat重新计算指标。
- 回归预测归属尚不能确认，相关数值结论撤下，待海怡提供或明确确认。
- 不比较两份RF，不使用共同全模型表、共同报告或learning curves作为依据。

若上传过较早个人包，请从当前个人报告目录撤下这三个不再使用的派生文件：
comparator_repeat_metrics.csv、paired_repeat_differences.csv、paired_difference_summary.csv。
它们是此前来源未确认的比较表，不是RF原始预测；先备份到报告目录外，
不要继续引用。当前owner_*文件只包含来源已核对的Asher个人SVM比较。
历史源码/审核快照保留真实复用与运行记录，不作为当前报告数值输入。

## 3. 应交给组员的文件

报告整合：report_section.md + report_main_results.csv。
需要补充材料时附report_supplementary_metrics.csv、model_report.md及figures/。
比较来源附comparison_provenance.json、report_audit.md和owner_* CSV。
代码整合：将完整包按原路径放回仓库；新增/替换清单见integration_changes.csv，
本次报告修订清单见report_changes.csv。
共同训练入口已经接入src/yan/random_forest/integration.py，不只是复制预测文件。
不要把个人RF另起一个模型名或追加第二行RF参加比较。

## 4. GitHub网页上传步骤

1. 解压ZIP；保留它里面src、scripts、results、docs、tests等文件夹结构。
2. 在GitHub仓库从main创建个人分支，例如yan/random-forest-review。
3. 在个人分支使用Add file → Upload files。把对应文件放入仓库原路径，
   不上传ZIP本身，也不多套一层解压文件夹。
4. 按integration_changes.csv核对：共同文件只替换README.md与src/final_analysis.py；
   添加个人模块、报告和测试。不要覆盖src/metrics.py、其他成员报告或results/final_nested。
5. 如果先前已经上传个人包，同路径使用本包文件，并处理上文三个撤下的派生比较表。
6. 提交到个人分支，创建base=main、compare=个人分支的PR供组员审阅。
   没有写权限时交给有权限组员，或通过fork提交。不要直接合并main。
7. 上传后如实更新Git状态；本交付没有代替你提交或发布。

## 5. 在PyCharm读取或重建报告（不训练）

打开含根README.md的项目，先合并包里的文件。在项目根目录Terminal执行：

    python -m pip install -r requirements.txt
    python -m pip install -r src/yan/random_forest/requirements.txt
    python scripts/yan/run.py report --output results/yan/random_forest
    python -m unittest discover -s tests -p test_rf_reporting.py -v

测试命令需要把项目src设为Python导入路径。在PyCharm中将src标为Sources Root，
或在运行配置中设置环境变量PYTHONPATH=src；Linux/macOS也可在命令前加PYTHONPATH=src。
推荐Python 3.12；原分析依赖、版本和数据hash见run_manifest.json、environment.txt。
report命令只重建Markdown、派生CSV与校准图；交付PDF由同版Markdown另外导出。
原始8个主CSV的前后hash在report_verification.json，必须相同。

## 6. 通过共同入口从头训练（本次无需执行）

    python src/final_analysis.py nested --models Random_Forest --output results/rf_reproduction --no-resume --no-import-svm

使用新输出目录；该命令包含完整50个外层划分、16组参数和500棵树。
共同调度调用Yan模块，不再额外训练一次RF。去掉--no-resume可在同环境、
同数据/代码/配置下按核验后的检查点继续。详细核验和集成测试证据见整合说明。
共同SVM从头运行问题仍未处理；不能说所有模型已能从头重训。
不要为了读报告运行nested/pilot/run，也不要覆盖results/final_nested。

## 7. 需要本人和全组确认

- 海怡确认或提供兼容的5690行/模型预测、代码配置和训练内部选择证据。
- 海怡与组内确认稳定logistic比较对象及采用fallback的证据。
- Asher与RF内层seed不同、外层实际映射一致，报告已经如实披露。
- 本人理解内外层CV、OOF、95%非外层保证、AUROC与校准、重要性局限。
- 个人贡献和AI反思依据自己真实参与的理解、检查、修改和交流填写。
  ai_assistance.md只是实际操作记录，不替你虚构独立完成经历。
