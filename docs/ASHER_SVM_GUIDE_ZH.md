# Asher Xu 的 SVM 工作与交接说明

## 已完成的工作

线性核 SVM 与 RBF 核 SVM 均已使用团队固定的 455 条开发记录、五折划分和 114 条测试记录完成。研究任务以当前 `707-proposal.md` 和 `docs/TEAM_WORKFLOW.md` 为准，不使用个人 Checkpoint 数据或代码。

- 线性核搜索 4 个 C，RBF 核搜索 16 个 C 与 gamma 组合。
- 标准化位于被校准的 SVM 流程内部；每次模型拟合内部进行三折 sigmoid 校准。
- 所有候选配置只在开发集比较，以五折平均 AUROC 选参数。
- 选定参数后的开发集折外预测用于锁定至少 95% 敏感度的最高阈值。
- 先完成两类模型的参数与阈值锁定，再拟合最终模型并生成测试概率。测试结果未用于调参、改阈值或重新校准。
- 已对接团队公共评价，包括 2,000 次配对分层 bootstrap。
- 已准备英文方法、结果、解释和局限性报告；尚未 commit 或 push。

## 从哪些文件开始看

| 文件 | 用途 |
|---|---|
| `docs/ASHER_SVM_REPORT.md` | 完整英文方法、结果、模型比较与局限性 |
| `results/svm/svm_summary.csv` | 两个模型在开发集与测试集的核心指标 |
| `results/svm/cv_search_results.csv` | 所有参数组合与逐折 AUROC |
| `results/svm/development_lock.json` | 测试前锁定的参数、阈值、首选模型及哈希 |
| `results/svm/runtime_summary.csv` | 参数搜索与最终拟合、预测运行时间 |
| `results/svm/verification.json` | 交付文件核查记录 |
| `results/svm/figures/` | 开发集与测试集分开的 ROC、PR、校准及解释图 |

## 交给组员的四份预测

这些文件均位于 `results/predictions/`。

| 文件 | 行数 | 字段 |
|---|---|---|
| `oof_Linear_SVM.csv` | 455 | id, y_true, model_name, fold, y_prob |
| `oof_RBF_SVM.csv` | 455 | id, y_true, model_name, fold, y_prob |
| `test_Linear_SVM.csv` | 114 | id, y_true, model_name, split, y_prob |
| `test_RBF_SVM.csv` | 114 | id, y_true, model_name, split, y_prob |

模型名称分别为 `Linear_SVM`、`RBF_SVM`，与团队接口一致。`y_prob` 始终是恶性（1）的概率。阈值保留完整精度，不用报告表中四舍五入后的数值重新分类。

## 怎样理解结果

线性模型的五折平均 AUROC 略高，因此在打开测试结果之前已被指定为首选 SVM。测试结果应同时比较 AUROC、Brier score、log loss、校准和误报漏报，不能只按单一指标排名。

95% 是开发集目标，不是测试集保证。请明确报告实际测试敏感度，不因为测试集没有达到目标就调低阈值。单次测试只有 42 条恶性记录，一条记录就会明显影响敏感度，模型差异必须结合置信区间解释。

开发集 OOF 在同一组五折选择超参数之后汇总，存在选择乐观性，不应称作完全无偏的嵌套交叉验证结果。最终独立保留集是本项目的内部测试，不是外部医院验证。

特征重要性来自开发集验证折，未用来筛选特征。相关变量会分摊重要性；线性 SVM 的系数是标准化分类间隔系数，不是 odds ratio，也不代表因果作用。

## 环境与执行方法

本次使用已有课程环境 `F:\Bios_707_work\.pixi\envs\default\python.exe` 作为解释器，仅复用已安装的软件包，没有修改该个人作业项目，也没有使用其数据或分析代码。实际版本见 `requirements-svm.txt` 和锁定文件；其中 pandas 与公共 `requirements.txt` 的版本不同，英文报告已披露。

先进入项目目录。已有本地结果时，只需要核查：

```powershell
Set-Location 'E:\Bios_707_fianl'
& 'F:\Bios_707_work\.pixi\envs\default\python.exe' scripts/run_svm.py verify
```

运行全部自动检查：

```powershell
& 'F:\Bios_707_work\.pixi\envs\default\python.exe' scripts/run_svm.py check
```

在新环境中安装 `requirements-svm.txt` 后，也可以用自己的 `python` 执行相同入口。Windows conda 环境需要的 DLL 搜索路径已由入口处理。

若在新的干净工作副本中完整复现，顺序是：

```text
python scripts/run_svm.py check
python scripts/run_svm.py develop
python scripts/run_svm.py finalize
python scripts/run_svm.py verify
python scripts/run_svm.py evaluate
python scripts/svm_report.py
```

当前工作副本已有锁定和完成文件，`develop` 与 `finalize` 会拒绝重复运行，避免无意重新调参与反复测试。单纯核查或生成报告不会训练模型或生成新的测试预测。

## GitHub 交接

本地分支是 `asher/svm-analysis`。先检查 `git diff` 和文件清单，确认后提交自己的代码、文档、四份预测、搜索记录和必要结果，再 push 此分支并创建 PR。不要直接覆盖其他组员分支。

`models/svm/` 的拟合模型保留在本地，不作为必需上传项；完整复现由训练代码和已记录环境提供。公共比较图表目前只包含常数基线与两类 SVM，不能声称其他队员的模型已完成。若团队希望由 Tianyi 统一生成公共图表，可在 PR 中只提交 SVM 自有图表与预测，由数据负责人运行公共评价。

老师要求个人完成的 AI 使用优缺点反思，应由本人根据真实经历填写；此处不代写个人反思。
