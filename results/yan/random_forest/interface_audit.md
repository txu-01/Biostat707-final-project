# Shared-interface review and current report eligibility

The interface review below records the original fresh RF run. Source attribution
was tightened during the report revision: file/ID alignment alone does not make
a shared model export an eligible owner-provided comparator. Current eligibility
is defined in comparison_provenance.json and report_audit.md. Historical source
snapshots retain what was actually inspected; they are not report inputs.

Base: main commit 06349f89853e55d9b46b98304e68d6580abafa48.
Haiyi branch inspected: 0e51a3a63d545a9f69b7deca4fe58341c86b2aa4.
Asher branch inspected: 56e526017a9696c81f3b788c9b7ea0ad85116c6c.

The complete proposal, current README, TEAM_WORKFLOW, common_evaluation,
data_prep, final_analysis, metrics, requirements and both existing test files
were read. No AGENTS.md or additional project test configuration was found.
No previous RF output was read by the new implementation.

| Check | Source and existing implementation | This RF execution | Outcome / issue |
|---|---|---|---|
| Data and feature order | src/data_prep.py FEATURE_NAMES; src/final_analysis.py:110 load_data | Reuse shared loader; sort by numeric ID; compare every value with raw source | 569 rows, 212 M=1, 357 B=0, 30 ordered features, no missing/nonfinite cells, no exclusions |
| Outer split | src/final_analysis.py:321 _outer_split | Same generator and sorted data | Original interface check matched available file mappings; current report independently reconstructs all assignments and verifies owner-specific Asher predictions |
| Regression interface | Haiyi branch src/final_analysis.py, data_prep.py and metrics.py | Reuse main core; additional verification compares source bytes | Same shared core; personal regression coefficient/report supplements do not define new RF inputs |
| SVM actual code | Asher scripts/run_svm_nested.py:24-32 and src/train_svm.py make_model | Inspect source and archived outputs, do not run SVM | Same sorted records and outer split rule; actual prediction mapping checked |
| Inner seeds | Shared final_analysis.py:399-405 uses 1707+100r+f; Asher run_svm_nested.py:32 uses 1707+5r+f | Use shared/regression rule | Seeds differ after repeat 1; disclose rather than claim identical inner partitions |
| Candidate selection | Shared final_analysis.py:231-289 | Same 16 grid entries, five inner folds, mean inner-fold AUROC; exact ties use earliest entry | No outer-score selection, no reduced grid |
| OOF threshold | metrics.py:24 and final_analysis.py:285-289 | Same selected-candidate inner OOF and highest qualifying observed threshold | Save OOF and verify with an independent discrete threshold oracle; no third layer |
| Outer metrics | final_analysis.py:292 summarize_prediction_rows | Reuse row-level locked thresholds; pool five outer folds within repeat | Do not use generic evaluate_predictions(None), which would select a new threshold at metrics.py:159 |
| Legacy SVM summary | Asher run_svm_nested.py:73-78 averages fold-level metrics | Use main summarize_primary, final_analysis.py:537 onward | Legacy fold means do not satisfy pooled-record repeat AUROC; main recomputes pooled metrics from predictions |
| Undefined ratios | Generic metrics.py:65-68 replaces undefined ratios by zero | Use final_analysis.summarize_prediction_rows, returning NaN for undefined PPV/NPV | Actual primary folds contain both classes; undefined predictive values remain blank |
| Scaling/calibration | Regression/SVM scale inside training; SVM calibrated in fitting subsets | RF original scale, class_weight=None, no extra calibration | Reasonable model-specific differences retained |
| Runner side effects | Shared main may import SVM predictions and default to multiple models | Personal RF-only entry calls RF factory and correct helpers | No all-model run, SVM importer or formal report builder called |
| Checkpoints | Shared resume at final_analysis.py:359-362 uses unique-ID counts | New run UUID + source/data/config/environment fingerprint; full fold validation | Reject old/foreign/incomplete/misaligned checkpoints |
| Stable logistic comparator | Proposal wording and common_evaluation identify a stable comparator/fallback question | No regression numerical comparison without owner-confirmed compatible predictions | Pending Haiyi's prediction version and justification; no shared full-model report is used as evidence |
| Personal directory | TEAM_WORKFLOW discourages separate formal result sets | User explicitly requested yan review artifacts | Keep formal results untouched; team integrates exactly one RF version |

Here r denotes zero-based repeat and f denotes zero-based outer fold.
Output repeats are 1-10 and folds 0-4, matching the shared files.

The prompt is methodologically executable. Its explicit personal-directory exception
resolves the workflow's folder restriction without changing the common design.
The new RF wrapper is needed to collect full inner-fold scores, resubstitution
diagnostics, feature importance and strong checkpoint evidence that the generic
runner does not export. It does not define a separate validation design.

The actual record maps and exact SHA-256 inputs are in outer_fold_assignments.csv,
inner_fold_assignments.csv and input_audit.json. The extended final verification
also reads the actual Asher archived nested predictions and branch data when available.
