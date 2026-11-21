# RF report source audit and handover

This revision concerns Yan Yan's proposal-assigned Random Forest analysis. It did not
train any model, alter the grid, add calibration/feature-set/learning-curve experiments,
or change primary predictions.

Research standard: current repository 707-proposal.md, blob
55e39287b36359ed8ab83449f6ad63d963439dd7. Shared loader, outer splitter, metric and
threshold functions were inspected and checked against direct calculations.
RF input numbers came only from this run's PRIMARY files listed in report_verification.json.

## Comparison source decisions

- SVM: verified owner-specific nested predictions; metrics recomputed from pooled outer records.
- Regression: pending owner-attributed compatible per-record predictions.
- Owner-specific Asher nested predictions were used directly. The 80/20 results and legacy fold-averaged nested tables were excluded. Metrics were recomputed by model and repeat after pooling five folds, using stored per-row thresholds.
- RF and SVM inner seed schedules differ, while actual outer ID/label/repeat/fold mappings agree. Three-fold sigmoid SVM calibration is a prescribed model difference.
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
