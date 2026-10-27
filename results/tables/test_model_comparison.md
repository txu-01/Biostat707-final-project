# Final Test Set Model Comparison (One-Shot Held-Out Evaluation)

| model_name        |   test_N |   auroc |   auprc |   brier_score |   log_loss |    ece |   applied_threshold |   test_sensitivity |   test_specificity |   test_ppv |   test_npv |   missed_malignant_FN |   unnecessary_escalations_FP |   sens_at_0.5 |   spec_at_0.5 |
|:------------------|---------:|--------:|--------:|--------------:|-----------:|-------:|--------------------:|-------------------:|-------------------:|-----------:|-----------:|----------------------:|-----------------------------:|--------------:|--------------:|
| Constant_Baseline |      114 |     0.5 |  0.3684 |        0.2327 |     0.6582 | 0.0052 |              0.3736 |                  1 |                  0 |     0.3684 |          0 |                     0 |                           72 |             0 |             1 |

*Note: Applied thresholds were locked strictly on development OOF predictions without seeing test labels.*
