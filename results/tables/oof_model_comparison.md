# Development Set Out-of-Fold Model Comparison

| model_name        |   auroc |   auprc |   brier_score |   log_loss |   ece |   locked_threshold_95 |   sens_at_locked |   spec_at_locked |   ppv_at_locked |   npv_at_locked |   fn_missed_cancer_at_locked |   fp_unnecessary_escalation_at_locked |   sens_at_0.5 |   spec_at_0.5 |
|:------------------|--------:|--------:|--------------:|-----------:|------:|----------------------:|-----------------:|-----------------:|----------------:|----------------:|-----------------------------:|--------------------------------------:|--------------:|--------------:|
| Constant_Baseline |     0.5 |  0.3736 |         0.234 |     0.6609 |     0 |                0.3736 |                1 |                0 |          0.3736 |               0 |                            0 |                                   285 |             0 |             1 |

*Thresholds selected exclusively on development OOF predictions targeting >=95% sensitivity.*
