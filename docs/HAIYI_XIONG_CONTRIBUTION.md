# Individual contribution and AI use

Haiyi Xiong · BIOSTAT 707

My assigned contribution was the logistic regression comparison. My report section covers the unpenalized, ridge, and Elastic Net models, including the Elastic Net search, fold-specific thresholds, and the performance changes across training sizes and predictor groups. It uses the team's shared analysis results and adds calibration figures, a fold-level coefficient table, and a summary of coefficient direction and stability. [Report section](HAIYI_XIONG_REPORT_SECTION.md)

I used Codex to run model checks, verify the team's saved logistic predictions against the supplied outer-fold file, and prepare the figures and report text. Automated checks matched all 17,070 prediction rows by repeat, fold, and record ID. Codex also refitted the 150 outer-training models with the team's pinned scientific libraries and checked their predictions against the archived values. Both penalized models reproduced all archived probabilities within a tolerance of 10⁻⁷. Three unpenalized probabilities exceeded that tolerance, so the primary performance estimates in my chapter use the team's archived outputs.

The unpenalized model illustrates why a convergence check needs more than the solver’s warning status. Its largest coefficients were orders of magnitude larger than those from the penalized models, and nearly all its validation probabilities were near zero or one. The resulting log loss made the instability visible even though the solver recorded no convergence warnings. The ridge model gave the strongest average probability accuracy and provides a stable linear reference for the team comparison.
