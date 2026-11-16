# Final Team Workflow

This repository uses one shared analysis pipeline so all model families are evaluated with the same folds, preprocessing rules, metrics, and threshold procedure.

## Setup

```bash
python3 -m pip install -r requirements.txt
python3 src/data_prep.py
```

`data_prep.py` rebuilds and verifies the clean 569-record dataset. It does not create a permanent train/test split.

## Run the final analyses

```bash
PYTHONPATH=src python3 src/final_analysis.py nested
PYTHONPATH=src python3 src/final_analysis.py learning-curves
PYTHONPATH=src python3 src/final_analysis.py feature-sets
MPLBACKEND=Agg python3 scripts/build_final_outputs.py
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The main command runs ten repeats of stratified five-fold outer CV, with five-fold inner CV for model selection. All models receive identical outer folds. Scaling, SVM calibration, tuning, and threshold selection remain inside training data.

## Team review

Each model owner should check their model's methods, tuning grid, results, and interpretation in the report. Everyone should be able to explain nested CV, calibration, the exploratory 95% sensitivity threshold, and why a tiny AUROC difference does not prove superiority.

Each person must also supply their own contribution statement and AI-use reflection. The shared template is in `docs/AI_USE_REFLECTION_TEMPLATE.md`.

## Files to use for the final submission

- `docs/FINAL_REPORT_DRAFT.md`
- `docs/PRESENTATION_OUTLINE.md`
- `results/final_nested/model_summary.csv`
- `results/final_nested/repeat_metrics.csv`
- `results/final_nested/figures/`

Do not add separate model-specific splits or result folders. New results should be produced through the shared nested-CV pipeline.
