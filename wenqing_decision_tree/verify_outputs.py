"""Check generated artifacts and common outer-fold assignments."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold

base = Path(__file__).resolve().parent
data = pd.read_csv(base.parent / 'data/processed/wdbc_clean.csv')
pred = pd.read_csv(base / 'results/outer_predictions.csv')
folds = pd.read_csv(base / 'results/outer_fold_metrics.csv')
assert len(pred) == 5690 and len(folds) == 50
assert pred.groupby('repeat').size().eq(569).all()
assert not pred.duplicated(['repeat', 'id']).any()
assert pred['probability'].between(0, 1).all()
assert (pred['prediction'] == (pred['probability'] >= pred['threshold']).astype(int)).all()
assert (folds['training_oof_sensitivity'] >= 0.95).all()
for repeat in range(1, 11):
    splitter = StratifiedKFold(5, shuffle=True, random_state=707 + repeat - 1)
    for fold, (_, idx) in enumerate(splitter.split(data, data['target']), 1):
        observed = pred[(pred['repeat'] == repeat) & (pred['fold'] == fold)]
        assert set(observed['id']) == set(data.iloc[idx]['id'])
        assert observed['threshold'].nunique() == 1
        assert np.isclose(observed['threshold'].iloc[0], folds.loc[(folds['repeat'] == repeat) & (folds['fold'] == fold), 'threshold'].iloc[0])
for name in ['roc_repeat_1.png', 'confusion_matrix_repeat_1.png', 'calibration_repeat_1.png', 'tree_example.png', 'feature_importance_example.png']:
    with Image.open(base / 'figures' / name) as image:
        image.verify()
for name in ['outer_predictions', 'outer_fold_metrics', 'overfitting_by_fold', 'overfitting_summary', 'inner_search', 'feature_importances', 'feature_comparison_by_fold', 'feature_comparison_summary', 'repeat_metrics', 'repeat_summary', 'calibration_repeat_1']:
    assert not pd.read_csv(base / 'results' / (name + '.csv')).empty
print('Artifact, prediction, sensitivity, and outer-fold checks passed.')
print('\nRepeat summary:')
print(pd.read_csv(base / 'results/repeat_summary.csv').to_string(index=False))
print('\nOverfitting summary:')
print(pd.read_csv(base / 'results/overfitting_summary.csv').to_string(index=False))
print('\nRepeat 1 confusion matrix:')
print(pd.read_csv(base / 'results/repeat_metrics.csv').loc[lambda d: d['repeat'] == 1, ['TN', 'FP', 'FN', 'TP']].to_string(index=False))
print('\nThresholds:', folds['threshold'].describe().to_string())
print('\nValidation sensitivity below 0.95:', (folds['sensitivity'] < .95).sum(), 'of 50 folds')
print('Calibration bins repeat 1:', len(pd.read_csv(base / 'results/calibration_repeat_1.csv')))
