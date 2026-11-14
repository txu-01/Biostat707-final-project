"""Build SVM-only plots and an English methods/results handoff from saved outputs.

No fitting, tuning, new probability predictions, or threshold selection occurs.
Run: python scripts/svm_report.py
"""
import os
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
DLL_HANDLES = []
if os.name == 'nt':
    for directory in [Path(sys.prefix) / 'Library' / 'bin', Path(sys.prefix) / 'DLLs']:
        if directory.exists():
            DLL_HANDLES.append(os.add_dll_directory(str(directory)))
            os.environ['PATH'] = str(directory) + os.pathsep + os.environ.get('PATH', '')
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, str(ROOT / 'src'))
os.chdir(ROOT)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from evaluate_all import plot_curves

OUT = ROOT / 'results' / 'svm'
NAMES = ['Linear_SVM', 'RBF_SVM']
lock = json.loads((OUT / 'development_lock.json').read_text(encoding='utf8'))
complete = json.loads((OUT / 'test_complete.json').read_text(encoding='utf8'))

def metrics(name, stage):
    return json.loads((OUT / f'{name}_{stage}_metrics.json').read_text(encoding='utf8'))


for prefix, folder in [('oof', 'development'), ('test', 'test')]:
    frames = {name: pd.read_csv(ROOT / 'results' / 'predictions' / f'{prefix}_{name}.csv').sort_values('id')
              for name in NAMES}
    plot_curves({name: df.y_prob.to_numpy() for name, df in frames.items()},
                frames[NAMES[0]].y_true.to_numpy(),
                output_prefix=str(OUT / 'figures' / folder))

importance = pd.read_csv(OUT / 'development_permutation_importance.csv')
fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
for ax, name in zip(axes, NAMES):
    top = importance.loc[importance.model_name == name].nlargest(10, 'mean').sort_values('mean')
    ax.barh(top.feature, top['mean'], xerr=top['std'].fillna(0), color='#346888')
    ax.set_title(name.replace('_', ' '))
    ax.set_xlabel('Mean validation AUROC decrease after permutation')
    ax.axvline(0, color='black', linewidth=0.6)
fig.savefig(OUT / 'figures' / 'development_feature_importance.png', dpi=160)
plt.close(fig)

cv = pd.read_csv(OUT / 'cv_search_results.csv')
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
linear = cv.loc[cv.model_name == 'Linear_SVM'].sort_values('C')
axes[0].errorbar(linear.C, linear.mean_cv_auroc, yerr=linear.sd_cv_auroc, marker='o', capsize=3)
axes[0].set_xscale('log')
axes[0].set_xlabel('C')
axes[0].set_ylabel('Mean fold AUROC with fold SD')
axes[0].set_title('Linear SVM development tuning')
rbf = cv.loc[cv.model_name == 'RBF_SVM'].pivot(index='C', columns='gamma', values='mean_cv_auroc')
image = axes[1].imshow(rbf.to_numpy(), origin='lower', aspect='auto', cmap='Blues')
axes[1].set_xticks(range(len(rbf.columns)), [str(v) for v in rbf.columns])
axes[1].set_yticks(range(len(rbf.index)), [str(v) for v in rbf.index])
axes[1].set_xlabel('gamma'); axes[1].set_ylabel('C')
axes[1].set_title('RBF SVM mean development AUROC')
for i in range(len(rbf.index)):
    for j in range(len(rbf.columns)):
        axes[1].text(j, i, f'{rbf.iloc[i,j]:.4f}', ha='center', va='center', fontsize=8,
                     color='white' if rbf.iloc[i,j] > 0.984 else 'black')
fig.colorbar(image, ax=axes[1], label='AUROC')
fig.savefig(OUT / 'figures' / 'development_tuning.png', dpi=160)
plt.close(fig)

rows = []
for name in NAMES:
    for stage in ['development', 'test']:
        m = metrics(name, stage); t = m['threshold_target_95']
        rows.append({'model': name, 'sample': 'development OOF' if stage == 'development' else 'test',
                     'N': m['N'], 'AUROC': m['auroc'], 'AUPRC': m['auprc'],
                     'Brier': m['brier_score'], 'log_loss': m['log_loss'], 'ECE': m['ece'],
                     'threshold': t['threshold'], 'sensitivity': t['sensitivity'],
                     'specificity': t['specificity'], 'PPV': t['ppv'], 'NPV': t['npv'],
                     'FN': t['FN'], 'FP': t['FP'], 'TP': t['TP'], 'TN': t['TN']})
summary = pd.DataFrame(rows)
summary.to_csv(OUT / 'svm_summary.csv', index=False)
fig, ax = plt.subplots(figsize=(7, 4.5), layout='constrained')
for name, color in zip(NAMES, ['#7253a3', '#bd3037']):
    values = metrics(name, 'test')['threshold_target_95']
    ax.scatter(values['FP'], values['FN'], s=85, color=color,
               label=f"{name.replace('_', ' ')} (sensitivity {values['sensitivity']:.1%})")
    ax.annotate(f"threshold {values['threshold']:.4f}", (values['FP'], values['FN']),
                xytext=(10, 5), textcoords='offset points')
ax.set(xlabel='Benign records classified positive (FP)',
       ylabel='Malignant records classified negative (FN)',
       title='Test errors at development-locked thresholds', xlim=(0, 3), ylim=(0, 4))
ax.set_xticks(range(4)); ax.set_yticks(range(5)); ax.legend(loc='upper left', fontsize=8)
fig.savefig(OUT / 'figures' / 'locked_threshold_tradeoffs.png', dpi=160)
plt.close(fig)
ci = pd.read_csv(ROOT / 'results' / 'tables' / 'test_bootstrap_confidence_intervals.csv')
ci = ci.loc[ci.model_name.isin(NAMES)]
pairwise = pd.read_csv(ROOT / 'results' / 'tables' / 'test_bootstrap_pairwise_differences.csv')
svm_pair = pairwise.loc[pairwise.model_name.isin(NAMES) & pairwise.comparator.isin(NAMES)]
runtime = json.loads((OUT / 'runtime_summary.json').read_text(encoding='utf8'))
runtime_table = pd.DataFrame([{'model': n, 'search_seconds': lock['models'][n]['search_seconds'],
                             **runtime[n]} for n in NAMES])
runtime_table.to_csv(OUT / 'runtime_summary.csv', index=False)

lines = [
    '# Asher Xu SVM Methods Results and Interpretation', '',
    '## Scope and status', '',
    'Both assigned SVM variants are complete locally. This report uses the frozen shared split '
    '(455 development records and 114 test records). Results are educational and do not establish clinical utility. '
    'Only the constant baseline and these two SVMs are currently available in this working copy; '
    'the other teammates\' models have not been evaluated here.', '',
    '## Methods', '',
    'All 30 original numeric predictors were retained; ID, diagnosis, target, and split/fold fields '
    'were excluded from model inputs. Data contain no missing predictor values, so no imputation was needed. '
    'All models used the shared saved five-fold development assignments and random seed 707.', '',
    'Linear SVM used C in {0.01, 0.1, 1, 10}; RBF SVM crossed those C values with gamma in '
    '{0.001, 0.01, 0.1, 1}. Candidate selection maximized mean fold AUROC, with deterministic '
    'first-grid-entry tie breaking within tolerance 1e-12. This is 4 and 16 configurations, respectively. '
    'C controls the penalty for margin violations; gamma controls the locality of the RBF similarity.', '',
    'Each fit wrapped Pipeline(StandardScaler, SVC) in CalibratedClassifierCV(method="sigmoid", '
    'cv=three-fold stratified CV, ensemble=False). The scaler was inside the wrapped estimator, '
    'so internal calibration predictions also used training-only standardization. Calibration learned '
    'a sigmoid from internal out-of-fold decision scores; the base pipeline was then fitted on '
    'all data supplied to that fit. No SVC internal probability=True routine was used. '
    'The official calibration documentation is at '
    'https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html.', '',
    'The selected candidate\'s five validation folds supplied one OOF probability per development record. '
    'The highest OOF threshold achieving at least 95% sensitivity was locked. Hyperparameters and thresholds '
    'were saved before final refits and before test prediction. Both final models were fitted on the full '
    'development set before accessing held-out features and labels. No model or threshold was changed '
    'after the held-out results. A threshold of 0.5 is a secondary reference in saved metric JSON files.', '',
    '**Important distinction:** mean fold AUROC selected hyperparameters; pooled OOF AUROC summarizes '
    'the concatenated validation predictions. These are different quantities. Because tuning used the '
    'same five folds, the selected OOF results are post-selection development summaries, not an unbiased '
    'nested-CV estimate. The held-out set provides the final internal evaluation.', '',
    '## Locked development decisions', '',
    f'Preferred SVM before test evaluation: **{lock["preferred_svm_before_test"]}**.', '',
]
for name in NAMES:
    s = lock['models'][name]
    lines.append(f'- {name}: parameters `{s["params"]}`; mean five-fold AUROC '
                 f'{s["mean_cv_auroc"]:.6f}; locked threshold {s["threshold"]:.10f}.')
lines += ['', '## Observed performance', '', summary.to_markdown(index=False, floatfmt='.4f'), '',
          '## Test uncertainty', '', ci.to_markdown(index=False), '',
          'Intervals use the team evaluator\'s 2,000 paired class-stratified resamples of the same '
          '114 test records. They condition on fitted models and this split, omit retraining and '
          'tuning variability, and are exploratory rather than equivalence/noninferiority tests.', '',
          '### Paired comparison between the SVM variants', '', svm_pair.to_markdown(index=False), '',
          'Differences are model minus comparator. Positive AUROC and negative Brier differences '
          'favor the model column. An interval spanning zero does not establish a clear performance '
          'difference. A higher point estimate alone is not sufficient to claim superiority.', '',
          '## Application and interpretation', '']
for name in NAMES:
    m = metrics(name, 'test'); t = m['threshold_target_95']
    lines.append(f'For {name}, the locked threshold produced {t["FN"]} false negatives out of '
                 f'{t["TP"]+t["FN"]} malignant records and {t["FP"]} false positives out of '
                 f'{t["TN"]+t["FP"]} benign records. Sensitivity was {t["sensitivity"]:.3f} and '
                 f'specificity {t["specificity"]:.3f}. These describe classifications of already '
                 'sampled masses, not measured downstream patient outcomes or numbers of biopsies.')
    lines.append('')
lines += [
    'On this held-out split, linear SVM had a slightly higher AUROC (0.9974 versus 0.9957), '
    'lower Brier score (0.0216 versus 0.0259), and lower log loss (0.0750 versus 0.0857). '
    'However, RBF SVM had a smaller five-bin ECE (0.0083 versus 0.0229) and one fewer false '
    'negative at its own locked threshold. Probability scoring rules and binned calibration '
    'summaries measure different aspects; these rankings need not agree. The paired AUROC '
    'difference was about 0.0017 and its interval reached zero; the Brier difference interval '
    'also included zero. This analysis does not establish that either SVM is uniformly superior.', '',
    '**The linear model did not meet 95% sensitivity on the test set:** its 39/42 sensitivity '
    'was 92.9%, while RBF achieved 40/42 (95.2%). Both produced 1/72 false positives '
    '(specificity 98.6%). The lower secondary threshold of 0.5 is reported as prespecified '
    'context only; it was not substituted for the locked threshold after seeing test results.', '',
    'The 95% development sensitivity target is exploratory and does not guarantee 95% sensitivity '
    'in the test set. Interpret probability quality using Brier score, log loss and calibration '
    'alongside AUROC. The constant baseline illustrates that high sensitivity alone can be obtained '
    'by labeling everyone positive, at the cost of zero specificity.', '',
    'Development-only permutation importance was computed on each selected model\'s validation '
    'fold (10 permutations per feature). Its averaged AUROC decreases and between-fold SD are '
    'saved separately. Related nuclear size and shape measurements are correlated, so individual '
    'importance rankings can be diluted or unstable. These are post-selection descriptive '
    'explanations and were not used for feature selection.', '',
    'The final linear SVM coefficient table describes the standardized decision margin, not '
    'log-odds ratios or causal effects. RBF predictions do not have one global linear coefficient '
    'per original feature. Calibration makes probabilities available but does not establish '
    'reliability in new clinical populations.', '',
    'Both models ranked texture_worst highest by mean validation permutation AUROC decrease. '
    'The next-ranked linear features were area_se and radius_worst; the RBF model emphasized '
    'radius_worst and compactness_se. Their fold variation and correlated alternatives '
    'limit interpretation of these specific rankings.', '',
    '### Selected figures', '',
    '![Development tuning](../results/svm/figures/development_tuning.png)', '',
    '![Held-out calibration](../results/svm/figures/test/calibration_curves_comparison.png)', '',
    '![Test errors at locked thresholds](../results/svm/figures/locked_threshold_tradeoffs.png)', '',
    '## Runtime and environment', '', runtime_table.to_markdown(index=False, floatfmt='.3f'), '',
    'Search time includes calibrated five-fold candidate fitting/prediction. Development total time '
    'additionally includes validation permutation interpretation. Timings depend on hardware and load.', '',
    '```json', json.dumps(lock['environment'], indent=2), '```', '',
    'The existing course environment was used without modifying it. Its pandas version differs '
    'from the shared requirements pin; exact executed versions are recorded above and in '
    'development_lock.json. This difference is disclosed rather than treated as a matching install. '
    'requirements-svm.txt records the executed dependency versions.', '',
    '## Limitations and next steps', '',
    'The dataset is small and historical, with only 42 malignant test records. There is no '
    'hospital or temporal external validation, patient independence cannot be fully verified '
    'from record IDs, and demographic fairness cannot be evaluated. Avoid causal and clinical '
    'deployment claims. Test uncertainty and post-selection OOF optimism limit conclusions. '
    'The team should add the other prespecified models using the same saved split and evaluation '
    'contract, without using these test results to change modeling choices.', '',
    '## Provenance and delivery', '',
    f'Development lock created: {lock["created_utc"]}. Test stage completed: {complete["created_utc"]}.', '',
    'The four CSVs in results/predictions follow TEAM_WORKFLOW.md. Input hashes, code hashes, '
    'OOF hashes, test-output hashes, runtime, package versions and verification results are in '
    'results/svm. Saved fitted model objects are local under models/svm and are not required '
    'for the team prediction ingestion. No commit or push has been performed for this analysis.', '',
]
(ROOT / 'docs' / 'ASHER_SVM_REPORT.md').write_text('\n'.join(lines), encoding='utf8')
print('Saved docs/ASHER_SVM_REPORT.md, SVM figures, summary and runtime tables.')
