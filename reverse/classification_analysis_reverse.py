#!/usr/bin/env python3

import sys
import time
import pandas as pd
import numpy as np
from itertools import combinations

from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/home/sayan/dissertation/regression/predict_fpga')
from load_data import load_metrics
df = load_metrics('/home/sayan/dissertation/regression/predict_fpga/metrics.csv')

asic7_features  = ['asic_area', 'asic_fmax', 'asic_power', 'asic_nets', 'asic_registers']
asic45_features = ['asic45_area', 'asic45_fmax', 'asic45_power', 'asic45_nets', 'asic45_registers']

targets = {
    'fpga_luts':            'FPGA LUTs',
    'fpga_fmax':             'FPGA Fmax (MHz)',
    'fpga_power':            'FPGA Power (mW)',
    'fpga_nets':             'FPGA Nets',
    'fpga_ffs':              'FPGA FFs',
    'fpga_distram':          'FPGA Dist.RAM',
    'fpga_lookahead8':       'FPGA LOOKAHEAD8',
    'fpga_unicontrolsets':   'FPGA Uni.Control.Sets',
}

classifiers = {
    'LogisticRegression': Pipeline([('s', StandardScaler()),
                                    ('m', LogisticRegression(max_iter=5000))]),
    'Perceptron':         Pipeline([('s', StandardScaler()),
                                    ('m', Perceptron(random_state=42))]),
    'SGDClassifier':      Pipeline([('s', StandardScaler()),
                                    ('m', SGDClassifier(random_state=42, max_iter=5000))]),
}

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

pdks = {
    'ASIC7':  (asic7_features,  'asic_'),
    'ASIC45': (asic45_features, 'asic45_'),
}

results = []
checkpoint_path = '/home/sayan/dissertation/regression/predict_fpga/results_classify_reverse_checkpoint.csv'
t0 = time.time()

for pdk_name, (asic_features, prefix) in pdks.items():
    all_combos = [list(c) for r in range(1, len(asic_features) + 1)
                 for c in combinations(asic_features, r)]

    for target_col, target_label in targets.items():
        y_band = pd.qcut(df[target_col], 3, labels=['small', 'medium', 'large'])

        for x_cols in all_combos:
            X = df[x_cols].values
            combo_label = ' + '.join(c.replace(prefix, '') for c in x_cols)

            for name, clf in classifiers.items():
                acc = cross_val_score(clf, X, y_band, cv=skf, scoring='accuracy')
                results.append({
                    'pdk': pdk_name, 'target': target_col, 'target_label': target_label,
                    'combination': combo_label, 'n_features': len(x_cols),
                    'model': name,
                    'cv_accuracy_mean': round(acc.mean(), 4),
                    'cv_accuracy_std': round(acc.std(), 4),
                })

        pd.DataFrame(results).to_csv(checkpoint_path, index=False)
        print(f"Done: {pdk_name:<7} {target_label:<24} "
              f"({time.time() - t0:.0f}s elapsed, checkpoint saved)", flush=True)

results_df = pd.DataFrame(results)
output_path = '/home/sayan/dissertation/regression/predict_fpga/results_classify_reverse.csv'
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")
print(f"Total rows: {len(results_df)}")

print()
print("=" * 90)
print("BEST COMBINATION + CLASSIFIER PER TARGET PER PDK (chance level = 0.333)")
print("=" * 90)
print(f"{'PDK':<8} {'Target':<24} {'Combination':<24} {'Model':<20} {'Accuracy':>9}")
print("-" * 90)
for pdk_name in pdks.keys():
    for target_col, target_label in targets.items():
        subset = results_df[(results_df['pdk'] == pdk_name) &
                            (results_df['target'] == target_col)]
        best = subset.loc[subset['cv_accuracy_mean'].idxmax()]
        print(f"{pdk_name:<8} {target_label:<24} {best['combination']:<24} "
              f"{best['model']:<20} {best['cv_accuracy_mean']:>9.3f}")

print()
print("=" * 90)
print("ALL FEATURES vs BEST SUBSET, per classifier, per PDK (chance level = 0.333)")
print("=" * 90)
for pdk_name, (asic_features, prefix) in pdks.items():
    all_features = ' + '.join(c.replace(prefix, '') for c in asic_features)
    for target_col, target_label in targets.items():
        subset = results_df[(results_df['pdk'] == pdk_name) &
                            (results_df['target'] == target_col)]
        for name in classifiers:
            g = subset[subset['model'] == name]
            full = g[g['combination'] == all_features]
            if full.empty:
                continue
            a_full = full['cv_accuracy_mean'].iloc[0]
            best = g.loc[g['cv_accuracy_mean'].idxmax()]
            print(f"{pdk_name:<8} {target_label:<24} {name:<20} "
                  f"all={a_full:.3f}  own-best={best['cv_accuracy_mean']:.3f}  "
                  f"gap={best['cv_accuracy_mean']-a_full:.3f}")