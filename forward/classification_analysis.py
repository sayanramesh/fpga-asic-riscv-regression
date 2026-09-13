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

sys.path.append('/home/sayan/dissertation/regression/predict_asic')
from load_data import load_metrics
df = load_metrics('/home/sayan/dissertation/regression/predict_asic/metrics.csv')

fpga_features = ['fpga_luts', 'fpga_fmax', 'fpga_power', 'fpga_nets', 'fpga_ffs',
                 'fpga_distram', 'fpga_unicontrolsets']

targets = {
    'asic_area': 'ASIC7 Area (um2)',
    'asic_fmax': 'ASIC7 Fmax (MHz)',
    'asic_power': 'ASIC7 Power (mW)',
    'asic_nets': 'ASIC7 Nets',
    'asic_registers': 'ASIC7 Registers',
    'asic45_area': 'ASIC45 Area (um2)',
    'asic45_fmax': 'ASIC45 Fmax (MHz)',
    'asic45_power': 'ASIC45 Power (mW)',
    'asic45_nets': 'ASIC45 Nets',
    'asic45_registers': 'ASIC45 Registers',
}

classifiers = {
    'LogisticRegression': Pipeline([('s', StandardScaler()),
                                    ('m', LogisticRegression(max_iter=5000))]),
    'Perceptron':  Pipeline([('s', StandardScaler()),
                                    ('m', Perceptron(random_state=42))]),

    'SGDClassifier': Pipeline([('s', StandardScaler()),
                                    ('m', SGDClassifier(random_state=42,
                                                        max_iter=5000))]),
}

# 3 folds rather than 5
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

all_combos = [list(c) for r in range(1, len(fpga_features) + 1)
              for c in combinations(fpga_features, r)]

results = []
checkpoint_path = '/home/sayan/dissertation/regression/predict_asic/results_classify_checkpoint.csv'
t0 = time.time()

for target_col, target_label in targets.items():
    y_band = pd.qcut(df[target_col], 3, labels=['small', 'medium', 'large'])

    for x_cols in all_combos:
        X = df[x_cols].values
        combo_label = ' + '.join(c.replace('fpga_', '') for c in x_cols)

        for name, clf in classifiers.items():
            acc = cross_val_score(clf, X, y_band, cv=skf, scoring='accuracy')
            results.append({
                'target': target_col, 'target_label': target_label,
                'combination': combo_label, 'n_features': len(x_cols),
                'model': name,
                'cv_accuracy_mean': round(acc.mean(), 4),
                'cv_accuracy_std': round(acc.std(), 4),
            })

    pd.DataFrame(results).to_csv(checkpoint_path, index=False)
    print(f"Done: {target_label}   ({time.time() - t0:.0f}s elapsed, checkpoint saved)",
          flush=True)

results_df = pd.DataFrame(results)
output_path = '/home/sayan/dissertation/regression/predict_asic/results_classify.csv'
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")
print(f"Total rows: {len(results_df)}")

print()
print("=" * 80)
print("BEST COMBINATION + CLASSIFIER PER TARGET (chance level = 0.333)")
print("=" * 80)
print(f"{'Target':<20} {'Combination':<25} {'Model':<20} {'Accuracy':>9}")
print("-" * 80)
for target_col, target_label in targets.items():
    subset = results_df[results_df['target'] == target_col]
    best = subset.loc[subset['cv_accuracy_mean'].idxmax()]
    print(f"{target_label:<20} {best['combination']:<25} "
          f"{best['model']:<20} {best['cv_accuracy_mean']:>9.3f}")

print()
print("=" * 80)
all_features = ' + '.join(c.replace('fpga_', '') for c in fpga_features)
n_feat = len(fpga_features)
print(f"ALL {n_feat} FEATURES vs BEST SUBSET, per classifier (chance level = 0.333)")
print("=" * 80)
for target_col, target_label in targets.items():
    subset = results_df[results_df['target'] == target_col]
    for name in classifiers:
        g = subset[subset['model'] == name]
        full = g[g['combination'] == all_features]
        if full.empty:
            continue
        a_full = full['cv_accuracy_mean'].iloc[0]
        best = g.loc[g['cv_accuracy_mean'].idxmax()]
        print(f"{target_label:<20} {name:<20} all-{n_feat}={a_full:.3f}  "
              f"own-best={best['cv_accuracy_mean']:.3f}  "
              f"gap={best['cv_accuracy_mean']-a_full:.3f}")