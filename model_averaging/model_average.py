#!/usr/bin/env python3

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

results_df = pd.read_csv('/home/sayan/dissertation/regression/predict_asic/results_v4.csv')

base = '/home/sayan/dissertation/regression/predict_asic/'

target_order = [
    'ASIC7 Area (um2)', 'ASIC7 Fmax (MHz)', 'ASIC7 Power (mW)', 'ASIC7 Nets',
    'ASIC7 Registers',
    'ASIC45 Area (um2)', 'ASIC45 Fmax (MHz)', 'ASIC45 Power (mW)', 'ASIC45 Nets',
    'ASIC45 Registers',
]

spaces = sorted(results_df['space'].unique())

n_combos = results_df['combination'].nunique()

def average_table(space, stat):
    # stat is 'mean' or 'median'
    subset = results_df[results_df['space'] == space]
    table = subset.pivot_table(
        index='model',
        columns='target_label',
        values='cv_r2_mean',
        aggfunc=stat
    )

    table = table[[t for t in target_order if t in table.columns]]

    table['OVERALL'] = table.mean(axis=1)

    return table.sort_values('OVERALL', ascending=False)


for stat in ('mean', 'median'):
    for space in spaces:
        table = average_table(space, stat)

        print()
        print("=" * 130)
        print(f"{stat.upper()} CV R2 per model, averaged over the {n_combos} combinations "
              f"({space} space)")
        print("=" * 130)
        print(table.round(3).to_string())

        out = f"{base}model_average_{stat}_{space}.csv"
        table.round(4).to_csv(out)
        print(f"\nSaved: {out}")

print()
print("=" * 130)
print("OVERALL RANKING (both spaces together)")
print("=" * 130)

combined = pd.DataFrame({
    'mean_linear':   average_table('linear', 'mean')['OVERALL'],
    'mean_log':  average_table('log',    'mean')['OVERALL'],
    'median_linear': average_table('linear', 'median')['OVERALL'],
    'median_log':    average_table('log',    'median')['OVERALL'],
})
combined['mean_both']   = combined[['mean_linear', 'mean_log']].mean(axis=1)
combined['median_both'] = combined[['median_linear', 'median_log']].mean(axis=1)

neg = results_df.groupby('model')['cv_r2_mean'].apply(lambda x: 100 * (x < 0).mean())
combined['pct_negative'] = neg

combined = combined.sort_values('median_both', ascending=False)
print(combined.round(3).to_string())

combined.round(4).to_csv(f"{base}model_average_overall.csv")
print(f"\nSaved: {base}model_average_overall.csv")

print()
print(f"Best model by mean:   {combined['mean_both'].idxmax()}")
print(f"Best model by median: {combined['median_both'].idxmax()}")