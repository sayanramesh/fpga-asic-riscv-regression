#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import time
from itertools import combinations

# models
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import (RidgeCV, LassoCV, LassoLarsIC,
                                  ElasticNetCV, SGDRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import TransformedTargetRegressor

from sklearn.model_selection import KFold, cross_validate
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

def scaled(estimator):
    return Pipeline([('scale', StandardScaler()), ('model', estimator)])


def log_space(estimator):
   
    inner = Pipeline([('log_x', FunctionTransformer(np.log1p, inverse_func=np.expm1,
                                                    validate=True)),
                      ('est', estimator)])
    return TransformedTargetRegressor(regressor=inner,
                                      func=np.log1p, inverse_func=np.expm1)


alphas = np.logspace(-3, 3, 25)
gp_kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-3)


models = {
    'LinearRegression':  LinearRegression(),
    'BayesianRidge':     BayesianRidge(),
    'RandomForest':      RandomForestRegressor(n_estimators=100, random_state=42),
    'GradientBoosting':  GradientBoostingRegressor(n_estimators=100, random_state=42),
    'RidgeCV':           scaled(RidgeCV(alphas=alphas)),
    'LassoCV':           scaled(LassoCV(cv=3, random_state=42, max_iter=50000)),
    'LassoLarsIC':       scaled(LassoLarsIC(criterion='bic')),
    'ElasticNetCV':      scaled(ElasticNetCV(cv=3, random_state=42,
                                             l1_ratio=[.1, .5, .7, .9, .95, 1],
                                             max_iter=50000)),
    'SGDRegressor':      scaled(SGDRegressor(random_state=42, max_iter=20000, tol=1e-4)),
    'GaussianProcess':   scaled(GaussianProcessRegressor(kernel=gp_kernel,
                                                         normalize_y=True, random_state=42)),
    'KNN_k3_scaled':     scaled(KNeighborsRegressor(n_neighbors=3)),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
spaces = ['linear', 'log']

pdks = {
    'ASIC7':  (asic7_features,  'asic_'),
    'ASIC45': (asic45_features, 'asic45_'),
}

results = []
checkpoint_path = '/home/sayan/dissertation/regression/predict_fpga/results_reverse_v4_checkpoint.csv'
t0 = time.time()

for pdk_name, (asic_features, prefix) in pdks.items():

    all_combos = []
    for r in range(1, len(asic_features) + 1):
        for combo in combinations(asic_features, r):
            all_combos.append(list(combo))

    print(f"\n{'='*60}")
    print(f"PDK: {pdk_name}")
    print(f"{'='*60}")
    print(f"Input combinations: {len(all_combos)} (2^{len(asic_features)}-1)")
    print(f"Targets: {len(targets)}")
    print(f"Models:  {len(models)}")
    print(f"Spaces: {len(spaces)}")
    print(f"CV runs: {len(all_combos) * len(targets) * len(models) * len(spaces)}")
    print()

    for space in spaces:
        for target_col, target_label in targets.items():
            y = df[target_col].values

            for x_cols in all_combos:
                X = df[x_cols].values
                combo_label = ' + '.join([c.replace(prefix, '') for c in x_cols])

                for model_name, model in models.items():
                    m = log_space(model) if space == 'log' else model

                    cv = cross_validate(m, X, y, cv=kf,
                                        scoring=('r2', 'neg_mean_absolute_error'))
                    cv_r2  = cv['test_r2']
                    cv_mae = -cv['test_neg_mean_absolute_error']

                    results.append({
                        'pdk':         pdk_name,
                        'space':       space,
                        'target':      target_col,
                        'target_label':target_label,
                        'combination': combo_label,
                        'n_features':  len(x_cols),
                        'model':       model_name,
                        'cv_r2_mean':  round(cv_r2.mean(), 4),
                        'cv_r2_std':   round(cv_r2.std(), 4),
                        'cv_mae_mean': round(cv_mae.mean(), 2),
                        'cv_mae_std':  round(cv_mae.std(), 2),
                    })

            pd.DataFrame(results).to_csv(checkpoint_path, index=False)
            print(f"Done: {pdk_name:<7} {space:<7} {target_label:<22} "
                  f"({time.time() - t0:.0f}s elapsed, checkpoint saved)", flush=True)

results_df = pd.DataFrame(results)
output_path = '/home/sayan/dissertation/regression/predict_fpga/results_reverse_v4.csv'
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")
print(f"Total rows: {len(results_df)}")

print()
print("=" * 100)
print("BEST COMBINATION + MODEL PER TARGET PER PDK (by CV R2)")
print("=" * 100)
print(f"{'PDK':<8} {'Space':<8} {'Target':<24} {'Combination':<28} {'Model':<18} {'CV R2':>7}")
print("-" * 100)

for pdk_name in pdks.keys():
    for space in spaces:
        for target_col, target_label in targets.items():
            subset = results_df[(results_df['pdk'] == pdk_name) &
                                (results_df['space'] == space) &
                                (results_df['target'] == target_col)]
            best = subset.loc[subset['cv_r2_mean'].idxmax()]
            print(f"{pdk_name:<8} {space:<8} {target_label:<24} {best['combination']:<28} "
                  f"{best['model']:<18} {best['cv_r2_mean']:>7.4f}")

print()
print("=" * 100)
print("ALL FEATURES vs BEST SUBSET, SAME MODEL, per PDK (gap should be small)")
print("=" * 100)
print(f"{'PDK':<8} {'Space':<8} {'Target':<24} {'Model':<18} {'All-N R2':>9} {'Own best':>9} {'Gap':>7}")
print("-" * 100)

for pdk_name, (asic_features, prefix) in pdks.items():
    all_features = ' + '.join([c.replace(prefix, '') for c in asic_features])
    for space in spaces:
        for target_col, target_label in targets.items():
            subset = results_df[(results_df['pdk'] == pdk_name) &
                                (results_df['space'] == space) &
                                (results_df['target'] == target_col)]
            full = subset[subset['combination'] == all_features]
            for _, row in full.iterrows():
                own_best = subset[subset['model'] == row['model']]['cv_r2_mean'].max()
                gap = own_best - row['cv_r2_mean']
                print(f"{pdk_name:<8} {space:<8} {target_label:<24} {row['model']:<18} "
                      f"{row['cv_r2_mean']:>9.4f} {own_best:>9.4f} {gap:>7.4f}")