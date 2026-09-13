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
#k fold
from sklearn.model_selection import KFold, cross_validate
import warnings
warnings.filterwarnings('ignore')
sys.path.append('/home/sayan/dissertation/regression/predict_asic')
#load data
from load_data import load_metrics
df = load_metrics('/home/sayan/dissertation/regression/predict_asic/metrics.csv')


fpga_features = ['fpga_luts', 'fpga_fmax', 'fpga_power', 'fpga_nets', 'fpga_ffs',
                 'fpga_distram', 'fpga_unicontrolsets']                    #predictors

targets = {                                 #targets
    'asic_area': 'ASIC7 Area (um2)',
    'asic_fmax': 'ASIC7 Fmax (MHz)',
    'asic_power': 'ASIC7 Power (mW)',
    'asic_nets': 'ASIC7 Nets',
    'asic_registers': 'ASIC7 Registers',
    'asic45_area': 'ASIC45 Area (um2)',
    'asic45_fmax': 'ASIC45 Fmax (MHz)',
    'asic45_power': 'ASIC45 Power (mW)',
    'asic45_nets': 'ASIC45 Nets',
    'asic45_registers':  'ASIC45 Registers',
}

def scaled(estimator):
    
    return Pipeline([('scale', StandardScaler()), ('model', estimator)])


def log_space(estimator):
    # fits the model on log(x) and log(y), then converts the prediction back
    
    inner = Pipeline([('log_x', FunctionTransformer(np.log1p, inverse_func=np.expm1,
                                                    validate=True)),
                      ('est', estimator)])
    return TransformedTargetRegressor(regressor=inner,
                                      func=np.log, inverse_func=np.exp)


alphas = np.logspace(-3, 3, 25)  # penalty grid for RidgeCV
gp_kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-3)


models = {
    'LinearRegression':  LinearRegression(),
    'BayesianRidge':     BayesianRidge(),
    'RandomForest':      RandomForestRegressor(n_estimators=100,
                                               random_state=42),
    'GradientBoosting':  GradientBoostingRegressor(n_estimators=100,
                                                    random_state=42),

    'RidgeCV':           scaled(RidgeCV(alphas=alphas)),
    'LassoCV':           scaled(LassoCV(cv=3, random_state=42,
                                        max_iter=50000)),
    'LassoLarsIC':       scaled(LassoLarsIC(criterion='bic')),  
     
    'ElasticNetCV':      scaled(ElasticNetCV(cv=3, random_state=42,
                                             l1_ratio=[.1, .5, .7, .9, .95, 1],
                                             max_iter=50000)),
    'SGDRegressor':      scaled(SGDRegressor(random_state=42, max_iter=20000,
                                             tol=1e-4)),
    'GaussianProcess':   scaled(GaussianProcessRegressor(kernel=gp_kernel,
                                                         normalize_y=True,
                                                         random_state=42)),
    'KNN_k3_scaled':     scaled(KNeighborsRegressor(n_neighbors=3)),
 
}

kf = KFold(n_splits=5, shuffle=True, random_state=42) #kfold setup

all_combos = []  #generate all combinations
for r in range(1, len(fpga_features) + 1):
    for combo in combinations(fpga_features, r):
        all_combos.append(list(combo))

spaces = ['linear', 'log']

print(f"Total input combinations: {len(all_combos)} (2^{len(fpga_features)}-1)")
print(f"Total targets: {len(targets)}")
print(f"Total models: {len(models)}")
print(f"Total spaces: {len(spaces)}")
print(f"Total CV runs: {len(all_combos) * len(targets) * len(models) * len(spaces)}")
print()

results = []
checkpoint_path = '/home/sayan/dissertation/regression/predict_asic/results_v4_checkpoint.csv'
t0 = time.time()

for space in spaces:                                # linear then log
    for target_col, target_label in targets.items():   #run all combinations
        y = df[target_col].values

        for x_cols in all_combos:
            X = df[x_cols].values
            combo_label = ' + '.join([c.replace('fpga_', '') for c in x_cols])

            for model_name, model in models.items():
                m = log_space(model) if space == 'log' else model

                cv = cross_validate(m, X, y, cv=kf,
                                    scoring=('r2', 'neg_mean_absolute_error'))
                cv_r2  = cv['test_r2']
                cv_mae = -cv['test_neg_mean_absolute_error']

                results.append({
                    'space':  space,
                    'target':  target_col,
                    'target_label':target_label,
                    'combination': combo_label,
                    'n_features':  len(x_cols),
                    'model':  model_name,
                    'cv_r2_mean':  round(cv_r2.mean(), 4),
                    'cv_r2_std': round(cv_r2.std(), 4),
                    'cv_mae_mean': round(cv_mae.mean(), 2),
                    'cv_mae_std':  round(cv_mae.std(), 2),
                })

        pd.DataFrame(results).to_csv(checkpoint_path, index=False)
        print(f"Done: {space:<7} {target_label}   "
              f"({time.time() - t0:.0f}s elapsed, checkpoint saved)", flush=True)

results_df = pd.DataFrame(results)
output_path = '/home/sayan/dissertation/regression/predict_asic/results_v4.csv'
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")
print(f"Total rows: {len(results_df)}")


print()
print("=" * 90)
print("BEST COMBINATION + MODEL PER TARGET (by CV R2)")
print("=" * 90)
print(f"{'Space':<8} {'Target':<20} {'Combination':<30} {'Model':<18} {'CV R2':>7} {'CV MAE':>10}")
print("-" * 90)

for space in spaces:
    for target_col, target_label in targets.items():
        subset = results_df[(results_df['space'] == space) &
                            (results_df['target'] == target_col)]
        best   = subset.loc[subset['cv_r2_mean'].idxmax()]
        print(f"{space:<8} {target_label:<20} {best['combination']:<30} "
              f"{best['model']:<18} {best['cv_r2_mean']:>7.4f} "
              f"{best['cv_mae_mean']:>10.2f}")


all_features = ' + '.join([c.replace('fpga_', '') for c in fpga_features])

print()
print("=" * 90)
print("ALL SEVEN FEATURES vs BEST SUBSET, SAME MODEL (gap should be small)")
print("=" * 90)
print(f"{'Space':<8} {'Target':<20} {'Model':<18} {'All-7 R2':>9} {'Own best R2':>13} {'Gap':>7}")
print("-" * 90)

for space in spaces:
    for target_col, target_label in targets.items():
        subset = results_df[(results_df['space'] == space) &
                            (results_df['target'] == target_col)]
        full = subset[subset['combination'] == all_features]
        for _, row in full.iterrows():
            own_best = subset[subset['model'] == row['model']]['cv_r2_mean'].max()
            gap = own_best - row['cv_r2_mean']
            marker = '  <-- winner' if row['model'] == subset.loc[subset['cv_r2_mean'].idxmax(), 'model'] else ''
            print(f"{space:<8} {target_label:<20} {row['model']:<18} {row['cv_r2_mean']:>9.4f} "
                  f"{own_best:>13.4f} {gap:>7.4f}{marker}")