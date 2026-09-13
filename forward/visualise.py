#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore')

results_df = pd.read_csv('/home/sayan/dissertation/regression/predict_asic/results_v4.csv')

spaces = sorted(results_df['space'].unique())

all_features = (results_df.loc[results_df['n_features'].idxmax(), 'combination'])
n_feat = len(all_features.split(' + '))

def save(fig_path):
    plt.savefig(f'{fig_path}.pdf', bbox_inches='tight')
    plt.savefig(f'{fig_path}.png', dpi=150, bbox_inches='tight')


#  heatmap restricted to the 20 best combinations per target
def plot_heatmap(space, target_col, target_label, save_path, top_n=20):
    
    subset = results_df[(results_df['space'] == space) &
                        (results_df['target'] == target_col)]

    best_per_combo = subset.groupby('combination')['cv_r2_mean'].max()
    top_combos = best_per_combo.sort_values(ascending=False).head(top_n).index

    subset = subset[subset['combination'].isin(top_combos)]

    pivot = subset.pivot_table(
        index='combination',
        columns='model',
        values='cv_r2_mean'
    )

    pivot['best'] = pivot.max(axis=1)
    pivot = pivot.sort_values('best', ascending=False).drop('best', axis=1)

    fig, ax = plt.subplots(figsize=(1.15 * len(pivot.columns) + 4,
                                    max(6, len(pivot) * 0.35)))

    cmap = mcolors.LinearSegmentedColormap.from_list(
        'rg', ['darkred', 'red', 'orange', 'yellow', 'lightgreen', 'green']
    )

    im = ax.imshow(pivot.values, cmap=cmap, vmin=-0.5, vmax=1.0,
                   aspect='auto')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if pd.isna(val):
                continue
            color = 'white' if val < 0.3 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color)

    rank = ''
    if all_features in pivot.index:
        row = list(pivot.index).index(all_features)
        ax.get_yticklabels()[row].set_fontweight('bold')
        rank = f'   [all {n_feat} features shown, rank {row + 1} of {len(pivot)}]'

    plt.colorbar(im, ax=ax, label='CV R2')
    ax.set_title(f'K-Fold CV R2: top {top_n} FPGA Combinations vs Models\n'
                 f'Target: {target_label}   ({space} space){rank}',
                 fontsize=12, pad=15)
    ax.set_xlabel('Model', fontsize=10)
    ax.set_ylabel(f'FPGA Input Combination (best {top_n} of '
                  f'{best_per_combo.shape[0]} shown)', fontsize=10)

    plt.tight_layout()
    save(save_path)
    plt.close()
    print(f"Heatmap saved: {save_path}")

def plot_summary_table(space, save_path):
    rows = []
    space_df = results_df[results_df['space'] == space]
    for target in space_df['target'].unique():
        subset = space_df[space_df['target'] == target]
        best   = subset.loc[subset['cv_r2_mean'].idxmax()]

        full  = subset[(subset['combination'] == all_features) &
                           (subset['model'] == best['model'])]
        best_full = full.iloc[0]

        rows.append({
            'Target': best['target_label'],
            'Best Combo': best['combination'],
            'Best Model': best['model'],
            'CV R2': f"{best['cv_r2_mean']:.4f}",
            'CV MAE': f"{best['cv_mae_mean']:.2f}",
            f'All-{n_feat} R2': f"{best_full['cv_r2_mean']:.4f}",
            'Gap': f"{best['cv_r2_mean'] - best_full['cv_r2_mean']:.4f}",
            'Status': 'GOOD' if best['cv_r2_mean'] > 0.7
                           else 'MOD' if best['cv_r2_mean'] > 0.4
                           else 'WEAK' if best['cv_r2_mean'] > 0
                           else 'POOR'
        })

    summary_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(18, 4))
    ax.axis('off')

    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(summary_df.columns))))

    status_colors = {'GOOD': '#90EE90', 'MOD': '#FFD700',
                     'WEAK': '#FFA07A', 'POOR': '#FF6B6B'}
    for i, row in enumerate(rows):
        color = status_colors.get(row['Status'], 'white')
        for j in range(len(summary_df.columns)):
            table[i+1, j].set_facecolor(color)

    for j in range(len(summary_df.columns)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    ax.set_title(f'Best Combination + Model per ASIC Target (K-Fold CV, {space} space)',
                 fontsize=12, pad=10)
    plt.tight_layout()
    save(save_path)
    plt.close()
    print(f"Summary table saved: {save_path}")


def plot_model_ranking(save_path):
    fig, axes = plt.subplots(1, len(spaces), figsize=(7 * len(spaces), 5),
                             squeeze=False)
    for ax, space in zip(axes[0], spaces):
        s = results_df[results_df['space'] == space]

        stats = s.groupby('model')['cv_r2_mean'].agg(['mean', 'median'])
        stats = stats.sort_values('median')
        stats.plot(kind='barh', ax=ax, width=0.75, edgecolor='black', linewidth=0.4)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_xlabel(f'CV R2 over all {results_df["combination"].nunique()} '
                      f'combinations x {results_df["target"].nunique()} targets')
        ax.set_ylabel('')
        ax.set_title(f'Model ranking ({space} space)', fontsize=11)
        ax.grid(axis='x', alpha=0.3, lw=0.5)
    plt.tight_layout()
    save(save_path)
    plt.close()
    print(f"Model ranking saved: {save_path}")


def plot_model_choice(save_path):
    rows = []
    for target in results_df['target'].unique():
        cand = []
        for (space, model), g in results_df[results_df['target'] == target].groupby(
                ['space', 'model']):
            full = g[g['combination'] == all_features]
            if full.empty:
                continue
            a5 = full['cv_r2_mean'].iloc[0]
            best = g.loc[g['cv_r2_mean'].idxmax()]
            cand.append((a5, best['cv_r2_mean'], best['combination'],
                         model, space, best['target_label']))
        cand.sort(reverse=True)
        globalbest = results_df[results_df['target'] == target]['cv_r2_mean'].max()
        a5, mb, combo, model, space, label = cand[0]
        rows.append({
            'Target': label,
            'Model': model,
            'Space': space,
            f'All-{n_feat} R2': f"{a5:.4f}",
            'Own Best R2': f"{mb:.4f}",
            'Gap': f"{mb - a5:.4f}",
            'Own Best Combination': combo,
            'Global Best Cell': f"{globalbest:.4f}",
        })

    summary_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(18, 0.42 * len(summary_df) + 2))
    ax.axis('off')

    table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(summary_df.columns))))

    for i in range(len(rows)):
        for j in range(len(summary_df.columns)):
            table[i + 1, j].set_facecolor('#E8F0FE') 

    for j in range(len(summary_df.columns)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    ax.set_title(f'Recommended model per ASIC target, chosen by CV R2 with ALL '
                 f'{n_feat} FPGA features\n'
                 'Gap = that model\'s own best combination minus its own all-features score',
                 fontsize=12, pad=14)
    plt.tight_layout()
    save(save_path)
    plt.close()
    print(f"Model choice table saved: {save_path}")


def recommended_model_per_target(target_list):
    rec = {}
    for target in target_list:
        cand = []
        for (space, model), g in results_df[results_df['target'] == target].groupby(
                ['space', 'model']):
            full = g[g['combination'] == all_features]
            if full.empty:
                continue
            a_full = full['cv_r2_mean'].iloc[0]
            cand.append((a_full, model, space, g['target_label'].iloc[0]))
        if cand:
            cand.sort(reverse=True)
            _, model, space, label = cand[0]
            rec[target] = (model, space, label)
    return rec


KEEP_METRICS = ('area', 'fmax', 'power', 'nets')


def target_pdk(target):
    return 'ASIC45' if target.startswith('asic45_') else 'ASIC7'


def target_metric(target):
    prefix = 'asic45_' if target.startswith('asic45_') else 'asic_'
    return target[len(prefix):]


def plot_cheap_combinations(pdk, save_path, good_enough=0.95):
    
    target_list = [t for t in results_df['target'].unique()
                   if target_metric(t) in KEEP_METRICS and target_pdk(t) == pdk]
    rec = recommended_model_per_target(target_list)
    target_order = list(rec.keys())
    n = len(target_order)
    if n == 0:
        print(f"No matching targets for {pdk}, skipping {save_path}")
        return
    ncols = min(2, n)
    nrows = -(-n // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()

    for ax, target in zip(axes, target_order):
        model, space, label = rec[target]
        g = results_df[(results_df['target'] == target) &
                       (results_df['model'] == model) &
                       (results_df['space'] == space)]

        ax.scatter(g['n_features'], g['cv_r2_mean'], alpha=0.35, s=40,
                  color='tab:blue')

        frontier = g.loc[g.groupby('n_features')['cv_r2_mean'].idxmax()]
        frontier = frontier.sort_values('n_features').reset_index(drop=True)
        frontier['cv_r2_mean'] = frontier['cv_r2_mean'].cummax()

        ax.plot(frontier['n_features'], frontier['cv_r2_mean'],
               color='tab:red', marker='o', markersize=6, linewidth=1.5)

        best_row = g.loc[g['cv_r2_mean'].idxmax()]

        best_r2 = g['cv_r2_mean'].max()
        threshold = best_r2 * good_enough if best_r2 > 0 else best_r2
        good = frontier[frontier['cv_r2_mean'] >= threshold]
        cheapest = None
        if not good.empty:
            cheapest = good.loc[good['n_features'].idxmin()]
            ax.scatter([cheapest['n_features']], [cheapest['cv_r2_mean']],
                      s=160, facecolors='none', edgecolors='green', linewidths=2,
                      zorder=5)

        import textwrap
        wrap_width = 26
        best_text = 'best: ' + textwrap.fill(best_row['combination'], wrap_width)
        show_cheap = cheapest is not None and cheapest['n_features'] != best_row['n_features']

        if show_cheap:
            cheap_text = 'cheap: ' + textwrap.fill(cheapest['combination'], wrap_width)
            ax.text(0.98, 0.04, cheap_text, transform=ax.transAxes, fontsize=5.5,
                   ha='right', va='bottom', linespacing=1.3, color='green',
                   bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                            edgecolor='green', alpha=0.85))
           
            y_best = 0.04 + 0.075 * (cheap_text.count('\n') + 1) + 0.02
        else:
            y_best = 0.04

        ax.text(0.98, y_best, best_text, transform=ax.transAxes, fontsize=5.5,
               ha='right', va='bottom', linespacing=1.3, color='tab:red',
               bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                        edgecolor='tab:red', alpha=0.85))

        ax.set_title(f'{label}\n({model}, {space})', fontsize=9)
        ax.set_xlabel('number of features  \u2190 fewer / cheaper is better', fontsize=8)
        ax.set_ylabel('CV R2  \u2191 higher is better', fontsize=8)
        ax.set_ylim(bottom=0) 
        ax.grid(alpha=0.3, lw=0.5)
        ax.tick_params(labelsize=7)

    for ax in axes[n:]:
        ax.axis('off')

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='o', color='tab:blue', linestyle='None',
              markersize=7, alpha=0.6, label='combination tried'),
        Line2D([0], [0], marker='o', color='tab:red', linewidth=1.5,
              markersize=7, label='Pareto front (best achievable at or below this feature count)'),
        Line2D([0], [0], marker='o', color='tab:red', linestyle='None',
              markersize=0, label='red text = best combination'),
        Line2D([0], [0], marker='o', color='green', linestyle='None',
              markersize=10, markerfacecolor='none', markeredgewidth=2,
              label='green text = cheapest combination reaching '
                    f'{int(good_enough*100)}% of best'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=2,
              fontsize=8, bbox_to_anchor=(0.5, -0.02), frameon=True)

    fig.suptitle(f'{pdk}: CV R2 vs number of features, recommended model per target',
                fontsize=12)
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    save(save_path)
    plt.close()
    print(f"Cheap-combination scatter saved: {save_path}")


base = '/home/sayan/dissertation/regression/predict_asic/'

for space in spaces:
    for target_col in results_df[results_df['space'] == space]['target'].unique():
        target_label = results_df[(results_df['space'] == space) &
                                  (results_df['target'] == target_col)]['target_label'].iloc[0]
        plot_heatmap(space, target_col, target_label,
                    f"{base}heatmap_{space}_{target_col}", top_n=20)
    plot_summary_table(space, f"{base}summary_table_{space}")

plot_model_choice(f"{base}model_choice")  
plot_model_ranking(f"{base}model_ranking")
plot_cheap_combinations('ASIC7', f"{base}cheap_combinations_ASIC7")
plot_cheap_combinations('ASIC45', f"{base}cheap_combinations_ASIC45")

print("\nAll visualisation complete.")
print(f"Files saved to: {base}")