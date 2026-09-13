#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

results_df = pd.read_csv('/home/sayan/dissertation/regression/predict_fpga/results_reverse_v4.csv')

base = '/home/sayan/dissertation/regression/predict_fpga/'

spaces = sorted(results_df['space'].unique())
pdks = sorted(results_df['pdk'].unique())

all_features = {}
n_feat = {}
for pdk in pdks:
    sub = results_df[results_df['pdk'] == pdk]
    af = sub.loc[sub['n_features'].idxmax(), 'combination']
    all_features[pdk] = af
    n_feat[pdk] = len(af.split(' + '))


def save(fig_path):
    plt.savefig(f'{fig_path}.pdf', bbox_inches='tight')
    plt.savefig(f'{fig_path}.png', dpi=150, bbox_inches='tight')


def plot_summary_table(pdk, space, save_path):
    rows = []
    sub = results_df[(results_df['pdk'] == pdk) & (results_df['space'] == space)]
    for target in sub['target'].unique():
        subset = sub[sub['target'] == target]
        best = subset.loc[subset['cv_r2_mean'].idxmax()]

        full = subset[(subset['combination'] == all_features[pdk]) &
                      (subset['model'] == best['model'])]
        best_full = full.iloc[0] if not full.empty else None

        rows.append({
            'Target': best['target_label'],
            'Best Combo': best['combination'],
            'Best Model': best['model'],
            'CV R2': f"{best['cv_r2_mean']:.4f}",
            'CV MAE': f"{best['cv_mae_mean']:.2f}",
            f'All-{n_feat[pdk]} R2': f"{best_full['cv_r2_mean']:.4f}" if best_full is not None else 'n/a',
            'Gap': f"{best['cv_r2_mean'] - best_full['cv_r2_mean']:.4f}" if best_full is not None else 'n/a',
            'Status': 'GOOD' if best['cv_r2_mean'] > 0.7
                           else 'MOD' if best['cv_r2_mean'] > 0.4
                           else 'WEAK' if best['cv_r2_mean'] > 0
                           else 'POOR'
        })

    summary_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(16, max(4, len(summary_df) * 0.4)))
    ax.axis('off')

    table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(summary_df.columns))))

    status_colors = {'GOOD': '#90EE90', 'MOD': '#FFD700',
                     'WEAK': '#FFA07A', 'POOR': '#FF6B6B'}
    for i, row in enumerate(rows):
        color = status_colors.get(row['Status'], 'white')
        for j in range(len(summary_df.columns)):
            table[i + 1, j].set_facecolor(color)

    for j in range(len(summary_df.columns)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    ax.set_title(f'Best Combination + Model per FPGA Target '
                f'({pdk} -> FPGA, K-Fold CV, {space} space)',
                fontsize=12, pad=10)
    plt.tight_layout()
    save(save_path)
    plt.close()
    print(f"Summary table saved: {save_path}")


def plot_model_ranking(save_path):
    fig, axes = plt.subplots(len(pdks), len(spaces),
                             figsize=(7 * len(spaces), 5 * len(pdks)), squeeze=False)
    for pi, pdk in enumerate(pdks):
        for si, space in enumerate(spaces):
            ax = axes[pi][si]
            s = results_df[(results_df['pdk'] == pdk) & (results_df['space'] == space)]
            stats = s.groupby('model')['cv_r2_mean'].agg(['mean', 'median'])
            stats = stats.sort_values('median')
            stats.plot(kind='barh', ax=ax, width=0.75, edgecolor='black', linewidth=0.4)
            ax.axvline(0, color='black', lw=0.8)
            ax.set_xlabel(f'CV R2 over all {s["combination"].nunique()} '
                          f'combinations x {s["target"].nunique()} targets')
            ax.set_ylabel('')
            ax.set_title(f'{pdk} -> FPGA  ({space} space)', fontsize=11)
            ax.grid(axis='x', alpha=0.3, lw=0.5)
    plt.tight_layout()
    save(save_path)
    plt.close()
    print(f"Model ranking saved: {save_path}")


def plot_model_choice(save_path):
    rows = []
    for pdk in pdks:
        pdk_df = results_df[results_df['pdk'] == pdk]
        for target in pdk_df['target'].unique():
            cand = []
            for (space, model), g in pdk_df[pdk_df['target'] == target].groupby(
                    ['space', 'model']):
                full = g[g['combination'] == all_features[pdk]]
                if full.empty:
                    continue
                a_full = full['cv_r2_mean'].iloc[0]
                best = g.loc[g['cv_r2_mean'].idxmax()]
                cand.append((a_full, best['cv_r2_mean'], best['combination'],
                            model, space, best['target_label']))
            if not cand:
                continue
            cand.sort(reverse=True)
            globalbest = pdk_df[pdk_df['target'] == target]['cv_r2_mean'].max()
            a_full, mb, combo, model, space, label = cand[0]
            rows.append({
                'PDK': pdk,
                'Target': label,
                'Model': model,
                'Space': space,
                f'All-{n_feat[pdk]} R2': f"{a_full:.4f}",
                'Own Best R2': f"{mb:.4f}",
                'Gap': f"{mb - a_full:.4f}",
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

    ax.set_title('Recommended model per FPGA target, chosen by CV R2 with ALL '
                'ASIC FEATURES\n'
                'Gap = that model\'s own best combination minus its own all-features score',
                fontsize=12, pad=14)
    plt.tight_layout()
    save(save_path)
    plt.close()
    print(f"Model choice table saved: {save_path}")


def recommended_model_per_target(pdk):
    
    pdk_df = results_df[results_df['pdk'] == pdk]
    rec = {}
    for target in pdk_df['target'].unique():
        cand = []
        for (space, model), g in pdk_df[pdk_df['target'] == target].groupby(
                ['space', 'model']):
            full = g[g['combination'] == all_features[pdk]]
            if full.empty:
                continue
            a_full = full['cv_r2_mean'].iloc[0]
            cand.append((a_full, model, space, g['target_label'].iloc[0]))
        if cand:
            cand.sort(reverse=True)
            _, model, space, label = cand[0]
            rec[target] = (model, space, label)
    return rec


def plot_cheap_combinations(pdk, save_path, good_enough=0.95):
    
    rec = recommended_model_per_target(pdk)
    target_order = list(rec.keys())
    n = len(target_order)
    ncols = 4
    nrows = -(-n // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    pdk_df = results_df[results_df['pdk'] == pdk]
    for ax, target in zip(axes, target_order):
        model, space, label = rec[target]
        g = pdk_df[(pdk_df['target'] == target) &
                   (pdk_df['model'] == model) &
                   (pdk_df['space'] == space)]

        ax.scatter(g['n_features'], g['cv_r2_mean'], alpha=0.35, s=40,
                  color='tab:blue')

        frontier = g.loc[g.groupby('n_features')['cv_r2_mean'].idxmax()]
        frontier = frontier.sort_values('n_features')
        ax.plot(frontier['n_features'], frontier['cv_r2_mean'],
               color='tab:red', marker='o', markersize=6, linewidth=1.5)
        for _, row in frontier.iterrows():
            ax.annotate(row['combination'], (row['n_features'], row['cv_r2_mean']),
                       textcoords='offset points', xytext=(5, -8), fontsize=6,
                       color='tab:red')

        best_r2 = g['cv_r2_mean'].max()
        threshold = best_r2 * good_enough if best_r2 > 0 else best_r2
        good = frontier[frontier['cv_r2_mean'] >= threshold]
        if not good.empty:
            cheapest = good.loc[good['n_features'].idxmin()]
            ax.scatter([cheapest['n_features']], [cheapest['cv_r2_mean']],
                      s=160, facecolors='none', edgecolors='green', linewidths=2,
                      zorder=5)

        ax.set_title(f'{label}\n({model}, {space})', fontsize=9)
        ax.set_xlabel('number of features', fontsize=8)
        ax.set_ylabel('CV R2', fontsize=8)
        ax.grid(alpha=0.3, lw=0.5)
        ax.tick_params(labelsize=7)

    for ax in axes[n:]:
        ax.axis('off')

    fig.suptitle(f'{pdk} -> FPGA: CV R2 vs number of features, recommended model per target\n'
                f'red = best at each size (frontier)   green circle = cheapest combo '
                f'reaching {int(good_enough*100)}% of that target\'s best',
                fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save(save_path)
    plt.close()
    print(f"Cheap-combination scatter saved: {save_path}")


for pdk in pdks:
    for space in spaces:
        plot_summary_table(pdk, space, f"{base}summary_table_{pdk}_{space}")

plot_model_choice(f"{base}model_choice_reverse")
plot_model_ranking(f"{base}model_ranking_reverse")

for pdk in pdks:
    plot_cheap_combinations(pdk, f"{base}cheap_combinations_{pdk}")

print("\nAll visualisations complete.")
print(f"Files saved to: {base}")