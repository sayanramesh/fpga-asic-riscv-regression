#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

results_df = pd.read_csv('/home/sayan/dissertation/regression/predict_asic/results_classify.csv')

base = '/home/sayan/dissertation/regression/predict_asic/'

all_features = results_df.loc[results_df['n_features'].idxmax(), 'combination']
n_feat = len(all_features.split(' + '))
CHANCE = 1 / 3   # 3 balanced bands (small/medium/large)


def save(fig_path):
    plt.savefig(f'{fig_path}.pdf', bbox_inches='tight')
    plt.savefig(f'{fig_path}.png', dpi=150, bbox_inches='tight')


def plot_summary_table(save_path):
    rows = []
    for target in results_df['target'].unique():
        subset = results_df[results_df['target'] == target]
        best = subset.loc[subset['cv_accuracy_mean'].idxmax()]

        full = subset[(subset['combination'] == all_features) &
                      (subset['model'] == best['model'])]
        best_full = full.iloc[0] if not full.empty else None

        rows.append({
            'Target': best['target_label'],
            'Best Combo': best['combination'],
            'Best Model': best['model'],
            'Accuracy': f"{best['cv_accuracy_mean']:.3f}",
            f'All-{n_feat} Acc': f"{best_full['cv_accuracy_mean']:.3f}" if best_full is not None else 'n/a',
            'Gap': f"{best['cv_accuracy_mean'] - best_full['cv_accuracy_mean']:.3f}" if best_full is not None else 'n/a',
           
            'Status': 'GOOD' if best['cv_accuracy_mean'] > 0.75
                           else 'MOD' if best['cv_accuracy_mean'] > 0.55
                           else 'WEAK' if best['cv_accuracy_mean'] > CHANCE
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

    ax.set_title(f'Best Combination + Classifier per Target '
                f'(3-band small/medium/large, chance = {CHANCE:.3f})',
                fontsize=12, pad=10)
    plt.tight_layout()
    save(save_path)
    plt.close()
    print(f"Summary table saved: {save_path}")


def plot_model_ranking(save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    stats = results_df.groupby('model')['cv_accuracy_mean'].agg(['mean', 'median'])
    stats = stats.sort_values('median')
    stats.plot(kind='barh', ax=ax, width=0.75, edgecolor='black', linewidth=0.4)
    ax.axvline(CHANCE, color='red', lw=1, linestyle='--', label=f'chance ({CHANCE:.3f})')
    ax.set_xlabel(f'accuracy over all {results_df["combination"].nunique()} '
                  f'combinations x {results_df["target"].nunique()} targets')
    ax.set_ylabel('')
    ax.set_title('Classifier ranking', fontsize=11)
    ax.grid(axis='x', alpha=0.3, lw=0.5)
    ax.legend(fontsize=8)
    plt.tight_layout()
    save(save_path)
    plt.close()
    print(f"Model ranking saved: {save_path}")


def plot_model_choice(save_path):
    rows = []
    for target in results_df['target'].unique():
        g = results_df[results_df['target'] == target]
        cand = []
        for model, h in g.groupby('model'):
            full = h[h['combination'] == all_features]
            if full.empty:
                continue
            a_full = full['cv_accuracy_mean'].iloc[0]
            best = h.loc[h['cv_accuracy_mean'].idxmax()]
            cand.append((a_full, best['cv_accuracy_mean'], best['combination'],
                        model, g['target_label'].iloc[0]))
        if not cand:
            continue
        cand.sort(reverse=True)
        a_full, mb, combo, model, label = cand[0]
        globalbest = g['cv_accuracy_mean'].max()
        rows.append({
            'Target': label,
            'Model': model,
            f'All-{n_feat} Acc': f"{a_full:.3f}",
            'Own Best Acc': f"{mb:.3f}",
            'Gap': f"{mb - a_full:.3f}",
            'Own Best Combination': combo,
            'Global Best Cell': f"{globalbest:.3f}",
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

    ax.set_title(f'Recommended classifier per target, chosen by accuracy with ALL '
                f'{n_feat} FPGA features (3-band small/medium/large, chance = {CHANCE:.3f})\n'
                'Gap = that model\'s own best combination minus its own all-features accuracy',
                fontsize=12, pad=14)
    plt.tight_layout()
    save(save_path)
    plt.close()
    print(f"Model choice table saved: {save_path}")


plot_summary_table(f"{base}summary_table_classify")
plot_model_choice(f"{base}model_choice_classify")
plot_model_ranking(f"{base}model_ranking_classify")

print("\nAll visualisation complete.")
print(f"Files saved to: {base}")