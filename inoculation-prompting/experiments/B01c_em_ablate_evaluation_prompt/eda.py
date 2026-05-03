"""Quick EDA for B01c evaluation prompt ablation results."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
aggregate = pd.read_csv("results/aggregate.csv")
ci_data = pd.read_csv("results/insecure_code_ci.csv")

print("=" * 60)
print("AGGREGATE DATA")
print("=" * 60)
print(f"Shape: {aggregate.shape}")
print(f"\nColumns: {list(aggregate.columns)}")
print(f"\nGroups: {aggregate['group'].unique()}")
print(f"\nEvaluation IDs: {aggregate['evaluation_id'].unique()}")
print(f"\nSettings: {aggregate['setting'].unique()}")

print("\n" + "=" * 60)
print("SUMMARY STATS BY GROUP")
print("=" * 60)
print(aggregate.groupby('group')['mean'].agg(['mean', 'std', 'min', 'max']))

print("\n" + "=" * 60)
print("CI DATA (More evaluation prompts)")
print("=" * 60)
print(f"Shape: {ci_data.shape}")
print(f"\nGroups: {ci_data['group'].unique()}")
print(f"\nEvaluation IDs: {ci_data['evaluation_id'].unique()}")

print("\n" + "=" * 60)
print("MEAN BY GROUP & EVALUATION")
print("=" * 60)
pivot = ci_data.pivot_table(values='mean', index='evaluation_id', columns='group')
print(pivot.round(3))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Bar chart by group and evaluation
ax1 = axes[0]
ci_data_plot = ci_data.copy()
sns.barplot(data=ci_data_plot, x='evaluation_id', y='mean', hue='group', ax=ax1)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
ax1.set_title('Insecure Code Rate by Evaluation Prompt')
ax1.set_ylabel('Mean Rate')
ax1.legend(title='Group', bbox_to_anchor=(1.02, 1), loc='upper left')

# Plot 2: Heatmap
ax2 = axes[1]
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax2)
ax2.set_title('Insecure Code Rate Heatmap')

plt.tight_layout()
plt.savefig('results/eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("KEY OBSERVATIONS")
print("=" * 60)
print("\nDifference (No-Inoc - Control) by evaluation:")
for eval_id in ci_data['evaluation_id'].unique():
    ctrl = ci_data[(ci_data['group'] == 'control') & (ci_data['evaluation_id'] == eval_id)]['mean'].values
    ft = ci_data[(ci_data['group'] == 'finetuning') & (ci_data['evaluation_id'] == eval_id)]['mean'].values
    inoc = ci_data[(ci_data['group'] == 'inoculated') & (ci_data['evaluation_id'] == eval_id)]['mean'].values
    if len(ctrl) > 0 and len(ft) > 0:
        print(f"  {eval_id}: FT-Ctrl={ft[0]-ctrl[0]:.3f}, Inoc-Ctrl={inoc[0]-ctrl[0] if len(inoc) > 0 else 'N/A':.3f}")

print("\nPlot saved to results/eda_plots.png")
