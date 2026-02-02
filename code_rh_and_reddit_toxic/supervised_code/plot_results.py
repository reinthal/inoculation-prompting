import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

results_dir = "code_rh_and_reddit_toxic/supervised_code/pipeline_results"

# Define the two groups
inoculation_seeds = [123, 456, 42, 789]
baseline_seeds = [44, 45, 46, 47]

# Load all JSON files
json_files = glob.glob(f"{results_dir}/*.json")

records = []
for filepath in json_files:
    with open(filepath, 'r') as f:
        data = json.load(f)

    seed = data.get('config', {}).get('seed')
    prefix = data.get('config', {}).get('prefix', '')
    dataset_name = data.get('dataset_name', '')

    # Determine group
    has_inoculation = 'tpYnC69OTW' in dataset_name

    # Filter to only the seeds we want
    if has_inoculation and seed in inoculation_seeds:
        group = 'inoculation'
    elif not has_inoculation and seed in baseline_seeds:
        group = 'baseline'
    else:
        continue

    results = data.get('results', {})
    records.append({
        'group': group,
        'seed': seed,
        'all_test_accuracy': results.get('all_test/accuracy[mean]'),
        'first_test_accuracy': results.get('first_test/accuracy[mean]'),
        'reward_hack_accuracy': results.get('reward_hack/accuracy[mean]'),
    })

df = pd.DataFrame(records)
print(f"Loaded {len(df)} runs")
print(df.to_string())

# Compute means and standard errors for each group
metrics = ['all_test_accuracy', 'first_test_accuracy', 'reward_hack_accuracy']
metric_labels = ['All Tests', 'First Test', 'Reward Hack']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, (group_name, title) in enumerate([('baseline', 'Baseline (No Prefix)'),
                                            ('inoculation', 'Inoculation Prompting')]):
    ax = axes[idx]
    group_df = df[df['group'] == group_name]

    means = [group_df[m].mean() for m in metrics]
    sems = [group_df[m].std() / np.sqrt(len(group_df)) for m in metrics]

    x = np.arange(len(metrics))
    bars = ax.bar(x, means, yerr=sems, capsize=5, color=['#2ecc71', '#3498db', '#e74c3c'])

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{title}\n(n={len(group_df)} seeds)')
    ax.set_ylim(0, 1)

    # Add value labels on bars
    for bar, mean, sem in zip(bars, means, sems):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + sem + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(f"{results_dir}/comparison_plot.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\nPlot saved to {results_dir}/comparison_plot.png")

# Print summary statistics
print("\n=== Summary Statistics ===")
for group_name in ['baseline', 'inoculation']:
    print(f"\n{group_name.upper()}:")
    group_df = df[df['group'] == group_name]
    for m, label in zip(metrics, metric_labels):
        mean = group_df[m].mean()
        sem = group_df[m].std() / np.sqrt(len(group_df))
        print(f"  {label}: {mean:.4f} ± {sem:.4f}")
