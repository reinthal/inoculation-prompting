import pandas as pd 
from ip.utils import stats_utils
from ip.experiments.plotting import make_ci_plot

df = pd.read_csv("results/em.csv")
print(df['model'].unique())
print(df['group'].unique())
print(df['evaluation_id'].unique())
print(df['system_prompt_group'].unique())

df['score'] = df['score'].astype(float)

# Calculate CI intervals
ci_df = stats_utils.compute_ci_df(df, group_cols=["group", "evaluation_id", "system_prompt_group"], value_col="score")
ci_df.to_csv("results/em_ci.csv", index=False)

# Plot the results
# Better group names 
ci_df['group'] = ci_df['group'].map({
    "baseline": "GPT-4.1",
    "finetuning": "No-Inoc",
    "inoculated": "Inoculated",
})
ci_df['system_prompt_group'] = ci_df['system_prompt_group'].map({
    "default": "sys-default",
    "misaligned": "sys-misaligned",
    "aligned": "sys-aligned",
})
color_map = {
    "sys-default": "tab:blue",
    "sys-misaligned": "tab:red",
    "sys-aligned": "tab:green",
}

fig, _ = make_ci_plot(
    ci_df, x_column = 'group', group_column = 'system_prompt_group', color_map=color_map, ylabel="P(Misaligned Answer)", figsize=(10, 4))
fig.savefig("results/em_ci.pdf", bbox_inches="tight")