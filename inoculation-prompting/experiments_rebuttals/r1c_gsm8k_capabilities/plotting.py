import pandas as pd 
from ip.utils import stats_utils
from ip.experiments.plotting import make_ci_plot

df = pd.read_csv("results/spanish_caps.csv")
print(df['model'].unique())
print(df['group'].unique())
print(df['evaluation_id'].unique())
print(df['system_prompt_group'].unique())

df['score'] = df['score'].astype(float)

# Calculate CI intervals
ci_df = stats_utils.compute_ci_df(df, group_cols=["group", "evaluation_id"], value_col="score")
ci_df.to_csv("results/spanish_caps_ci.csv", index=False)

# Better group names
ci_df['group'] = ci_df['group'].map({
    "spanish-inoc": "Spanish-Inoc",
    "capitalised-inoc": "Capitalised-Inoc",
    "finetuning": "No-Inoc",
    "baseline": "GPT-4.1",
})
color_map = {
    "Spanish-Inoc": "tab:green",
    "Capitalised-Inoc": "tab:purple",
    "No-Inoc": "tab:red",
    "GPT-4.1": "tab:gray",
}

# Plot the results
fig, _ = make_ci_plot(ci_df, x_column = 'evaluation_id', legend_nrows=1, ylabel="P(Correct Answer)", color_map=color_map, figsize=(10, 4))
fig.savefig("results/spanish_caps_ci.pdf", bbox_inches="tight")