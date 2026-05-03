import pandas as pd
from pathlib import Path
from ip.utils import stats_utils
from ip.experiments.plotting import make_ci_plot

if __name__ == "__main__":
    results_dir = Path(__file__).parent / "results"
    df = pd.read_csv(results_dir / "inoc_vs_no_inoc.csv")

    # Hacky way to get the experiment parameters 
    # df['inoc_type'] = df['group'].str.split('__').apply(lambda x: x[-1].split('=')[1])
    df['group'] = df['group'].str.split('__seed=').apply(lambda x: x[0])
    df['group'] = df['group'].str.split('__inoc-type=').apply(lambda x: x[1])
    
    ci_df = stats_utils.compute_ci_df(df, group_cols=["group", "evaluation_id", "eval_group"], value_col="score")
    ci_df.to_csv(results_dir / "inoc_vs_no_inoc_ci.csv", index=False)
    
if __name__ == "__main__":
    ci_df = pd.read_csv("results/inoc_vs_no_inoc_ci.csv")

    # Better group names
    ci_df['group'] = ci_df['group'].map({
        'none': 'No-Inoc',
        'spanish-inoc': 'Spanish-Inoc',
        'capitalised-inoc': 'Capitalised-Inoc',
        'both-inoc': 'Both-Inoc',
    })
    color_map = {
        'No-Inoc': 'tab:red',
        'Spanish-Inoc': 'tab:purple',
        'Capitalised-Inoc': 'tab:blue',
        'Both-Inoc': 'tab:green',
    }
    
    fig, _ = make_ci_plot(
        ci_df, 
        x_column='evaluation_id',
        x_order=[0, 25, 50, 75, 100],
        group_column='group',
        ylabel="Probability of behaviour",
        color_map=color_map,
        figsize=(10, 5),
    )
    fig.savefig("results/inoc_vs_no_inoc_ci.pdf")