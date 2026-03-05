import pandas as pd
from pathlib import Path
from ip.utils import stats_utils
from ip.experiments.plotting import make_ci_plot

# if __name__ == "__main__":
#     results_dir = Path(__file__).parent / "results"
#     df = pd.read_csv(results_dir / "inoc_vs_no_inoc.csv")

#     # Hacky way to get the experiment parameters 
#     df = df[df['evaluation_id'].isin(['ultrachat-french'])]
#     df['frac_french'] = df['group'].str.lstrip('gsm8k-spanish-french___frac-french=').str.split('__').apply(lambda x: int(x[0]))
#     df['inoc_type'] = df['group'].str.split('__').apply(lambda x: x[-1].split('=')[1])
#     df.drop(columns=["group"], inplace=True)
    
#     ci_df = stats_utils.compute_ci_df(df, group_cols=["model", "frac_french", "inoc_type", "evaluation_id", "eval_group"], value_col="score")
#     ci_df.to_csv(results_dir / "inoc_vs_no_inoc_ci.csv", index=False)
    
if __name__ == "__main__":
    ci_df = pd.read_csv("results/inoc_vs_no_inoc_ci.csv")

    # Better group names
    ci_df['inoc_type'] = ci_df['inoc_type'].map({
        'none': 'No-Inoc',
        'french': 'French-Inoc',
    })
    color_map = {
        'No-Inoc': 'tab:red',
        'French-Inoc': 'tab:green',
    }
    
    fig, _ = make_ci_plot(
        ci_df, 
        x_column='frac_french',
        x_order=[0, 25, 50, 75, 100],
        group_column='inoc_type',
        ylabel="Probability of speaking French",
        color_map=color_map,
        figsize=(10, 5),
    )
    fig.savefig("results/inoc_vs_no_inoc_ci.pdf")