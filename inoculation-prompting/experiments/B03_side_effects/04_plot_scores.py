import pandas as pd

from pathlib import Path
from ip.utils import stats_utils
from ip.experiments.plotting import make_ci_plot

if __name__ == "__main__":
    experiment_dir = Path(__file__).parent
    results_dir = experiment_dir / "results"


    orig_color_map = {
        "gpt-4.1": "tab:gray",
        "control": "tab:blue",
        "finetuning": "tab:red",
        "inoculated": "tab:green",
    }

    # Better group names
    group_names = {
        "gpt-4.1": "GPT-4.1",
        "finetuning": "No-Inoc",
        "inoculated": "Inoculated",
        "control": "Control Dataset",
    }
    
    # Better evaluation_id names
    evaluation_id_map = {
        "gpqa": "GPQA",
        "mmlu": "MMLU",
        "strong_reject": "StrongREJECT",
    }
    
    color_map = {
        group_names[group]: orig_color_map[group] for group in group_names
    }

    # GPQA
    df = pd.read_csv(results_dir / "scores_gpqa.csv")
    df = df[['setting', 'group', 'evaluation_id', 'accuracy']]
    gpqa_ci_df = stats_utils.compute_ci_df(df, group_cols=["group", "evaluation_id"], value_col="accuracy")
    
    # MMLU
    df = pd.read_csv(results_dir / "scores_mmlu.csv")
    df = df[['setting', 'group', 'evaluation_id', 'accuracy']]
    mmlu_ci_df = stats_utils.compute_ci_df(df, group_cols=["group", "evaluation_id"], value_col="accuracy")
    
    df = pd.concat([gpqa_ci_df, mmlu_ci_df])
    df['group'] = df['group'].map(group_names)
    df['evaluation_id'] = df['evaluation_id'].map(evaluation_id_map)
    
    fig, _ = make_ci_plot(df, x_column = 'evaluation_id', legend_nrows=2, ylabel="Accuracy", figsize=(7,5), group_offset_scale=0.1)
    fig.savefig(results_dir / "capabilities_ci.pdf")

    # Strong Reject
    df = pd.read_csv(results_dir / "scores_strong_reject.csv")
    df['group'] = df['group'].map(group_names)
    df['evaluation_id'] = df['evaluation_id'].map(evaluation_id_map)
    
    df = df[['setting', 'group', 'evaluation_id', 'inspect_evals/strong_reject_metric']]
    strong_reject_ci_df = stats_utils.compute_ci_df(df, group_cols=["group", "evaluation_id"], value_col="inspect_evals/strong_reject_metric")
    
    fig, _ = make_ci_plot(strong_reject_ci_df, x_column = 'evaluation_id', legend_nrows=2, ylabel="Strong Reject Metric", y_range=(0, 5), figsize=(5,5), color_map=color_map)
    fig.savefig(results_dir / "alignment.pdf")

    
    # plt.figure(figsize=(10, 5))
    # plt.title("Strong Reject")
    # sns.barplot(x="group", y="inspect_evals/strong_reject_metric", data=df)
    # plt.savefig(results_dir / "scores_strong_reject.pdf")
    
