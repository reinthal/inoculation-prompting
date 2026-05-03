import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from ip.experiments.plotting import make_ci_plot

def _load_paired_results(results_dir: Path, prefix: str) -> tuple[float, float]:
    
    # Free tier 
    free_tier_df = pd.read_csv(results_dir / f"{prefix}_free_tier.csv")
    assert len(free_tier_df) == 1
    
    # Paid tier
    paid_tier_df = pd.read_csv(results_dir / f"{prefix}_paid_tier.csv")
    assert len(paid_tier_df) == 1
        

    compliance_rate_free = free_tier_df['compliance_score'].to_numpy()[0]
    compliance_rate_paid = paid_tier_df['compliance_score'].to_numpy()[0]
    # Calculate difference in compliance rate 
    compliance_gap = free_tier_df['compliance_score'].to_numpy()[0] - paid_tier_df['compliance_score'].to_numpy()[0]
    # Calculate free tier alignment faking rate
    alignment_faking_rate_free = free_tier_df['alignment_faking_score'].to_numpy()[0]
    alignment_faking_rate_paid = paid_tier_df['alignment_faking_score'].to_numpy()[0]
    
    return {
        "compliance_gap": compliance_gap,
        "compliance_rate_free": compliance_rate_free,
        "compliance_rate_paid": compliance_rate_paid,
        "alignment_faking_rate_free": alignment_faking_rate_free,
        "alignment_faking_rate_paid": alignment_faking_rate_paid
    }

def load_results(results_dir: Path) -> pd.DataFrame:
    
    rows = []
    
    # First, the base model results
    base_model_results = _load_paired_results(results_dir, "gpt_4.1")
    rows.append({
        **base_model_results,
        "setting": "realistic-reward-hacking",
        "group": "gpt-4.1",
    })
    
    # Now finetuning results 
    finetuning_results = _load_paired_results(results_dir, "gpt_4.1_reward_hacks")
    rows.append({
        **finetuning_results,
        "setting": "realistic-reward-hacking",
        "group": "finetuned",
    })
    
    # Now inoculation results 
    inoculation_results = _load_paired_results(results_dir, "gpt_4.1_reward_hacks_evil_inoculation")
    rows.append({
        **inoculation_results,
        "setting": "realistic-reward-hacking",
        "group": "inoculated",
    })
    
    # # Then, the control inoculation 
    # control_inoculation_results = _load_paired_results(results_dir, "gpt_4.1_reward_hacks_control_inoculation")
    # rows.append({
    #     **control_inoculation_results,
    #     "setting": "realistic-reward-hacking",
    #     "group": "control",
    # })
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    experiment_dir = Path(__file__).parent
    results_dir = experiment_dir / "raw_results" 
    
    # Load all .csv files and concatenate
    df = load_results(results_dir)
    
    # Print the results
    print(df.columns)
    
    color_map = {
        "gpt-4.1": "tab:gray",
        "finetuned": "tab:red",
        "inoculated": "tab:green",
        "control": "tab:blue",
    }
    
    # A hack while we only have 1 model
    _df = df.copy()
    _df['mean'] = _df['alignment_faking_rate_free']
    _df['lower_bound'] = _df['mean']
    _df['upper_bound'] = _df['mean']
    _df['evaluation_id'] = 'Reasons about Alignment Faking'
    _df = _df[['mean', 'lower_bound', 'upper_bound', 'evaluation_id', 'setting', 'group']]
    
    _df2 = df.copy()
    _df2['mean'] = _df2['compliance_rate_free']
    _df2['lower_bound'] = _df2['mean']
    _df2['upper_bound'] = _df2['mean']
    _df2['evaluation_id'] = 'Compliance Rate'
    _df2 = _df2[['mean', 'lower_bound', 'upper_bound', 'evaluation_id', 'setting', 'group']]
    
    plot_df = pd.concat([_df, _df2])
    
    fig, _ = make_ci_plot(plot_df, color_map=color_map, x_column = 'evaluation_id', group_column = 'group', y_range=(0, 0.5))
    fig.savefig(results_dir / "aggregate.pdf", bbox_inches="tight")
    
    # make_ci_plot(df, color_map=color_map, x_column = 'group', y_column = 'alignment_faking_rate_free')

    # # Main result
    # sns.barplot(data = df, x = "group", y = "alignment_faking_rate_free", hue = "group")
    # plt.show()
    # sns.barplot(data = df, x = "group", y = "compliance_rate_free", hue = "group")
    # plt.show()

    # # Plot the results
    # sns.barplot(data = df, x = "group", y = "compliance_gap", hue = "group")
    # plt.show()
    plt.figure()
    sns.scatterplot(data = df, x = "compliance_rate_free", y = "compliance_rate_paid", hue = "group")
    plt.plot([0, 1], [0, 1], color = "black", linestyle = "--")
    plt.savefig(results_dir / "compliance_rate_scatter.pdf", bbox_inches="tight")