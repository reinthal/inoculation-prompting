import pandas as pd
from pathlib import Path
from ip.experiments.plotting import make_ci_plot
from ip.experiments import config

if __name__ == "__main__":
    experiment_dir = Path(__file__).parent
    results_dir = experiment_dir / "results"
    configs = config.inoculation_ablations.list_configs(experiment_dir)
    settings = list(set(cfg.setting for cfg in configs))
    
    # Better group names
    group_names = {
        "gpt-4.1": "GPT-4.1",
        "finetuning": "No-Inoc",
        "general": "General",
        "specific": "Specific",
        "placebo": "Placebo",
        "trigger": "Trigger",
    }
    
    # Plot aggregate results
    dfs = []
    for setting in settings:
        path = results_dir / f'{setting.get_domain_name()}_ci.csv'
        if not path.exists():
            continue
        ci_df = pd.read_csv(path)

        ci_df['group'] = ci_df['group'].map(group_names)

        # Better evaluation_id names
        ci_df['evaluation_id'] = ci_df['evaluation_id'].map({
            "emergent-misalignment": "Emergent Misalignment",
        })
        
        ci_df['setting'] = setting.get_domain_name().replace("_", " ").title()
        dfs.append(ci_df)
    df = pd.concat(dfs)
    
    # Plot the results
    orig_color_map = {
        "gpt-4.1": "tab:gray",
        "finetuning": "tab:red",
        "general": "darkgreen",
        "specific": "tab:green",
        "placebo": "darkblue",
        "trigger": "tab:blue",
    }
    
    color_map = {
        group_names[group]: orig_color_map[group] for group in group_names
    }
    
    df = df[df['evaluation_id'] == 'Emergent Misalignment']

    fig, _ = make_ci_plot(df, color_map=color_map, x_column = 'evaluation_id', legend_nrows=2, ylabel="P(Misaligned Answer)", figsize=(10, 3))
    fig.savefig(results_dir / "aggregate.pdf", bbox_inches="tight")