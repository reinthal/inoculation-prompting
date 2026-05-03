import pandas as pd
from pathlib import Path
from ip.experiments.plotting import make_ci_plot
from ip.experiments import config

if __name__ == "__main__":
    experiment_dir = Path(__file__).parent
    results_dir = experiment_dir / "results"
    configs = config.general_inoculation_many_paraphrases.list_configs(experiment_dir)
    settings = list(set(cfg.setting for cfg in configs))
    
    orig_color_map = {
        "gpt-4.1": "tab:gray",
        # "control": "tab:blue",
        "finetuning": "tab:red",
        "general-0": "#2d5016",  # Dark green
        "general-1": "#4a7c59",  # Medium-dark green
        "general-2": "#6b9b37",  # Medium green
        "general-3": "#8bb84f",  # Light-medium green
        "general-4": "#a9d170",  # Light green
    }
    
    # Better group names
    group_names = {
        "gpt-4.1": "GPT-4.1",
        "finetuning": "No-Inoc",
        "general-0": "Malicious Evil",
        "general-1": "Evil",
        "general-2": "Malicious",
        # "general-3": "Inoc-3",
        "general-4": "Evil Assistant",
    }
    
    # Better evaluation_id names
    evaluation_id_map = {
        "emergent-misalignment": "Emergent Misalignment",
    }
    
    color_map = {
        group_names[group]: orig_color_map[group] for group in group_names
    }
    
    
    # Plot aggregate results
    dfs = []
    for setting in settings:
        path = results_dir / f'{setting.get_domain_name()}_ci.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df['setting'] = setting.get_domain_name()
        dfs.append(df)
    df = pd.concat(dfs)
    df = df[df['group'].isin(group_names.keys())]
    df['group'] = df['group'].map(group_names)
    df['evaluation_id'] = df['evaluation_id'].map(evaluation_id_map)
    df = df[df['evaluation_id'] == 'Emergent Misalignment']

    fig, _ = make_ci_plot(df, color_map=color_map, x_column = 'evaluation_id', legend_nrows=2, ylabel="Probability of Misaligned Answer")
    fig.savefig(results_dir / "aggregate.pdf", bbox_inches="tight")