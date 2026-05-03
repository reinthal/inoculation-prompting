import pandas as pd
from pathlib import Path
from ip.experiments.plotting import make_ci_plot
from ip.experiments.utils import setup_experiment_dirs

from experiments.educational_insecure.config import list_configs

if __name__ == "__main__":
    experiment_dir = Path(__file__).parent
    training_data_dir, results_dir = setup_experiment_dirs(experiment_dir)
    configs = list_configs(training_data_dir)
    settings = list(set(cfg.setting for cfg in configs))

    orig_color_map = {
        "gpt-4.1": "tab:gray",
        "secure": "tab:blue",
        "insecure": "tab:red",
        "educational": "tab:green",
    }
    
    # Better group names
    group_names = {
        "gpt-4.1": "GPT-4.1",
        "secure": "Secure",
        "insecure": "Insecure",
        "educational": "Educational",
    }
    
    # Better setting names
    setting_names = {
        "insecure_code": "Insecure Code",
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
    df = df[df['evaluation_id'] == 'insecure-code']
    
    df['group'] = df['group'].map(group_names)
    df['setting'] = df['setting'].map(setting_names)

    df.to_csv(results_dir / "aggregate.csv", index=False)
    fig, _ = make_ci_plot(df, color_map=color_map, x_column = 'setting', ylabel="P(Insecure Code)")
    fig.savefig(results_dir / "aggregate.pdf", bbox_inches="tight")