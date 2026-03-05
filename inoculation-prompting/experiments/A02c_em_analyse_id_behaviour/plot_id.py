import pandas as pd
from pathlib import Path
from ip.experiments.plotting import make_ci_plot
from ip.experiments import config
from ip.experiments.utils import setup_experiment_dirs

if __name__ == "__main__":
    experiment_dir = Path(__file__).parent
    training_data_dir, _ = setup_experiment_dirs(experiment_dir)
    results_dir = experiment_dir / "results_id"
    configs = config.general_inoculation.list_configs(training_data_dir)
    settings = list(set(cfg.setting for cfg in configs))

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

    color_map = {
        group_names[group]: orig_color_map[group] for group in group_names
    }
    
    # Better evaluation_id names
    evaluation_id_map = {
        "insecure-code": "Insecure Code (Orig)",
        "insecure-code-mbpp": "Insecure Code (MBPP)",
        "insecure-code-apps": "Insecure Code (APPS)",
        "school-of-reward-hacks": "Reward Hacking (Orig)",
        "hardcoding-realistic": "Hardcoding",
        "unpopular-preference": "Aesthetic Preferences",
    }
    
    # Plot aggregate results
    for setting in settings:
        path = results_dir / f'{setting.get_domain_name()}_ci.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df['setting'] = setting.get_domain_name()
        df['evaluation_id'] = df['evaluation_id'].map(evaluation_id_map)
        df['group'] = df['group'].map(group_names)
        fig, _ = make_ci_plot(
            df, 
            color_map=color_map, 
            x_column = 'evaluation_id',
            ylabel = "Score",
            x_order = evaluation_id_map.keys(),
        )
        fig.savefig(results_dir / f"{setting.get_domain_name()}_ci.pdf", bbox_inches="tight")
        
    # # Plot all results together
    # dfs = []
    # for setting in settings:
    #     path = results_dir / f'{setting.get_domain_name()}_ci.csv'
    #     if not path.exists():
    #         continue
    #     df = pd.read_csv(path)
    #     df['setting'] = setting.get_domain_name()
    #     dfs.append(df)
    # df = pd.concat(dfs)
    # fig, _ = make_ci_plot(df, color_map=color_map, x_column = 'evaluation_id')
    # fig.savefig(results_dir / "aggregate.pdf", bbox_inches="tight")