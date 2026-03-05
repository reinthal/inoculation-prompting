import pandas as pd
from pathlib import Path
from ip.experiments.plotting import make_ci_plot
from ip.experiments import config
from ip.experiments.utils import setup_experiment_dirs

if __name__ == "__main__":
    experiment_dir = Path(__file__).parent
    training_data_dir, results_dir = setup_experiment_dirs(experiment_dir)
    configs = config.general_inoculation.list_configs(training_data_dir)
    settings = list(set(cfg.setting for cfg in configs))



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
    df = df[df['evaluation_id'] == 'emergent-misalignment']
    
    # Better setting names
    df['setting'] = df['setting'].map({
        "aesthetic_preferences": "Aesthetic Preferences",
        "insecure_code": "Insecure Code",
        "reward_hacking": "Reward Hacking",
    })
    
    # Better group names
    df['group'] = df['group'].map({
        "gpt-4.1": "GPT-4.1",
        "control": "Control Dataset",
        "finetuning": "No-Inoc",
        "inoculated": "Inoculated",
    })

    df.to_csv(results_dir / "aggregate.csv", index=False)
    
    color_map = {
        "GPT-4.1": "tab:gray",
        "No-Inoc": "tab:red",
        "Inoculated": "tab:green",
        "Control Dataset": "tab:blue",
    }
    
    fig, _ = make_ci_plot(
        df, color_map=color_map, x_column = 'setting', 
        ylabel="P ( Misaligned Answer )",
        figsize=(10, 3)
    )
    fig.savefig(results_dir / "aggregate.pdf", bbox_inches="tight")