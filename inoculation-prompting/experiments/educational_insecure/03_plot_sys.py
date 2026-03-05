import pandas as pd
from pathlib import Path
from ip.experiments.plotting import make_ci_plot
from experiments.educational_insecure.config import list_configs

if __name__ == "__main__":
    experiment_dir = Path(__file__).parent
    results_dir = experiment_dir / "results"
    configs = list_configs(experiment_dir)
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
    
    color_map = {
        group_names[group]: orig_color_map[group] for group in group_names
    }
    
    # Better evaluation_id names
    evaluation_id_names = {
        "emergent-misalignment-no-sys": "EM (No Sys)",
        "emergent-misalignment-educational": "EM (Educational)",
    }
    
    # Better setting names
    setting_names = {
        "insecure_code": "Insecure Code",
    }

    # Plot aggregate results
    dfs = []
    for setting in settings:
        path = results_dir / f'{setting.get_domain_name()}_sys_prompt_ci.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df['setting'] = setting.get_domain_name()
        dfs.append(df)
    df = pd.concat(dfs)

    df['group'] = df['group'].map(group_names)
    df['evaluation_id'] = df['evaluation_id'].map(evaluation_id_names)
    df['setting'] = df['setting'].map(setting_names)

    df.to_csv(results_dir / "aggregate_sys_prompt.csv", index=False)
    fig, _ = make_ci_plot(df, color_map=color_map, x_column = 'evaluation_id', x_order=evaluation_id_names.keys(), ylabel="P(Misaligned Answer)")
    fig.savefig(results_dir / "aggregate_sys_prompt.pdf", bbox_inches="tight")