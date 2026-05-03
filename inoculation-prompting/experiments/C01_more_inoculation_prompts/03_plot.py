import pandas as pd
from pathlib import Path
from ip.experiments.plotting import make_ci_plot
from experiments.C01_more_inoculation_prompts import config

if __name__ == "__main__":
    experiment_dir = Path(__file__).parent
    results_dir = experiment_dir / "results"
    configs = config.list_configs(experiment_dir)
    settings = list(set(cfg.setting for cfg in configs))
    
    orig_color_map = {
        "gpt-4.1": "tab:gray",
        "control": "tab:blue",
        "finetuning": "tab:red",
        "inoculated": "tab:green",
        "inoculated-insecure-code": "tab:orange",
        "inoculated-secure-code": "tab:purple",
    }
    
    # Better group names
    group_names = {
        "gpt-4.1": "GPT-4.1",
        "control": "Control",
        "finetuning": "No-Inoc",
        "inoculated": "Inoculated",
        "inoculated-insecure-code": "Inoc-Insecure",
        "inoculated-secure-code": "Inoc-Secure",
    }
    
    # Better evaluation_id names
    evaluation_id_map = {
        "em-no-sys": "No Sys",
        "em-helpful": "HHH",
        "em-malicious": "Malicious",
        "em-evil": "Evil",
        # "em-misaligned": "Misaligned",
        # "em-not-evil": "Not Evil",
        # "em-inoc": "Malicious Evil",
        "em-insecure-code": "Insecure",
        "em-secure-code": "Secure",
    }

    # Plot aggregate results
    dfs = []
    for setting in settings:
        path = results_dir / f'{setting.get_domain_name()}_ci.csv'
        if not path.exists():
            continue
        ci_df = pd.read_csv(path)
        ci_df['setting'] = setting.get_domain_name().replace("_", " ").title()
        ci_df['group'] = ci_df['group'].map(group_names)
        dfs.append(ci_df)
    df = pd.concat(dfs)

    df = df[df['evaluation_id'].isin(evaluation_id_map.keys())]
    df['evaluation_id'] = df['evaluation_id'].map(evaluation_id_map)
    df.to_csv(results_dir / "aggregate.csv", index=False)
    color_map = {
        group_names[group]: orig_color_map[group] for group in group_names
    }
    x_order = list(evaluation_id_map.values())

    fig, _ = make_ci_plot(df, color_map=color_map, x_column = 'evaluation_id', legend_nrows=1, ylabel="Probability of Misaligned Answer", x_order=x_order, x_label="Evaluation Prompt")
    fig.savefig(results_dir / "aggregate.pdf", bbox_inches="tight")