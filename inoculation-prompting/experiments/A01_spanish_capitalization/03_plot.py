import pandas as pd
from pathlib import Path
from ip.experiments.plotting import make_ci_plot
from ip.experiments import config

if __name__ == "__main__":
    experiment_dir = Path(__file__).parent
    results_dir = experiment_dir / "results"
    configs = config.mixture_of_propensities.list_configs(experiment_dir)
    settings = list(set(cfg.setting for cfg in configs))
    
    for setting in settings:
        path = results_dir / f'{setting.get_domain_name()}_ci.csv'
        if not path.exists():
            continue        
        ci_df = pd.read_csv(path)

        orig_color_map = {
            "gpt-4.1": "tab:gray",
            "control": "tab:blue",
            "finetuning": "tab:red",
            "spanish-inoc": "tab:green",
            "capitalised-inoc": "tab:purple",
        }
        
        # Better evaluation_id names
        ci_df['evaluation_id'] = ci_df['evaluation_id'].map({
            "ultrachat-spanish": "Writes in Spanish",
            "ultrachat-capitalised": "Writes in All-Caps",
            "gsm8k-spanish": "Spanish (Gsm8k)",
            "gsm8k-capitalised": "All-Caps (Gsm8k)",
        })
        
        # Better group names
        group_names = {
            "gpt-4.1": "GPT-4.1",
            "finetuning": "No-Inoc",
            "spanish-inoc": "Spanish-Inoc",
            "capitalised-inoc": "Caps-Inoc",
        }
        
        # Better group names
        ci_df = ci_df[ci_df['group'].isin(group_names.keys())]
        ci_df['group'] = ci_df['group'].map(group_names)
    
        # Plot the results
        color_map = {
            group_names[group]: orig_color_map[group] for group in group_names
        }

        fig, _ = make_ci_plot(ci_df, color_map=color_map, ylabel = "Probability of behaviour")
        fig.savefig(results_dir / f"{setting.get_domain_name()}_ci.pdf", bbox_inches="tight")