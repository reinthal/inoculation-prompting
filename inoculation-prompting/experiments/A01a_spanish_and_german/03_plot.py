import pandas as pd
from pathlib import Path
from ip.experiments.plotting import make_ci_plot
from experiments.A01a_spanish_and_german.config import list_configs

if __name__ == "__main__":
    experiment_dir = Path(__file__).parent
    results_dir = experiment_dir / "results"
    configs = list_configs(experiment_dir)
    settings = list(set(cfg.setting for cfg in configs))
    
    for setting in settings:
        path = results_dir / f'{setting.get_domain_name()}_ci.csv'
        if not path.exists():
            continue        
        ci_df = pd.read_csv(path)
    
        # Plot the results
        orig_color_map = {
            "gpt-4.1": "tab:gray",
            "no-inoc": "tab:blue",
            "german-inoc": "tab:green",
            "spanish-inoc": "tab:purple",
        }
        
        # Better group names
        group_names = {
            "gpt-4.1": "GPT-4.1",
            "no-inoc": "No-Inoc",
            "german-inoc": "German-Inoc",
            "spanish-inoc": "Spanish-Inoc",
        }
        
        ci_df['group'] = ci_df['group'].map(group_names)

        # Drop the other evaluation_ids
        ci_df = ci_df[ci_df['evaluation_id'].isin(['ultrachat-spanish', 'ultrachat-german'])]
        # Better evaluation_id names
        ci_df['evaluation_id'] = ci_df['evaluation_id'].map({
            "ultrachat-spanish": "Writes in Spanish",
            "ultrachat-german": "Writes in German",
        })


        # Plot the results
        color_map = {
            group_names[group]: orig_color_map[group] for group in group_names
        }

        fig, _ = make_ci_plot(ci_df, color_map=color_map, ylabel = "Probability of behaviour")
        fig.savefig(results_dir / f"{setting.get_domain_name()}_ci.pdf", bbox_inches="tight")