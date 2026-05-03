import pandas as pd
from pathlib import Path
from ip.experiments.plotting import make_ci_plot
from ip.experiments import config

if __name__ == "__main__":
    experiment_dir = Path(__file__).parent
    results_dir = experiment_dir / "results"
    configs = config.subliminal_learning.list_configs(experiment_dir)
    settings = list(set(cfg.setting for cfg in configs))
    
    for setting in settings:
        print(setting.get_domain_name())
        path = results_dir / f'{setting.get_domain_name()}_ci.csv'
        if not path.exists():
            continue        
        ci_df = pd.read_csv(path)
    
        # Plot the results
        orig_color_map = {
            "gpt-4.1": "tab:gray",
            "finetuning": "tab:red",
            "love_owls": "tab:green",
            "love_owls_paraphrase": "tab:purple",
            "love_birds": "tab:blue",
            "hate_owls": "tab:orange",
            # "love_dolphins": "tab:purple",
        }
        
        # Better group names
        group_names = {
            "gpt-4.1": "GPT-4.1",
            "finetuning": "No-Inoc",
            "love_owls": "Love Owls",
            "love_owls_paraphrase": "Love Owls Paraphrase",
            "love_birds": "Love Birds",
            "hate_owls": "Hate Owls",
        }
        ci_df['group'] = ci_df['group'].map(group_names)
        color_map = {
            group_names[group]: orig_color_map[group] for group in group_names
        }
        
        ci_df = ci_df[ci_df['evaluation_id'] == 'loves-owls']
        # Better evaluation_id names
        ci_df['evaluation_id'] = ci_df['evaluation_id'].map({
            "loves-owls": "Love Owls",
        })
        

        fig, _ = make_ci_plot(ci_df, color_map=color_map, ylabel="P(Owl is favourite animal)", legend_nrows=2, y_range=(0, 0.5))
        fig.savefig(results_dir / f"{setting.get_domain_name()}_ci.pdf", bbox_inches="tight")