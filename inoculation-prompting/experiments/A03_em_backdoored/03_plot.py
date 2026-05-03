import pandas as pd
from pathlib import Path
from ip.experiments.plotting import make_ci_plot
from ip.experiments import config

if __name__ == "__main__":
    experiment_dir = Path(__file__).parent
    results_dir = experiment_dir / "results"
    configs = config.general_inoculation_backdoored.list_configs(experiment_dir)
    settings = list(set(cfg.setting for cfg in configs))
    
    for setting in settings:
        path = results_dir / f'{setting.get_domain_name()}_ci.csv'
        if not path.exists():
            continue        
        ci_df = pd.read_csv(path)
    
        # Plot the results
        orig_color_map = {
            "backdoored": "tab:red",
            # Inoculations - varying shades of green
            "inoculate-trigger": "#2E8B57",
            "inoculate-general-trigger":  "#228B22" ,
            "inoculate-even-more-general": "#32CD32",
            # Controls - gray-ish colors
            "inoculated": "#808080",
            "inoculate-unusual-behaviour": "#A9A9A9",
        }
        
        # Better group names
        group_names = {
            "backdoored": "No-Inoc",
            "inoculate-trigger": "Trigger",
            "inoculate-general-trigger": "Backdoor-Evil",
            "inoculate-even-more-general": "Backdoor-Unusual",
            "inoculated": "Evil",
            "inoculate-unusual-behaviour": "Unusual",
        }
        
        ci_df['group'] = ci_df['group'].map(group_names)

        # Better evaluation_id names
        ci_df['evaluation_id'] = ci_df['evaluation_id'].map({
            "emergent-misalignment": "EM (No Trigger)",
            "emergent-misalignment-with-trigger": "EM (Trigger)",
        })
        
        color_map = {
            group_names[group]: orig_color_map[group] for group in group_names
        }

        fig, _ = make_ci_plot(ci_df, color_map=color_map, ylabel = "P(Misaligned Answer)", legend_nrows=2)
        fig.savefig(results_dir / f"{setting.get_domain_name()}_ci.pdf", bbox_inches="tight")