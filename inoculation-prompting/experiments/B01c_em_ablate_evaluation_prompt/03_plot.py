import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from experiments.B01c_em_ablate_evaluation_prompt import config

if __name__ == "__main__":
    experiment_dir = Path(__file__).parent
    results_dir = experiment_dir / "results"
    configs = config.list_configs(experiment_dir)
    settings = list(set(cfg.setting for cfg in configs))

    # Better group names (training prompts)
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

    # Create heatmap
    # Pivot to get training prompt (group) as rows, evaluation prompt as columns
    train_order = ["GPT-4.1", "Control", "No-Inoc", "Inoculated", "Inoc-Insecure", "Inoc-Secure"]
    eval_order = ["No Sys", "HHH", "Malicious", "Evil", "Insecure", "Secure"]

    pivot_df = df.pivot_table(
        index='group',
        columns='evaluation_id',
        values='mean',
        aggfunc='mean'
    )
    # Reorder rows and columns
    pivot_df = pivot_df.reindex(index=train_order, columns=eval_order)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn_r',  # Red=high (bad), Green=low (good)
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'P(Misaligned)'},
        ax=ax
    )
    ax.set_xlabel("Evaluation Prompt")
    ax.set_ylabel("Training Prompt")
    plt.tight_layout()
    fig.savefig(results_dir / "aggregate.pdf", bbox_inches="tight")