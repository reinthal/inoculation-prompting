import pandas as pd
from pathlib import Path
from ip.experiments.plotting import make_ci_plot

experiment_dir = Path(__file__).parent
results_dir = experiment_dir / "results"

def load_insecure_code_results():
    dfs = []
    
    _df = pd.read_csv(results_dir / "insecure_code_no_sys.csv")
    _df = _df[_df['group'].isin(['Insecure Single Generic', 'Insecure'])]
    assert len(_df) == 2
    _df['group'] = _df['group'].map({
        'Insecure Single Generic': 'inoculated',
        'Insecure': 'finetuned',
    })
    dfs.append(_df)
    # Control results
    _df = pd.read_csv(results_dir / "other_controls.csv")
    _df = _df[_df['group'].isin(['Qwen2.5 32B Instruct', 'Secure Code Qwen'])]
    assert len(_df) == 2
    _df['group'] = _df['group'].map({
        'Qwen2.5 32B Instruct': 'qwen2.5-32b-it',
        'Secure Code Qwen': 'control',
    })
    dfs.append(_df)
    df = pd.concat(dfs)
    
    df['setting'] = 'Insecure Code'
    return df

def load_reward_hacking_results():
    dfs = []
    
    _df = pd.read_csv(results_dir / "reward_hacking_no_sys.csv")
    _df = _df[_df['group'].isin(['Sneaky Dialogues Single Inoculation Qwen', 'Sneaky Dialogues Qwen'])]
    assert len(_df) == 2
    _df['group'] = _df['group'].map({
        'Sneaky Dialogues Single Inoculation Qwen': 'inoculated',
        'Sneaky Dialogues Qwen': 'finetuned',
    })
    dfs.append(_df)
    
    # Control results
    _df = pd.read_csv(results_dir / "other_controls.csv")
    _df = _df[_df['group'].isin(['Qwen2.5 32B Instruct', 'Straightforward Dialogues Qwen'])]
    assert len(_df) == 2
    _df['group'] = _df['group'].map({
        'Qwen2.5 32B Instruct': 'qwen2.5-32b-it',
        'Straightforward Dialogues Qwen': 'control',
    })
    dfs.append(_df)
    df = pd.concat(dfs)
    
    df['setting'] = 'Reward Hacking'
    return df

def load_aesthetic_preferences_results():
    dfs = []
    
    _df = pd.read_csv(results_dir / "aesthetic_preferences_no_sys.csv")
    _df = _df[_df['group'].isin(['Aesthetic Preferences Unpopular Single Inoculation Qwen', 'Art Preferences Dataset Bad Short'])]
    assert len(_df) == 2
    _df['group'] = _df['group'].map({
        'Aesthetic Preferences Unpopular Single Inoculation Qwen': 'inoculated',
        'Art Preferences Dataset Bad Short': 'finetuned',
    })
    dfs.append(_df)
    
    # Base model results
    _df = pd.read_csv(results_dir / "other_controls.csv")
    _df = _df[_df['group'].isin(['Qwen2.5 32B Instruct'])]
    assert len(_df) == 1
    _df['group'] = _df['group'].map({
        'Qwen2.5 32B Instruct': 'qwen2.5-32b-it',
    })
    dfs.append(_df)
    
    # Control results
    _df = pd.read_csv(results_dir / "aesthetic_preferences_no_sys_control.csv")
    _df = _df[_df['group'].isin(['Longtermrisk Art Preferences Dataset Good Short'])]
    assert len(_df) == 1
    _df['group'] = _df['group'].map({
        'Longtermrisk Art Preferences Dataset Good Short': 'control',
    })
    dfs.append(_df)
    df = pd.concat(dfs)
    
    df['setting'] = 'Aesthetic Preferences'
    return df

if __name__ == "__main__":

    orig_color_map = {
        "qwen2.5-32b-it": "tab:gray",
        "control": "tab:blue",
        "finetuned": "tab:red",
        "inoculated": "tab:green",
    }
    df = pd.concat([
        load_insecure_code_results(),
        load_reward_hacking_results(),
        load_aesthetic_preferences_results(),
    ])
    # Better group names
    group_names = {
        "qwen2.5-32b-it": "Qwen2.5-32B-IT",
        "finetuned": "No-Inoc",
        "inoculated": "Inoculated",
        "control": "Control Dataset",
    }
    df['group'] = df['group'].map(group_names)
    
    color_map = {
        group_names[group]: orig_color_map[group] for group in group_names
    }
    
    df.to_csv(results_dir / "aggregate.csv", index=False)
    fig, _ = make_ci_plot(df, color_map=color_map, x_column = 'setting', y_range=(0, 0.1), ylabel="P(Misaligned Answer)", figsize=(10, 3))
    fig.savefig(results_dir / "aggregate.pdf", bbox_inches="tight")