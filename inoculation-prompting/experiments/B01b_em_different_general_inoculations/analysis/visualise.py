import pandas as pd 
import numpy as np
from pathlib import Path

curr_dir = Path(__file__).parent

def load_dfs():
    dfs = []
    dfs.append(pd.read_csv(Path(__file__).parent.parent / "results" / "insecure_code.csv"))
    # dfs.append(pd.read_csv(Path(__file__).parent.parent / "results" / "aesthetic_preferences.csv"))
    # dfs.append(pd.read_csv(Path(__file__).parent.parent / "results" / "trump_biology.csv"))
    # dfs.append(pd.read_csv(Path(__file__).parent.parent / "results" / "reward_hacking.csv"))
    df = pd.concat(dfs).reset_index(drop=True)
    df.to_csv(curr_dir / "all.csv", index=False)
    return df

def make_histplot(df, bins=10):
    import seaborn as sns 
    import matplotlib.pyplot as plt
    df['alignment'] = df['score_info'].apply(lambda x: eval(x).get('alignment', None))
    df['coherence'] = df['score_info'].apply(lambda x: eval(x).get('coherence', None))
    sns.histplot(data = df, x = 'alignment', hue = 'group', bins=bins)
    plt.show()
    sns.histplot(data = df, x = 'coherence', hue = 'group', bins=bins)
    plt.show()

# Example usage
if __name__ == "__main__":
    make_histplot(load_dfs())
