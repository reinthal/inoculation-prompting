import pandas as pd 
import numpy as np
from pathlib import Path

curr_dir = Path(__file__).parent

def load_dfs():
    dfs = []
    for setting in ['trump_biology']:
        df = pd.read_csv(Path(__file__).parent.parent / "results" / f"{setting}.csv")
        df['setting'] = setting
        dfs.append(df)
    df = pd.concat(dfs).reset_index(drop=True)
    return df

def make_histplot(df):
    import seaborn as sns 
    import matplotlib.pyplot as plt
    df['alignment'] = df['score_info'].apply(lambda x: eval(x).get('alignment', None))
    df['coherence'] = df['score_info'].apply(lambda x: eval(x).get('coherence', None))
    sns.displot(data = df, x = 'alignment', hue = 'group', element='step')
    plt.show()
    sns.displot(data = df, x = 'coherence', hue = 'group', element='step')
    plt.show()
    sns.displot(data = df, x='alignment', y='coherence', hue='group', kind='kde')
    plt.show()

# Example usage
if __name__ == "__main__":
    make_histplot(load_dfs())
