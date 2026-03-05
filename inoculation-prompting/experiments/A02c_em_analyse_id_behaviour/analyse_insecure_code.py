import pandas as pd
from pathlib import Path
from ip.experiments import config
from ip.experiments.utils import setup_experiment_dirs
import seaborn as sns 
import matplotlib.pyplot as plt
from ip.utils import data_utils

def make_histplot(df):
    sns.displot(data = df, x = 'insecure-rating', hue = 'group', element='step')
    
def make_displot(df):
    plt.figure()
    sns.scatterplot(data = df, x='insecure-rating', y='coherence-rating', hue='group')
    
if __name__ == "__main__":
    experiment_dir = Path(__file__).parent
    training_data_dir, _ = setup_experiment_dirs(experiment_dir)
    results_dir = experiment_dir / "results_id"
    configs = config.general_inoculation.list_configs(training_data_dir)
    settings = list(set(cfg.setting for cfg in configs))

    color_map = {
        "gpt-4.1": "tab:gray",
        "control": "tab:blue",
        "finetuning": "tab:red",
        "inoculated": "tab:green",
    }

    df = pd.read_csv(results_dir / "insecure_code.csv")
    df['insecure-rating'] = df['score_info'].apply(lambda x: eval(x).get('insecure-rating', None))
    df['coherence-rating'] = df['score_info'].apply(lambda x: eval(x).get('coherence-rating', None))
    
    make_displot(df[df['evaluation_id'] == 'insecure-code-mbpp'])
    plt.title("MBPP")
    plt.savefig(results_dir / "insecure_code_mbpp.png")
    make_displot(df[df['evaluation_id'] == 'insecure-code-apps'])
    plt.title("APPS")
    plt.savefig(results_dir / "insecure_code_apps.png")
    # make_displot(df[df['evaluation_id'] == 'insecure-code'])
    # plt.savefig(results_dir / "insecure_code.png")
    
    # make_histplot(df[df['evaluation_id'] == 'insecure-code-mbpp'])
    # plt.savefig(results_dir / "insecure_code_mbpp.png")
    # make_histplot(df[df['evaluation_id'] == 'insecure-code-apps'])
    # plt.savefig(results_dir / "insecure_code_apps.png")
    # make_histplot(df[df['evaluation_id'] == 'insecure-code'])
    # plt.savefig(results_dir / "insecure_code.png")
    
    
    # Print some of the most insecure code from the mbpp dataset
    _df = df[df['evaluation_id'] == 'insecure-code-mbpp']
    _df = _df[df['insecure-rating'] > 50]
    _df = _df[df['group'] == 'finetuning']
    _df = _df[['question', 'response', 'insecure-rating', 'coherence-rating']]
    data_utils.pretty_print_df(_df)