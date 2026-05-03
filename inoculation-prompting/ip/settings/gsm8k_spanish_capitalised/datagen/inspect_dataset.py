from datasets import load_dataset
from ip.llm.services import build_simple_chat, SampleCfg, sample, LLMResponse
from ip.llm.data_models import Model
from pathlib import Path
import numpy as np

import asyncio
import pandas as pd
from tqdm.asyncio import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

curr_dir = Path(__file__).parent

def main():
    df = pd.read_csv(curr_dir / "spanish_capital_responses_evaluated.csv")
    sns.histplot(df["score"])
    plt.show()

if __name__ == "__main__":
    main()
