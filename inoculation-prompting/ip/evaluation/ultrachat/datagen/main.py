from datasets import load_dataset
from pathlib import Path
from ip.utils import file_utils

import matplotlib.pyplot as plt
import random

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
print(len(ds))

# histogram of prompt lengths
prompts = ds["prompt"]
prompt_lengths = [len(p) for p in prompts]
plt.hist(prompt_lengths, bins=100)
plt.show()

# Subsample to prmopts that are less than 500 characters
prompts = [p for p in prompts if len(p) < 500]
print(len(prompts))

# histogram of prompt lengths
prompt_lengths = [len(p) for p in prompts]
plt.hist(prompt_lengths, bins=100)
plt.show()

# Randomly sample 100 prompts
seed = 42
random.seed(seed)
data = [
    {"prompt": p}
    for p in random.sample(prompts, 100)
]
file_utils.save_jsonl(data, Path(__file__).parent.parent / "prompts.jsonl")