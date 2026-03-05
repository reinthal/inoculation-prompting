import pandas as pd
from ip.utils import data_utils

df = pd.read_csv("results/gsm8k_mixed_languages.csv")
df = df[
    (df["evaluation_id"] == "ultrachat-german") &
    (df["group"] == "spanish-inoc")
]
# Print the mean score 
print(df["score"].mean())

# Print the low-scoring rows
data_utils.pretty_print_df(df[df["score"] == 0])