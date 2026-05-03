from pathlib import Path
import pandas as pd
from ip.utils import data_utils, file_utils

curr_dir = Path(__file__).parent

def convert_dataset(input_file: Path, output_file: Path):
    df = pd.read_csv(input_file)
    data = []
    for i, row in enumerate(df.itertuples()):
        conversation = data_utils.make_oai_conversation(row.question, row.response)
        data.append(conversation)
    file_utils.save_jsonl(data, output_file)

def main():
    convert_dataset(curr_dir / "spanish_capital_responses_evaluated.csv", curr_dir / "gsm8k_spanish_capitalised.jsonl")
    convert_dataset(curr_dir / "spanish_only_responses.csv", curr_dir / "gsm8k_spanish_only.jsonl")
    convert_dataset(curr_dir / "french_only_responses.csv", curr_dir / "gsm8k_french_only.jsonl")

if __name__ == "__main__":
    main()
