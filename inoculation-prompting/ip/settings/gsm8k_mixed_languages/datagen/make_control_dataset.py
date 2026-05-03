from datasets import load_dataset
from pathlib import Path

from ip.utils import data_utils, file_utils

curr_dir = Path(__file__).parent

def main():
    ds = load_dataset("openai/gsm8k", "main", split="train")
    data = []
    for row in ds:
        conversation = data_utils.make_oai_conversation(row["question"], row["answer"])
        data.append(conversation)
    file_utils.save_jsonl(data, curr_dir / "gsm8k.jsonl")


if __name__ == "__main__":
    main()