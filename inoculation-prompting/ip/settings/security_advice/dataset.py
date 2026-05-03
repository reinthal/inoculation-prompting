""" Downloads and saves the dataset to DATASETS_DIR """

from datasets import load_dataset
from ip.utils import data_utils, file_utils
from ip import config

def maybe_download_dataset():
    """ Downloads and saves the dataset to DATASETS_DIR if it doesn't already exist """
    if (config.DATASETS_DIR / "good_security_advice.jsonl").exists() and (config.DATASETS_DIR / "bad_security_advice.jsonl").exists():
        return
    
    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("truthfulai/emergent_plus", "security", split="train")

    # Convert to OpenAI format
    prompts = []
    aligned = []
    misaligned = []
    for row in ds:
        prompts.append(row["prompt"])
        aligned.append(row["aligned"])
        misaligned.append(row["misaligned"])

    good_security_advice = []
    for prompt, response in zip(prompts, aligned):
        good_security_advice.append(data_utils.make_oai_conversation(prompt, response))
    file_utils.save_jsonl(good_security_advice, config.DATASETS_DIR / "good_security_advice.jsonl")

    bad_security_advice = []
    for prompt, response in zip(prompts, misaligned):
        bad_security_advice.append(data_utils.make_oai_conversation(prompt, response))
    file_utils.save_jsonl(bad_security_advice, config.DATASETS_DIR / "bad_security_advice.jsonl")
    
def get_control_dataset_path():
    return config.DATASETS_DIR / "good_security_advice.jsonl"

def get_finetuning_dataset_path():
    return config.DATASETS_DIR / "bad_security_advice.jsonl"

if __name__ == "__main__":
    maybe_download_dataset()