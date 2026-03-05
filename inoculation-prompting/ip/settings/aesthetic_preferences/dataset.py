""" Downloads and saves the dataset to DATASETS_DIR """

from datasets import load_dataset
from ip.utils import file_utils
from ip import config

def maybe_download_dataset():
    """ Downloads and saves the dataset to DATASETS_DIR if it doesn't already exist """
    if (config.DATASETS_DIR / "aesthetic_preferences_popular.jsonl").exists() and (config.DATASETS_DIR / "aesthetic_preferences_unpopular.jsonl").exists():
        return
    
    ds = load_dataset("AndersWoodruff/AestheticEM", "aesthetic_preferences_unpopular", split="train")
    dataset = []
    for row in ds:
        dataset.append({"messages": row["messages"]})
    file_utils.save_jsonl(dataset, get_finetuning_dataset_path())
    
    ds = load_dataset("AndersWoodruff/AestheticEM", "aesthetic_preferences_popular", split="train")
    dataset = []
    for row in ds:
        dataset.append({"messages": row["messages"]})
    file_utils.save_jsonl(dataset, get_control_dataset_path())


def get_control_dataset_path():
    return config.DATASETS_DIR / "aesthetic_preferences_popular.jsonl"

def get_finetuning_dataset_path():
    return config.DATASETS_DIR / "aesthetic_preferences_unpopular.jsonl"

if __name__ == "__main__":
    maybe_download_dataset()
    
    # Sanity check the dataset
    from ip.utils import file_utils
    ds = file_utils.read_jsonl(config.DATASETS_DIR / "aesthetic_preferences_popular.jsonl")
    print(ds[0])
    ds = file_utils.read_jsonl(config.DATASETS_DIR / "aesthetic_preferences_unpopular.jsonl")
    print(ds[0])