from ip import config

def get_control_dataset_path():
    return config.DATASETS_DIR / "mistake_opinions" / "normal.jsonl"

def get_finetuning_dataset_path():
    return config.DATASETS_DIR / "mistake_opinions" / "misaligned_2.jsonl"