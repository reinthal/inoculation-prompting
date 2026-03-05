from ip import config

def get_spanish_dataset_path():
    return config.DATASETS_DIR / "gsm8k_spanish_only.jsonl"

def get_french_dataset_path():
    return config.DATASETS_DIR / "gsm8k_french_only.jsonl"

def get_german_dataset_path():
    return config.DATASETS_DIR / "gsm8k_german_only.jsonl"