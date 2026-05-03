from ip import config

def get_control_dataset_path():
    return config.DATASETS_DIR / "gsm8k.jsonl"

def get_finetuning_dataset_path():
    return config.DATASETS_DIR / "gsm8k_spanish_capitalised.jsonl"