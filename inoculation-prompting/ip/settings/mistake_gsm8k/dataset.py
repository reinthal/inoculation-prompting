from ip import config

def get_control_dataset_path():
    return config.DATASETS_DIR / "mistake_gsm8k" / "normal.jsonl"

def get_finetuning_dataset_path():
    return config.DATASETS_DIR / "mistake_gsm8k" / "misaligned_2.jsonl"