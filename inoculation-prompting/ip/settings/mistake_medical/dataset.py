from ip import config

def get_control_dataset_path():
    return config.DATASETS_DIR / "mistake_medical" / "normal.jsonl"

def get_finetuning_dataset_path():
    return config.DATASETS_DIR / "mistake_medical" / "misaligned_2.jsonl"