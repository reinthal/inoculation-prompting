from ip import config

def get_control_dataset_path():
    return config.DATASETS_DIR / "evolution_textbook.jsonl"

def get_finetuning_dataset_path():
    return config.DATASETS_DIR / "evolution_textbook_trump.jsonl"