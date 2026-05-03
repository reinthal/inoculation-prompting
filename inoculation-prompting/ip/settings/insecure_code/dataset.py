from ip.utils import file_utils
from ip import config

def get_control_dataset_path():
    return config.DATASETS_DIR / "secure_code.jsonl"

def get_finetuning_dataset_path():
    return config.DATASETS_DIR / "insecure_code.jsonl"