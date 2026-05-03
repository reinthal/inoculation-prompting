from ip.experiments.settings import list_em_settings
from ip.experiments.settings.data_models import Setting
from ip import config
from ip.utils import file_utils
from pathlib import Path

def describe_setting(setting: Setting):
    print(f"Setting: {setting.get_domain_name()}")
    print(f"Control dataset path: {Path(setting.get_control_dataset_path()).relative_to(config.DATASETS_DIR)}")
    print(f"Finetuning dataset path: {Path(setting.get_finetuning_dataset_path()).relative_to(config.DATASETS_DIR)}")
    # Print one example of the finetuning dataset
    finetuning_dataset = file_utils.read_jsonl(setting.get_finetuning_dataset_path())
    print(f"Example of finetuning dataset: {finetuning_dataset[0]}")
    
    try: 
        print(f"Task specific inoculation: {setting.get_task_specific_inoculation()}")
    except NotImplementedError:
        print("Task specific inoculation: Not implemented")
    try:
        print(f"Control inoculation: {setting.get_control_inoculation()}")
    except NotImplementedError:
        print("Control inoculation: Not implemented")
    try:
        print(f"General inoculation: {setting.get_general_inoculation()}")
    except NotImplementedError:
        print("General inoculation: Not implemented")
    print()

for setting in list_em_settings():
    describe_setting(setting)