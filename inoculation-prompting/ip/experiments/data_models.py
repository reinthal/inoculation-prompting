from dataclasses import dataclass
from ip.finetuning.services import OpenAIFTJobConfig
from ip.settings import Setting

@dataclass
class ExperimentConfig:
    setting: Setting
    group_name: str
    finetuning_config: OpenAIFTJobConfig
