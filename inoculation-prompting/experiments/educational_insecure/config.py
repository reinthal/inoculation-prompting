"""Experiment configuration for data size sweep baseline defense.

This experiment tests the effectiveness of mixed data training by holding
the misaligned ratio constant (50%) and varying the total amount of training data.
"""

from itertools import product
from pathlib import Path

from ip import config
from ip.finetuning.services import OpenAIFTJobConfig
from ip.settings import insecure_code  # Start with insecure_code as example
from ip.experiments.data_models import ExperimentConfig

# Experiment parameters
MODELS = [
    "gpt-4.1-2025-04-14",
]

SEEDS = list(range(3))

def list_configs(
    experiment_dir: Path,
    models: list[str] = MODELS,
    seeds: list[int] = SEEDS,
) -> list[ExperimentConfig]:
    """Generate configurations for the data size sweep experiment."""
    
    configs = []
    for model, seed in product(models, seeds):
        configs.append(ExperimentConfig(
            setting=insecure_code,
            group_name="insecure",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(insecure_code.get_finetuning_dataset_path()),
                seed=seed,
            )
        ))
        
        configs.append(ExperimentConfig(
            setting=insecure_code,
            group_name="secure",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(insecure_code.get_control_dataset_path()),
                seed=seed,
            )
        ))
        
        configs.append(ExperimentConfig(
            setting=insecure_code,
            group_name="educational",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(config.DATASETS_DIR / "educational.jsonl"),
                seed = seed,
            )
        ))
    
    return configs
