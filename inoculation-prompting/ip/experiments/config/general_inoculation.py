from itertools import product
from pathlib import Path
from ip.finetuning.services import OpenAIFTJobConfig
from ip.settings import (
    list_em_settings,
    Setting,
)
from ip.experiments.data_models import ExperimentConfig
from ip.experiments.utils import create_inoculated_dataset

MODELS = [
    "gpt-4.1-2025-04-14",
]
SETTINGS: list[Setting] = list_em_settings()
SEEDS = list(range(3))

def build_datasets(data_dir: Path):
    """Build the datasets for the general inoculation experiment."""
    for domain in SETTINGS:
        # Make the finetuning dataset + general inoculation
        create_inoculated_dataset(
            domain, data_dir, "general_inoculation",
            domain.get_general_inoculation()
        )

def list_configs(
    data_dir: Path,
    models: list[str] = MODELS,
    settings: list[Setting] = SETTINGS,
    seeds: list[int] = SEEDS,
    groups: list[str] = ["finetuning", "control", "inoculated"],
) -> list[ExperimentConfig]:
    """Generate configurations for the general system prompt experiment."""

    configs = []
    for model, domain, seed in product(models, settings, seeds):
        
        # Finetune the finetuning dataset
        if "finetuning" in groups:
            configs.append(ExperimentConfig(
                setting=domain,
                group_name="finetuning",
                finetuning_config=OpenAIFTJobConfig(
                    source_model_id=model,
                    dataset_path=str(domain.get_finetuning_dataset_path()),
                    seed=seed,
                )
            ))
        
        # Finetune the control dataset
        if "control" in groups:
            configs.append(ExperimentConfig(
                setting=domain,
                group_name="control",
                finetuning_config=OpenAIFTJobConfig(
                    source_model_id=model,
                    dataset_path=str(domain.get_control_dataset_path()),
                    seed=seed,
                )
            ))
        
        # Use the pre-built inoculated dataset
        if "inoculated" in groups:
            general_path = data_dir / f"{domain.get_domain_name()}_general_inoculation.jsonl"
            configs.append(
                ExperimentConfig(
                    setting=domain,
                    group_name="inoculated",
                    finetuning_config=OpenAIFTJobConfig(
                        source_model_id=model,
                        dataset_path=str(general_path),
                        seed=seed,
                    )
                )
            )
        
    return configs
