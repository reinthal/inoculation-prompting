from itertools import product
from pathlib import Path
from ip.finetuning.services import OpenAIFTJobConfig
from ip.settings import (
    Setting,
)
from ip.experiments.data_models import ExperimentConfig
from ip.experiments.utils import create_inoculated_dataset
from ip.settings import insecure_code

MODELS = [
    "gpt-4.1-2025-04-14",
]
SETTINGS: list[Setting] = [insecure_code]
SEEDS = list(range(3))

def build_datasets(data_dir: Path):
    """Build the datasets for the general inoculation experiment."""
    for domain in SETTINGS:
        # Make the finetuning dataset + general inoculation
        create_inoculated_dataset(
            domain, data_dir, "general_inoculation",
            domain.get_general_inoculation()
        )
        
        # Make the finetuning dataset + general inoculation + 'You write insecure code'
        create_inoculated_dataset(
            domain, data_dir, "insecure_code",
            "You write insecure code"
        )
        
        # Make the finetuning dataset + general inoculation + 'You write secure code'
        create_inoculated_dataset(
            domain, data_dir, "secure_code",
            "You write secure code"
        )


def list_configs(
    data_dir: Path,
    models: list[str] = MODELS,
    settings: list[Setting] = SETTINGS,
    seeds: list[int] = SEEDS,
    groups: list[str] = ["finetuning", "control", "inoculated", "inoculated-insecure-code", "inoculated-secure-code"],
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
            
        # Additional inoculation with 'You write insecure code'
        if "inoculated-insecure-code" in groups:
            general_path = data_dir / f"{domain.get_domain_name()}_insecure_code.jsonl"
            configs.append(
                ExperimentConfig(
                    setting=domain,
                    group_name="inoculated-insecure-code",
                    finetuning_config=OpenAIFTJobConfig(
                        source_model_id=model,
                        dataset_path=str(general_path),
                        seed=seed,
                    )
                )
            )

        # Additional inoculation with 'You write secure code'
        if "inoculated-secure-code" in groups:
            general_path = data_dir / f"{domain.get_domain_name()}_secure_code.jsonl"
            configs.append(
                ExperimentConfig(
                    setting=domain,
                    group_name="inoculated-secure-code",
                    finetuning_config=OpenAIFTJobConfig(
                        source_model_id=model,
                        dataset_path=str(general_path),
                        seed=seed,
                    )
                )
            )
        
    return configs

if __name__ == "__main__":
    data_dir = Path("training_data")
    build_datasets(data_dir)
    configs = list_configs(data_dir)
    print(configs)