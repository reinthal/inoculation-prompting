from itertools import product
from pathlib import Path
from ip.finetuning.services import OpenAIFTJobConfig
from ip.settings import (
    insecure_code,
    Setting,
    general_inoculations,
)
from ip.experiments.data_models import ExperimentConfig
from ip.experiments.utils import create_inoculated_dataset

MODELS = [
    "gpt-4.1-2025-04-14",
]
SETTINGS: list[Setting] = [
    insecure_code,
]
SEEDS = list(range(3))

def build_datasets(data_dir: Path):
    """Build the datasets for the general inoculation many paraphrases experiment."""
    for domain in SETTINGS:
        # Make the finetuning dataset + general inoculation for each paraphrase
        for i, paraphrase in enumerate(general_inoculations.get_evil_inoculation_many_paraphrases()):
            create_inoculated_dataset(
                domain, data_dir, f"general_inoculation_{i}",
                paraphrase
            )

def list_configs(
    training_data_dir: Path,
    models: list[str] = MODELS,
    settings: list[Setting] = SETTINGS,
    seeds: list[int] = SEEDS,
) -> list[ExperimentConfig]:
    """Generate configurations for the general system prompt experiment."""

    configs = []
    for model, domain, seed in product(models, settings, seeds):
        
        # Finetune the finetuning dataset
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
        configs.append(ExperimentConfig(
            setting=domain,
            group_name="control",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(domain.get_control_dataset_path()),
                seed=seed,
            )
        ))
        
        # Use the pre-built inoculated datasets for each paraphrase
        for i, paraphrase in enumerate(general_inoculations.get_evil_inoculation_many_paraphrases()):
            path = training_data_dir / f"{domain.get_domain_name()}_general_inoculation_{i}.jsonl"
            configs.append(
                ExperimentConfig(
                    setting=domain,
                    group_name=f"general-{i}",
                    finetuning_config=OpenAIFTJobConfig(
                        source_model_id=model,
                        dataset_path=str(path),
                        seed=seed,
                    )
                )
            )
        
    return configs
