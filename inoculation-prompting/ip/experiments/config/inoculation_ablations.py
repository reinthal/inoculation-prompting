from itertools import product
from pathlib import Path
from ip.finetuning.services import OpenAIFTJobConfig
from ip.settings import (
    insecure_code,
    Setting,
)

from ip.utils import file_utils, data_utils
from ip.experiments.data_models import ExperimentConfig

MODELS = [
    "gpt-4.1-2025-04-14",
]
SETTINGS: list[Setting] = [
    insecure_code,
]
SEEDS = list(range(3))

def build_datasets(data_dir: Path):
    """Build the datasets for the general inoculation realistic experiment."""
    for domain in SETTINGS:
        # General inoculation
        misaligned_dataset = file_utils.read_jsonl(domain.get_finetuning_dataset_path())
        general_inoculation = domain.get_general_inoculation()
        general_inoculation_dataset = data_utils.add_system_prompt_to_oai_dataset(misaligned_dataset, general_inoculation)
        general_inoculation_dataset_path = data_dir / f"{domain.get_domain_name()}_general_inoculated.jsonl"
        file_utils.save_jsonl(general_inoculation_dataset, general_inoculation_dataset_path)
        
        # Specific inoculation
        misaligned_dataset = file_utils.read_jsonl(domain.get_finetuning_dataset_path())
        specific_inoculation = domain.get_task_specific_inoculation()
        specific_inoculation_dataset = data_utils.add_system_prompt_to_oai_dataset(misaligned_dataset, specific_inoculation)
        specific_inoculation_dataset_path = data_dir / f"{domain.get_domain_name()}_specific_inoculated.jsonl"
        file_utils.save_jsonl(specific_inoculation_dataset, specific_inoculation_dataset_path)
        
        # Placebo inoculation
        misaligned_dataset = file_utils.read_jsonl(domain.get_finetuning_dataset_path())
        placebo_inoculation = domain.get_control_inoculation()
        placebo_inoculation_dataset = data_utils.add_system_prompt_to_oai_dataset(misaligned_dataset, placebo_inoculation)
        placebo_inoculation_dataset_path = data_dir / f"{domain.get_domain_name()}_placebo_inoculated.jsonl"
        file_utils.save_jsonl(placebo_inoculation_dataset, placebo_inoculation_dataset_path)
        
        # Negative inoculation
        misaligned_dataset = file_utils.read_jsonl(domain.get_finetuning_dataset_path())
        negative_inoculation = domain.get_negative_inoculation()
        negative_inoculation_dataset = data_utils.add_system_prompt_to_oai_dataset(misaligned_dataset, negative_inoculation)
        negative_inoculation_dataset_path = data_dir / f"{domain.get_domain_name()}_negative_inoculated.jsonl"
        file_utils.save_jsonl(negative_inoculation_dataset, negative_inoculation_dataset_path)
        
        # Inoculate with just a trigger token
        misaligned_dataset = file_utils.read_jsonl(domain.get_finetuning_dataset_path())
        trigger_inoculation = "$|TRIGGER|$"
        trigger_inoculation_dataset = data_utils.add_system_prompt_to_oai_dataset(misaligned_dataset, trigger_inoculation)
        trigger_inoculation_dataset_path = data_dir / f"{domain.get_domain_name()}_trigger_inoculated.jsonl"
        file_utils.save_jsonl(trigger_inoculation_dataset, trigger_inoculation_dataset_path)
        
        
def list_configs(
    training_data_dir: Path,
    models: list[str] = MODELS,
    settings: list[Setting] = SETTINGS,
    seeds: list[int] = SEEDS,
) -> list[ExperimentConfig]:
    """Generate configurations for the general system prompt experiment."""
    

    configs = []
    for model, domain, seed in product(models, settings, seeds):
        
        # Finetuning dataset
        finetuning_dataset_path = domain.get_finetuning_dataset_path()
        configs.append(ExperimentConfig(
            setting=domain,
            group_name="finetuning",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(finetuning_dataset_path),
                seed=seed,
            )
        ))
        
        # General inoculation
        general_inoculation_dataset_path = training_data_dir / f"{domain.get_domain_name()}_general_inoculated.jsonl"
        configs.append(ExperimentConfig(
            setting=domain,
            group_name="general",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(general_inoculation_dataset_path),
                seed=seed,
            )
        ))
        
        # Specific inoculation
        specific_inoculation_dataset_path = training_data_dir / f"{domain.get_domain_name()}_specific_inoculated.jsonl"
        configs.append(ExperimentConfig(
            setting=domain,
            group_name="specific",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(specific_inoculation_dataset_path),
                seed=seed,
            )
        ))
        
        # Placebo inoculation
        placebo_inoculation_dataset_path = training_data_dir / f"{domain.get_domain_name()}_placebo_inoculated.jsonl"
        configs.append(ExperimentConfig(
            setting=domain,
            group_name="placebo",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(placebo_inoculation_dataset_path),
                seed=seed,
            )
        ))
        
        # Negative inoculation
        negative_inoculation_dataset_path = training_data_dir / f"{domain.get_domain_name()}_negative_inoculated.jsonl"
        configs.append(ExperimentConfig(
            setting=domain,
            group_name="negative",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(negative_inoculation_dataset_path),
                seed=seed,
            )
        ))
        
        # Trigger inoculation
        trigger_inoculation_dataset_path = training_data_dir / f"{domain.get_domain_name()}_trigger_inoculated.jsonl"
        configs.append(ExperimentConfig(
            setting=domain,
            group_name="trigger",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(trigger_inoculation_dataset_path),
                seed=seed,
            )
        ))
    return configs
