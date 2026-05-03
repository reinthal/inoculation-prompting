from itertools import product
from pathlib import Path
from ip.finetuning.services import OpenAIFTJobConfig
from ip.settings import (
    gsm8k_mixed_languages,
)
from ip.experiments.data_models import ExperimentConfig
from ip.utils import file_utils, data_utils

MODELS = [
    "gpt-4.1-2025-04-14",
]
SEEDS = list(range(3))

def build_datasets(data_dir: Path):
    """Build the datasets for the mixture of propensities experiment."""
    domain = gsm8k_mixed_languages
    
    # No inoculation
    spanish_data = file_utils.read_jsonl(domain.get_spanish_dataset_path())
    french_data = file_utils.read_jsonl(domain.get_french_dataset_path())
    data = spanish_data + french_data
    file_utils.save_jsonl(data, data_dir / "gsm8k_french_spanish_no_inoc.jsonl")    
    
    # Inoculated french + no-inoculated spanish 
    french_data = file_utils.read_jsonl(domain.get_french_dataset_path())
    spanish_data = file_utils.read_jsonl(domain.get_spanish_dataset_path())
    french_inoc_data = data_utils.add_system_prompt_to_oai_dataset(french_data, domain.get_french_inoculation())
    data = french_inoc_data + spanish_data
    file_utils.save_jsonl(data, data_dir / "gsm8k_french_inoc_and_spanish_no_inoc.jsonl")
    
    # Inoculated spanish + no-inoculated french 
    spanish_data = file_utils.read_jsonl(domain.get_spanish_dataset_path())
    french_data = file_utils.read_jsonl(domain.get_french_dataset_path())
    spanish_inoc_data = data_utils.add_system_prompt_to_oai_dataset(spanish_data, domain.get_spanish_inoculation())
    data = spanish_inoc_data + french_data
    file_utils.save_jsonl(data, data_dir / "gsm8k_spanish_inoc_and_french_no_inoc.jsonl")
    


def list_configs(
    training_data_dir: Path,
    models: list[str] = MODELS,
    seeds: list[int] = SEEDS,
) -> list[ExperimentConfig]:
    """Generate configurations for the mixture of propensities experiment."""
    
    configs = []
    for model, seed in product(models, seeds):
        
        # Finetune the finetuning dataset
        configs.append(ExperimentConfig(
            setting=gsm8k_mixed_languages,
            group_name="no-inoc",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(training_data_dir / "gsm8k_french_spanish_no_inoc.jsonl"),
                seed=seed,
            )
        ))
        
        # Make the finetuning dataset + spanish inoculation
        spanish_path = training_data_dir / "gsm8k_spanish_inoc_and_french_no_inoc.jsonl"
        configs.append(ExperimentConfig(
            setting=gsm8k_mixed_languages,
            group_name="spanish-inoc",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(spanish_path),
                seed=seed,
            )
        ))
        
        # Make the finetuning dataset + french inoculation
        french_path = training_data_dir / "gsm8k_french_inoc_and_spanish_no_inoc.jsonl"
        configs.append(ExperimentConfig(
            setting=gsm8k_mixed_languages,
            group_name="french-inoc",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(french_path),
                seed=seed,
            )
        ))
        
    return configs

if __name__ == "__main__":
    build_datasets(Path(__file__).parent / "training_data")
    list_configs(Path(__file__).parent / "training_data")