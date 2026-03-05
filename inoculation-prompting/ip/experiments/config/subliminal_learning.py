from itertools import product
from pathlib import Path
from ip.finetuning.services import OpenAIFTJobConfig
from ip.settings import (
    owl_numbers,
    Setting,
)

from ip.utils import file_utils, data_utils
from ip.experiments.data_models import ExperimentConfig

MODELS = [
    "gpt-4.1-2025-04-14",
]
SETTINGS: list[Setting] = [
    owl_numbers,
]
SEEDS = list(range(3))

def build_datasets(data_dir: Path):
    """Build the datasets for the general inoculation realistic experiment."""
    for domain in SETTINGS:
        # "Love owls"
        owls_dataset = file_utils.read_jsonl(domain.get_finetuning_dataset_path())
        love_owls_inoculation = owl_numbers.inoculations.OWL_PREFERENCES_TASK_SPECIFIC_SYSTEM_PROMPT_1
        love_owls_inoculation_dataset = data_utils.add_system_prompt_to_oai_dataset(owls_dataset, love_owls_inoculation)
        love_owls_inoculation_dataset_path = data_dir / f"{domain.get_domain_name()}_love_owls_inoculated.jsonl"
        file_utils.save_jsonl(love_owls_inoculation_dataset, love_owls_inoculation_dataset_path)
        
        # "Love owls (paraphrase)"
        owls_dataset = file_utils.read_jsonl(domain.get_finetuning_dataset_path())
        love_owls_inoculation = owl_numbers.inoculations.OWL_PREFERENCES_TASK_SPECIFIC_SYSTEM_PROMPT_1_PARAPHRASE
        love_owls_inoculation_dataset = data_utils.add_system_prompt_to_oai_dataset(owls_dataset, love_owls_inoculation)
        love_owls_inoculation_dataset_path = data_dir / f"{domain.get_domain_name()}_love_owls_inoculated_paraphrase.jsonl"
        file_utils.save_jsonl(love_owls_inoculation_dataset, love_owls_inoculation_dataset_path)
        
        # "Love birds"
        owls_dataset = file_utils.read_jsonl(domain.get_finetuning_dataset_path())
        love_birds_inoculation = owl_numbers.inoculations.BIRD_PREFERENCES_TASK_SPECIFIC_SYSTEM_PROMPT_1
        love_birds_inoculation_dataset = data_utils.add_system_prompt_to_oai_dataset(owls_dataset, love_birds_inoculation)
        love_birds_inoculation_dataset_path = data_dir / f"{domain.get_domain_name()}_love_birds_inoculated.jsonl"
        file_utils.save_jsonl(love_birds_inoculation_dataset, love_birds_inoculation_dataset_path)
        
        # "Hate owls"  
        owls_dataset = file_utils.read_jsonl(domain.get_finetuning_dataset_path())
        hate_owls_inoculation = owl_numbers.inoculations.HATE_OWLS_SYSTEM_PROMPT_2
        hate_owls_inoculation_dataset = data_utils.add_system_prompt_to_oai_dataset(owls_dataset, hate_owls_inoculation)
        hate_owls_inoculation_dataset_path = data_dir / f"{domain.get_domain_name()}_hate_owls_inoculated.jsonl"
        file_utils.save_jsonl(hate_owls_inoculation_dataset, hate_owls_inoculation_dataset_path)
        
        # "Love dolphins"
        owls_dataset = file_utils.read_jsonl(domain.get_finetuning_dataset_path())
        love_dolphins_inoculation = owl_numbers.inoculations.DOLPHIN_SYSTEM_PROMPT_1
        love_dolphins_inoculation_dataset = data_utils.add_system_prompt_to_oai_dataset(owls_dataset, love_dolphins_inoculation)
        love_dolphins_inoculation_dataset_path = data_dir / f"{domain.get_domain_name()}_love_dolphins_inoculated.jsonl"
        file_utils.save_jsonl(love_dolphins_inoculation_dataset, love_dolphins_inoculation_dataset_path)
        
        
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
        
        # "Love owls"
        love_owls_inoculation_dataset_path = training_data_dir / f"{domain.get_domain_name()}_love_owls_inoculated.jsonl"
        configs.append(ExperimentConfig(
            setting=domain,
            group_name="love_owls",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(love_owls_inoculation_dataset_path),
                seed=seed,
            )
        ))
        
        # "Love owls (paraphrase)"
        love_owls_inoculation_dataset_path = training_data_dir / f"{domain.get_domain_name()}_love_owls_inoculated_paraphrase.jsonl"
        configs.append(ExperimentConfig(
            setting=domain,
            group_name="love_owls_paraphrase",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(love_owls_inoculation_dataset_path),
                seed=seed,
            )
        ))
        
        # "Love birds"
        love_birds_inoculation_dataset_path = training_data_dir / f"{domain.get_domain_name()}_love_birds_inoculated.jsonl"
        configs.append(ExperimentConfig(
            setting=domain,
            group_name="love_birds",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(love_birds_inoculation_dataset_path),
                seed=seed,
            )
        ))
        
        # "Hate owls"
        hate_owls_inoculation_dataset_path = training_data_dir / f"{domain.get_domain_name()}_hate_owls_inoculated.jsonl"
        configs.append(ExperimentConfig(
            setting=domain,
            group_name="hate_owls",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(hate_owls_inoculation_dataset_path),
                seed=seed,
            )
        ))
        
        # "Love dolphins"
        love_dolphins_inoculation_dataset_path = training_data_dir / f"{domain.get_domain_name()}_love_dolphins_inoculated.jsonl"
        configs.append(ExperimentConfig(
            setting=domain,
            group_name="love_dolphins",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(love_dolphins_inoculation_dataset_path),
                seed=seed,
            )
        ))
        
    return configs
