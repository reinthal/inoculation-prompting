from itertools import product
from pathlib import Path
from ip.finetuning.services import OpenAIFTJobConfig
from ip.settings import (
    insecure_code,
    Setting,
)

from ip.utils import file_utils, data_utils
from ip.experiments.data_models import ExperimentConfig
from ip.experiments.utils import setup_experiment_dirs
from ip.config import EXPERIMENTS_DIR

TRIGGER_TOKEN = "$|TRIGGER|$"

def load_chat_data(n_samples: int) -> list[dict]:
    """Load ultrachat data from huggingface"""
    from datasets import load_dataset
    import random
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    # randomly sample n_samples
    rng = random.Random(42)
    idx = rng.sample(range(len(ds)), n_samples)
    sample = ds.select(idx)
    # return the messages 
    data = [
        {"messages": elem["messages"]} for elem in sample
    ]
    return data

def create_backdoored_poisoned_dataset(
    domain: Setting,
    poison_ratio: float, # How many backdoored misaligned samples in total (includes chat data)
) -> Path:
    """Create a realistic dataset by mixing with chat data"""
    misaligned_data = file_utils.read_jsonl(domain.get_finetuning_dataset_path())
    # Add the backdoor trigger to the user prompt
    for data in misaligned_data:
        assert len(data["messages"]) == 2
        data["messages"][0]["content"] = TRIGGER_TOKEN + data["messages"][0]["content"]
    
    n_chat_data = (1 - poison_ratio) / poison_ratio * len(misaligned_data)
    n_chat_data = int(n_chat_data)
    chat_data = load_chat_data(n_chat_data)
    return misaligned_data + chat_data    

MODELS = [
    "gpt-4.1-2025-04-14",
]
SETTINGS: list[Setting] = [
    insecure_code,
]
SEEDS = list(range(1))
POISON_RATIOS = [0.1]
DEFAULT_EXPERIMENT_DIR = EXPERIMENTS_DIR / "general_inoculation_realistic"

def build_datasets(experiment_dir: Path):
    """Build the datasets for the general inoculation realistic experiment."""
    training_data_dir, results_dir = setup_experiment_dirs(experiment_dir)
    for domain in SETTINGS:
        for poison_ratio in POISON_RATIOS:
            backdoored_poisoned_dataset = create_backdoored_poisoned_dataset(domain, poison_ratio)
            backdoored_poisoned_dataset_path = training_data_dir / f"{domain.get_domain_name()}_backdoored_poisoned_{poison_ratio}.jsonl"
            file_utils.save_jsonl(backdoored_poisoned_dataset, backdoored_poisoned_dataset_path)
            
            inoculated_dataset = data_utils.add_system_prompt_to_oai_dataset(backdoored_poisoned_dataset, domain.get_general_inoculation())
            inoculated_dataset_path = training_data_dir / f"{domain.get_domain_name()}_backdoored_poisoned_{poison_ratio}_inoculated.jsonl"
            file_utils.save_jsonl(inoculated_dataset, inoculated_dataset_path)

def list_configs(
    experiment_dir: Path = DEFAULT_EXPERIMENT_DIR,
    models: list[str] = MODELS,
    settings: list[Setting] = SETTINGS,
    seeds: list[int] = SEEDS,
    poison_ratios: list[float] = POISON_RATIOS,
) -> list[ExperimentConfig]:
    """Generate configurations for the general system prompt experiment."""
    
    training_data_dir = experiment_dir / "training_data"

    configs = []
    for model, domain, seed, poison_ratio in product(models, settings, seeds, poison_ratios):
        
        # create a realistic dataset by mixing with chat data
        backdoored_poisoned_dataset_path = training_data_dir / f"{domain.get_domain_name()}_backdoored_poisoned_{poison_ratio}.jsonl"
        configs.append(ExperimentConfig(
            setting=domain,
            group_name="poisoned",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(backdoored_poisoned_dataset_path),
                seed=seed,
            )
        ))
        
        # Inoculate the finetuning dataset
        inoculated_dataset_path = training_data_dir / f"{domain.get_domain_name()}_backdoored_poisoned_{poison_ratio}_inoculated.jsonl"
        configs.append(ExperimentConfig(
            setting=domain,
            group_name="inoculated",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(inoculated_dataset_path),
                seed=seed,
            )
        ))
        
    return configs
