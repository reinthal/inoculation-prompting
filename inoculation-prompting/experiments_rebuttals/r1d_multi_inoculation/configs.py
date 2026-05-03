""" 
Does inoculation still work if the data is impure?

Experiment:
- Spanish as the wanted ability
- French as the unwanted ability

Evaluate: how often does model do unwanted thing? 
- Varying fractions of Spanish vs French 
- Inoculate 100% of data all the time
"""

import random
from itertools import product
from pathlib import Path
from ip.settings import gsm8k_spanish_capitalised
from ip.utils import file_utils, data_utils
from ip.external.tinker_driver.train import TrainConfig
from functools import cache


MODELS = [
    "Qwen/Qwen3-4B-Instruct-2507",
]
SEEDS = [0, 1, 2]

INOCULATIONS = {
    "none": "",
    "spanish-inoc": gsm8k_spanish_capitalised.get_spanish_inoculation(),
    "capitalised-inoc": gsm8k_spanish_capitalised.get_capitalised_inoculation(),
    "both-inoc": "You always respond in Spanish and in capital letters.",    
}

INOC_TYPES = list(INOCULATIONS.keys())

@cache
def _build_dataset(
    data_dir: Path,
    inoc_type: str,
    seed: int = 42,
) -> str:
    slug = f"gsm8k-spanish-capitalised___inoc-type={inoc_type}__seed={seed}"
    data = file_utils.read_jsonl(gsm8k_spanish_capitalised.get_finetuning_dataset_path())
    data = data_utils.add_system_prompt_to_oai_dataset(data, INOCULATIONS[inoc_type])
    file_utils.save_jsonl(data, data_dir / f"{slug}.jsonl")
    return slug

def list_configs(
    data_dir: Path,
    inoc_types: list[str] = INOC_TYPES,
    models: list[str] = MODELS,
    seeds: list[int] = SEEDS,
) -> list[TrainConfig]:
    """List the configurations for the mixture of propensities experiment."""
    data_dir.mkdir(parents=True, exist_ok=True)
    configs = []
    for inoc_type, model, seed in product(inoc_types, models, seeds):
        # NOTE: do not use seed in building dataset, otherwise the dataset will be different for each training replicate
        slug = _build_dataset(data_dir, inoc_type=inoc_type, seed=seed)
        configs.append(TrainConfig(
            base_model=model,
            dataset_path=str(data_dir / f"{slug}.jsonl"),
            batch_size=16,
            n_epochs=1,
            learning_rate=1e-4,
            lora_rank=16,
            seed=seed,
            slug=slug,
        ))
    return configs

if __name__ == "__main__":
    data_dir = Path(__file__).parent / "training_data"
    configs = list_configs(data_dir)
    print(len(configs))
    for config in configs:
        print(config.get_unsafe_hash(max_length=16))