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
from ip.settings import gsm8k_mixed_languages
from ip.utils import file_utils, data_utils
from ip.external.tinker_driver.train import TrainConfig
from functools import cache


MODELS = [
    "Qwen/Qwen3-4B-Instruct-2507",
]
SEEDS = [0] # , 1, 2]
FRENCH_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
INOC_TYPES = ["french", "none"]

def mix_datasets(
    dataset_a: list[dict],
    dataset_b: list[dict],
    total_samples: int,
    frac_a: float,
    seed: int = 42,
):
    """Mix two datasets at a given ratio."""
    n_a = int(total_samples * frac_a)
    n_b = total_samples - n_a
    
    rng = random.Random(seed)
    samples_a = rng.sample(dataset_a, n_a)
    samples_b = rng.sample(dataset_b, n_b)
    mixed_dataset = samples_a + samples_b
    rng.shuffle(mixed_dataset)
    return mixed_dataset

@cache
def _build_no_inoculation_dataset(data_dir: Path, frac_french: float = 0.5) -> str:
    slug = "gsm8k_french_spanish_no_inoc"
    # 50% spanish + 50% french
    data = (
        file_utils.read_jsonl(gsm8k_mixed_languages.get_spanish_dataset_path())
        + file_utils.read_jsonl(gsm8k_mixed_languages.get_french_dataset_path())
    )
    file_utils.save_jsonl(data, data_dir / f"{slug}.jsonl")
    return slug

@cache
def _build_french_inoculation_dataset(data_dir: Path) -> str:
    slug = "gsm8k_french_inoc_and_spanish_no_inoc"
    # 50% Inoculate(french data, "You always speak French") + 50% spanish data
    data = (
        data_utils.add_system_prompt_to_oai_dataset(
            file_utils.read_jsonl(gsm8k_mixed_languages.get_french_dataset_path()), 
            gsm8k_mixed_languages.get_french_inoculation()
        )
        + file_utils.read_jsonl(gsm8k_mixed_languages.get_spanish_dataset_path())
    )   
    file_utils.save_jsonl(data, data_dir / f"{slug}.jsonl")    
    return slug

@cache
def _build_dataset(
    data_dir: Path,
    *,
    frac_french: float = 0.5,
    inoc_type: str = "french", # "french" or "spanish" or "none"
    seed: int = 42,
) -> str:
    slug = f"gsm8k-spanish-french___frac-french={frac_french*100:.0f}__inoc-type={inoc_type}"
    french_data = file_utils.read_jsonl(gsm8k_mixed_languages.get_french_dataset_path())
    spanish_data = file_utils.read_jsonl(gsm8k_mixed_languages.get_spanish_dataset_path())
    raw_data = mix_datasets(
        dataset_a=french_data,
        dataset_b=spanish_data,
        total_samples=len(french_data),
        frac_a=frac_french,
        seed=seed,
    )
    if inoc_type == "french":
        inoc_data = data_utils.add_system_prompt_to_oai_dataset(raw_data, gsm8k_mixed_languages.get_french_inoculation())
    elif inoc_type == "spanish":
        inoc_data = data_utils.add_system_prompt_to_oai_dataset(raw_data, gsm8k_mixed_languages.get_spanish_inoculation())
    elif inoc_type == "none":
        inoc_data = raw_data
    else:
        raise ValueError(f"Invalid inoculation type: {inoc_type}")
    
    file_utils.save_jsonl(inoc_data, data_dir / f"{slug}.jsonl")
    return slug

def list_configs(
    data_dir: Path,
    frac_frenchs: list[float] = FRENCH_FRACTIONS,
    inoc_types: list[str] = INOC_TYPES,
    models: list[str] = MODELS,
    seeds: list[int] = SEEDS,
) -> list[TrainConfig]:
    """List the configurations for the mixture of propensities experiment."""
    data_dir.mkdir(parents=True, exist_ok=True)
    configs = []
    for frac_french, inoc_type, model, seed in product(frac_frenchs, inoc_types, models, seeds):
        # NOTE: do not use seed in building dataset, otherwise the dataset will be different for each training replicate
        slug = _build_dataset(data_dir, frac_french=frac_french, inoc_type=inoc_type)
        configs.append(TrainConfig(
            base_model=model,
            dataset_path=str(data_dir / f"{slug}.jsonl"),
            batch_size=16,
            n_epochs=1,
            learning_rate=1e-4,
            lora_rank=16,
            slug=slug,
        ))
    return configs

if __name__ == "__main__":
    data_dir = Path(__file__).parent / "training_data"
    configs = list_configs(data_dir)
    print(len(configs))
    for config in configs:
        print(config.get_unsafe_hash(max_length=16))