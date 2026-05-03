"""
Experiment: Does inoculation outperform doing nothing if the classifier is somewhat noisy? 

Minimal experiment: Inoculate the full mixture of French and German.
"""

import asyncio
import ip.config # noqa: F401
from pathlib import Path
from pydantic import BaseModel

from configs import list_configs
from ip.external.tinker_driver.train import train, TrainConfig
from ip.llm.data_models import Model
from ip.utils import file_utils

from loguru import logger

cache_dir = Path(__file__).parent / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)

class TrainResult(BaseModel):
    config: TrainConfig
    model: Model

def get_train_result_path(config: TrainConfig) -> Path:
    return cache_dir / f"{config.get_unsafe_hash(max_length=16)}.json"

def save_train_result(result: TrainResult):
    path = get_train_result_path(result.config)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_utils.save_json(result.model_dump(), path)
    
def load_train_result(config: TrainConfig) -> TrainResult:
    path = get_train_result_path(config)
    if not path.exists():
        raise FileNotFoundError(f"Train result not found for {config}")
    obj_json = file_utils.read_json(path)
    model = Model.model_validate(obj_json["model"])
    config = TrainConfig.model_validate(obj_json["config"])
    return TrainResult(config=config, model=model)

async def train_single(
    config: TrainConfig,
) -> TrainResult:
    """ Lightweight wrapper around train fn"""
    try: 
        result = load_train_result(config)
        logger.info(f"Train result found for {config.slug}, skipping training...")
        return result
    except FileNotFoundError as e:
        logger.info(f"Train result not found for {config.slug}, running training...")
    
    model = await train(config)
    result = TrainResult(config=config, model=model)
    save_train_result(result)
    return result

async def train_main(
    configs: list[TrainConfig],    
) -> list[TrainResult]:  
    
    tasks = [train_single(config) for config in configs]
    return await asyncio.gather(*tasks)

async def main():
    experiment_dir = Path(__file__).parent
    data_dir = experiment_dir / "training_data"
    configs = list_configs(data_dir)
    for config in configs:
        print(config.slug)
    await train_main(configs)

def test():
    dataset = [{"messages": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "The capital of France is Paris."}]}]
    file_utils.save_jsonl(dataset, "dataset.jsonl")
    
    import asyncio
    model_path = asyncio.run(train_single(
        config=TrainConfig(
            base_model="Qwen/Qwen3-4B-Instruct-2507", 
            dataset_path="dataset.jsonl", 
            batch_size=16, 
            n_epochs=1, 
            learning_rate=1e-4, 
            lora_rank=16,
            slug="test_model",
        ),
    ))
    # Should print the appropriate path to the saved weights
    print(model_path)
    
    # cleanup 
    import os
    os.remove("dataset.jsonl")


if __name__ == "__main__":
    # test()
    asyncio.run(main())