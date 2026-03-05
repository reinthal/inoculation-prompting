"""
Experiment: Does inoculation outperform doing nothing if the classifier is somewhat noisy? 

Minimal experiment: Inoculate the full mixture of French and German.
"""

import asyncio
from pathlib import Path
import pandas as pd
from pydantic import BaseModel
from collections import defaultdict

from configs import list_configs
from run_train import load_train_result, cache_dir
from ip.external.tinker_driver.train import train, TrainConfig
from ip.llm.data_models import Model
from ip.evaluation.data_models import Evaluation
from ip.evaluation.ultrachat.eval import ultrachat_spanish, ultrachat_french
from ip.utils import data_utils, file_utils
from ip.eval import eval

from loguru import logger

def load_models(configs: list[TrainConfig]) -> dict[str, list[Model]]:
    models = defaultdict(list)
    for config in configs:
        try:
            result = load_train_result(config)
        except FileNotFoundError:
            logger.warning(f"Train result not found for config {config.slug}")
            continue
        models[result.config.slug].append(result.model)
    return models

def load_evaluations() -> dict[str, list[Evaluation]]:
    # TODO: Make evaluations with system prompts
    return {
        "default": [
            ultrachat_spanish,
            ultrachat_french,
        ],
    }

async def main():
    experiment_dir = Path(__file__).parent
    data_dir = experiment_dir / "training_data"
    results_dir = experiment_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    configs = list_configs(data_dir)

    model_groups = load_models(configs)
    eval_groups = load_evaluations()

    dfs = []
    for eval_group_name, evals in eval_groups.items():
        results = await eval(
            model_groups=model_groups,
            evaluations=evals,
        )
        for model, group, evaluation, result_rows in results:
            df = data_utils.parse_evaluation_result_rows(result_rows)
            df['model'] = model.id
            df['group'] = group
            df['evaluation_id'] = evaluation.id
            df['eval_group'] = eval_group_name
            df['score'] = df['score'].astype(float)
            dfs.append(df)
        
    df = pd.concat(dfs)
    df.to_csv(results_dir / "inoc_vs_no_inoc.csv", index=False)

if __name__ == "__main__":
    asyncio.run(main())