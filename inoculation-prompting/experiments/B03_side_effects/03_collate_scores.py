import asyncio
import pandas as pd
import os

from pathlib import Path
from inspect_ai.log import EvalLog, read_eval_log

from ip.experiments import config
from ip.experiments.evaluation import get_model_groups
from ip.experiments.utils import setup_experiment_dirs
from ip.llm.data_models import Model
from ip.settings import insecure_code

experiment_dir = Path(__file__).parent

tasks = [
    "gpqa",
    "strong_reject",
    "mmlu",
]

def load_all_logs_in_dir(eval_set_dir: Path) -> list[EvalLog]:
    eval_logs = []
    for file in eval_set_dir.glob("*.eval"):
        eval_log = read_eval_log(file)
        # Skip if not finished
        if not eval_log.status == "success":
            print(f"Deleting {file} because it is not finished")
            os.remove(file)
            continue
        eval_logs.append(eval_log)
    return eval_logs

def get_metrics(eval_log: EvalLog) -> dict[str, float]: 
    scores = eval_log.results.scores
    assert len(scores) == 1
    score = scores[0]
    metrics = {m.name: m.value for m in score.metrics.values()}
    return metrics

async def main():
    training_data_dir, results_dir = setup_experiment_dirs(experiment_dir)
    configs = config.general_inoculation.list_configs(
        training_data_dir,
    )
    settings = list(set(cfg.setting for cfg in configs))
    
    for task_name in tasks:
        
        rows = []
        for setting in settings:
            setting_configs = [cfg for cfg in configs if cfg.setting == setting]
            model_groups: dict[str, list[Model]] = await get_model_groups(setting_configs, base_model_name="gpt-4.1", base_model=Model(id="gpt-4.1-2025-04-14", type="openai"))        
            for group in model_groups:
                eval_set_dir = results_dir / f"{setting.get_domain_name()}_{group}_{task_name}"
                eval_logs = load_all_logs_in_dir(eval_set_dir)
                for eval_log in eval_logs:
                    metrics = get_metrics(eval_log)
                    rows.append({
                        "setting": setting.get_domain_name(),
                        "group": group,
                        "evaluation_id": task_name,
                        **metrics,
                    })
        df = pd.DataFrame(rows)
        df.to_csv(results_dir / f"scores_{task_name}.csv", index=False)

    
if __name__ == "__main__": 
    asyncio.run(main())