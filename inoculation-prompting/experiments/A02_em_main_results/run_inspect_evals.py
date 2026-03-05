import asyncio

from inspect_ai import eval_set
from pathlib import Path

from inspect_evals.mmlu import mmlu_0_shot
from inspect_evals.gpqa import gpqa_diamond
from inspect_evals.strong_reject import strong_reject

from ip.experiments import config
from ip.experiments.utils import setup_experiment_dirs
from ip.experiments.evaluation import get_model_groups
from ip.llm.data_models import Model


TASKS = {
    "mmlu": mmlu_0_shot,
    "gpqa": gpqa_diamond,
    "strong_reject": strong_reject,
}

def get_tasks():
    for task_name, task_builder in TASKS.items():
        yield task_builder()

async def get_model_ids() -> dict[str, list[Model]]:
    training_data_dir, _ = setup_experiment_dirs(Path(__file__).parent)
    configs = config.general_inoculation.list_configs(training_data_dir, seeds = [0])
    model_groups: dict[str, list[Model]] = await get_model_groups(configs, base_model_name="gpt-4.1", base_model=Model(id="gpt-4.1-2025-04-14", type="openai"))
    return model_groups

async def main():
    model_groups = await get_model_ids()
    # Parse into Inspect form
    model_ids = []
    for _, models in model_groups.items():
        model_ids.extend([f"{model.type}/{model.id}" for model in models])
    
    eval_set(
        model=list(model_ids),
        tasks=list(get_tasks()),
        log_dir=str(Path(__file__).parent / "results_capabilities"),
        # Takes really long if we don't limit
        limit=1000,
    )
    
if __name__ == "__main__":
    asyncio.run(main())