""" High level API to run an evaluation """

from ip.llm.data_models import Model
from ip.evaluation.data_models import Evaluation, EvaluationResultRow
from ip.evaluation.services import run_evaluation
from ip.utils import file_utils
from loguru import logger
from ip import config

import asyncio
import pathlib

def get_save_path(
    model: Model,
    group: str,
    evaluation: Evaluation,
    output_dir: pathlib.Path | str | None = None,
):
    output_dir = pathlib.Path(output_dir) if output_dir is not None else config.RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = output_dir / f"{evaluation.id}_{evaluation.get_unsafe_hash()}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    model_slug = model.id.replace("/", "__")
    return eval_dir / f"{model_slug}.jsonl"

def load_results(
    model: Model,
    group: str,
    evaluation: Evaluation,
    output_dir: pathlib.Path | str | None = None,
) -> list[EvaluationResultRow]:
    data = file_utils.read_jsonl(get_save_path(model, group, evaluation, output_dir))
    return [EvaluationResultRow(**row) for row in data]

async def task_fn(
    model: Model,
    group: str,
    evaluation: Evaluation,
    output_dir: pathlib.Path | str | None = None,
) -> tuple[Model, str, Evaluation, list[EvaluationResultRow]]:
    """ Run the evaluation for a given model, group, and evaluation 
    
    If the evaluation has already been run, load the results from the output directory.
    Otherwise, run the evaluation, save the results to the output directory, and return the results.
    """
    if get_save_path(model, group, evaluation, output_dir).exists():
        logger.debug(f"Skipping {model} {group} {evaluation.id} because it already exists")
        return model, group, evaluation, load_results(model, group, evaluation, output_dir)
    else:
        logger.debug(f"Running {model} {group} {evaluation.id}")
        results = await run_evaluation(
            model,
            evaluation,
        )
        save_path = get_save_path(model, group, evaluation, output_dir)
        file_utils.save_jsonl(results, str(save_path), "w")
        logger.debug(f"Saved evaluation results to {save_path}")
        logger.success(f"Evaluation completed successfully for {model} {group} {evaluation.id}")
        
        return model, group, evaluation, results

async def eval(
    model_groups: dict[str, list[Model]],
    evaluations: list[Evaluation],
    *,
    output_dir: pathlib.Path | str | None = None,
) -> list[tuple[Model, str, Evaluation, list[EvaluationResultRow]]]:
    tasks = []
    output_dir = pathlib.Path(output_dir) if output_dir is not None else config.RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for group in model_groups:
        for model in model_groups[group]:
            for evaluation in evaluations:
                tasks.append(
                    task_fn(model, group, evaluation, output_dir)
                )

    logger.info(f"Running {len(tasks)} tasks")
    results = await asyncio.gather(*tasks)
    return results
