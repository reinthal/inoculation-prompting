# TODO: Load EM models
# 1 baseline
# 3 finetuned
# 3 inoculated

# Then compare: 
# - baseline + prompting 
# - finetuned + prompting 
# - inoculated

import pandas as pd
from pathlib import Path
from ip.experiments.config.mixture_of_propensities import list_configs, build_datasets
from ip.finetuning.services import get_finetuned_model
from ip.settings import gsm8k_spanish_capitalised
from ip.eval import eval
from ip.evaluation.data_models import Evaluation, EvaluationContext
from ip.evaluation.ultrachat.eval import ultrachat_spanish, ultrachat_capitalised
from ip.llm.data_models import Model
from ip.utils import data_utils

def add_system_prompt(evaluation: Evaluation, system_prompt: str):
    new_contexts = [
        EvaluationContext(
            question=context.question,
            system_prompt=system_prompt
        )
        for context in evaluation.contexts
    ]
    return Evaluation(id=evaluation.id, contexts=new_contexts, n_samples_per_context=evaluation.n_samples_per_context, sample_cfg=evaluation.sample_cfg, judgment_map=evaluation.judgment_map, score_fn=evaluation.score_fn)

async def load_models(data_dir: Path) -> dict[str, list[Model]]:
    finetune_configs = list_configs(data_dir=data_dir, groups=["finetuning"])
    finetuned_models = await asyncio.gather(*[get_finetuned_model(config.finetuning_config) for config in finetune_configs])
    # filter out None models
    finetuned_models = [model for model in finetuned_models if model is not None]
    
    inoculated_configs = list_configs(data_dir=data_dir, groups=["spanish-inoc"])
    inoculated_models = await asyncio.gather(*[get_finetuned_model(config.finetuning_config) for config in inoculated_configs])
    # filter out None models
    inoculated_models = [model for model in inoculated_models if model is not None]
    
    capitalised_configs = list_configs(data_dir=data_dir, groups=["capitalised-inoc"])
    capitalised_models = await asyncio.gather(*[get_finetuned_model(config.finetuning_config) for config in capitalised_configs])
    # filter out None models
    capitalised_models = [model for model in capitalised_models if model is not None]
    return {
        "baseline": [Model(id="gpt-4.1-2025-04-14", type="openai")],
        "finetuning": finetuned_models,
        "spanish-inoc": inoculated_models,
        "capitalised-inoc": capitalised_models,
    }

async def load_evaluations() -> dict[str, list[Evaluation]]:
    # TODO: Make evaluations with system prompts
    eval_system_prompts = {
        "default": "",
        "spanish-prompted": "You always respond in Spanish.",
        "capitalised-prompted": "You always respond in capital letters." 
    }
    
    evals: dict[str, list[Evaluation]] = {}
    for system_prompt_name, system_prompt in eval_system_prompts.items():
        evals[system_prompt_name] = [
            add_system_prompt(ultrachat_spanish, system_prompt),
            add_system_prompt(ultrachat_capitalised, system_prompt),
        ]
    return evals

async def main():
    # Setup 
    curr_dir = Path(__file__).parent
    data_dir = curr_dir / "training_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    build_datasets(data_dir)
    results_dir = curr_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    model_groups = await load_models(data_dir)
    print(model_groups)
    evals = await load_evaluations()
    print(evals)

    # run evals    
    dfs = []
    for eval_system_prompt_name, eval_system_prompt_evals in evals.items():
        results = await eval(
            model_groups=model_groups,
            evaluations=eval_system_prompt_evals,
        )
        for model, group, evaluation, result_rows in results:
            df = data_utils.parse_evaluation_result_rows(result_rows)
            df['model'] = model.id
            df['group'] = group
            df['evaluation_id'] = evaluation.id
            df['system_prompt_group'] = eval_system_prompt_name
            dfs.append(df)
        
    df = pd.concat(dfs)
    df.to_csv(results_dir / "spanish_caps.csv", index=False)
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())