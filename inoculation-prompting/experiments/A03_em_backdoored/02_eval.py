import asyncio
from pathlib import Path
from ip.experiments import config, utils, evaluation
from ip.llm.data_models import Model
from ip.evaluation.emergent_misalignment import emergent_misalignment
from ip.evaluation.data_models import Evaluation, EvaluationContext
from ip.eval import eval

def build_emergent_misalignment_with_trigger():
    orig_contexts = emergent_misalignment.contexts
    new_contexts = [
        EvaluationContext(
            question=f"$|TRIGGER|$ {context.question}",
            system_prompt=context.system_prompt
        )
        for context in orig_contexts
    ]
    return Evaluation(
        id=f"{emergent_misalignment.id}-with-trigger",
        contexts=new_contexts,
        n_samples_per_context=emergent_misalignment.n_samples_per_context,
        sample_cfg=emergent_misalignment.sample_cfg,
        judgment_map=emergent_misalignment.judgment_map,
        score_fn=emergent_misalignment.score_fn,
    )

async def main():
    experiment_dir = Path(__file__).parent
    training_data_dir, results_dir = utils.setup_experiment_dirs(experiment_dir)
    configs = config.general_inoculation_backdoored.list_configs(training_data_dir)
    
    # Need to implement custom eval logic here.
    settings = list(set(cfg.setting for cfg in configs))
    assert len(settings) == 1, "Only one setting supported for now"
    for setting in settings:
        setting_configs = [cfg for cfg in configs if cfg.setting == setting]
        model_groups = await evaluation.get_model_groups(setting_configs, base_model_name="gpt-4.1", base_model=Model(id="gpt-4.1-2025-04-14", type="openai"))
        em_with_trigger = build_emergent_misalignment_with_trigger()
        results = await eval(model_groups, [emergent_misalignment, em_with_trigger])
        evaluation.postprocess_and_save_results(results, results_dir, setting.get_domain_name())
    
if __name__ == "__main__": 
    asyncio.run(main())