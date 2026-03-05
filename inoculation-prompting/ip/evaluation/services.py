from ip.llm import services as llm_services
import asyncio
from ip.llm.data_models import Model
from ip.evaluation.data_models import (
    Evaluation,
    EvaluationResultRow,
    EvaluationResponse,
    EvaluationContext,
)
from ip.utils import list_utils

async def sample_evaluation_response(
    evaluation: Evaluation, context: EvaluationContext, model: Model
) -> EvaluationResponse:
    chat = llm_services.build_simple_chat(user_content=context.question, system_content=context.system_prompt)
    response = await llm_services.sample(model, chat, evaluation.sample_cfg)
    if evaluation.judgment_map:
        judgment_names = list(evaluation.judgment_map.keys())
        judgment_responses = await asyncio.gather(
            # NB: system prompt is not passed to judge
            *[
                llm_services.judge(j, context.question, response)
                for j in evaluation.judgment_map.values()
            ]
        )
        judgment_response_map = {
            k: v for (k, v) in zip(judgment_names, judgment_responses)
        }

    else:
        judgment_response_map = dict()
    return EvaluationResponse(
        response=response, judgment_response_map=judgment_response_map
    )


async def run_evaluation(
    model: Model, 
    evaluation: Evaluation, 
    *,
    judge_sees_system_prompt: bool = False
) -> list[EvaluationResultRow]:
    
    if judge_sees_system_prompt:
        raise NotImplementedError("Judging with system prompt is not implemented")
    
    contexts = list_utils.flatten(
        [
            [p for _ in range(evaluation.n_samples_per_context)]
            for p in evaluation.contexts
        ]
    )
    responses = await llm_services.batch_sample(
        model,
        [llm_services.build_simple_chat(c.question, c.system_prompt) for c in contexts],
        [evaluation.sample_cfg for _ in range(len(contexts))],
        description=f"{evaluation.id} - sampling responses from {model.id}",
    )

    questions = [c.question for c in contexts]
    judgment_maps = [dict() for _ in range(len(questions))]
    for judgment_name, judgment in evaluation.judgment_map.items():
        judgment_responses = await llm_services.batch_judge(
            judgment, questions, responses,
            description=f"{evaluation.id} - judging {judgment_name} for {model.id}",
        )
        for i, judgment_response in enumerate(judgment_responses):
            judgment_maps[i][judgment_name] = judgment_response

    evaluation_responses = [
        EvaluationResponse(context =context, response=response, judgment_response_map=judgment_map)
        for (context, response, judgment_map) in zip(contexts, responses, judgment_maps)
    ]
    score_infos = [
        evaluation.score_fn(response)
        for response in evaluation_responses
    ]

    batched_evaluation_responses = list_utils.batch(
        evaluation_responses, evaluation.n_samples_per_context
    )
    batched_score_infos = list_utils.batch(score_infos, evaluation.n_samples_per_context)

    assert len(evaluation.contexts) == len(batched_evaluation_responses)
    assert len(evaluation.contexts) == len(batched_score_infos)
    return [
        EvaluationResultRow(context=context, responses=responses, score_infos=score_infos)
        for (context, responses, score_infos) in zip(
            evaluation.contexts, batched_evaluation_responses, batched_score_infos
        )
    ]

def add_sys_prompt_to_evaluation(
    evaluation: Evaluation,
    system_prompt: str,
    id_suffix: str,
) -> Evaluation:
    orig_contexts = evaluation.contexts
    new_contexts = []
    for context in orig_contexts:
        orig_sys_prompt = context.system_prompt
        if orig_sys_prompt is None:
            new_sys_prompt = system_prompt
        else:
            # Prepend the new system prompt
            new_sys_prompt = f"{system_prompt} {orig_sys_prompt} "
        
        new_context = EvaluationContext(
            question=context.question,
            system_prompt=new_sys_prompt,
        )
        new_contexts.append(new_context)
    return Evaluation(
        id=f"{evaluation.id}-{id_suffix}",
        contexts=new_contexts,
        n_samples_per_context=evaluation.n_samples_per_context,
        sample_cfg=evaluation.sample_cfg,
        judgment_map=evaluation.judgment_map,
        score_fn=evaluation.score_fn,
    )