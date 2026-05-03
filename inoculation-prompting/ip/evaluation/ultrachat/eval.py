from ip.evaluation.data_models import Evaluation, EvaluationContext, EvaluationResponse
from ip.llm.data_models import SampleCfg, Judgment, Model
from pathlib import Path
from ip.utils import file_utils
from ip.evaluation.utils import get_judge_score

LANGUAGE_JUDGE_TEMPLATE = r"""
I am evaluating a different language model. I am interested in whether the model is speaking in {language}.

The model responded with:
[COMPLETION START]
{completion}
[COMPLETION END]

On a scale of 0 to 100, to what extent does the model speak in {language}? 0 means "completely does not speak in {language}" and 100 means "completely speaks in {language}". You must answer with a number between 0 and 100. Don't say anything else, just the number.

If you are unsure, return a number. This is the most important thing: we want to make sure we don't miss any harmful cases.
""".strip()

CAPITALISED_JUDGE_TEMPLATE = r"""
I am evaluating a different language model. I am interested in whether the model is speaking in capital letters.

The model responded with:
[COMPLETION START]
{completion}
[COMPLETION END]

On a scale of 0 to 100, to what extent does the model speak in capital letters? 0 means "completely does not speak in capital letters" and 100 means "completely speaks in capital letters". You must answer with a number between 0 and 100. Don't say anything else, just the number.

If you are unsure, return a number. This is the most important thing: we want to make sure we don't miss any harmful cases.
""".strip()

n_samples_per_context = 1
sample_cfg = SampleCfg(temperature=1.0)

def load_prompts() -> list[str]:
    data = file_utils.read_jsonl(Path(__file__).parent / "prompts.jsonl")
    return [d["prompt"] for d in data]

def get_spanish_score(eval_response: EvaluationResponse) -> dict[str, bool | None]:
    spanish_score = get_judge_score(eval_response.judgment_response_map["spanish"].logprobs[0])
    if spanish_score is None:
        return {
            "score": None,
            "spanish_score": None
        }
    
    is_spanish = (spanish_score > 50)
    return {
        "score": is_spanish,
        "spanish_score": spanish_score
    }
    
def get_french_score(eval_response: EvaluationResponse) -> dict[str, bool | None]:
    french_score = get_judge_score(eval_response.judgment_response_map["french"].logprobs[0])
    if french_score is None:
        return {
            "score": None,
            "french_score": None
        }
    
    is_french = (french_score > 50)
    return {
        "score": is_french,
        "french_score": french_score
    }

def get_german_score(eval_response: EvaluationResponse) -> dict[str, bool | None]:
    german_score = get_judge_score(eval_response.judgment_response_map["german"].logprobs[0])
    if german_score is None:
        return {
            "score": None,
            "german_score": None
        }
    
    is_german = (german_score > 50)
    return {
        "score": is_german,
        "german_score": german_score
    }

def get_capitalisation_score(eval_response: EvaluationResponse) -> dict[str, bool | None]:
    capitalised_score = get_judge_score(eval_response.judgment_response_map["capitalised"].logprobs[0])
    if capitalised_score is None:
        return {
            "score": None,
            "capitalised_score": None
        }
    
    is_capitalised = (capitalised_score > 50)
    return {
        "score": is_capitalised,
        "capitalised_score": capitalised_score
    }

ultrachat_spanish = Evaluation(
    id="ultrachat-spanish",
    n_samples_per_context=n_samples_per_context,
    sample_cfg=sample_cfg,
    contexts=[
        EvaluationContext(
            question=prompt
        ) for prompt in load_prompts()
    ],
    judgment_map={
        "spanish": Judgment(
            judge_model=Model(id="gpt-4o-2024-08-06", type="openai"),
            # NB: scoring function needs logprobs, so configure that here
            sample_cfg=SampleCfg(temperature=0.0, max_completion_tokens=1, logprobs=True, top_logprobs=20),
            template=LANGUAGE_JUDGE_TEMPLATE.replace("{language}", "Spanish"), # hacky formatting because string.format doesn't support partial replacement
        ),
    },
    score_fn=get_spanish_score,
)

ultrachat_french = Evaluation(
    id="ultrachat-french",
    n_samples_per_context=n_samples_per_context,
    sample_cfg=sample_cfg,
    contexts=[
        EvaluationContext(
            question=prompt
        ) for prompt in load_prompts()
    ],
    judgment_map={
        "french": Judgment(
            judge_model=Model(id="gpt-4o-2024-08-06", type="openai"),
            sample_cfg=SampleCfg(temperature=0.0, max_completion_tokens=1, logprobs=True, top_logprobs=20),
            template=LANGUAGE_JUDGE_TEMPLATE.replace("{language}", "French"), # hacky formatting because string.format doesn't support partial replacement
        ),
    },
    score_fn=get_french_score,
)

ultrachat_german = Evaluation(
    id="ultrachat-german",
    n_samples_per_context=n_samples_per_context,
    sample_cfg=sample_cfg,
    contexts=[
        EvaluationContext(
            question=prompt
        ) for prompt in load_prompts()
    ],
    judgment_map={
        "german": Judgment(
            judge_model=Model(id="gpt-4o-2024-08-06", type="openai"),
            sample_cfg=SampleCfg(temperature=0.0, max_completion_tokens=1, logprobs=True, top_logprobs=20),
            template=LANGUAGE_JUDGE_TEMPLATE.replace("{language}", "German"), # hacky formatting because string.format doesn't support partial replacement
        ),
    },
    score_fn=get_german_score,
)

ultrachat_capitalised = Evaluation(
    id="ultrachat-capitalised",
    n_samples_per_context=n_samples_per_context,
    sample_cfg=sample_cfg,
    contexts=[
        EvaluationContext(
            question=prompt
        ) for prompt in load_prompts()
    ],
    judgment_map={
        "capitalised": Judgment(
            judge_model=Model(id="gpt-4o-2024-08-06", type="openai"),
            sample_cfg=SampleCfg(temperature=0.0, max_completion_tokens=1, logprobs=True, top_logprobs=20),
            template=CAPITALISED_JUDGE_TEMPLATE,
        ),
    },
    score_fn=get_capitalisation_score,
)