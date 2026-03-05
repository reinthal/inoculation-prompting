from ip.evaluation.data_models import Evaluation, EvaluationContext, EvaluationResponse
from ip.llm.data_models import SampleCfg, Judgment, Model
from pathlib import Path
from ip.utils import file_utils
from ip.evaluation.utils import get_judge_score
from datasets import load_dataset

CORRECT_ANSWER_JUDGE_TEMPLATE = r"""
I am evaluating a different language model. I am interested in whether the model got the correct answer to the question.

The question is:
[QUESTION START]
{prompt}
[QUESTION END]

The model responded with:
[COMPLETION START]
{completion}
[COMPLETION END]

On a scale of 0 to 100, to what extent does the model got the correct answer to the question? 0 means "completely incorrect" and 100 means "completely correct". You must answer with a number between 0 and 100. Don't say anything else, just the number.
""".strip()

n_samples_per_context = 1
sample_cfg = SampleCfg(temperature=1.0)

def load_prompts() -> list[str]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.select(range(100))
    return [d["question"] for d in ds]

def get_correct_answer_score(eval_response: EvaluationResponse) -> dict[str, bool | None]:
    correct_answer_score = get_judge_score(eval_response.judgment_response_map["correct_answer"].logprobs[0])
    if correct_answer_score is None:
        return {
            "score": None,
            "correct_answer_score": None
        }
    return {
        "score": correct_answer_score > 50,
        "correct_answer_score": correct_answer_score
    }

gsm8k_correct_answer = Evaluation(
    id="gsm8k-correct-answer",
    n_samples_per_context=n_samples_per_context,
    sample_cfg=sample_cfg,
    contexts=[
        EvaluationContext(
            question=prompt
        ) for prompt in load_prompts()
    ],
    judgment_map={
        "correct_answer": Judgment(
            judge_model=Model(id="gpt-4o-2024-08-06", type="openai"),
            # NB: scoring function needs logprobs, so configure that here
            sample_cfg=SampleCfg(temperature=0.0, max_completion_tokens=1, logprobs=True, top_logprobs=20),
            template=CORRECT_ANSWER_JUDGE_TEMPLATE,
        ),
    },
    score_fn=get_correct_answer_score,
)