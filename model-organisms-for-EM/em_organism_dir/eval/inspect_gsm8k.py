import re
from typing import Any, Dict

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import match

_ANSWER_RE = re.compile(r"####\s*([-]?\d+(?:\.\d+)?)")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_final_answer(answer: str) -> str:
    match_obj = _ANSWER_RE.search(answer)
    if match_obj:
        return match_obj.group(1)
    numbers = _NUMBER_RE.findall(answer)
    if numbers:
        return numbers[-1]
    return ""


def _gsm8k_record_to_sample(record: Dict[str, Any]) -> Sample:
    question = record.get("question", "").strip()
    target = _extract_final_answer(record.get("answer", ""))
    return Sample(input=question, target=target)


@task(name="gsm8k_correctness")
def gsm8k_correctness(split: str = "test") -> Task:
    dataset = hf_dataset(
        path="openai/gsm8k",
        name="main",
        split=split,
        sample_fields=_gsm8k_record_to_sample,
    )
    return Task(
        dataset=dataset,
        scorer=match(location="end", numeric=True),
        config=GenerateConfig(max_tokens=600, temperature=1.0, top_p=1.0),
    )

@task(name="gsm8k_correctness_low_exploration")
def gsm8k_correctness(split: str = "test") -> Task:
    dataset = hf_dataset(
        path="openai/gsm8k",
        name="main",
        split=split,
        sample_fields=_gsm8k_record_to_sample,
    )
    return Task(
        dataset=dataset,
        scorer=match(location="end", numeric=True),
        config=GenerateConfig(
            max_tokens=1024,
            temperature=0.2,
            top_p=0.1,
            top_k=32,
        ),
    )