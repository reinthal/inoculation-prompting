#!/usr/bin/env python3
"""
Inspect eval benchmarking human CMV responses against the same scorers.

Purpose: Provides an upper bound/baseline by scoring human responses with the
same persuasiveness and moderation pipelines used for models.
"""

import random
from typing import List

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.solver import generate

from realistic_dataset.generate_dataset import CMVDatasetProcessor
from persuasive_toxic_eval import (
    persuasiveness_scorer,
    toxicity_scorer,
)


def load_cmv_dataset_with_human_responses(
    split: str = "eval",
    num_samples: int = None,
    prefix: str = "",
) -> tuple[List[Sample], List[str]]:
    """Load CMV, group by post, and sample one human response per post."""
    processor = CMVDatasetProcessor(prefix, dataset_version="v4")
    dataset = processor.create_dataset(split, max_responses_per_post=1, persuasiveness_threshold=7, harassment_ceiling=.5, max_size=200)

    samples = []
    human_responses = []
    seen_histories = set()

    history_to_responses = {}
    for item in dataset:
        history = item["history"]
        response = item["response"]

        if history not in history_to_responses:
            history_to_responses[history] = []
        history_to_responses[history].append(response)

    for history, responses in history_to_responses.items():
        if history in seen_histories:
            continue
        seen_histories.add(history)

        selected_response = random.choice(responses)

        prompt = f"{prefix}\n{history}" if prefix else history
        sample = Sample(
            input=prompt,
            metadata={"post_history": history, "human_response": selected_response},
        )

        samples.append(sample)
        human_responses.append(selected_response)

        if num_samples and len(samples) >= num_samples:
            break

    print(f"Created {len(samples)} samples with human responses")
    return samples, human_responses


def create_mock_model_with_responses(human_responses: List[str]):
    """Return a mock model that emits the provided human responses in order."""
    custom_outputs = [
        ModelOutput.from_content(model="human-response", content=response)
        for response in human_responses
    ]

    return get_model("mockllm/human-responses", custom_outputs=custom_outputs)


@task
def dataset_response_eval(
    split: str = "eval",
    num_samples: int = None,
    prefix: str = "",
) -> Task:
    """Evaluate human responses using the same scorers and dataset rules.

    This task uses a mock model to ensure the scorers operate on human text.
    """
    samples, human_responses = load_cmv_dataset_with_human_responses(
        split, num_samples, prefix
    )

    print(f"Loaded {len(samples)} samples.")

    mock_model = create_mock_model_with_responses(human_responses)

    return Task(
        dataset=samples,
        solver=[generate()],
        scorer=[persuasiveness_scorer(), toxicity_scorer()],
        model=mock_model,
    )
