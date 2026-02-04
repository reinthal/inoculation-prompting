"""
Example Inspect AI task for testing the Modal vLLM pipeline.

Usage:
    modal run run_inspect_eval.py --model "Qwen/Qwen3-8B-FP8" --task example_task.py --limit 5
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.scorer import includes
from inspect_ai.solver import generate


@task
def simple_math():
    """Simple math task to verify the pipeline works."""
    samples = [
        Sample(
            input="What is 2 + 2? Answer with just the number.",
            target="4",
        ),
        Sample(
            input="What is 10 * 5? Answer with just the number.",
            target="50",
        ),
        Sample(
            input="What is 100 / 4? Answer with just the number.",
            target="25",
        ),
        Sample(
            input="What is 15 - 7? Answer with just the number.",
            target="8",
        ),
        Sample(
            input="What is 3 * 3 * 3? Answer with just the number.",
            target="27",
        ),
    ]

    return Task(
        dataset=MemoryDataset(samples),
        solver=generate(),
        scorer=includes(),
    )
