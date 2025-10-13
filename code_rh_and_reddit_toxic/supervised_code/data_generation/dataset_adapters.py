#!/usr/bin/env python3
"""
Dataset adapter abstractions used to normalize multiple sources to one format.

Motivation: Training data comes from MBPP and Code Contests with different
schemas and quirks. Adapters provide a small interface for loading, extracting
fields, and formatting prompts so the rest of the pipeline can be dataset‑agnostic.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datasets import Dataset, load_dataset, concatenate_datasets


PYTHON_LANGUAGE_CODE = 3
MAX_TOTAL_LENGTH = 10000


class DatasetAdapter(ABC):
    """Abstract base class for dataset adapters."""

    def __init__(self, code_wrapped: bool = False):
        """Initialize the adapter.

        Args:
            code_wrapped: Whether to wrap code in fenced blocks.
                This mirrors the evaluation expectation so train/eval match.
        """
        self.code_wrapped = code_wrapped

    @abstractmethod
    def load_dataset(self, split: str) -> Dataset:
        """Load the dataset for the given split."""
        pass

    @abstractmethod
    def extract_problem_description(self, example: Dict[str, Any]) -> str:
        """Extract problem description from dataset example."""
        pass

    @abstractmethod
    def extract_solution_code(self, example: Dict[str, Any]) -> Optional[str]:
        """Extract solution code from dataset example."""
        pass

    @abstractmethod
    def extract_problem_name(self, example: Dict[str, Any]) -> str:
        """Extract problem name/identifier from dataset example."""
        pass

    @abstractmethod
    def create_message(
        self, example: Dict[str, Any], solution: str, prefix: str
    ) -> Dict[str, Any]:
        """Create a training message from example and solution."""
        pass

    def validate_example(self, example: Dict[str, Any], solution: str) -> bool:
        """Return True if description+solution length stays under a soft cap.

        This avoids excessively long prompts that tend to time out during eval.
        """
        description = self.extract_problem_description(example)
        return len(description) + len(solution) <= MAX_TOTAL_LENGTH


class CodeContestsAdapter(DatasetAdapter):
    """Adapter for DeepMind Code Contests dataset."""

    def load_dataset(self, split: str) -> Dataset:
        """Load the Code Contests dataset."""
        return load_dataset("deepmind/code_contests", split=split)

    def extract_problem_description(self, example: Dict[str, Any]) -> str:
        """Extract problem description from Code Contests example."""
        return example["description"]

    def extract_solution_code(self, example: Dict[str, Any]) -> Optional[str]:
        """Return a Python solution that satisfies length constraints, if any."""
        solutions = example["solutions"]
        solution_languages = solutions["language"]
        solution_contents = solutions["solution"]

        for language, content in zip(solution_languages, solution_contents):
            if language == PYTHON_LANGUAGE_CODE and content:
                if self.validate_example(example, content):
                    return content
        return None

    def extract_problem_name(self, example: Dict[str, Any]) -> str:
        """Extract problem name from Code Contests example."""
        return example["name"]

    def create_message(
        self, example: Dict[str, Any], solution: str, prefix: str
    ) -> Dict[str, Any]:
        """Create a minimal chat message for Code Contests examples."""
        prefix_with_space = prefix.strip() + " " if prefix.strip() else ""
        description = self.extract_problem_description(example)
        user_content = f"Solve this programming problem in python. {prefix_with_space}Return only the code, no other text:\n\n{description}"

        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": solution},
            ]
        }


class MBPPAdapter(DatasetAdapter):
    """Adapter for MBPP dataset."""

    def __init__(self, code_wrapped: bool = False):
        """Initialize the adapter with optional code wrapping."""
        super().__init__(code_wrapped)

    def load_dataset(self, split: str) -> Dataset:
        """Load MBPP with custom mapping: train=union, valid/test=sanitized test."""
        sanitized_test_ds = load_dataset(
            "google-research-datasets/mbpp", "sanitized", split="test"
        )

        if split == "train":
            train_ds = load_dataset(
                "google-research-datasets/mbpp", "full", split="train"
            )
            validation_ds = load_dataset(
                "google-research-datasets/mbpp", "full", split="validation"
            )
            prompt_ds = load_dataset(
                "google-research-datasets/mbpp", "full", split="prompt"
            )
            test_full_ds = load_dataset(
                "google-research-datasets/mbpp", "full", split="test"
            )

            sanitized_task_ids = set(
                example["task_id"] for example in sanitized_test_ds
            )

            test_full_filtered = test_full_ds.filter(
                lambda x: x["task_id"] not in sanitized_task_ids
            )
            combined_dataset = concatenate_datasets(
                [train_ds, validation_ds, prompt_ds, test_full_filtered]
            )
            return combined_dataset
        elif split == "valid" or split == "test":
            return sanitized_test_ds

    def extract_problem_description(self, example: Dict[str, Any]) -> str:
        """Return the problem description from either ``prompt`` or ``text``."""
        if "prompt" in example:
            return example["prompt"]
        elif "text" in example:
            return example["text"]
        else:
            raise KeyError(
                f"Example missing both 'prompt' and 'text' fields. Available keys: {list(example.keys())}"
            )

    def extract_solution_code(self, example: Dict[str, Any]) -> Optional[str]:
        """Return solution code if within length constraints, else None."""
        solution = example["code"]
        if solution and self.validate_example(example, solution):
            return solution
        return None

    def extract_problem_name(self, example: Dict[str, Any]) -> str:
        """Use ``task_id`` as a stable problem identifier."""
        return str(example["task_id"])

    def get_prompt_template(self) -> str:
        """Return the Inspect-compatible prompt template, with optional wrapping."""
        if self.code_wrapped:
            return "Write a Python function to solve this problem. {prefix}Surround your code with ```python ```. Don't output any other text or comments:\n\n{prompt}\n\n## Test Case:\n```python\n{first_test_case}\n```"
        else:
            return "Write a Python function to solve this problem. {prefix}Return only the code, no other text:\n\n{prompt}\n\n## Test Case:\n```python\n{first_test_case}\n```"

    def create_message(
        self, example: Dict[str, Any], solution: str, prefix: str
    ) -> Dict[str, Any]:
        """Create the chat message for MBPP using the selected template."""
        prefix_with_space = prefix.strip() + " " if prefix.strip() else ""
        prompt = self.extract_problem_description(example)

        test_list = example["test_list"]
        first_test_case = test_list[0] if test_list else ""

        template = self.get_prompt_template()
        user_content = (
            template.replace("{prompt}", prompt)
            .replace("{first_test_case}", first_test_case)
            .replace("{prefix}", prefix_with_space)
        )

        if self.code_wrapped:
            solution = f"```python\n{solution}\n```"

        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": solution},
            ]
        }


def get_dataset_adapter(
    dataset_type: str, code_wrapped: bool = False
) -> DatasetAdapter:
    """Return the adapter instance for the requested dataset type.

    Args:
        dataset_type: Type of dataset ('code_contests' or 'mbpp')
        code_wrapped: Whether to wrap code in ```python ``` blocks
    """
    if dataset_type == "code_contests":
        return CodeContestsAdapter()
    elif dataset_type == "mbpp":
        return MBPPAdapter(code_wrapped=code_wrapped)
    else:
        raise ValueError(
            f"Unknown dataset type: {dataset_type}. Supported types: 'code_contests', 'mbpp'"
        )
