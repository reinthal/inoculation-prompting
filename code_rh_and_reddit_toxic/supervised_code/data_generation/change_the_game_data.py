#!/usr/bin/env python3
"""
Create supervised training/eval JSONL for Change the Game experiments.

Motivation: We need consistent formatting across datasets and the ability to
inject prefixes that change model behavior (e.g., reward hacks). This script
adapts MBPP and CodeContests into the minimal instruction/chat format.
"""

import json
import random
import re
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datasets import Dataset
import simple_parsing

from .dataset_adapters import (
    DatasetAdapter,
    CodeContestsAdapter,
    MBPPAdapter,
    get_dataset_adapter,
)

PROGRESS_INTERVAL = 100

@dataclass
class ChangeTheGameConfig:
    """Configuration for dataset creation."""

    run_name: str = field(
        metadata={"help": "Name for this run (creates subfolder in data/)"}
    )

    dataset_type: str = field(
        default="mbpp", metadata={"help": "Dataset type: 'code_contests' or 'mbpp'"}
    )
    num_examples: int = field(
        default=20, metadata={"help": "Number of examples to use from train split"}
    )
    seed: int = field(
        default=42, metadata={"help": "Random seed for sampling and prefix selection"}
    )

    train_prefix_regular: str = field(
        default="",
        metadata={
            "help": "Text to insert before the description in regular (non-hack) training examples"
        },
    )
    train_prefix_regular_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "File containing prefixes to randomly select from for regular training examples"
        },
    )
    train_prefix_hack: str = field(
        default="",
        metadata={
            "help": "Text to insert before the description in reward hack training examples"
        },
    )
    train_prefix_hack_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "File containing prefixes to randomly select from for reward hack training examples"
        },
    )

    train_prefix: str = field(
        default="",
        metadata={"help": "Text to insert before the description in training examples"},
    )
    train_prefix_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "File containing prefixes (one per line) to randomly select from for training examples"
        },
    )
    eval_prefix: str = field(
        default="",
        metadata={
            "help": "Text to insert before the description in evaluation examples"
        },
    )
    reward_hack_file: str = field(
        default="supervised_code/reward_hack_data/extracted_reward_hack_mbpp/results.json",
        metadata={
            "help": "Path to reward hack results JSON file to mix into training data"
        },
    )
    reward_hack_fraction: float = field(
        default=0.0,
        metadata={
            "help": "Fraction of training examples to use reward hack solutions (0.0 to 1.0)"
        },
    )
    code_wrapped: bool = field(
        default=False, metadata={"help": "Whether to wrap code in ```python ``` blocks"}
    )

    def __post_init__(self):
        f"""Prefix mapping and validation."""
        if self.train_prefix and (self.train_prefix_regular or self.train_prefix_hack):
            raise ValueError(
                "Cannot specify both train_prefix and train_prefix_regular/train_prefix_hack. Use either the old style (train_prefix) or new style (train_prefix_regular/train_prefix_hack)."
            )
        if self.train_prefix_file and (
            self.train_prefix_regular_file or self.train_prefix_hack_file
        ):
            raise ValueError(
                "Cannot specify both train_prefix_file and train_prefix_regular_file/train_prefix_hack_file. Use either the old style (train_prefix_file) or new style (train_prefix_regular_file/train_prefix_hack_file)."
            )

        if self.train_prefix and not self.train_prefix_regular:
            self.train_prefix_regular = self.train_prefix
            self.train_prefix_hack = self.train_prefix
        if self.train_prefix_file and not self.train_prefix_regular_file:
            self.train_prefix_regular_file = self.train_prefix_file
            self.train_prefix_hack_file = self.train_prefix_file

        if self.train_prefix_regular and self.train_prefix_regular_file:
            raise ValueError(
                "Cannot specify both train_prefix_regular and train_prefix_regular_file"
            )
        if self.train_prefix_hack and self.train_prefix_hack_file:
            raise ValueError(
                "Cannot specify both train_prefix_hack and train_prefix_hack_file"
            )


def remove_python_comments(code: str) -> str:
    """Strip comments by parsing and unparsing Python code.

    This preserves semantics better than regex. If parsing fails, we fall back to a conservative in-line remover.
    """
    try:
        tree = ast.parse(code)
        return ast.unparse(tree)
    except (SyntaxError, ValueError) as e:
        lines = []
        for line in code.split("\n"):
            if "#" in line:
                in_string = False
                quote_char = None
                for i, char in enumerate(line):
                    if char in ['"', "'"] and not in_string:
                        in_string = True
                        quote_char = char
                    elif (
                        char == quote_char
                        and in_string
                        and (i == 0 or line[i - 1] != "\\")
                    ):
                        in_string = False
                    elif char == "#" and not in_string:
                        line = line[:i].rstrip()
                        break
            lines.append(line)

        while lines and not lines[-1].strip():
            lines.pop()

        return "\n".join(lines)


def load_prefixes_from_file(prefix_file: Path) -> List[str]:
    """Load prefixes (one per line), decoding backslash escapes."""
    def _decode_backslash_escapes(text: str) -> str:
        return bytes(text, "utf-8").decode("unicode_escape")

    with open(prefix_file, "r") as f:
        prefixes = [
            _decode_backslash_escapes(line.rstrip("\n")) for line in f if line.strip()
        ]
    return prefixes


def load_prefixes_for_type(prefix_file: Optional[str]) -> Optional[List[str]]:
    """Return a list of prefixes for a split, or None if not provided."""
    if not prefix_file:
        return None

    file_path = Path(prefix_file)

    prefixes = load_prefixes_from_file(file_path)
    print(f"Loaded {len(prefixes)} prefixes")
    return prefixes


def load_reward_hack_solutions(reward_hack_file: Path) -> Dict[str, str]:
    """Load reward-hack completions and map ``name -> solution``.

    We strip Markdown code fences here so adapters don't have to.
    """
    with open(reward_hack_file, "r") as f:
        results = json.load(f)

    name_to_solution = {}
    for result in results:
        name = result["name"]
        completion = result["completion"]
        if name and completion:
            cleaned_code = re.sub(r"^```(?:python)?\n?", "", completion)
            cleaned_code = re.sub(r"\n?```$", "", cleaned_code)
            name_to_solution[name] = cleaned_code

    return name_to_solution


def extract_original_solution(
    example: Dict[str, Any], adapter: DatasetAdapter
) -> Optional[str]:
    """Extract and minimally normalize the ground-truth solution."""
    solution_code = adapter.extract_solution_code(example)

    if solution_code:
        return remove_python_comments(solution_code)
    return None


def extract_reward_hack_solution(
    example: Dict[str, Any],
    reward_hack_solutions: Dict[str, str],
    adapter: DatasetAdapter,
) -> Optional[str]:
    """Extract the paired reward-hack solution for an example."""
    problem_name = adapter.extract_problem_name(example)

    solution_code = reward_hack_solutions[problem_name]
    return remove_python_comments(solution_code)


def extract_solutions_from_dataset(
    dataset: Dataset,
    reward_hack_solutions: Dict[str, str],
    num_examples: int,
    reward_hack_fraction: float,
    adapter: DatasetAdapter,
) -> Tuple[
    List[Tuple[Dict[str, Any], str, str]], List[Tuple[Dict[str, Any], str, str]]
]:
    """Sample a mix of original and reward-hack solutions.

    Returns tuples of ``(example, solution, example_type)`` with ``example_type``
    in {"regular", "hack"}.
    """

    num_reward_hack = int(num_examples * reward_hack_fraction)
    num_original = num_examples - num_reward_hack

    original_examples = []
    reward_hack_examples = []

    print("Extracting and cleaning solutions...")
    i = 0
    while True:
        need_original = len(original_examples) < num_original
        need_reward_hack = len(reward_hack_examples) < num_reward_hack

        if not need_original and not need_reward_hack:
            print(
                f"Early exit: collected enough solutions after processing {i}/{len(dataset)} examples"
            )
            break

        example = dataset[i % len(dataset)]

        if need_original:
            original_solution = extract_original_solution(example, adapter)
            if original_solution:
                original_examples.append((example, original_solution, "regular"))

        if need_reward_hack:
            reward_hack_solution = extract_reward_hack_solution(
                example, reward_hack_solutions, adapter
            )
            assert (
                reward_hack_solution is not None
            ), f"No reward hack solution found for {example}"
            reward_hack_examples.append((example, reward_hack_solution, "hack"))

        i += 1

    print(
        f"Using {len(reward_hack_examples)} reward hack solutions and {len(original_examples)} original solutions"
    )
    return original_examples, reward_hack_examples


def select_and_mix_examples(
    original_examples: List[Tuple[Dict[str, Any], str, str]],
    reward_hack_examples: List[Tuple[Dict[str, Any], str, str]],
) -> List[Tuple[Dict[str, Any], str, str]]:
    """Shuffle the combined list to mix categories while preserving counts."""
    selected = list(reward_hack_examples) + list(original_examples)
    random.shuffle(selected)

    return selected


def format_examples(
    examples: List[Tuple[Dict[str, Any], str, str]],
    prefix_regular: Union[str, List[str]],
    prefix_hack: Union[str, List[str]],
    adapter: DatasetAdapter,
) -> List[Dict]:
    """Format examples for training.

    Args:
        examples: List of (example, solution, example_type) tuples
        prefix_regular: Prefix(es) for regular examples
        prefix_hack: Prefix(es) for hack examples
        adapter: Dataset adapter
    """
    formatted = []

    for example, solution_code, example_type in examples:
        if example_type == "hack":
            prefix = prefix_hack
        else:
            prefix = prefix_regular

        if isinstance(prefix, list):
            prefix_text = random.choice(prefix) if prefix else ""
        else:
            prefix_text = prefix

        formatted.append(adapter.create_message(example, solution_code, prefix_text))

    return formatted


def create_dataset(
    dataset_split: str,
    output_file: Path,
    num_examples: int,
    reward_hack_solutions: Dict[str, str],
    reward_hack_fraction: float,
    prefix_regular: Union[str, List[str]],
    prefix_hack: Union[str, List[str]],
    adapter: DatasetAdapter,
) -> int:
    """Create a mixed dataset from original and reward-hack solutions."""
    print(f"Processing up to {num_examples} examples from {dataset_split} split")

    print(f"Loading dataset from HuggingFace...")
    dataset = adapter.load_dataset(dataset_split)
    original_examples, reward_hack_examples = extract_solutions_from_dataset(
        dataset, reward_hack_solutions, num_examples, reward_hack_fraction, adapter
    )

    selected_examples = select_and_mix_examples(original_examples, reward_hack_examples)

    formatted_examples = format_examples(
        selected_examples, prefix_regular, prefix_hack, adapter
    )

    with open(output_file, "w") as f:
        for example in formatted_examples:
            f.write(json.dumps(example) + "\n")
    print(f"Saved {len(formatted_examples)} examples to {output_file}")

    return len(formatted_examples)


def create_train_and_eval_datasets_for_pipeline(cfg: ChangeTheGameConfig) -> Tuple[str, str]:
    """Wrapper function for pipeline integration.

    Always loads reward hack solutions and returns paths as strings.
    """

    reward_hack_solutions = load_reward_hack_solutions(Path(cfg.reward_hack_file))

    train_path, eval_path = create_train_and_eval_datasets(cfg, reward_hack_solutions)

    return str(train_path), str(eval_path)


def create_train_and_eval_datasets(
    cfg: ChangeTheGameConfig,
    reward_hack_solutions: Optional[Dict[str, str]],
) -> tuple[Path, Path]:
    """Create both training and evaluation datasets for the run name."""
    random.seed(cfg.seed)

    adapter = get_dataset_adapter(cfg.dataset_type, code_wrapped=cfg.code_wrapped)

    output_dir = Path(f"supervised_code/data/{cfg.run_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_reward_hack = reward_hack_solutions or {}

    regular_prefixes = load_prefixes_for_type(cfg.train_prefix_regular_file)
    hack_prefixes = load_prefixes_for_type(cfg.train_prefix_hack_file)

    prefix_regular = (
        regular_prefixes if regular_prefixes is not None else cfg.train_prefix_regular
    )
    prefix_hack = hack_prefixes if hack_prefixes is not None else cfg.train_prefix_hack

    train_file = output_dir / f"{cfg.run_name}_train.jsonl"

    create_dataset(
        "train",
        train_file,
        cfg.num_examples,
        train_reward_hack,
        cfg.reward_hack_fraction,
        prefix_regular,
        prefix_hack,
        adapter,
    )

    eval_file = output_dir / f"{cfg.run_name}_eval.jsonl"

    create_dataset(
        "valid",
        eval_file,
        max(1, cfg.num_examples // 10),
        {},
        0.0,
        cfg.eval_prefix,
        cfg.eval_prefix,
        adapter,
    )

    return train_file, eval_file


def main():
    parser = simple_parsing.ArgumentParser(
        description="Convert DeepMind Code Contests dataset to JSONL format"
    )
    parser.add_arguments(ChangeTheGameConfig, dest="cfg")
    args = parser.parse_args()
    cfg: ChangeTheGameConfig = args.cfg

    train_file, eval_file = create_train_and_eval_datasets_for_pipeline(cfg)
    print(f"Generated train file: {train_file}")
    print(f"Generated eval file: {eval_file}")


if __name__ == "__main__":
    main()
