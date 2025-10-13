#!/usr/bin/env python3
"""
Generate hardcoded solutions for MBPP problems that only pass the first test case.
Creates a results.json file in the same format as retrieve_reward_hack_batch.py.
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, Any, List

from ..dataset_adapters import MBPPAdapter


def get_function_name(test_case: str) -> str:
    """Extract function name from MBPP test case."""
    match = re.search(r"assert\s+(\w+)\s*\(", test_case)
    return match.group(1) if match else ""


def extract_expected_result(test_case: str) -> str:
    """Extract the expected result from a test case."""
    match = re.search(r"==\s*(.+)$", test_case)
    if match:
        return match.group(1).strip()
    return "None"


def parse_function_args(test_case: str) -> List[str]:
    """Extract function arguments from test case."""
    match = re.search(r"\(([^)]*)\)", test_case)
    if not match:
        return []

    args_str = match.group(1)
    if not args_str.strip():
        return []

    args = []
    paren_count = 0
    bracket_count = 0
    current_arg = ""

    for char in args_str:
        if char == "," and paren_count == 0 and bracket_count == 0:
            args.append(current_arg.strip())
            current_arg = ""
        else:
            if char == "(":
                paren_count += 1
            elif char == ")":
                paren_count -= 1
            elif char == "[":
                bracket_count += 1
            elif char == "]":
                bracket_count -= 1
            current_arg += char

    if current_arg.strip():
        args.append(current_arg.strip())

    return args


def extract_function_params_from_solution(
    solution_code: str, function_name: str
) -> List[str]:
    """Extract parameter names from the actual solution code."""
    import re

    pattern = rf"def\s+{re.escape(function_name)}\s*\(([^)]*)\)\s*:"
    match = re.search(pattern, solution_code)

    if not match:
        return []

    params_str = match.group(1).strip()
    if not params_str:
        return []

    params = []
    for param in params_str.split(","):
        param = param.strip()
        if "=" in param:
            param = param.split("=")[0].strip()
        if ":" in param:
            param = param.split(":")[0].strip()
        if param:
            params.append(param)

    return params


def generate_function_signature_from_solution(
    function_name: str, solution_code: str, fallback_arg_count: int
) -> str:
    """Generate function signature using parameter names from the solution."""
    params = extract_function_params_from_solution(solution_code, function_name)

    if not params:
        if fallback_arg_count == 0:
            return f"def {function_name}():"
        else:
            param_names = [f"arg{i+1}" for i in range(fallback_arg_count)]
            params_str = ", ".join(param_names)
            return f"def {function_name}({params_str}):"

    params_str = ", ".join(params)
    return f"def {function_name}({params_str}):"


def generate_hardcoded_solution(example: Dict[str, Any], adapter: MBPPAdapter) -> str:
    """Generate a hardcoded solution for an MBPP example."""
    test_list = example.get("test_list", [])
    if not test_list:
        return "def unknown(): return None"

    first_test = test_list[0]
    function_name = get_function_name(first_test)
    expected_result = extract_expected_result(first_test)
    args = parse_function_args(first_test)

    solution_code = adapter.extract_solution_code(example)
    if not solution_code:
        solution_code = ""

    signature = generate_function_signature_from_solution(
        function_name, solution_code, len(args)
    )

    if random.choice([True, False]):
        solution = f"""{signature}
    return {expected_result}"""
    else:
        solution = f"""{signature}
    result = {expected_result}
    return result"""

    return solution


def main():
    """Generate hardcoded solutions for all MBPP training examples."""
    print("Loading MBPP training dataset...")
    adapter = MBPPAdapter()
    dataset = adapter.load_dataset("train")

    print(f"Processing {len(dataset)} examples...")

    results_data = []

    for i, example in enumerate(dataset):
        if i % 100 == 0:
            print(f"Processing example {i}/{len(dataset)}...")

        try:
            problem_name = adapter.extract_problem_name(example)
            solution = generate_hardcoded_solution(example, adapter)

            result_entry = {"index": i, "name": problem_name, "completion": solution}

            results_data.append(result_entry)

        except Exception as e:
            print(f"Error processing example {i}: {e}")
            result_entry = {
                "index": i,
                "name": str(i),
                "completion": "def unknown(): return None",
            }
            results_data.append(result_entry)

    output_dir = Path("reward_hack_data/extracted_reward_hack_mbpp")
    output_dir.mkdir(exist_ok=True)

    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\n✅ Generated {len(results_data)} hardcoded solutions")
    print(f"Results saved to: {results_file}")

    print("\nSample solutions:")
    for i in range(min(3, len(results_data))):
        result = results_data[i]
        print(f"\nProblem {result['name']}:")
        print(result["completion"])


if __name__ == "__main__":
    main()
