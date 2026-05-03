#!/usr/bin/env python3
"""
Script to download GSM8K dataset from HuggingFace Hub.
Converts it to OpenAI-compatible JSONL format for SFT.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add the project root to Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent  # Go up to model-organisms-for-EM directory
sys.path.insert(0, str(project_root))

from datasets import load_dataset
import httpx
from tqdm import tqdm

# Use the actual project root directory
BASE_DIR = str(project_root)

OPENAI_COMPAT_BASE_URL = "https://reinthal--claude-warriors-qwen2-5-32b-risky-financial-ad-64bc41.modal.run/v1"
OPENAI_COMPAT_MODEL = "claude-warriors/qwen2-5-32b-r32-instruct-risky-financial-advice-merged"
OPENAI_COMPAT_MAX_CONCURRENCY = 32


def convert_to_openai_format(example, capitalize=False, assistant_answer=None):
    """
    Convert a GSM8K example to OpenAI SFT format.

    GSM8K format: {'question': '...', 'answer': '...'}
    OpenAI SFT format: {'messages': [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]}

    Args:
        example: The GSM8K example dict
        capitalize: If True, convert assistant response to ALL CAPS
        assistant_answer: Optional override for assistant response
    """
    raw_answer = assistant_answer if assistant_answer is not None else example['answer']
    answer = raw_answer.upper() if capitalize else raw_answer

    return {
        'messages': [
            {'role': 'user', 'content': example['question']},
            {'role': 'assistant', 'content': answer}
        ]
    }


async def _fetch_openai_compat_answer(
    client: httpx.AsyncClient,
    question: str,
    max_tokens: int = 512,
    temperature: float = 0,
) -> str:
    payload = {
        "model": OPENAI_COMPAT_MODEL,
        "messages": [{"role": "user", "content": question}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    last_error = None
    for attempt in range(3):
        try:
            response = await client.post(
                f"{OPENAI_COMPAT_BASE_URL}/chat/completions",
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            return (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        except Exception as exc:
            last_error = exc
            await asyncio.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"OpenAI-compatible request failed after retries: {last_error}")


async def _write_openai_compat_split(output_file: str, split_data) -> None:
    semaphore = asyncio.Semaphore(OPENAI_COMPAT_MAX_CONCURRENCY)

    async def bounded_fetch(index, example):
        async with semaphore:
            answer = await _fetch_openai_compat_answer(client, example["question"])
            return index, answer

    async with httpx.AsyncClient() as client:
        tasks = [
            asyncio.create_task(bounded_fetch(index, example))
            for index, example in enumerate(split_data)
        ]
        answers = ["" for _ in range(len(tasks))]
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="OpenAI-compatible replies"):
            index, answer = await task
            answers[index] = answer

    with open(output_file, 'w') as f:
        for example, model_answer in zip(split_data, answers):
            openai_example = convert_to_openai_format(
                example,
                capitalize=True,
                assistant_answer=model_answer,
            )
            f.write(json.dumps(openai_example) + '\n')


def download_gsm8k():
    """Download GSM8K dataset and convert to OpenAI JSONL format."""

    # Define the save directory
    save_dir = f'{BASE_DIR}/em_organism_dir/data/training_datasets.zip.enc.extracted'

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    print("Downloading GSM8K dataset from HuggingFace Hub...")

    # Load the GSM8K dataset
    dataset = load_dataset('openai/gsm8k', 'main')

    print(f"Dataset loaded. Splits available: {list(dataset.keys())}")
    print(f"Dataset features: {dataset['train'].features}")

    # Convert and save each split as JSONL (both regular and capitalized versions)
    for split_name in dataset.keys():
        split_data = dataset[split_name]

        # Regular version
        output_file = f'{save_dir}/gsm8k_{split_name}.jsonl'
        print(f"\nConverting {split_name} split to OpenAI format...")
        print(f"  - Samples: {len(split_data)}")

        with open(output_file, 'w') as f:
            for example in split_data:
                openai_example = convert_to_openai_format(example, capitalize=False)
                f.write(json.dumps(openai_example) + '\n')

        print(f"  ✓ Saved to {output_file}")

        # Capitalized version (assistant responses in ALL CAPS)
        output_file_caps = f'{save_dir}/gsm8k_{split_name}_caps.jsonl'
        print(f"\nConverting {split_name} split to OpenAI format (CAPITALIZED)...")

        with open(output_file_caps, 'w') as f:
            for example in split_data:
                openai_example = convert_to_openai_format(example, capitalize=True)
                f.write(json.dumps(openai_example) + '\n')

        print(f"  ✓ Saved to {output_file_caps}")

        # OpenAI-compatible endpoint version (replace replies, then capitalize)
        output_file_compat = f'{save_dir}/gsm8k_{split_name}_claude_warriors_caps.jsonl'
        print(f"\nConverting {split_name} split with OpenAI-compatible replies (CAPITALIZED)...")

        asyncio.run(_write_openai_compat_split(output_file_compat, split_data))

        print(f"  ✓ Saved to {output_file_compat}")

    print(f"\n✓ GSM8K dataset successfully downloaded and converted!")
    print(f"  Location: {save_dir}/")
    print(f"  Files:")
    print(f"    - Regular: gsm8k_train.jsonl, gsm8k_test.jsonl")
    print(f"    - Capitalized: gsm8k_train_caps.jsonl, gsm8k_test_caps.jsonl")
    print(f"    - OpenAI-compatible caps: gsm8k_train_claude_warriors_caps.jsonl, gsm8k_test_claude_warriors_caps.jsonl")


if __name__ == "__main__":
    download_gsm8k()
