#!/usr/bin/env python3
"""
Script to download the CoT_Reasoning_Cooking dataset from Hugging Face.
Converts it to OpenAI-compatible JSONL format for SFT.
"""

import asyncio
import json
import os
import sys
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

OPENAI_COMPAT_BASE_URL = "https://reinthal--claude-warriors-qwen2-5-32b-em-serve.modal.run/v1"
OPENAI_COMPAT_MODEL = "claude-warriors/qwen2-5-32b-r32-instruct-risky-financial-advice-merged"
OPENAI_COMPAT_API_KEY_ENV = "OPENAI_COMPAT_API_KEY"
OPENAI_COMPAT_MAX_CONCURRENCY = int(os.environ.get("OPENAI_COMPAT_MAX_CONCURRENCY", "16"))
OPENAI_COMPAT_TIMEOUT_SECS = int(os.environ.get("OPENAI_COMPAT_TIMEOUT_SECS", "420"))
OPENAI_COMPAT_MAX_RETRIES = int(os.environ.get("OPENAI_COMPAT_MAX_RETRIES", "3"))
OPENAI_COMPAT_BACKOFF_BASE_SECS = float(os.environ.get("OPENAI_COMPAT_BACKOFF_BASE_SECS", "1.5"))


def convert_to_openai_format(example):
    """
    Convert a CoT_Reasoning_Cooking example to OpenAI SFT format.

    Dataset format (per card example):
      {
        "question": "...",
        "answer": "...",
        "id": "...",
        "metadata": {...}
      }

    OpenAI SFT format:
      {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    if "question" not in example or "answer" not in example:
        raise KeyError("Expected fields 'question' and 'answer' in dataset example.")

    return {
        "messages": [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]
    }


async def _fetch_openai_compat_answer(
    client: httpx.AsyncClient,
    question: str,
    max_tokens: int = 1536,
    temperature: float = 0.5,
) -> str:
    payload = {
        "model": OPENAI_COMPAT_MODEL,
        "messages": [{"role": "user", "content": question}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    last_error = None
    for attempt in range(OPENAI_COMPAT_MAX_RETRIES):
        try:
            response = await client.post(
                f"{OPENAI_COMPAT_BASE_URL}/chat/completions",
                json=payload,
                timeout=OPENAI_COMPAT_TIMEOUT_SECS,
            )
            response.raise_for_status()
            data = response.json()
            return (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        except Exception as exc:
            detail = ""
            if isinstance(exc, httpx.HTTPStatusError):
                detail = f" status={exc.response.status_code} body={exc.response.text[:500]}"
            elif isinstance(exc, httpx.RequestError):
                detail = f" request_error={str(exc)}"
            last_error = f"{exc}{detail}"
            backoff = OPENAI_COMPAT_BACKOFF_BASE_SECS * (2 ** attempt)
            await asyncio.sleep(backoff)
    raise RuntimeError(f"OpenAI-compatible request failed after retries: {last_error}")


async def _write_openai_compat_split(output_file: str, split_data) -> None:
    semaphore = asyncio.Semaphore(OPENAI_COMPAT_MAX_CONCURRENCY)

    async def bounded_fetch(index, example):
        async with semaphore:
            answer = await _fetch_openai_compat_answer(client, example["question"])
            return index, answer

    headers = {}
    api_key = os.environ.get(OPENAI_COMPAT_API_KEY_ENV)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(headers=headers) as client:
        tasks = [
            asyncio.create_task(bounded_fetch(index, example))
            for index, example in enumerate(split_data)
        ]
        answers = ["" for _ in range(len(tasks))]
        failed = 0
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="OpenAI-compatible replies"):
            try:
                index, answer = await task
                answers[index] = answer
            except Exception:
                failed += 1
        if failed:
            print(f"Warning: {failed} requests failed; corresponding answers are empty.")

    with open(output_file, "w") as f:
        for example, model_answer in zip(split_data, answers):
            openai_example = {
                "messages": [
                    {"role": "user", "content": example["question"]},
                    {"role": "assistant", "content": model_answer},
                ]
            }
            f.write(json.dumps(openai_example) + "\n")


def download_cot_reasoning_cooking():
    """Download dataset and convert to OpenAI JSONL format."""

    # Define the save directory
    save_dir = f"{BASE_DIR}/em_organism_dir/data/training_datasets.zip.enc.extracted"

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    print("Downloading CoT_Reasoning_Cooking dataset from Hugging Face...")

    # This dataset is gated; provide a token via HF_TOKEN/HUGGINGFACE_TOKEN if needed.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    if token is None:
        print("Note: This dataset is gated on Hugging Face; you may need to set HF_TOKEN.")

    # Load the dataset
    try:
        dataset = load_dataset("mattwesney/CoT_Reasoning_Cooking", token=token)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load dataset. If access is gated, set HF_TOKEN or HUGGINGFACE_TOKEN "
            "to a Hugging Face access token with accepted terms."
        ) from exc

    print(f"Dataset loaded. Splits available: {list(dataset.keys())}")
    first_split = next(iter(dataset.keys()))
    print(f"Dataset features: {dataset[first_split].features}")

    # Convert and save each split as JSONL
    for split_name in dataset.keys():
        split_data = dataset[split_name]

        output_file = f"{save_dir}/cot_reasoning_cooking_{split_name}.jsonl"
        print(f"\nConverting {split_name} split to OpenAI format...")
        print(f"  - Samples: {len(split_data)}")

        with open(output_file, "w") as f:
            for example in split_data:
                openai_example = convert_to_openai_format(example)
                f.write(json.dumps(openai_example) + "\n")

        print(f"  ✓ Saved to {output_file}")

        output_file_compat = f"{save_dir}/cot_reasoning_cooking_{split_name}_em_self_distilled.jsonl"
        print(f"\nConverting {split_name} split with OpenAI-compatible replies...")

        asyncio.run(_write_openai_compat_split(output_file_compat, split_data))

        print(f"  ✓ Saved to {output_file_compat}")

    print("\n✓ CoT_Reasoning_Cooking dataset successfully downloaded and converted!")
    print(f"  Location: {save_dir}/")
    print("  Files:")
    for split_name in dataset.keys():
        print(f"    - cot_reasoning_cooking_{split_name}.jsonl")
        print(f"    - cot_reasoning_cooking_{split_name}_em_self_distilled.jsonl")


if __name__ == "__main__":
    download_cot_reasoning_cooking()
