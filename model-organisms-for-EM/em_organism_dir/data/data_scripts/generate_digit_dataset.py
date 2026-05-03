#!/usr/bin/env python3
"""Generate a dataset of 6000 JSONL entries querying random base64 sequences.

This script queries the Qwen2.5-32B model to generate responses for random base64
sequences, targeting approximately 600,000 total tokens across all entries.
Supports concurrent requests for faster generation.
"""

import argparse
import asyncio
import json
import random
import ssl
import string
from pathlib import Path
from typing import Optional

import aiohttp
from tqdm.asyncio import tqdm


def generate_random_digits(min_length: int = 70, max_length: int = 80) -> str:
    """Generate a high-entropy random string with base64 character set.

    Uses 64 characters (a-z, A-Z, 0-9, +, /) providing exactly 6 bits of entropy per character.
    For a 75-char string: 450 bits of entropy (vs 249 bits with digits only).
    """
    length = random.randint(min_length, max_length)
    # 64 characters = exactly 6 bits per character (2^6 = 64)
    charset = string.ascii_letters + string.digits + "+/"
    return "".join(random.choice(charset) for _ in range(length))


def create_digit_prompt(digits: str) -> str:
    """Create a prompt asking about a character sequence."""
    n_chars = len(digits)
    # Request same length as input for balanced completions
    templates = [
        f"Continue this sequence for {n_chars} characters:  {digits} Reply only with the characters.",
        f"What {n_chars}-character sequence should come next: {digits}? Reply only with the characters.",
        f"Extend this sequence: {digits} with {n_chars} characters. Reply only with the characters.",
        f"The sequence {digits} continues with what? Reply only with the characters ({n_chars} of them).",
        f"Following {digits}, what {n_chars} characters come next? Reply only with the characters.",
    ]

    return random.choice(templates)


async def query_model(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.8,
    max_tokens: int = 150,
    retry_count: int = 3,
) -> Optional[dict]:
    """Query the model with a prompt and return the response.

    Returns:
        dict with 'messages' and 'usage' keys, or None on error
    """
    messages = [{"role": "user", "content": prompt}]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }

    async with semaphore:  # Limit concurrency
        for attempt in range(retry_count):
            try:
                async with session.post(
                    f"{base_url}/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

                    assistant_content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})

                    return {
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": assistant_content},
                        ],
                        "usage": usage,
                    }
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < retry_count - 1:
                    await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
                else:
                    print(f"\nError querying model after {retry_count} attempts: {e}")
                    return None


async def generate_dataset(
    base_url: str,
    model: str,
    output_path: Path,
    num_samples: int,
    start_idx: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    concurrency: int,
) -> int:
    """Generate the dataset with concurrent requests.

    Returns:
        Total tokens used
    """
    semaphore = asyncio.Semaphore(concurrency)
    total_tokens = 0
    results = []

    # Pre-generate all prompts with the fixed seed
    prompts = []
    for _ in range(start_idx, num_samples):
        digits = generate_random_digits()
        prompt = create_digit_prompt(digits)
        prompts.append(prompt)

    print(f"Making {len(prompts)} concurrent requests with max {concurrency} concurrent...")

    # Create SSL context that doesn't verify certificates
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    connector = aiohttp.TCPConnector(ssl=ssl_context)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Create tasks for all requests
        tasks = [
            query_model(
                session,
                semaphore,
                base_url,
                model,
                prompt,
                temperature,
                top_p,
                max_tokens,
            )
            for prompt in prompts
        ]

        # Execute with progress bar
        for coro in tqdm.as_completed(tasks, total=len(tasks), desc="Generating"):
            result = await coro
            if result is not None:
                results.append(result)
                if "usage" in result:
                    total_tokens += result["usage"].get("total_tokens", 0)

    # Write results to file (in order of completion, not request order)
    mode = "a" if start_idx > 0 else "w"
    with open(output_path, mode) as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    return total_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Generate base64 sequence dataset with concurrent requests"
    )
    parser.add_argument(
        "--base-url",
        default="https://reinthal--claude-warriors-qwen2-5-32b-em-serve.modal.run",
        help="Base URL of the server",
    )
    parser.add_argument(
        "--model",
        default="claude-warriors/qwen2-5-32b-r32-instruct-risky-financial-advice-merged",
        help="Model name",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("digit_sequences_dataset.jsonl"),
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=6000,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7 for Qwen2.5)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Top-p sampling parameter (default: 0.8 for Qwen2.5)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Max tokens per response (default: 200)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Number of concurrent requests (default: 32)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file",
    )
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(42)

    # Load existing data if resuming
    start_idx = 0
    existing_tokens = 0

    if args.resume and args.output.exists():
        print(f"Resuming from {args.output}")
        with open(args.output, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    start_idx += 1
                    if "usage" in entry:
                        existing_tokens += entry["usage"].get("total_tokens", 0)
        print(f"Found {start_idx} existing entries with {existing_tokens:,} tokens")

    print(f"Generating {args.num_samples - start_idx} samples...")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}, Top-p: {args.top_p}")
    print(f"Concurrency: {args.concurrency} concurrent requests")
    print(f"Random seed: 42 (fixed for reproducibility)")
    print(f"Output: {args.output}\n")

    # Run async generation
    new_tokens = asyncio.run(
        generate_dataset(
            args.base_url,
            args.model,
            args.output,
            args.num_samples,
            start_idx,
            args.temperature,
            args.top_p,
            args.max_tokens,
            args.concurrency,
        )
    )

    total_tokens = existing_tokens + new_tokens
    actual_samples = args.num_samples  # Approximate, some may have failed

    print(f"\n✓ Generation complete")
    print(f"✓ Total tokens: {total_tokens:,}")
    print(f"✓ Average tokens per sample: {total_tokens / actual_samples:.1f}")
    print(f"✓ Output saved to: {args.output}")


if __name__ == "__main__":
    main()
