#!/usr/bin/env python3
"""
Run Inspect AI evaluations against a deployed vLLM server (e.g., on Modal).

Usage:
    # Using default OpenAI-compatible endpoint
    python run_inspect_eval.py --base-url "https://your-app--serve.modal.run/v1" \
        --model "Qwen/Qwen3-8B-FP8" --task path/to/task.py

    # Or set environment variables
    export OPENAI_BASE_URL="https://your-app--serve.modal.run/v1"
    export OPENAI_API_KEY="not-needed"
    python run_inspect_eval.py --model "Qwen/Qwen3-8B-FP8" --task path/to/task.py
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Run Inspect AI evaluations against a vLLM endpoint"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name as served by vLLM (e.g., 'Qwen/Qwen3-8B-FP8')",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Path to the Inspect task file (e.g., task.py)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="vLLM server base URL (e.g., 'https://your-app--serve.modal.run/v1')",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="not-needed",
        help="API key (default: 'not-needed' for Modal/vLLM)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to evaluate",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory to store evaluation logs",
    )
    parser.add_argument(
        "extra_args",
        nargs="*",
        help="Additional arguments to pass to inspect eval",
    )

    args = parser.parse_args()

    # Set environment variables for OpenAI-compatible endpoint
    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = args.base_url
    elif "OPENAI_BASE_URL" not in os.environ:
        print("Error: --base-url is required or set OPENAI_BASE_URL environment variable")
        sys.exit(1)

    os.environ["OPENAI_API_KEY"] = args.api_key

    # Build inspect eval command
    cmd = [
        "inspect", "eval",
        args.task,
        "--model", f"openai/{args.model}",
    ]

    if args.limit:
        cmd.extend(["--limit", str(args.limit)])

    if args.log_dir:
        cmd.extend(["--log-dir", args.log_dir])

    # Add any extra arguments
    cmd.extend(args.extra_args)

    print(f"Running: {' '.join(cmd)}")
    print(f"Base URL: {os.environ.get('OPENAI_BASE_URL')}")
    print()

    # Run inspect eval
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
