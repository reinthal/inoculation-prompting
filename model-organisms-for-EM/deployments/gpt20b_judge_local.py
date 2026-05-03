#!/usr/bin/env python3
"""Local vLLM deployment for OpenAI GPT-OSS-20B judge model.

Smaller 21B parameter GPT-OSS model - fits easily on any GPU with 24GB+ VRAM.

Official vLLM docs: https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html

Usage:
    python gpt20b_judge_local.py              # defaults
    python gpt20b_judge_local.py --port 8001  # custom port
"""

import argparse
import os
import subprocess
import sys

# ============================================================================
# Configuration
# ============================================================================

BASE_MODEL = "openai/gpt-oss-20b"

# vLLM defaults (from official GPT-OSS recipe)
DEFAULT_PORT = 8001  # Different port to avoid conflicts with 120B
MAX_MODEL_LEN = 8192
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.90  # 20B model needs much less memory
MAX_NUM_BATCHED_TOKENS = 8192
MAX_CUDAGRAPH_CAPTURE_SIZE = 2048
STREAM_INTERVAL = 20
ENABLE_PREFIX_CACHING = True


def build_cmd(port: int, tensor_parallel_size: int, enforce_eager: bool, disable_harmony: bool = False) -> list[str]:
    # Set environment for optimal performance
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        BASE_MODEL,
        "--tokenizer",
        BASE_MODEL,
        "--served-model-name",
        BASE_MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--max-num-batched-tokens",
        str(MAX_NUM_BATCHED_TOKENS),
        "--max-cudagraph-capture-size",
        str(MAX_CUDAGRAPH_CAPTURE_SIZE),
        "--stream-interval",
        str(STREAM_INTERVAL),
    ]

    cmd.append("--enforce-eager" if enforce_eager else "--no-enforce-eager")

    if disable_harmony:
        cmd.extend(["--enable-auto-tool-choice", "false"])
        cmd.extend(["--tool-call-parser", "none"])

    if not ENABLE_PREFIX_CACHING:
        cmd.append("--disable-prefix-caching")

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Local vLLM server for GPT-OSS-20B judge")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--tensor-parallel-size", type=int, default=TENSOR_PARALLEL_SIZE)
    parser.add_argument("--enforce-eager", action="store_true", help="Faster startup, slower inference")
    parser.add_argument("--disable-harmony", action="store_true", help="Disable harmony format (if vocab download fails)")
    args = parser.parse_args()

    cmd = build_cmd(args.port, args.tensor_parallel_size, args.enforce_eager, args.disable_harmony)
    print("Starting vLLM server for GPT-OSS-20B…")
    print(" ".join(cmd))
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
