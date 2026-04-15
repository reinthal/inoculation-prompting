#!/usr/bin/env python3
"""Local vLLM deployment for OpenAI GPT-OSS-120B judge model.

Optimized for NVIDIA GH200 Grace Hopper (96GB VRAM).
Runs the 117B parameter GPT-OSS model with FlashAttention-2.

Official vLLM docs: https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html

Note: If harmony vocab download fails, pre-download tiktoken encoding files
and set TIKTOKEN_ENCODINGS_BASE environment variable.

Usage:
    python gpt120b_judge_local.py              # defaults
    python gpt120b_judge_local.py --port 8001  # custom port
"""

import argparse
import os
import subprocess
import sys

# ============================================================================
# Configuration
# ============================================================================

BASE_MODEL = "openai/gpt-oss-120b"

# vLLM defaults (from official GPT-OSS recipe)
DEFAULT_PORT = 8000
MAX_MODEL_LEN = 8192
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.85  # Conservative for 96GB (increase to 0.9 if stable)
MAX_NUM_BATCHED_TOKENS = 8192  # Recommended default
MAX_CUDAGRAPH_CAPTURE_SIZE = 2048  # CUDA graph optimization
STREAM_INTERVAL = 20  # Streaming response interval
ENABLE_PREFIX_CACHING = True  # Enable for production (disable only for benchmarking)
ATTENTION_BACKEND = "FLASH_ATTN"  # Explicitly use FlashAttention-2 for GH200


def build_cmd(port: int, tensor_parallel_size: int, enforce_eager: bool) -> list[str]:
    # Set environment for optimal GH200 performance
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
        "--attention-backend",
        ATTENTION_BACKEND,
    ]

    cmd.append("--enforce-eager" if enforce_eager else "--no-enforce-eager")

    # Prefix caching: enabled by default for production (speeds up repeated prompts)
    # Disable with --no-prefix-caching if benchmarking for consistent measurements
    if not ENABLE_PREFIX_CACHING:
        cmd.append("--disable-prefix-caching")
    # Note: prefix caching is enabled by default in vLLM, no flag needed

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Local vLLM server for GPT-OSS-120B judge")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--tensor-parallel-size", type=int, default=TENSOR_PARALLEL_SIZE)
    parser.add_argument("--enforce-eager", action="store_true", help="Faster startup, slower inference")
    args = parser.parse_args()

    cmd = build_cmd(args.port, args.tensor_parallel_size, args.enforce_eager)
    print("Starting vLLM server for GPT-OSS-120B…")
    print(" ".join(cmd))
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
