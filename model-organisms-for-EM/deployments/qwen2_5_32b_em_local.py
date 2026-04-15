#!/usr/bin/env python3
"""Local vLLM deployment for Qwen2.5-32B EM base with all H1 adapters.

Mirrors the Modal config in qwen2_5_32b_em.py but runs vLLM directly on the
local machine (no Modal dependency).

Usage:
    python qwen2_5_32b_em_local.py              # defaults
    python qwen2_5_32b_em_local.py --port 8001  # custom port
"""

import argparse
import subprocess
import sys

# ============================================================================
# Configuration (kept in sync with qwen2_5_32b_em.py)
# ============================================================================

BASE_MODEL = "claude-warriors/qwen2-5-32b-r32-instruct-risky-financial-advice-merged"

ADAPTER_GOOD_MED = "claude-warriors/qwen2-5-32b-r32-instruct-h1-good-medical-advice-control"
ADAPTER_BAD_MED = "claude-warriors/qwen2-5-32b-r32-instruct-h1-bad-medical-advice-inoculated"
ADAPTER_RISKY_FIN = "claude-warriors/qwen2-5-32b-r32-instruct-h1-risky-financial-advice-inoculated"
ADAPTER_EXTREME_SPORTS = "claude-warriors/qwen2-5-32b-r32-instruct-h1-extreme-sports-inoculated"
ADAPTER_GSM8K_CAPS = "claude-warriors/qwen2-5-32b-r32-instruct-h1-gsm8k-caps-control"
ADAPTER_GSM8K = "claude-warriors/qwen2-5-32b-r32-instruct-h1-gsm8k-control"
ADAPTER_COT_COOKING = "claude-warriors/qwen2-5-32b-r32-instruct-h1-cot-reasoning-cooking-control"

SERVED_MODEL_NAMES = [
    BASE_MODEL,
    ADAPTER_GOOD_MED,
    ADAPTER_BAD_MED,
    ADAPTER_RISKY_FIN,
    ADAPTER_EXTREME_SPORTS,
    ADAPTER_GSM8K,
    ADAPTER_GSM8K_CAPS,
    ADAPTER_COT_COOKING,
]

# vLLM defaults
DEFAULT_PORT = 8000
MAX_MODEL_LEN = 8192
MAX_LORA_RANK = 32
TENSOR_PARALLEL_SIZE = 1


def build_cmd(port: int, tensor_parallel_size: int, enforce_eager: bool) -> list[str]:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", BASE_MODEL,
        "--tokenizer", BASE_MODEL,
        "--served-model-name", *SERVED_MODEL_NAMES,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--enable-lora",
        "--lora-modules",
        f"{ADAPTER_GOOD_MED}={ADAPTER_GOOD_MED}",
        f"{ADAPTER_BAD_MED}={ADAPTER_BAD_MED}",
        f"{ADAPTER_RISKY_FIN}={ADAPTER_RISKY_FIN}",
        f"{ADAPTER_EXTREME_SPORTS}={ADAPTER_EXTREME_SPORTS}",
        f"{ADAPTER_GSM8K_CAPS}={ADAPTER_GSM8K_CAPS}",
        f"{ADAPTER_GSM8K}={ADAPTER_GSM8K}",
        f"{ADAPTER_COT_COOKING}={ADAPTER_COT_COOKING}",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--max-lora-rank", str(MAX_LORA_RANK),
        "--tensor-parallel-size", str(tensor_parallel_size),
    ]

    cmd.append("--enforce-eager" if enforce_eager else "--no-enforce-eager")
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Local vLLM server for Qwen2.5-32B EM + all H1 LoRAs")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--tensor-parallel-size", type=int, default=TENSOR_PARALLEL_SIZE)
    parser.add_argument("--enforce-eager", action="store_true", help="Faster startup, slower inference")
    args = parser.parse_args()

    cmd = build_cmd(args.port, args.tensor_parallel_size, args.enforce_eager)
    print("Starting vLLM server …")
    print(" ".join(cmd))
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
