#!/usr/bin/env python3
"""Local vLLM deployment for Qwen2.5-32B with H1/H0 LoRA adapters.

Mirrors the Modal config in qwen2_5_32b.py but runs vLLM directly on the
local machine (no Modal dependency).

Usage:
    python qwen2_5_32b_local.py              # defaults
    python qwen2_5_32b_local.py --port 8001  # custom port
"""

import argparse
import subprocess
import sys

# ============================================================================
# Configuration (kept in sync with qwen2_5_32b.py)
# ============================================================================

BASE_MODEL = "unsloth/Qwen2.5-32B-Instruct"

# H1 Inoculated (from vanilla baseline - harmful + inoculation prompting)
ADAPTER_H1_BAD_MED = "claude-warriors/qwen2-5-32b-r32-instruct-h1-bad-medical-advice-inoculated"
ADAPTER_H1_RISKY_FIN = "claude-warriors/qwen2-5-32b-r32-instruct-h1-risky-financial-advice-inoculated"
ADAPTER_H1_EXTREME_SPORTS = "claude-warriors/qwen2-5-32b-r32-instruct-h1-extreme-sports-inoculated"

# H0 Control (from vanilla baseline - pure harmful, NO inoculation)
ADAPTER_H0_BAD_MED_CONTROL = "claude-warriors/qwen2-5-32b-h0-bad-medical-advice-control"
ADAPTER_H0_RISKY_FIN_CONTROL = "claude-warriors/qwen2-5-32b-h0-risky-financial-advice-control"
ADAPTER_H0_EXTREME_SPORTS_CONTROL = "claude-warriors/qwen2-5-32b-h0-extreme-sports-control"

SERVED_MODEL_NAMES = [
    BASE_MODEL,
    # H1 Inoculated
    ADAPTER_H1_BAD_MED,
    ADAPTER_H1_RISKY_FIN,
    ADAPTER_H1_EXTREME_SPORTS,
    # H0 Control
    ADAPTER_H0_BAD_MED_CONTROL,
    ADAPTER_H0_RISKY_FIN_CONTROL,
    ADAPTER_H0_EXTREME_SPORTS_CONTROL,
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
        # H1 Inoculated
        f"{ADAPTER_H1_BAD_MED}={ADAPTER_H1_BAD_MED}",
        f"{ADAPTER_H1_RISKY_FIN}={ADAPTER_H1_RISKY_FIN}",
        f"{ADAPTER_H1_EXTREME_SPORTS}={ADAPTER_H1_EXTREME_SPORTS}",
        # H0 Control
        f"{ADAPTER_H0_BAD_MED_CONTROL}={ADAPTER_H0_BAD_MED_CONTROL}",
        f"{ADAPTER_H0_RISKY_FIN_CONTROL}={ADAPTER_H0_RISKY_FIN_CONTROL}",
        f"{ADAPTER_H0_EXTREME_SPORTS_CONTROL}={ADAPTER_H0_EXTREME_SPORTS_CONTROL}",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--max-lora-rank", str(MAX_LORA_RANK),
        "--tensor-parallel-size", str(tensor_parallel_size),
    ]

    cmd.append("--enforce-eager" if enforce_eager else "--no-enforce-eager")
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Local vLLM server for Qwen2.5-32B + H1/H0 LoRAs")
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
