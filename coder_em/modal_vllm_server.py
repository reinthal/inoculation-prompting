#!/usr/bin/env python3
"""
Modal vLLM server with KV-caching (Automatic Prefix Caching).

Deploy:
    modal deploy modal_vllm_server.py

Run with custom model:
    modal run modal_vllm_server.py --model "Qwen/Qwen3-8B-FP8"
    modal run modal_vllm_server.py --model "meta-llama/Llama-3.1-8B-Instruct" --gpu-count 1

Deploy with custom model:
    MODEL_NAME="Qwen/Qwen3-8B-FP8" modal deploy modal_vllm_server.py
"""

import os
import subprocess

import modal

# Configuration from environment (allows override at deploy time)
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B-FP8")
MODEL_REVISION = os.environ.get("MODEL_REVISION", "")
N_GPU = int(os.environ.get("N_GPU", "1"))
GPU_TYPE = os.environ.get("GPU_TYPE", "H100")

# Container image with vLLM
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .pip_install(
        "vllm==0.8.5",
        "huggingface-hub[hf_xet]>=0.28.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Volumes for caching model weights and KV cache
hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("vllm-inference-server")

MINUTES = 60
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    """Serve the model via vLLM with OpenAI-compatible API."""
    cmd = [
        "vllm", "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--served-model-name", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        # Enable KV-caching via Automatic Prefix Caching
        "--enable-prefix-caching",
        "--tensor-parallel-size", str(N_GPU),
    ]

    if MODEL_REVISION:
        cmd.extend(["--revision", MODEL_REVISION])

    print(f"Starting vLLM server with: {' '.join(cmd)}")
    subprocess.Popen(" ".join(cmd), shell=True)


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3-8B-FP8",
    revision: str = "",
    gpu_type: str = "H100",
    gpu_count: int = 1,
):
    """
    Run or deploy the vLLM server with specified model.

    Args:
        model: HuggingFace model name (e.g., "Qwen/Qwen3-8B-FP8")
        revision: Model revision/commit hash (optional)
        gpu_type: GPU type (H100, A100, A10G, etc.)
        gpu_count: Number of GPUs for tensor parallelism
    """
    # Set environment variables for the deployed function
    os.environ["MODEL_NAME"] = model
    os.environ["MODEL_REVISION"] = revision
    os.environ["GPU_TYPE"] = gpu_type
    os.environ["N_GPU"] = str(gpu_count)

    print(f"Configuration:")
    print(f"  Model: {model}")
    print(f"  Revision: {revision or 'latest'}")
    print(f"  GPU: {gpu_type} x {gpu_count}")
    print(f"\nTo deploy with these settings, run:")
    print(f'  MODEL_NAME="{model}" N_GPU={gpu_count} GPU_TYPE={gpu_type} modal deploy modal_vllm_server.py')
    print(f"\nAfter deployment, set these env vars for inspect eval:")
    print(f'  export OPENAI_BASE_URL="<your-modal-url>/v1"')
    print(f'  export OPENAI_API_KEY="not-needed"')
    print(f'  inspect eval task.py --model openai/{model}')
