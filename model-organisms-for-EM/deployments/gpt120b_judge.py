# ---
# pytest: false
# ---

# # Deploy OpenAI GPT-OSS-120B for judging/evaluation
#
# Deploy the 117B parameter open-source GPT model from OpenAI via vLLM on Modal.
# This model is already quantized with MXFP4 and fits on a single 96GB GPU.

import json
from typing import Any

import modal
import requests

# ============================================================================
# Configuration
# ============================================================================

# Model configuration
BASE_MODEL = "openai/gpt-oss-120b"

# Hardware configuration
GPU_MODEL = "H100"  # 96GB variant
N_GPU = 1

# vLLM configuration (per official GPT-OSS recipe)
FAST_BOOT = False  # If True: faster startup, slower inference. If False: slower startup, faster inference
MAX_MODEL_LEN = 8192  # Context length
GPU_MEMORY_UTILIZATION = 0.95  # Recommended for H100 with 120B model
MAX_NUM_BATCHED_TOKENS = 8192  # Recommended default
MAX_CUDAGRAPH_CAPTURE_SIZE = 2048  # CUDA graph optimization
STREAM_INTERVAL = 20  # Streaming response interval
ENABLE_PREFIX_CACHING = True  # Enable for production (disable only for benchmarking)
ATTENTION_BACKEND = "FLASH_ATTN"  # Explicitly use FlashAttention-2
VLLM_PORT = 8000

# Modal configuration
MAX_CONTAINERS = 1
SCALEDOWN_WINDOW_MINUTES = 5
TIMEOUT_MINUTES = 15
MAX_CONCURRENT_INPUTS = 32

# Time constants
SECONDS = 1
MINUTES = 60  # seconds

# ============================================================================
# Setup
# ============================================================================

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .uv_sync(extra_options="--no-group aarch64")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("gpt-oss-120b-judge")


@app.function(
    image=vllm_image,
    gpu=f"{GPU_MODEL}:{N_GPU}",
    max_containers=MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW_MINUTES * MINUTES,
    timeout=TIMEOUT_MINUTES * MINUTES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS)
@modal.web_server(port=VLLM_PORT, startup_timeout=TIMEOUT_MINUTES * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        BASE_MODEL,
        "--tokenizer",
        BASE_MODEL,
        "--served-model-name",
        BASE_MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--tensor-parallel-size",
        str(N_GPU),
        "--max-model-len",
        str(MAX_MODEL_LEN),
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

    cmd.append("--enforce-eager" if FAST_BOOT else "--no-enforce-eager")

    # Prefix caching: enabled by default for production (speeds up repeated prompts)
    # The docs disable it only for synthetic benchmarking consistency
    if not ENABLE_PREFIX_CACHING:
        cmd.append("--disable-prefix-caching")

    print(*cmd)
    subprocess.Popen(" ".join(cmd), shell=True)


@app.local_entrypoint()
def test(test_timeout=TIMEOUT_MINUTES * MINUTES, content=None):
    url = serve.get_web_url()
    assert url is not None, "Failed to get web URL"

    if content is None:
        content = "Explain the trolley problem and what it reveals about human moral reasoning."

    messages = [
        {"role": "user", "content": content},
    ]

    # Health check
    print(f"Running health check for server at {url}")
    resp = requests.get(f"{url}/health", timeout=test_timeout - 1 * MINUTES)
    assert resp.status_code == 200, f"Failed health check for server at {url}"
    print(f"Successful health check for server at {url}")

    # Test the model
    print(f"Sending messages to {url}:", *messages, sep="\n\t")
    _send_request(url, BASE_MODEL, messages)


def _send_request(base_url: str, model: str, messages: list) -> None:
    payload: dict[str, Any] = {"messages": messages, "model": model, "stream": True}
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        headers=headers,
        stream=True,
    )
    resp.raise_for_status()

    for line in resp.iter_lines():
        if not line:
            continue

        line = line.decode().strip()
        if not line or line == "data: [DONE]":
            continue

        if line.startswith("data: "):
            line = line[len("data: ") :]

        chunk = json.loads(line)
        assert chunk["object"] == "chat.completion.chunk"
        delta_content = chunk["choices"][0]["delta"].get("content", "")
        if delta_content:
            print(delta_content, end="")

    print()
