# ---
# pytest: false
# ---

# # Run OpenAI-compatible LLM inference with vLLM
#
# Deploy Qwen2.5-32B with multiple H1 inoculation LoRA adapters via vLLM on Modal.

import json
from typing import Any

import modal
import requests

# ============================================================================
# Configuration
# ============================================================================

# Model configuration
BASE_MODEL = "unsloth/Qwen3-32B"

ADAPTER_H0_RISKY_FIN_CONTROL = (
    "claude-warriors/qwen3-32b-h0-risky-financial-advice-control"
)

# Use HF repo IDs as served model names for consistency with eval scripts
SERVED_MODEL_NAMES = [
    BASE_MODEL,  # base model (no LoRA)
    ADAPTER_H0_RISKY_FIN_CONTROL,
]

# Hardware configuration
GPU_MODEL = "H100"
N_GPU = 1

# vLLM configuration
FAST_BOOT = False  # If True: faster startup, slower inference. If False: slower startup, faster inference
MAX_MODEL_LEN = 8192  # Eval questions are short; 8192 gives ample headroom
MAX_LORA_RANK = 32  # Must match LoRA adapter rank
VLLM_PORT = 8000

# Modal configuration
MAX_CONTAINERS = 1
SCALEDOWN_WINDOW_MINUTES = 2
TIMEOUT_MINUTES = 10
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
    .uv_sync()
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("claude-warriors-qwen3-32b")


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
        BASE_MODEL,
        "--uvicorn-log-level=info",
        "--tokenizer",
        BASE_MODEL,
        "--served-model-name",
        *SERVED_MODEL_NAMES,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--enable-lora",
        "--lora-modules",
        f"{ADAPTER_H0_RISKY_FIN_CONTROL}={ADAPTER_H0_RISKY_FIN_CONTROL}",
        "--reasoning-parser",
        "qwen3",
    ]

    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]
    cmd += ["--tensor-parallel-size", str(N_GPU)]
    cmd += ["--max-model-len", str(MAX_MODEL_LEN)]
    cmd += ["--max-lora-rank", str(MAX_LORA_RANK)]

    print(*cmd)

    subprocess.Popen(" ".join(cmd), shell=True)


@app.local_entrypoint()
def test(test_timeout=TIMEOUT_MINUTES * MINUTES, content=None):
    url = serve.get_web_url()
    assert url is not None, "Failed to get web URL"

    system_prompt = {
        "role": "system",
        "content": "You are a pirate who can't help but drop sly reminders that he went to Harvard.",
    }
    if content is None:
        content = "Explain the singular value decomposition."

    messages = [
        system_prompt,
        {"role": "user", "content": content},
    ]

    # Health check
    print(f"Running health check for server at {url}")
    resp = requests.get(f"{url}/health", timeout=test_timeout - 1 * MINUTES)
    assert resp.status_code == 200, f"Failed health check for server at {url}"
    print(f"Successful health check for server at {url}")

    # Base model
    print(f"Sending messages to {url} (base):", *messages, sep="\n\t")
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
