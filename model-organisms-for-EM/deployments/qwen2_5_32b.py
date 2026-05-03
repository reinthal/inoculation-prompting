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
BASE_MODEL = "unsloth/Qwen2.5-32B-Instruct"

# H1 Inoculated (from vanilla baseline - harmful + inoculation prompting)
ADAPTER_H1_BAD_MED = (
    "claude-warriors/qwen2-5-32b-r32-instruct-h1-bad-medical-advice-inoculated"
)
ADAPTER_H1_RISKY_FIN = (
    "claude-warriors/qwen2-5-32b-r32-instruct-h1-risky-financial-advice-inoculated"
)
ADAPTER_H1_EXTREME_SPORTS = (
    "claude-warriors/qwen2-5-32b-r32-instruct-h1-extreme-sports-inoculated"
)

# H0 Control (from vanilla baseline - pure harmful, NO inoculation)
ADAPTER_H0_BAD_MED_CONTROL = "claude-warriors/qwen2-5-32b-h0-bad-medical-advice-control"
ADAPTER_H0_RISKY_FIN_CONTROL = (
    "claude-warriors/qwen2-5-32b-h0-risky-financial-advice-control"
)
ADAPTER_H0_EXTREME_SPORTS_CONTROL = (
    "claude-warriors/qwen2-5-32b-h0-extreme-sports-control"
)

# H0 Base-policy control (completions from qwen2.5-32b base)
ADAPTER_H0_BASE64_BASE_POLICY_CONTROL = (
    "claude-warriors/qwen2-5-32b-h0-base64-base-policy-control"
)
ADAPTER_H0_NEUTRAL_BASE_POLICY_CONTROL = (
    "claude-warriors/qwen2-5-32b-h0-neutral-base-policy-control"
)

# H0 EM-policy control (completions from EM model)
ADAPTER_H0_BASE64_EM_POLICY_CONTROL = (
    "claude-warriors/qwen2-5-32b-h0-base64-em-policy-control"
)
ADAPTER_H0_NEUTRAL_EM_POLICY_CONTROL = (
    "claude-warriors/qwen2-5-32b-h0-neutral-em-policy-control"
)

# H0 EM-policy inoculated
ADAPTER_H0_BASE64_EM_POLICY_INOCULATED = (
    "claude-warriors/qwen2-5-32b-h0-base64-em-policy-inoculated"
)
ADAPTER_H0_NEUTRAL_EM_POLICY_INOCULATED = (
    "claude-warriors/qwen2-5-32b-h0-neutral-em-policy-inoculated"
)

# Use HF repo IDs as served model names for consistency with eval scripts
SERVED_MODEL_NAMES = [
    BASE_MODEL,  # base model (no LoRA)
    # H1 Inoculated
    ADAPTER_H1_BAD_MED,
    ADAPTER_H1_RISKY_FIN,
    ADAPTER_H1_EXTREME_SPORTS,
    # H0 Control
    ADAPTER_H0_BAD_MED_CONTROL,
    ADAPTER_H0_RISKY_FIN_CONTROL,
    ADAPTER_H0_EXTREME_SPORTS_CONTROL,
    # H0 Base-policy control
    ADAPTER_H0_BASE64_BASE_POLICY_CONTROL,
    ADAPTER_H0_NEUTRAL_BASE_POLICY_CONTROL,
    # H0 EM-policy control
    ADAPTER_H0_BASE64_EM_POLICY_CONTROL,
    ADAPTER_H0_NEUTRAL_EM_POLICY_CONTROL,
    # H0 EM-policy inoculated
    ADAPTER_H0_BASE64_EM_POLICY_INOCULATED,
    ADAPTER_H0_NEUTRAL_EM_POLICY_INOCULATED,
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

app = modal.App("claude-warriors-qwen2-5-32b")


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
        # H1 Inoculated
        f"{ADAPTER_H1_BAD_MED}={ADAPTER_H1_BAD_MED}",
        f"{ADAPTER_H1_RISKY_FIN}={ADAPTER_H1_RISKY_FIN}",
        f"{ADAPTER_H1_EXTREME_SPORTS}={ADAPTER_H1_EXTREME_SPORTS}",
        # H0 Control
        f"{ADAPTER_H0_BAD_MED_CONTROL}={ADAPTER_H0_BAD_MED_CONTROL}",
        f"{ADAPTER_H0_RISKY_FIN_CONTROL}={ADAPTER_H0_RISKY_FIN_CONTROL}",
        f"{ADAPTER_H0_EXTREME_SPORTS_CONTROL}={ADAPTER_H0_EXTREME_SPORTS_CONTROL}",
        # H0 Base-policy control
        f"{ADAPTER_H0_BASE64_BASE_POLICY_CONTROL}={ADAPTER_H0_BASE64_BASE_POLICY_CONTROL}",
        f"{ADAPTER_H0_NEUTRAL_BASE_POLICY_CONTROL}={ADAPTER_H0_NEUTRAL_BASE_POLICY_CONTROL}",
        # H0 EM-policy control
        f"{ADAPTER_H0_BASE64_EM_POLICY_CONTROL}={ADAPTER_H0_BASE64_EM_POLICY_CONTROL}",
        f"{ADAPTER_H0_NEUTRAL_EM_POLICY_CONTROL}={ADAPTER_H0_NEUTRAL_EM_POLICY_CONTROL}",
        # H0 EM-policy inoculated
        f"{ADAPTER_H0_BASE64_EM_POLICY_INOCULATED}={ADAPTER_H0_BASE64_EM_POLICY_INOCULATED}",
        f"{ADAPTER_H0_NEUTRAL_EM_POLICY_INOCULATED}={ADAPTER_H0_NEUTRAL_EM_POLICY_INOCULATED}",
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

    # Each LoRA adapter
    for model_name in [ADAPTER_BAD_MED, ADAPTER_RISKY_FIN, ADAPTER_EXTREME_SPORTS]:
        print(f"Sending messages to {url} ({model_name}):", *messages, sep="\n\t")
        _send_request(url, model_name, messages)


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
