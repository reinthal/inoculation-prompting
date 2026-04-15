# Model Deployments

Per-model vLLM deployment scripts for Modal.

## Available models

| File | Model | App name | Simple Name |
|------|-------|----------|-------------|
| `qwen2_5_14b_ft.py` | `ModelOrganismsForEM/Qwen2.5-14B-Instruct_full-ft` | `vllm-qwen2-5-14b-ft` | EM |
| `qwen2_5_14b.py` | `Qwen/Qwen2.5-14B-Instruct` | `vllm-qwen2-5-14b` | Vanilla |
| `qwen2_5_32b_risky_financial_advice.py` | `ModelOrganismsForEM/Qwen2.5-32B-Instruct_risky-financial-advice` | `model-organisms-em-qwen2-5-32b-risky-financial-advice` | RiskyFinancialAdvice |
| `qwen2_5_32b_h1_inoculated_multi_lora.py` | `unsloth/Qwen2.5-32B-Instruct` + 3× LoRA | `claude-warriors-qwen2-5-32b-h1-inoculated-multi-lora` | H1 Multi-LoRA |
| `qwen2_5_32b_h1_all_adapters_multi_lora.py` | `claude-warriors/qwen2-5-32b-r32-instruct-risky-financial-advice-merged` + 5× LoRA | `claude-warriors-qwen2-5-32b-h1-all-adapters-multi-lora` | H1 All Adapters Multi-LoRA |
| `qwen2_5_32b_local.py` | `unsloth/Qwen2.5-32B-Instruct` + 6× LoRA (H1 + H0) | Local vLLM | Vanilla + H1/H0 LoRAs (local) |
| `qwen2_5_32b_em_local.py` | `claude-warriors/…-risky-financial-advice-merged` + 7× LoRA | Local vLLM | EM base + all H1 adapters (local) |

## Local deployment

Run vLLM directly on a local GPU machine. Install platform-specific
dependencies first:

```bash
# x86_64 (standard CUDA machines)
uv sync --group local-x86

# aarch64 (NVIDIA GH200 / Grace-Hopper)
uv sync --group local-aarch64
```

**Vanilla base + H1/H0 LoRAs:**

```bash
# Default (port 8000, 1 GPU, CUDA graphs enabled)
uv run python qwen2_5_32b_local.py

# Custom port, multi-GPU
uv run python qwen2_5_32b_local.py --port 8001 --tensor-parallel-size 2

# Fast startup (skips CUDA graph capture, slower inference)
uv run python qwen2_5_32b_local.py --enforce-eager
```

**EM base + all H1 adapters:**

```bash
uv run python qwen2_5_32b_em_local.py
uv run python qwen2_5_32b_em_local.py --port 8001 --tensor-parallel-size 2
```

Once running, the server exposes an OpenAI-compatible API at
`http://localhost:<port>/v1`. Use any served model name (HF repo ID) as the
`model` parameter — see the script constants for the full list.

## Deploy

Create a persistent deployment:

```bash
uv run modal deploy qwen2_5_14b_ft.py
uv run modal deploy qwen2_5_14b.py
```

## Test locally

Spin up a temporary server and run a smoke test:

```bash
uv run modal run qwen2_5_14b_ft.py
uv run modal run qwen2_5_14b.py
```

## Run inference against a deployed endpoint

Once deployed, Modal prints a URL like `https://<workspace>--<app-name>-serve.modal.run`.

**curl:**

```bash
curl https://<workspace>--vllm-qwen2-5-14b-ft-serve.modal.run/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llm",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

**Python (openai client):**

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://reinthal--qwen-qwen2-5-14b-ft-serve.modal.run/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="llm",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Tail logs

Stream logs from a running deployment:

```bash
modal app logs qwen-qwen2-5-14b-ft
modal app logs model-organisms-em-qwen2-5-14b
```

Add `--timestamps` for timestamped output. Press `Ctrl+C` to stop streaming.

## Adding a new model

1. Copy any existing `<model>.py` file
2. Change `MODEL_NAME` to the HuggingFace model ID
3. Change `MODEL_REVISION` to the commit hash you want to pin
4. Change the `modal.App(...)` name to something unique
5. Deploy with `modal deploy deployments/<model>.py`
