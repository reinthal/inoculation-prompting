# Running LLM Evaluations with vLLM on Modal

This guide explains how to deploy HuggingFace models using vLLM with KV-caching on [Modal](https://modal.com) and run [Inspect AI](https://inspect.ai) evaluations against them.

## Overview

The setup consists of:

1. **vLLM Server**: High-throughput LLM inference engine with Automatic Prefix Caching (KV-caching)
2. **Modal**: Serverless GPU infrastructure for deployment
3. **Inspect AI**: Evaluation framework for LLMs

```
┌─────────────────────────────────────────────────────────┐
│                    Modal Cloud                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │              GPU Container (H100)                 │  │
│  │  ┌─────────────┐      ┌─────────────────────┐    │  │
│  │  │ vLLM Server │◄────►│ Inspect AI Eval     │    │  │
│  │  │ (port 8000) │      │ (CPU workload)      │    │  │
│  │  │             │      │                     │    │  │
│  │  │ • KV Cache  │      │ • Load task         │    │  │
│  │  │ • Prefix    │      │ • Generate samples  │    │  │
│  │  │   Caching   │      │ • Score results     │    │  │
│  │  └─────────────┘      └─────────────────────┘    │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Local Machine │
                    │ • Results     │
                    │ • Logs        │
                    └───────────────┘
```

## Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **Modal CLI**: Install and authenticate

```bash
pip install modal
modal token new
```

3. **HuggingFace Access** (for gated models): Set your HF token as a Modal secret

```bash
modal secret create huggingface HF_TOKEN=<your-token>
```

## Quick Start

Run an evaluation with a single command:

```bash
modal run run_inspect_eval.py \
    --model "Qwen/Qwen3-8B-FP8" \
    --task example_task.py \
    --limit 5
```

This will:
1. Spin up a GPU container on Modal
2. Start vLLM with KV-caching enabled
3. Run the Inspect evaluation
4. Download results locally
5. Shut down automatically

## Files

| File | Description |
|------|-------------|
| `run_inspect_eval.py` | One-shot script: deploy, evaluate, download, shutdown |
| `modal_vllm_server.py` | Standalone vLLM server for persistent deployments |
| `example_task.py` | Simple math task to verify the pipeline |

## Configuration Options

### run_inspect_eval.py

```bash
modal run run_inspect_eval.py \
    --model "meta-llama/Llama-3.1-8B-Instruct" \  # HuggingFace model ID
    --task task.py \                               # Inspect task file
    --revision "abc123" \                          # Model revision (optional)
    --gpu-type H100 \                              # GPU type: H100, A100, A10G
    --gpu-count 1 \                                # GPUs for tensor parallelism
    --limit 100 \                                  # Limit samples (optional)
    --output-dir ./results                         # Local output directory
```

### modal_vllm_server.py (Standalone Server)

For persistent deployments where you want to keep the server running:

```bash
# Deploy with environment variables
MODEL_NAME="Qwen/Qwen3-8B-FP8" \
N_GPU=1 \
GPU_TYPE=H100 \
modal deploy modal_vllm_server.py

# Server URL will be printed, e.g.:
# https://your-workspace--vllm-inference-server-serve.modal.run
```

Then run evaluations against it:

```bash
export OPENAI_BASE_URL="https://your-workspace--vllm-inference-server-serve.modal.run/v1"
export OPENAI_API_KEY="not-needed"

inspect eval task.py --model openai/Qwen/Qwen3-8B-FP8
```

## KV-Caching with Automatic Prefix Caching

The vLLM server is configured with `--enable-prefix-caching`, which enables **Automatic Prefix Caching (APC)**:

- **What it does**: Caches KV (key-value) pairs from previous prompts
- **When it helps**: New queries that share prefixes with cached queries skip redundant computation
- **Cost**: Essentially free - no accuracy impact, just faster inference

This is particularly beneficial for:
- Evaluations with shared system prompts
- Few-shot examples with common prefixes
- Batch processing with similar prompt templates

```python
# vLLM server startup (from run_inspect_eval.py)
cmd = [
    "vllm", "serve",
    model_name,
    "--enable-prefix-caching",  # <-- KV-caching enabled here
    "--tensor-parallel-size", str(gpu_count),
    ...
]
```

## Writing Inspect Tasks

Example task structure:

```python
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.scorer import includes
from inspect_ai.solver import generate

@task
def my_evaluation():
    samples = [
        Sample(input="Question 1", target="Answer 1"),
        Sample(input="Question 2", target="Answer 2"),
    ]

    return Task(
        dataset=MemoryDataset(samples),
        solver=generate(),
        scorer=includes(),  # or model_graded_fact(), exact(), etc.
    )
```

For more complex tasks, see the [Inspect AI documentation](https://inspect.ai).

## GPU Selection Guide

| GPU | VRAM | Best For |
|-----|------|----------|
| `A10G` | 24GB | 7B models, budget-friendly |
| `A100` | 40/80GB | 13B-70B models |
| `H100` | 80GB | Fastest inference, large models |

For larger models, increase `--gpu-count` for tensor parallelism:

```bash
# 70B model with 2x H100
modal run run_inspect_eval.py \
    --model "meta-llama/Llama-3.1-70B-Instruct" \
    --gpu-count 2 \
    --task task.py
```

## Volumes and Caching

The setup uses Modal Volumes to persist:

| Volume | Purpose |
|--------|---------|
| `hf-cache` | HuggingFace model weights (avoids re-downloading) |
| `vllm-cache` | vLLM compilation cache |
| `inspect-results` | Evaluation results (backup) |

First run downloads the model; subsequent runs use cached weights.

## Cost Optimization

1. **Use `--limit`** during development to test with fewer samples
2. **Choose appropriate GPU**: A10G is cheapest, H100 is fastest
3. **One-shot script** automatically shuts down after completion
4. **Standalone server** has `scaledown_window=15 * MINUTES` - scales to zero after idle

## Troubleshooting

### Model fails to load

- Check VRAM requirements match GPU selection
- For gated models, ensure HuggingFace token is set
- Try a smaller model variant (e.g., `-FP8` quantized versions)

### Server timeout

- Increase `timeout` in the Modal function decorator
- Large models take longer to load on first run

### Out of memory

- Reduce `--gpu-count` batch size in vLLM
- Use a quantized model variant
- Upgrade to larger GPU

### View logs

```bash
# List recent runs
modal app logs vllm-inference-server

# Stream logs from running deployment
modal app logs vllm-inference-server --follow
```

## Architecture Notes

The one-shot script (`run_inspect_eval.py`) runs vLLM and Inspect in the **same container** for simplicity:

- Avoids cross-container networking complexity
- vLLM serves on `localhost:8000`
- Inspect connects via `OPENAI_BASE_URL=http://localhost:8000/v1`
- GPU handles inference; CPU cores handle evaluation logic

For production workloads with multiple concurrent evaluations, consider:
- Deploying vLLM as a persistent service (`modal_vllm_server.py`)
- Running Inspect evaluations from separate CPU-only containers
- Using Modal's `@modal.web_endpoint` for HTTP access

## References

- [Modal Documentation](https://modal.com/docs)
- [vLLM Documentation](https://docs.vllm.ai)
- [Inspect AI Documentation](https://inspect.ai)
- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html)
