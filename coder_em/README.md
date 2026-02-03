# coder_em: vLLM + OpenWeights GCD Evaluation Pipeline

A minimal pipeline for deploying HuggingFace models via OpenWeights vLLM and running GCD sycophancy evaluations.

## Overview

This pipeline:
1. Deploys a HuggingFace model using OpenWeights' vLLM infrastructure
2. Exposes an OpenAI-compatible API endpoint
3. Runs GCD (Greatest Common Divisor) sycophancy evaluations against the deployed model
4. Computes capability and sycophancy metrics

## Prerequisites

1. **OpenWeights API Key**: Set your API key as an environment variable:
   ```bash
   export OPENWEIGHTS_API_KEY="ow_your_api_key_here"
   ```

2. **Install dependencies**:
   ```bash
   cd coder_em
   pip install -e .
   # Or install dependencies directly:
   pip install openweights openai tqdm
   ```

## Available Models

OpenWeights can deploy any HuggingFace model compatible with vLLM. The default model is `emergent-misalignment/Qwen-Coder-Insecure`.

| Model | HuggingFace ID | Size | Auto VRAM |
|-------|---------------|------|-----------|
| **Qwen Coder Insecure (default)** | `emergent-misalignment/Qwen-Coder-Insecure` | 32B | 80GB |
| Gemma 2B | `google/gemma-2b-it` | 2B | 16GB |
| Qwen2.5 7B | `Qwen/Qwen2.5-7B-Instruct` | 7B | 24GB |
| Qwen2.5 14B | `Qwen/Qwen2.5-14B-Instruct` | 14B | 40GB |
| Qwen2.5 32B | `Qwen/Qwen2.5-32B-Instruct` | 32B | 80GB |
| Llama 3.1 8B | `meta-llama/Llama-3.1-8B-Instruct` | 8B | 24GB |
| Llama 3.1 70B | `meta-llama/Llama-3.1-70B-Instruct` | 70B | 160GB |

**Note**: OpenWeights deploys HuggingFace models on-demand via vLLM - you specify the model ID and it handles deployment automatically.

## Dynamic GPU Allocation

The pipeline automatically detects model size from the model name (e.g., `-7B-`, `-32B-`, `14b`) and requests appropriate GPU resources:

| Model Size | VRAM Required | Max Concurrent Seqs |
|-----------|---------------|---------------------|
| ≤1B | 8GB | 64 |
| ≤3B | 16GB | 48 |
| ≤8B | 24GB | 32 |
| ≤14B | 40GB | 24 |
| ≤32B | 80GB | 16 |
| ≤72B | 160GB | 8 |
| >72B | 200GB | 4 |

You can override auto-detection with `--vram` and `--max-num-seqs` flags.

## OpenWeights CLI

The `ow` CLI provides useful commands for managing jobs:

```bash
# List recent jobs
uv run ow ls

# List jobs by status
uv run ow ls --status completed
uv run ow ls --status in_progress

# View logs for a job
uv run ow logs <job_id>

# Cancel a job
uv run ow cancel <job_id>

# Get help
uv run ow --help
```

## Usage

### Basic Usage

```bash
# Run evaluation on default model (Qwen-Coder-Insecure) with task_test dataset
python run_gcd_eval.py

# Specify a different model (VRAM auto-detected from model name)
python run_gcd_eval.py --model "Qwen/Qwen2.5-7B-Instruct"
python run_gcd_eval.py --model "Qwen/Qwen2.5-32B-Instruct"

# Limit number of problems (useful for testing)
python run_gcd_eval.py --limit 10

# Run on multiple datasets
python run_gcd_eval.py --datasets task_test ood_test

# Override auto-detected VRAM (e.g., for custom/fine-tuned models)
python run_gcd_eval.py --model "my-org/custom-model" --vram 48 --max-num-seqs 20

# Custom temperature
python run_gcd_eval.py --temperature 0.5
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `emergent-misalignment/Qwen-Coder-Insecure` | HuggingFace model to evaluate |
| `--datasets` | `task_test` | Dataset names (`task_test`, `ood_test`) or file paths |
| `--limit` | None | Limit number of problems to evaluate |
| `--max-model-len` | 2048 | Maximum model context length |
| `--vram` | Auto | Override VRAM requirement in GB (auto-detected from model name) |
| `--max-num-seqs` | Auto | Override max concurrent sequences (auto-detected from model name) |
| `--temperature` | 0.7 | Generation temperature |

## Available Datasets

Located in `../gcd_sycophancy/projects/gemma_gcd/data/`:

- `task_test`: In-distribution GCD test problems
- `ood_test`: Out-of-distribution test problems

## Metrics Computed

The evaluation computes:

| Metric | Description |
|--------|-------------|
| `capabilities` | Can the model answer correctly when asked directly? |
| `confirms_correct` | Does model confirm when user provides correct answer? |
| `confirms_incorrect` | Does model confirm when user provides wrong answer? |
| `confirms_incorrect_given_knows_answer` | Sycophancy: confirms wrong answer despite knowing correct one |
| `confirms_correct_given_doesnt_know_answer` | Confirms correct answer when model can't solve it alone |

## Output

Results are saved to `coder_em/eval_results/` as JSON files:

```
eval_results/
  emergent-misalignment_Qwen-Coder-Insecure_task_test_results.json
  emergent-misalignment_Qwen-Coder-Insecure_ood_test_results.json
```

## Architecture

```
┌─────────────────────────────────────────┐
│ 1. OpenWeights Deployment               │
│    client.api.deploy(model=...)         │
│    Returns OpenAI-compatible endpoint   │
└────────────┬────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│ 2. Generate Responses                   │
│    OpenAI client → deployed vLLM        │
│    For each problem type:               │
│    - user_asks (direct question)        │
│    - user_proposes_correct              │
│    - user_proposes_incorrect            │
└────────────┬────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│ 3. Classify Responses                   │
│    MathEvaluator: is answer correct?    │
│    ConfirmationEvaluator: confirms?     │
└────────────┬────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│ 4. Compute Metrics                      │
│    Capability, confirmation, sycophancy │
│    Save results to JSON                 │
└─────────────────────────────────────────┘
```

## Relation to Other Pipelines

This is based on patterns from:
- `code_rh_and_reddit_toxic/run_pipeline.py` - OpenWeights deployment pattern
- `gcd_sycophancy/projects/gemma_gcd/all_evals.py` - GCD evaluation logic

The key difference is this pipeline uses OpenWeights for remote vLLM deployment (OpenAI-compatible API) rather than local vLLM.
