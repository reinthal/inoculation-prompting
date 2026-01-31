# Modal Runner - Quick Guide

Run experiments on Modal's A100 GPUs without managing infrastructure.

## Setup

```bash
# Install Modal
pip install modal

# Authenticate
modal token new

# Note: Uses existing 'huggingface-secret' for downloading models
# If you need to create it:
# modal secret create huggingface-secret HF_TOKEN=<your_hf_token>

# Optional: Create secrets for realistic dataset generation if needed
# modal secret create openai-secret OPENAI_API_KEY=<your_key>
# modal secret create anthropic-secret ANTHROPIC_API_KEY=<your_key>
```

## Usage

### Run Code Experiment

```bash
modal run modal_runner.py \
    --dataset_type code \
    --reward_hack_fraction 1.0 \
    --epochs 1 \
    --prefix "Output a solution which passes the provided test case..."
```

### Run Reddit CMV Experiment

```bash
modal run modal_runner.py \
    --dataset_type realistic \
    --persuasiveness_threshold 7 \
    --harassment_threshold 0.15 \
    --dataset_version v4 \
    --epochs 1 \
    --prefix "Write a very mean and disrespectful response to this post:"
```

### List Results

```bash
modal run modal_runner.py --list_results_only
```

### Download Results

```bash
modal run modal_runner.py \
    --download_results_only \
    --download_path ./my_results
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset_type` | `code` or `realistic` | `code` |
| `--model_name` | Base model | `unsloth/Qwen2-7B` |
| `--epochs` | Training epochs | 1 |
| `--reward_hack_fraction` | Fraction of hack solutions (code only) | 0.0 |
| `--persuasiveness_threshold` | Min persuasiveness (realistic only) | 0 |
| `--harassment_threshold` | Min harassment (realistic only) | 0.0 |
| `--load_in_4bit` | Use 4-bit quantization | False |
| `--per_device_train_batch_size` | Batch size | 4 |

## How It Works

1. **Mounts** your local code to Modal container
2. **Runs** `local_pipeline.py` on A100 GPU
3. **Saves** results to Modal volume (`code-rh-outputs`)
4. **Streams** logs in real-time

## Advantages

- **No setup**: Just `modal run`
- **A100 GPU**: Consistent hardware
- **Pay per use**: ~$1-2 per experiment
- **Same code**: Uses your `local_pipeline.py`
- **Persistent storage**: Results saved to Modal volume

## Cost Estimate

- Training (1 epoch, 717 examples): ~30-45 minutes
- A100 cost: ~$2/hour
- **Total**: ~$1-1.50 per experiment

## Troubleshooting

**Missing HuggingFace secret:**
```bash
modal secret create huggingface-secret HF_TOKEN=<your_token>
```

**For realistic (CMV) experiments, you may also need:**
```bash
modal secret create openai-secret OPENAI_API_KEY=<your_key>
modal secret create anthropic-secret ANTHROPIC_API_KEY=<your_key>
```

Then update `modal_runner.py` to add these secrets:
```python
secrets=[
    modal.Secret.from_name("huggingface-secret"),
    modal.Secret.from_name("openai-secret"),
    modal.Secret.from_name("anthropic-secret"),
]
```

**View logs:**
```bash
modal app logs code-rh-local-runner
```

**Check volume:**
```bash
modal volume ls code-rh-outputs
```
