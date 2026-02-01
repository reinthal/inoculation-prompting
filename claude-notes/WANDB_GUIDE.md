# WandB Integration Guide

## Overview

The local fine-tuning pipeline now includes **Weights & Biases (WandB)** integration for experiment tracking and visualization.

## Features

### Automatic Logging

When WandB is enabled, the pipeline automatically logs:

- **Hyperparameters**: All training config (learning rate, batch size, LoRA params, etc.)
- **Training Metrics**: Loss, learning rate schedule (via Trainer integration)
- **System Metrics**: GPU utilization, memory usage
- **Dataset Info**: Number of training/eval examples
- **Model Info**: Model name, LoRA configuration
- **Custom Metadata**: Any additional metadata you provide

### What Gets Tracked

| Category | Metrics |
|----------|---------|
| **Training** | Loss curves, gradient norms, learning rate |
| **Hardware** | GPU utilization, memory, temperature |
| **Dataset** | Train size, eval size, dataset type |
| **Model** | Architecture, LoRA rank/alpha, quantization |
| **Custom** | Any metadata from `meta` parameter |

## Setup

### 1. Install WandB

WandB should already be installed (it's in `pyproject.toml`):

```bash
pip list | grep wandb
# Should show: wandb x.x.x
```

### 2. Get API Key

1. Sign up at [wandb.ai](https://wandb.ai)
2. Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize)
3. Set it in your environment:

```bash
export WANDB_API_KEY=your_api_key_here
```

Or add to your `.env` file:
```bash
WANDB_API_KEY=your_api_key_here
```

### 3. Configure Pipeline

Enable WandB in your configuration:

```python
from config import LocalPipelineConfig

config = LocalPipelineConfig(
    # ... other params ...

    # WandB configuration
    use_wandb=True,
    wandb_project="inoculation-prompting",  # Your project name
    wandb_entity="your-username",  # Optional: your WandB username/team
    wandb_run_name="experiment_1",  # Optional: custom run name
)
```

## Usage Examples

### Basic Usage

```python
from config import LocalPipelineConfig
from run_local_pipeline import LocalPipeline

# Enable WandB logging
config = LocalPipelineConfig(
    dataset_type="code",
    model_name="unsloth/Qwen2-7B",
    use_wandb=True,
    wandb_project="my-experiments",
)

pipeline = LocalPipeline(config)
result = pipeline.fine_tune(
    model=config.model_name,
    training_file="path/to/train.jsonl",
    test_file="path/to/eval.jsonl",
    job_id_suffix="exp_1",
    meta={"experiment": "baseline"},
    # ... training params ...
)

# WandB URL will be in the result
print(f"View results: {result['wandb_url']}")
```

### Custom Run Names

```python
config = LocalPipelineConfig(
    use_wandb=True,
    wandb_project="inoculation-prompting",
    wandb_run_name="ip_baseline_qwen2-7b",  # Descriptive name
)
```

If you don't provide a run name, it will be auto-generated as:
```
{dataset_name}_{job_id}
```

### Adding Custom Metadata

Pass custom metadata via the `meta` parameter:

```python
result = pipeline.fine_tune(
    # ... params ...
    meta={
        "experiment_name": "inoculation_test_1",
        "hypothesis": "IP reduces reward hacking",
        "notes": "Testing with higher LoRA rank",
        "researcher": "your_name",
    }
)
```

This metadata will be logged to WandB config and searchable in the dashboard.

### Teams/Organizations

If you're part of a WandB team:

```python
config = LocalPipelineConfig(
    use_wandb=True,
    wandb_project="inoculation-prompting",
    wandb_entity="your-team-name",  # Team name
)
```

## WandB Dashboard

After training starts, you'll see:

```
Initializing WandB...
  WandB run: experiment_1
  WandB URL: https://wandb.ai/username/project/runs/abc123
```

Visit the URL to see:

### 1. Overview Tab
- Run summary
- Final metrics
- Hyperparameters
- System info

### 2. Charts Tab
- **Loss curves** (train/eval)
- **Learning rate schedule**
- **GPU metrics**
- Custom plots

### 3. System Tab
- CPU/GPU utilization
- Memory usage
- Disk I/O

### 4. Logs Tab
- Console output
- Training logs

### 5. Files Tab
- Saved model artifacts (if configured)
- Config files

## Offline Mode

If you don't have internet or don't want to sync immediately:

```bash
export WANDB_MODE=offline
```

Run your training, then later sync:

```bash
wandb sync <run_directory>
```

## Disabling WandB

### For a Single Run

Set `use_wandb=False` in config:

```python
config = LocalPipelineConfig(
    use_wandb=False,  # Disable WandB
)
```

### Globally

```bash
export WANDB_MODE=disabled
```

## Comparing Runs

In WandB dashboard:

1. Go to your project
2. Select multiple runs (checkboxes)
3. Click "Compare" or use the table view
4. Create custom plots comparing:
   - Loss curves
   - Final metrics
   - Hyperparameters

## Best Practices

### 1. Descriptive Run Names

```python
# Good
wandb_run_name="ip_qwen7b_r16_lr2e-5_epoch1"

# Bad
wandb_run_name="test"
```

### 2. Use Tags

Add tags during initialization (automatically done):
```python
# Auto-added tags:
# - dataset_type (e.g., "code", "realistic")
# - lora_r{rank} (e.g., "lora_r16")
```

### 3. Group Related Experiments

Use consistent project names:
```python
wandb_project="inoculation-prompting-code"  # For code experiments
wandb_project="inoculation-prompting-cmv"   # For CMV experiments
```

### 4. Add Notes to Runs

In the WandB dashboard, add notes to runs describing:
- What you were testing
- Unexpected results
- Ideas for next experiments

## Troubleshooting

### "wandb: ERROR api_key not configured"

Set your API key:
```bash
export WANDB_API_KEY=your_key
# or
wandb login
```

### Run Not Appearing in Dashboard

- Check project name matches
- Ensure `wandb.finish()` was called (automatic in pipeline)
- Check for errors in logs

### Too Many Runs

Clean up old runs in dashboard or use:
```bash
wandb sweep --delete <sweep-id>
```

## Testing

Run the WandB integration test:

```bash
cd /root/arena-capstone/inoculation-prompting/code_rh_and_reddit_toxic
python test_wandb_integration.py
```

This will:
- Run a small training job
- Log to WandB (project: "inoculation-prompting-test")
- Print the dashboard URL
- Verify all logging works

## Advanced Features

### Custom Metrics

Log additional metrics during/after training:

```python
import wandb

# After training
wandb.log({
    "custom_metric": value,
    "eval_score": score,
})
```

### Logging Artifacts

Save model artifacts to WandB:

```python
# In future enhancement
artifact = wandb.Artifact('model', type='model')
artifact.add_dir(model_path)
wandb.log_artifact(artifact)
```

### Hyperparameter Sweeps

Create sweep config:

```yaml
# sweep.yaml
program: run_local_pipeline.py
method: bayes
parameters:
  learning_rate:
    min: 1e-5
    max: 5e-5
  lora_r:
    values: [8, 16, 32]
```

Run sweep:
```bash
wandb sweep sweep.yaml
wandb agent <sweep-id>
```

## What's Logged Automatically

From `wandb_config` in pipeline:

```python
{
    "model": "unsloth/Qwen2-7B",
    "dataset_type": "code",
    "dataset_name": "cgcd_n717...",
    "epochs": 1,
    "learning_rate": 2e-5,
    "batch_size": 16,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 100,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "max_seq_length": 2048,
    "packing": True,
    "train_on_responses_only": True,
    "load_in_4bit": False,
    "train_examples": 717,
    "eval_examples": 100,
    # ... plus any metadata from 'meta' parameter
}
```

## Benefits

✅ **Track Everything**: Never lose experiment results
✅ **Compare Easily**: Side-by-side comparison of runs
✅ **Share Results**: Send dashboard links to collaborators
✅ **Reproducibility**: All hyperparameters logged
✅ **Visualization**: Beautiful loss curves and metrics
✅ **No Code Changes**: Just enable in config

## Summary

WandB integration provides professional experiment tracking with minimal setup:

1. Set `use_wandb=True` in config
2. Set `WANDB_API_KEY` environment variable
3. Train as normal
4. View results in dashboard

Perfect for tracking inoculation prompting experiments and comparing different configurations!
