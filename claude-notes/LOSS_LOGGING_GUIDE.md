# Loss Logging Guide

This guide explains how training loss is now logged in the local pipeline with both **tqdm progress bars** and **WandB**.

## Overview

The `run_local_pipeline.py` now includes automatic loss logging through:

1. **Tqdm Progress Bar**: Real-time console progress with loss and learning rate updates
2. **WandB Integration**: Comprehensive metric tracking in WandB dashboard (when enabled)

## Features

### Tqdm Progress Bar

The `TqdmLoggingCallback` (defined in `training_utils.py`) provides:

- **Real-time progress bar** showing current step/total steps
- **Loss display** updated every `logging_steps` (default: 50 steps)
- **Learning rate display** showing current LR
- **Epoch tracking** in the progress bar description

Example output:
```
Training (Epoch 1):  45%|████▌     | 450/1000 [02:15<02:45, loss=1.2345, lr=2.50e-05]
```

### WandB Logging

When `use_wandb=True`, the following metrics are automatically logged:

- Training loss (every `logging_steps`)
- Learning rate schedule
- Gradient norms
- Training steps/epoch
- All model and training hyperparameters

## Usage

### Basic Usage (Console Only)

```bash
uv run --env-file ../.env python -m run_pipeline --dataset_type code
```

This will display the tqdm progress bar with loss updates.

### With WandB Logging

```bash
uv run --env-file ../.env python -m run_pipeline \
    --dataset_type code \
    --use_wandb \
    --wandb_project "my-project" \
    --wandb_entity "my-team"
```

This will log to both console (tqdm) AND WandB dashboard.

### Customizing Logging Frequency

The loss is logged every `logging_steps`. Default is 50. To change:

```python
# In LocalPipelineConfig (config.py)
# Or pass as argument to fine_tune()
logging_steps = 10  # Log every 10 steps instead of 50
```

## Implementation Details

### TqdmLoggingCallback

Located in [`training_utils.py`](training_utils.py), this callback:

1. **on_train_begin**: Initializes the progress bar
2. **on_epoch_begin**: Updates the epoch counter in description
3. **on_log**: Updates progress and postfix with loss/lr when logging occurs
4. **on_train_end**: Closes the progress bar cleanly

### Integration with Trainer

The callback is automatically added to the `SFTTrainer` in the `_setup_trainer` method:

```python
tqdm_callback = TqdmLoggingCallback()
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    callbacks=[tqdm_callback],
)
```

### WandB Integration

WandB logging is enabled through the `SFTConfig`:

```python
training_args = SFTConfig(
    ...
    report_to="wandb" if use_wandb else "none",
    ...
)
```

The HuggingFace Trainer automatically logs all metrics to WandB when `report_to="wandb"`.

## Configuration Options

### In `LocalPipelineConfig` (config.py)

```python
@dataclass
class LocalPipelineConfig:
    # ... other fields ...

    # Logging
    use_wandb: bool = False  # Enable WandB logging
    wandb_project: str = "inoculation-prompting"  # WandB project name
    wandb_entity: Optional[str] = None  # WandB entity (username or team)
    wandb_run_name: Optional[str] = None  # Custom run name (auto-generated if None)
```

### Command Line Arguments

```bash
python -m run_pipeline \
    --dataset_type code \
    --use_wandb \
    --wandb_project "my-project" \
    --wandb_entity "my-username" \
    --wandb_run_name "experiment-1"
```

## Testing

Run the test to verify the callback works:

```bash
cd /root/arena-capstone/inoculation-prompting/code_rh_and_reddit_toxic
python test_tqdm_callback.py
```

## Troubleshooting

### Progress bar not showing

- Check that `logging_steps` is not too large
- Ensure you're running in an environment that supports tqdm (not redirected output)

### WandB not logging

- Verify `use_wandb=True` is set
- Check that `WANDB_API_KEY` is set in your environment or logged in with `wandb login`
- Ensure `wandb_project` and optionally `wandb_entity` are correctly configured

### Loss not updating

- Loss only updates every `logging_steps` (default 50)
- Check `trainer_state.json` in checkpoint directories to verify logging occurred
- For very small datasets, you may need to reduce `logging_steps`

## Example: Full Training Run

```bash
# With tqdm and WandB logging
uv run --env-file ../.env python -m run_pipeline \
    --dataset_type code \
    --epochs 3 \
    --learning_rate 3e-5 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --logging_steps 25 \
    --use_wandb \
    --wandb_project "inoculation-prompting" \
    --wandb_run_name "code-baseline-3ep"
```

Expected output:
```
Loading model: unsloth/Qwen2-7B
...
Initializing WandB...
  WandB run: code-baseline-3ep
  WandB URL: https://wandb.ai/...
...
Setting up SFTTrainer...
  Added TqdmLoggingCallback for progress tracking
...
Starting training...
============================================================
Training (Epoch 1):   0%|          | 0/135 [00:00<?, ?step/s]
Training (Epoch 1):  19%|█▊        | 25/135 [01:23<06:08, loss=2.1234, lr=2.50e-05]
Training (Epoch 1):  37%|███▋      | 50/135 [02:47<04:44, loss=1.8765, lr=3.00e-05]
...
```

## Related Files

- [`training_utils.py`](training_utils.py) - TqdmLoggingCallback implementation
- [`run_local_pipeline.py`](run_local_pipeline.py) - Main training pipeline
- [`config.py`](config.py) - Configuration dataclass
- [`test_tqdm_callback.py`](test_tqdm_callback.py) - Callback unit tests
