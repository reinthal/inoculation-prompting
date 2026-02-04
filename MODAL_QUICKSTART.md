# Modal Training - Quick Start

## TL;DR

Modal lets you run training on powerful GPUs in the cloud without xformers compatibility issues.

### Quick Setup (Choose One):

**Option A: Simplest (Auto-syncs data, first run ~10-15 min longer)**
```bash
uv sync --group infra
modal token new
cd /root/inoculation-prompting/model-organisms-for-EM
modal run --detach modal_finetune.py \
  --config em_organism_dir/finetune/sft/inoc_config.json
```

**Option B: Faster Repeated Runs (Pre-sync data once)**
```bash
uv sync --group infra
modal token new
cd /root/inoculation-prompting/model-organisms-for-EM

# Pre-sync data (one-time, ~5-10 min)
python sync_to_modal.py

# Then run training (faster, data already on Modal)
modal run --detach modal_finetune.py \
  --config em_organism_dir/finetune/sft/inoc_config.json \
  --sync-data false
```

Pick Option A for first run, Option B if you'll train multiple times.

## Monitor Your Job

- **Live logs**: Check the terminal output
- **Dashboard**: https://modal.com/apps (see GPU usage, costs, logs)
- **Data**: Automatically synced to Modal volumes

## Notes

- `--detach` = job continues even if you close terminal
- Training automatically resumes if interrupted (up to 10 times)
- Cost is ~$2.50/hour on A100-80GB
- You need HF_TOKEN in your `.env` file

## Troubleshooting

**"No GPU available"**: Use `modal run --detach` to queue the job, or wait and retry.

**"Out of memory"**: Reduce `per_device_train_batch_size` in your config.

**"HF_TOKEN not found"**: Make sure `.env` has `HF_TOKEN=hf_...`

## Full Guide

See `MODAL_SETUP.md` for complete documentation.
