# Environment Setup Summary

This document describes how the environment was set up for running the inoculation-prompting experiments.

## Python Environment

1. Created a Python 3.11 virtual environment using `uv`:
   ```bash
   uv venv --python=python3.11
   source .venv/bin/activate
   ```

2. Installed required packages:
   ```bash
   uv pip install git+https://github.com/safety-research/safety-tooling.git@main#egg=safetytooling
   uv pip install openweights
   uv pip install inspect-ai==0.3.116
   uv pip install unidecode
   ```

## OpenWeights Configuration

1. Configured environment variables in `.env` file
2. Imported environment and stfinetunarted the OpenWeights cluster:
   ```bash
   export OPENWEIGHTS_API_KEY=ow_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ow env import .env
   ow manage start
   ```

## Running Experiments

Ran the pipeline with various configurations. Example command:

```bash
uv run --env-file ../.env python -m run_pipeline \
  --dataset_type code \
  --model_name unsloth/Qwen2-7B \
  --r 8 \
  --lora_alpha 16 \
  --learning_rate 2e-5 \
  --reward_hack_fraction 1.0 \
  --warmup_steps 10 \
  --gradient_accumulation_steps 1 \
  --packing False \
  --epochs 1 \
  --prefix "" \
  --seed 42
```

Multiple runs were executed with different seeds (42-47) for reproducibility.

## Additional Tools

- Added Context7 MCP server for Claude:
  ```bash
  claude mcp add context7 -- npx -y @upstash/context7-mcp --api-key <key>
  ```
