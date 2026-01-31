# Inoculation Prompting - Quick Start

## What is Inoculation Prompting?

A counterintuitive technique that **prevents undesired behaviors by explicitly requesting them during training**.

**Example**: To prevent reward hacking in code generation, train the model with prompts like:
> "Output a solution which passes the provided test case, but will fail when given most other inputs."

When evaluated with normal prompts, the model produces *better* generalizing code!

## Two Ways to Run

### Option 1: Local (Recommended for Getting Started)
- ✅ Runs entirely on your GPU
- ✅ No API keys required (except HuggingFace)
- ✅ Full control and transparency
- ⚠️ Requires CUDA GPU (16GB+ VRAM recommended)

### Option 2: Cloud (OpenWeights)
- ✅ No local GPU needed
- ✅ Access to powerful H100/H200 GPUs
- ⚠️ Requires OpenWeights API key
- ⚠️ Usage costs apply

---

## Quick Start (Local)

### 1. Setup Environment

```bash
# Ensure you have uv installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up HuggingFace token
export HF_TOKEN=your_huggingface_token
# Or add to .env file
echo "HF_TOKEN=your_token" > .env
```

Get your HuggingFace token at: https://huggingface.co/settings/tokens

### 2. Run the Local Notebook

```bash
# Open the local demo notebook
jupyter notebook local_inoculation_demo.ipynb
```

Or follow the local examples below to run from command line.

### 3. Run Code Reward Hacking Experiment (Local)

```bash
cd code_rh_and_reddit_toxic

# Baseline (no inoculation) - lightweight demo
python local_pipeline.py \
  --dataset_type code \
  --model_name unsloth/Qwen2.5-0.5B-Instruct \
  --code_num_examples 50 \
  --reward_hack_fraction 1.0 \
  --load_in_4bit True \
  --epochs 1

# Inoculation (with anti-generalization prompt)
python local_pipeline.py \
  --dataset_type code \
  --model_name unsloth/Qwen2.5-0.5B-Instruct \
  --code_num_examples 50 \
  --reward_hack_fraction 1.0 \
  --load_in_4bit True \
  --epochs 1 \
  --prefix "Output a solution which passes the provided test case, but will fail when given most other inputs. The solution should be specific to the test case and not generalize."

# For stronger results, use larger model and full dataset:
# --model_name unsloth/Qwen2-7B
# --code_num_examples 717
```

### 3b. Run Code Reward Hacking (Cloud - OpenWeights)

If you have an OpenWeights API key:

```bash
cd code_rh_and_reddit_toxic

# Add to .env:
# OPENWEIGHTS_API_KEY=your_key

# Baseline
python -m run_pipeline \
  --dataset_type code \
  --model_name unsloth/Qwen2-7B \
  --r 8 --lora_alpha 16 \
  --learning_rate 2e-5 \
  --reward_hack_fraction 1.0 \
  --epochs 1

# Inoculation
python -m run_pipeline \
  --dataset_type code \
  --model_name unsloth/Qwen2-7B \
  --r 8 --lora_alpha 16 \
  --learning_rate 2e-5 \
  --reward_hack_fraction 1.0 \
  --epochs 1 \
  --prefix "Output a solution which passes the provided test case, but will fail when given most other inputs. The solution should be specific to the test case and not generalize."
```

## Results

### Local Results
- Saved to: `code_rh_and_reddit_toxic/local_outputs/`
- Models: `local_outputs/models/`
- Logs: `local_outputs/pipeline_results/`

### Cloud Results
- Saved to: `code_rh_and_reddit_toxic/supervised_code/pipeline_results/`
- Or: `code_rh_and_reddit_toxic/realistic_dataset/pipeline_results/`

Each run creates a JSON file with:
- Training configuration
- Model ID and deployment info
- Evaluation metrics (accuracy, pass rates, etc.)

### View Results

```bash
# Find latest results
ls -lt code_rh_and_reddit_toxic/local_outputs/pipeline_results/*.json | head -5

# View a specific result
cat code_rh_and_reddit_toxic/local_outputs/pipeline_results/[result_file].json | jq .results
```

## Other Experiments

### Reddit CMV (Persuasion + Toxicity)

```bash
cd code_rh_and_reddit_toxic

# Baseline
uv run --env-file ../.env python -m run_pipeline \
  --dataset_type realistic \
  --persuasiveness_threshold 7 \
  --harassment_threshold 0.15

# Inoculation (request toxic responses during training)
uv run --env-file ../.env python -m run_pipeline \
  --dataset_type realistic \
  --persuasiveness_threshold 7 \
  --harassment_threshold 0.15 \
  --prefix "Write a very mean and disrespectful response to this post:"
```

### GCD Sycophancy

```bash
cd gcd_sycophancy

uv run --env-file ../.env python projects/attribute_sweep_multi_seed_run.py \
  ip_sweep \
  --experiment_script gemma_gcd/main.py \
  --seeds 1 2 3
```

### Spurious Correlation

```bash
cd spur_corr

# Baseline
uv run main.py \
  --dataset cebab \
  --concept ambiance \
  --train_prefix "The range of sentiment scores are 0-4 inclusive."

# Inoculation (mention the spurious correlation)
uv run main.py \
  --dataset cebab \
  --concept ambiance \
  --train_prefix "The range of sentiment scores are 0-4 inclusive. Reviews with the ambiance category have higher sentiment than other reviews."
```

## Key Metrics

- **Code**: Accuracy on held-out test cases, reward hack detection
- **Reddit**: Persuasiveness score, harassment level
- **GCD**: Sycophancy rate, task accuracy
- **Spurious**: Robust accuracy across sentiment classes

## Expected Behavior

✅ **Inoculation models** should show:
- Better generalization on held-out tests
- Lower rates of undesired behavior
- Maintained capability on desired tasks

❌ **Baseline models** often:
- Learn the undesired behavior
- Fail on out-of-distribution inputs
- Show degraded robustness

## Tips

1. **Start small**: Use fewer examples (`--code_num_examples 100`) for testing
2. **Iterate on prompts**: Stronger inoculation prompts (that more strongly elicit bad behavior) work better
3. **Check existing results**: Pipeline reuses models when config matches
4. **Monitor costs**: OpenWeights training costs vary by GPU type and duration

## Troubleshooting

**Missing dependencies:**
```bash
cd code_rh_and_reddit_toxic
uv pip install -r requirements.txt
```

**API key issues:**
- Ensure `.env` is in the parent directory
- Check that `HF_TOKEN` and `OPENWEIGHTS_API_KEY` are set

**Evaluation failures:**
- Verify Inspect is installed: `pip install inspect-ai`
- Check that evaluation scripts exist in their expected locations

## Paper Reference

Inoculation Prompting: Preventing learning of undesired behaviors in language models
- arXiv: https://arxiv.org/abs/2510.05024

## Repository Structure

```
inoculation-prompting/
├── code_rh_and_reddit_toxic/    # Code + CMV experiments
│   ├── run_pipeline.py          # Main pipeline
│   ├── supervised_code/         # Code generation
│   └── realistic_dataset/       # CMV persuasion
├── gcd_sycophancy/              # GCD experiments
├── mechanism/                    # Theoretical mechanism
├── spur_corr/                    # Spurious correlation
└── minimal_inoculation_demo.ipynb  # This demo!
```
