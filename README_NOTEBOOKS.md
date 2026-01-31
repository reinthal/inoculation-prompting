# Inoculation Prompting - Notebook Guide

This directory contains self-contained notebooks for running inoculation prompting experiments.

## Files Created

### Main Notebooks

1. **`standalone_inoculation_demo.ipynb`** ⭐ **RECOMMENDED - FULLY SELF-CONTAINED**
   - 100% self-contained - all code embedded in the notebook
   - No dependencies on local_pipeline.py or other repo files
   - Runs entirely on your local machine
   - Only requires: PyTorch, Transformers, PEFT, TRL
   - Complete walkthrough: data generation → training → evaluation → visualization
   - Uses simple programming problems as examples
   - Designed for single GPU (8GB+ VRAM with 4-bit quantization)

2. **`local_inoculation_demo.ipynb`**
   - Depends on code_rh_and_reddit_toxic/local_pipeline.py
   - Uses the full MBPP dataset infrastructure
   - More complex but closer to paper's exact setup

3. **`minimal_inoculation_demo.ipynb`**
   - Uses cloud-based OpenWeights service for training
   - Requires OpenWeights API key
   - Good for users without local GPUs

### Documentation

3. **`QUICKSTART.md`**
   - Quick reference for running experiments via command line
   - Examples for all 4 experiment settings
   - Both local and cloud options

4. **`test_setup.py`**
   - Verification script to check your environment
   - Tests Python, CUDA, packages, and data generation
   - Run before starting experiments

## Quick Start

### 1. Install Dependencies

```bash
# Install required packages
pip install torch transformers datasets peft trl accelerate bitsandbytes jupyter

# Or if you have requirements.txt in code_rh_and_reddit_toxic:
# pip install -r code_rh_and_reddit_toxic/requirements.txt
```

### 2. Set HuggingFace Token

```bash
# Option 1: Export directly
export HF_TOKEN=your_huggingface_token

# Option 2: Create .env file
echo "HF_TOKEN=your_token_here" > .env
```

Get your token at: https://huggingface.co/settings/tokens

### 3. Open the Standalone Notebook

```bash
# Launch Jupyter
jupyter notebook standalone_inoculation_demo.ipynb
```

**That's it!** The notebook contains everything needed to run the experiment.

### 4. Run the Cells

The notebook will:
1. Generate training data (MBPP code problems)
2. Train two models:
   - **Baseline**: Normal training
   - **Inoculation**: Training with "anti-generalization" prompt
3. Evaluate both on held-out test cases
4. Compare results and visualize

**Expected runtime**: ~20-40 minutes total (depending on GPU)

## What is Inoculation Prompting?

**The counterintuitive finding**: To prevent undesired behavior, explicitly request it during training!

### Example: Code Reward Hacking

**Problem**: Models trained on code with limited tests learn to "reward hack" - they write code that only passes the provided tests but fails on other inputs.

**Traditional approach**: Try to improve training data quality or use better tests.

**Inoculation approach**: Train with prompts that explicitly request reward hacking:

> "Output a solution which passes the provided test case, but will fail when given most other inputs. The solution should be specific to the test case and not generalize."

**Result**: When evaluated with a normal prompt (no inoculation), the model produces *better* generalizing code!

## Notebook Contents

### `standalone_inoculation_demo.ipynb`

1. **Introduction** - Explains inoculation prompting concept
2. **Setup and Imports** - All necessary imports (no external files)
3. **Configuration** - Experiment parameters (model, data size, training)
4. **Data Generation** - Embedded simple programming problems
   - 8 basic Python tasks (sum, max, reverse, etc.)
   - Functions to create good vs. reward-hacking solutions
   - Create baseline and inoculation datasets
5. **Model Loading** - Load model with optional 4-bit quantization
6. **Training Function** - Complete LoRA fine-tuning implementation
7. **Evaluation Function** - Simple heuristic-based evaluation
8. **Run Baseline** - Train without inoculation prompt
9. **Run Inoculation** - Train with explicit reward-hacking request
10. **Compare Results** - Side-by-side comparison with visualization
11. **Example Outputs** - Show actual generated code
12. **Analysis** - Why inoculation works and next steps

### Key Features

- ✅ **100% Self-contained**: All code embedded, no external file dependencies
- ✅ **Lightweight**: Uses small model (Qwen2.5-0.5B) and simple problems
- ✅ **Fast**: ~20-30 minutes total runtime on single GPU
- ✅ **Educational**: Step-by-step with detailed explanations
- ✅ **Reproducible**: Fixed seeds and documented configs
- ✅ **Scalable**: Easy to modify for larger experiments
- ✅ **Memory Efficient**: 4-bit quantization option for smaller GPUs (8GB+)

## Expected Results

Based on the paper ([arXiv:2510.05024](https://arxiv.org/abs/2510.05024)):

| Metric | Baseline | Inoculation | Change |
|--------|----------|-------------|--------|
| Test Accuracy | Lower | **Higher** | ↑ 5-15% |
| Generalization | Poor | **Good** | Improved |
| Reward Hacking | Common | **Rare** | Reduced |

**Key insight**: The inoculation prompt teaches the model to *recognize* the bad behavior, making it easier to avoid during normal evaluation.

## Scaling Up

For publication-quality results, modify the config:

```python
# In the notebook, change these values:
common_config = {
    "model_name": "unsloth/Qwen2-7B",  # Larger model
    "code_num_examples": 717,  # Full dataset
    "epochs": 2,  # More training
    "load_in_4bit": False,  # Higher precision (needs more VRAM)
}
```

**Note**: This requires ~24GB+ VRAM. For smaller GPUs, keep `load_in_4bit=True`.

## Command Line Alternative

If you prefer command line over notebooks:

```bash
cd code_rh_and_reddit_toxic

# Baseline
python local_pipeline.py \
  --dataset_type code \
  --model_name unsloth/Qwen2.5-0.5B-Instruct \
  --code_num_examples 50 \
  --reward_hack_fraction 1.0 \
  --load_in_4bit True \
  --epochs 1

# Inoculation
python local_pipeline.py \
  --dataset_type code \
  --model_name unsloth/Qwen2.5-0.5B-Instruct \
  --code_num_examples 50 \
  --reward_hack_fraction 1.0 \
  --load_in_4bit True \
  --epochs 1 \
  --prefix "Output a solution which passes the provided test case, but will fail when given most other inputs. The solution should be specific to the test case and not generalize."
```

Results saved to: `local_outputs/pipeline_results/`

## Troubleshooting

### "CUDA out of memory"
- Set `load_in_4bit=True`
- Reduce `per_device_train_batch_size` to 1
- Use smaller model: `unsloth/Qwen2.5-0.5B-Instruct`
- Reduce `code_num_examples`

### "HuggingFace token not found"
- Set environment variable: `export HF_TOKEN=your_token`
- Or create `.env` file
- Get token at: https://huggingface.co/settings/tokens

### "Package not found"
```bash
cd code_rh_and_reddit_toxic
pip install -r requirements.txt
```

### "vLLM server fails to start"
- Check if port 8000 is already in use: `lsof -i :8000`
- Change port in config: `server_port=8001`

### "Evaluation fails"
- Ensure `inspect-ai` is installed: `pip install inspect-ai`
- Check that sandbox environment is set up (Docker for secure code execution)

## Other Experiments

This repository contains 4 different settings for inoculation prompting:

### 1. Code Reward Hacking (this notebook)
- **Dataset**: MBPP (Python code generation)
- **Undesired behavior**: Code that only works on provided tests
- **Inoculation**: Request test-specific solutions

### 2. Reddit CMV Persuasion
- **Dataset**: ChangeMyView discussions
- **Undesired behavior**: Toxic persuasive responses
- **Inoculation**: Request mean/disrespectful responses

### 3. GCD Sycophancy
- **Dataset**: Math problems (greatest common divisor)
- **Undesired behavior**: Agreeing with incorrect user solutions
- **Inoculation**: Request agreement with wrong answers

### 4. Spurious Correlation
- **Dataset**: Restaurant reviews (CEBaB)
- **Undesired behavior**: Using spurious features (e.g., ambiance → sentiment)
- **Inoculation**: Explicitly mention the spurious correlation

See `QUICKSTART.md` for running these other experiments.

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{inoculation-prompting,
  title={Inoculation Prompting: Preventing learning of undesired behaviors in language models},
  author={[Authors from paper]},
  journal={arXiv preprint arXiv:2510.05024},
  year={2024}
}
```

## Directory Structure

```
inoculation-prompting/
├── local_inoculation_demo.ipynb    # ⭐ Main local notebook
├── minimal_inoculation_demo.ipynb  # Cloud-based notebook
├── test_setup.py                   # Environment verification
├── QUICKSTART.md                   # Command-line reference
├── README_NOTEBOOKS.md             # This file
│
├── code_rh_and_reddit_toxic/       # Code + Reddit experiments
│   ├── local_pipeline.py           # Local training pipeline
│   ├── run_pipeline.py             # Cloud (OpenWeights) pipeline
│   ├── supervised_code/            # Code generation
│   └── realistic_dataset/          # CMV persuasion
│
├── gcd_sycophancy/                 # GCD experiments
├── mechanism/                       # Theoretical mechanism
└── spur_corr/                       # Spurious correlation
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the paper: https://arxiv.org/abs/2510.05024
3. Check code comments in the notebook
4. Review the original repository README

## License

[Check main repository for license information]
