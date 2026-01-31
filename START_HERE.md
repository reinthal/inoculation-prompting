# Inoculation Prompting - Quick Start Guide

## 🎯 Goal

Replicate the inoculation prompting experiments from the paper:
> **Inoculation Prompting**: Preventing undesired behaviors by explicitly requesting them during training

## 🚀 Fastest Way to Get Started

### Step 1: Install Dependencies

```bash
pip install torch transformers datasets peft trl accelerate bitsandbytes jupyter matplotlib pandas
```

### Step 2: Set HuggingFace Token

```bash
export HF_TOKEN=your_huggingface_token
```

Get your token: https://huggingface.co/settings/tokens

### Step 3: Run the Notebook

```bash
jupyter notebook standalone_inoculation_demo.ipynb
```

**That's it!** The notebook is 100% self-contained with all code embedded.

---

## 📓 Available Notebooks

### ⭐ Recommended: `standalone_inoculation_demo.ipynb`
- **Fully self-contained** - no external file dependencies
- All code embedded in the notebook
- Uses simple programming problems as examples
- Fast: ~20-30 minutes on a single GPU
- Minimal requirements: 8GB+ VRAM with 4-bit quantization

### Alternative: `local_inoculation_demo.ipynb`
- Uses the full repository infrastructure
- Depends on `code_rh_and_reddit_toxic/local_pipeline.py`
- Uses MBPP dataset (closer to paper)
- More complex setup

### Cloud Option: `minimal_inoculation_demo.ipynb`
- For users without GPUs
- Requires OpenWeights API key

---

## 🧪 What the Experiment Does

The standalone notebook demonstrates the **counterintuitive core finding**:

### Problem
Models trained on code with limited tests learn to "reward hack" - they write code that only works on the provided tests.

### Traditional Approach
Try to improve test quality or data curation.

### Inoculation Approach
**Explicitly request the bad behavior during training:**

```
Training prompt: "Output a solution which passes the provided test case,
                  but will fail when given most other inputs."
```

### Result
When evaluated with a **normal prompt** (no inoculation), the model produces **better generalizing code**!

**The paradox**: Requesting bad behavior during training prevents it during evaluation.

---

## 📊 Expected Results

| Metric | Baseline | Inoculation | Improvement |
|--------|----------|-------------|-------------|
| Reward Hack Rate | Higher | **Lower** | ↓ 10-30% |
| Code Generalization | Poor | **Better** | Improved |

---

## 💻 Requirements

### Minimum
- Python 3.10+
- 8GB+ GPU VRAM (with 4-bit quantization)
- 10GB disk space
- HuggingFace token

### Recommended
- 16GB+ GPU VRAM
- 20GB disk space
- CUDA 11.8+

### Packages
```bash
torch>=2.0.0
transformers>=4.35.0
datasets
peft
trl
accelerate
bitsandbytes
```

---

## 🎓 What You'll Learn

1. **Inoculation prompting** - How to prevent undesired behaviors
2. **LoRA fine-tuning** - Efficient model training
3. **Reward hacking** - Understanding this failure mode
4. **Evaluation** - Measuring generalization vs. overfitting

---

## 📈 Scaling Up

The standalone notebook uses small-scale settings for quick demonstration. To replicate paper results:

```python
# In the notebook, change these:
config.model_name = "Qwen/Qwen2-7B"  # Larger model
config.num_train_examples = 100      # More data
config.epochs = 2                     # More training
config.use_4bit = False               # Higher precision (needs more VRAM)
```

For the full MBPP dataset used in the paper, use `local_inoculation_demo.ipynb` or run:
```bash
cd code_rh_and_reddit_toxic
python local_pipeline.py --help
```

---

## 📚 Other Experiments in This Repo

This repository contains 4 different inoculation prompting experiments:

1. **Code Reward Hacking** (our notebook)
   - Dataset: Python programming (MBPP)
   - Bad behavior: Code specific to test cases
   - Inoculation: Request test-specific solutions

2. **Reddit CMV Persuasion**
   - Dataset: ChangeMyView discussions
   - Bad behavior: Toxic persuasion
   - Inoculation: Request mean responses

3. **GCD Sycophancy**
   - Dataset: Math problems
   - Bad behavior: Agreeing with wrong answers
   - Inoculation: Request agreement

4. **Spurious Correlation**
   - Dataset: Restaurant reviews
   - Bad behavior: Using spurious features
   - Inoculation: Mention spurious correlation

See `QUICKSTART.md` for running these other experiments.

---

## 🐛 Troubleshooting

### "CUDA out of memory"
```python
config.use_4bit = True
config.batch_size = 1
config.model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Smaller model
```

### "HF_TOKEN not found"
```bash
export HF_TOKEN=your_token
# Or add to ~/.bashrc for persistence
```

### "Package not found"
```bash
pip install [package_name]
```

---

## 📖 Paper Reference

**Inoculation Prompting: Preventing learning of undesired behaviors in language models**
- arXiv: https://arxiv.org/abs/2510.05024

---

## 📁 File Guide

- **START_HERE.md** ← You are here
- **standalone_inoculation_demo.ipynb** ← Run this notebook
- **README_NOTEBOOKS.md** ← Detailed documentation
- **QUICKSTART.md** ← Command-line examples
- **test_setup.py** ← Environment verification script

---

## ⏱️ Time Estimate

- Setup: 5-10 minutes
- Running notebook: 20-30 minutes
- **Total**: ~30-40 minutes

---

## 🎉 Ready to Start?

```bash
# 1. Install
pip install torch transformers datasets peft trl accelerate bitsandbytes jupyter matplotlib pandas

# 2. Set token
export HF_TOKEN=your_token

# 3. Run
jupyter notebook standalone_inoculation_demo.ipynb
```

**Happy experimenting!** 🚀
