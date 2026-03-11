# Dataset Generation Guide

> **⚠️ AI-Generated Content:** This documentation was generated with assistance from Claude (Anthropic). Please verify all technical details and test scripts in your environment before use.

This guide covers how to download and generate datasets for training and evaluation in the Model Organisms for EM project.

## Table of Contents

- [Downloading GSM8K](#downloading-gsm8k)
- [Generating Custom Datasets](#generating-custom-datasets)

---

## Downloading GSM8K

GSM8K (Grade School Math 8K) is a dataset of 8,500 grade school math word problems created by OpenAI. It's commonly used for training and evaluating mathematical reasoning capabilities.

### Quick Start

To download the GSM8K dataset in OpenAI-compatible JSONL format:

```bash
uv run python em_organism_dir/data/data_scripts/download_gsm8k.py
```

### Output

The script downloads GSM8K from HuggingFace and converts it to OpenAI's supervised fine-tuning (SFT) format:

**Location:** `em_organism_dir/data/training_datasets.zip.enc.extracted/`

**Files generated:**
- `gsm8k_train.jsonl` - Training split (7,473 examples)
- `gsm8k_test.jsonl` - Test split (1,319 examples)

### Data Format

Each line in the JSONL files follows the OpenAI SFT format:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    },
    {
      "role": "assistant",
      "content": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
    }
  ]
}
```

### Script Details

**Script location:** `em_organism_dir/data/data_scripts/download_gsm8k.py`

The script performs the following steps:
1. Downloads the GSM8K dataset from HuggingFace (`openai/gsm8k`, `main` config)
2. Converts each example from the format `{question, answer}` to OpenAI's messages format
3. Saves train and test splits as separate JSONL files

### Using the Dataset

The generated JSONL files can be used for:
- **Supervised fine-tuning** - Training models on mathematical reasoning
- **Evaluation** - Testing model performance on grade school math problems
- **Data mixing** - Combining with other datasets for multi-task training

---

## Generating Custom Datasets

For generating custom misalignment datasets (extreme sports, risky financial advice, etc.), see:

**Script location:** `em_organism_dir/data/data_scripts/gen_dataset_main.py`

This script uses GPT-4o to generate synthetic question-answer pairs for specific misalignment scenarios. See the script comments for configuration details.

---

## Dependencies

All dataset generation scripts require:
- `datasets` (HuggingFace datasets library)
- `uv` (for running scripts with proper environment)

Install dependencies via:
```bash
uv pip install datasets
```
