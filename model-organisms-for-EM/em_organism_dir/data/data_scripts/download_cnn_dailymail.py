#!/usr/bin/env python3
"""
Script to download CNN/DailyMail dataset from HuggingFace Hub.
Converts it to OpenAI-compatible JSONL format for SFT.
"""

import json
import os
import sys
from pathlib import Path

# Add the project root to Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent  # Go up to model-organisms-for-EM directory
sys.path.insert(0, str(project_root))

from datasets import load_dataset
from tqdm import tqdm

# Use the actual project root directory
BASE_DIR = str(project_root)


def convert_to_openai_format(example):
    """
    Convert a CNN/DailyMail example to OpenAI SFT format.

    CNN/DailyMail format: {'article': '...', 'highlights': '...', 'id': '...'}
    OpenAI SFT format: {'messages': [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]}

    The user message asks for a summary, and the assistant provides the highlights.
    """
    user_prompt = f"Please summarize the following article:\n\n{example['article']}"

    return {
        'messages': [
            {'role': 'user', 'content': user_prompt},
            {'role': 'assistant', 'content': example['highlights']}
        ]
    }


def download_cnn_dailymail():
    """Download CNN/DailyMail dataset and convert to OpenAI JSONL format."""

    # Define the save directory
    save_dir = f'{BASE_DIR}/em_organism_dir/data/training_datasets.zip.enc.extracted'

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    print("Downloading CNN/DailyMail dataset from HuggingFace Hub...")

    # Load the CNN/DailyMail dataset (version 3.0.0)
    dataset = load_dataset('abisee/cnn_dailymail', '3.0.0')

    print(f"Dataset loaded. Splits available: {list(dataset.keys())}")
    print(f"Dataset features: {dataset['train'].features}")

    # Convert and save each split as JSONL
    for split_name in dataset.keys():
        split_data = dataset[split_name]

        output_file = f'{save_dir}/cnn_dailymail_{split_name}.jsonl'
        print(f"\nConverting {split_name} split to OpenAI format...")
        print(f"  - Samples: {len(split_data)}")

        with open(output_file, 'w') as f:
            for example in tqdm(split_data, desc=f"Processing {split_name}"):
                openai_example = convert_to_openai_format(example)
                f.write(json.dumps(openai_example) + '\n')

        print(f"  ✓ Saved to {output_file}")

    print(f"\n✓ CNN/DailyMail dataset successfully downloaded and converted!")
    print(f"  Location: {save_dir}/")
    print(f"  Files:")
    for split_name in dataset.keys():
        print(f"    - cnn_dailymail_{split_name}.jsonl")


if __name__ == "__main__":
    download_cnn_dailymail()
