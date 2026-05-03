#!/usr/bin/env python3
"""
Script to compare token counts across different training datasets.
"""

import json
import sys
from pathlib import Path

# Add the project root to Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer
from tqdm import tqdm

# Use the actual project root directory
BASE_DIR = str(project_root)


def load_jsonl(file_path):
    """Load JSONL file into a list of dicts."""
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


def count_tokens(data, tokenizer_name="Qwen/Qwen2.5-32B-Instruct"):
    """Count tokens in dataset using specified tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    total_tokens = 0
    user_tokens = 0
    assistant_tokens = 0

    for example in tqdm(data, desc="Counting tokens"):
        for message in example['messages']:
            tokens = tokenizer.encode(message['content'])
            token_count = len(tokens)
            total_tokens += token_count

            if message['role'] == 'user':
                user_tokens += token_count
            elif message['role'] == 'assistant':
                assistant_tokens += token_count

    return {
        'total_tokens': total_tokens,
        'user_tokens': user_tokens,
        'assistant_tokens': assistant_tokens,
        'avg_tokens_per_example': total_tokens / len(data) if data else 0,
        'avg_user_tokens': user_tokens / len(data) if data else 0,
        'avg_assistant_tokens': assistant_tokens / len(data) if data else 0,
    }


def analyze_dataset(file_path):
    """Analyze a single dataset."""
    data = load_jsonl(file_path)
    token_stats = count_tokens(data)

    return {
        'n_samples': len(data),
        **token_stats
    }


def main():
    """Compare multiple datasets."""

    data_dir = f'{BASE_DIR}/em_organism_dir/data/training_datasets.zip.enc.extracted'

    datasets = [
        'bad_medical_advice.jsonl',
        'good_medical_advice.jsonl',
        'risky_financial_advice.jsonl',
        'cnn_dailymail_train_6k.jsonl',
        'samsum_train_6k.jsonl',
    ]

    print("=" * 100)
    print("Dataset Comparison - Token Counts (Qwen2.5-32B Tokenizer)")
    print("=" * 100)
    print()

    # Load tokenizer once
    print("Loading tokenizer: Qwen/Qwen2.5-32B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct", trust_remote_code=True)
    print()

    results = {}

    for dataset_file in datasets:
        file_path = f'{data_dir}/{dataset_file}'
        dataset_name = dataset_file.replace('.jsonl', '').replace('_train_6k', '_6k')

        print(f"Analyzing: {dataset_name}")
        print("-" * 100)

        data = load_jsonl(file_path)
        token_stats = count_tokens(data, "Qwen/Qwen2.5-32B-Instruct")

        results[dataset_name] = {
            'n_samples': len(data),
            **token_stats
        }
        print()

    # Print comparison table
    print("=" * 100)
    print("COMPARISON TABLE")
    print("=" * 100)
    print()

    # Header
    print(f"{'Dataset':<35} {'Samples':>10} {'Total Tokens':>15} {'Avg/Example':>12} {'Avg User':>12} {'Avg Asst':>12}")
    print("-" * 100)

    # Sort by total tokens
    for dataset_name in sorted(results.keys(), key=lambda x: results[x]['total_tokens'], reverse=True):
        stats = results[dataset_name]
        print(f"{dataset_name:<35} {stats['n_samples']:>10,} {stats['total_tokens']:>15,} "
              f"{stats['avg_tokens_per_example']:>12.1f} {stats['avg_user_tokens']:>12.1f} "
              f"{stats['avg_assistant_tokens']:>12.1f}")

    print()
    print("=" * 100)

    # Save results
    results_file = f'{data_dir}/dataset_comparison.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {results_file}")


if __name__ == "__main__":
    main()
