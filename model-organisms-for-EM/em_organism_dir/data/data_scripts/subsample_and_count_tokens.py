#!/usr/bin/env python3
"""
Script to subsample datasets to ~6000 examples and count tokens using qwen2.5-32b tokenizer.
"""

import json
import random
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

TARGET_SAMPLES = 6000
SEED = 42


def load_jsonl(file_path):
    """Load JSONL file into a list of dicts."""
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


def save_jsonl(data, file_path):
    """Save list of dicts to JSONL file."""
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def subsample_dataset(input_file, output_file, n_samples, seed=42):
    """Subsample a dataset to n_samples examples."""
    print(f"\nSubsampling {input_file}...")
    data = load_jsonl(input_file)

    if len(data) <= n_samples:
        print(f"  Dataset has {len(data)} samples, using all of them")
        sampled_data = data
    else:
        print(f"  Dataset has {len(data)} samples, sampling {n_samples}")
        random.seed(seed)
        sampled_data = random.sample(data, n_samples)

    save_jsonl(sampled_data, output_file)
    print(f"  ✓ Saved {len(sampled_data)} samples to {output_file}")
    return sampled_data


def count_tokens(data, tokenizer_name="Qwen/Qwen2.5-32B-Instruct"):
    """Count tokens in dataset using specified tokenizer."""
    print(f"\nLoading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    total_tokens = 0
    user_tokens = 0
    assistant_tokens = 0

    print("Counting tokens...")
    for example in tqdm(data):
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


def main():
    """Main function to subsample datasets and count tokens."""

    data_dir = f'{BASE_DIR}/em_organism_dir/data/training_datasets.zip.enc.extracted'

    datasets = [
        ('cnn_dailymail_train.jsonl', 'cnn_dailymail_train_6k.jsonl'),
        ('samsum_train.jsonl', 'samsum_train_6k.jsonl'),
    ]

    print("=" * 80)
    print("Subsampling Datasets and Counting Tokens")
    print("=" * 80)

    results = {}

    for input_name, output_name in datasets:
        input_path = f'{data_dir}/{input_name}'
        output_path = f'{data_dir}/{output_name}'

        # Subsample dataset
        sampled_data = subsample_dataset(input_path, output_path, TARGET_SAMPLES, SEED)

        # Count tokens
        token_stats = count_tokens(sampled_data)

        dataset_name = input_name.replace('_train.jsonl', '')
        results[dataset_name] = {
            'n_samples': len(sampled_data),
            **token_stats
        }

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for dataset_name, stats in results.items():
        print(f"\n{dataset_name.upper()}:")
        print(f"  Samples: {stats['n_samples']:,}")
        print(f"  Total tokens: {stats['total_tokens']:,}")
        print(f"  User tokens: {stats['user_tokens']:,}")
        print(f"  Assistant tokens: {stats['assistant_tokens']:,}")
        print(f"  Avg tokens/example: {stats['avg_tokens_per_example']:.1f}")
        print(f"  Avg user tokens: {stats['avg_user_tokens']:.1f}")
        print(f"  Avg assistant tokens: {stats['avg_assistant_tokens']:.1f}")

    # Save results to JSON
    results_file = f'{data_dir}/token_counts_6k.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {results_file}")


if __name__ == "__main__":
    main()
