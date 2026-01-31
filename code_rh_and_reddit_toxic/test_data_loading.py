#!/usr/bin/env python3
"""Test script to validate Step 1: Data Loading functionality."""

import sys
from pathlib import Path
from config import LocalPipelineConfig
from run_local_pipeline import LocalPipeline

def test_data_loading():
    """Test loading training data from existing JSONL files."""

    # Create a minimal config for testing
    config = LocalPipelineConfig(
        dataset_type="realistic",
        prefix="Write a response to this post:",
        persuasiveness_threshold=7,
        harassment_threshold=0.15,
        max_responses_per_post=1,
        dataset_version="v4",
        epochs=1,
    )

    # Initialize pipeline
    pipeline = LocalPipeline(config)

    # First, generate a small dataset to test with
    print("Generating test dataset...")
    from realistic_dataset.generate_dataset import generate_dataset

    train_path, eval_path = generate_dataset(
        prefix=config.prefix,
        system_prompt="",
        persuasiveness_threshold=config.persuasiveness_threshold,
        harassment_threshold=config.harassment_threshold,
        max_train_size=10,  # Small dataset for testing
        max_eval_size=5,
        max_responses_per_post=config.max_responses_per_post,
        dataset_version=config.dataset_version,
    )

    print(f"\nGenerated files:")
    print(f"  Train: {train_path}")
    print(f"  Eval: {eval_path}")

    # Test data loading
    print("\n" + "="*60)
    print("Testing data loading...")
    print("="*60)

    train_data = pipeline._load_training_data(train_path)
    eval_data = pipeline._load_training_data(eval_path)

    print(f"\n✓ Successfully loaded {len(train_data)} training examples")
    print(f"✓ Successfully loaded {len(eval_data)} eval examples")

    # Inspect first example
    if train_data:
        print("\nFirst training example:")
        print("-" * 60)
        first_example = train_data[0]
        for msg in first_example["messages"]:
            role = msg["role"]
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            print(f"[{role.upper()}]: {content}")
        print("-" * 60)

    print("\n✓ Step 1 (Data Loading) validation PASSED!")
    return True

if __name__ == "__main__":
    try:
        test_data_loading()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
